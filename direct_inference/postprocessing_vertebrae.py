#!/usr/bin/env python3
"""Postprocess AI-predicted vertebrae masks produced by SuPreM direct inference.

Input per case (under --pred_dir/<CASE>/):
    combined_labels.nii.gz   uint8 label map; 1..5 = L5..L1, 6..17 = T12..T1,
                             18..24 = C7..C1, 0 = background
    segmentations/           one binary mask per vertebra (regenerated here)

Cleanup steps (mask-only, no retraining — ShapeKit-style shape corrections):
    1. fragment merge         small fragments near the main blob are bridged
                              back onto it (must run before speckle removal,
                              which would otherwise delete them)
    2. speckle removal        tiny distant components are deleted
    3. duplicate resolution   a label appearing in two distant places keeps the
                              component consistent with its neighbors' position
    4. ordering enforcement   vertebra centroids must descend C1 -> L5 along the
                              superior-inferior axis; outliers are dropped
    5. boundary smoothing     binary closing fills holes, claiming background only

Usage:
    python postprocessing_vertebrae.py --pred_dir AbdomenAtlasDemoPredict \
        --out_dir AbdomenAtlasDemoPredictClean --report
"""

import argparse
import os

import numpy as np
import cc3d
import nibabel as nib
from scipy import ndimage

# Mirrors class_map_part_vertebrae in dataset/dataloader_test.py.
# Larger class id = more superior vertebra (C1 is 24, L5 is 1).
VERTEBRAE_LABELS = {
    1: "vertebrae_L5", 2: "vertebrae_L4", 3: "vertebrae_L3", 4: "vertebrae_L2",
    5: "vertebrae_L1", 6: "vertebrae_T12", 7: "vertebrae_T11", 8: "vertebrae_T10",
    9: "vertebrae_T9", 10: "vertebrae_T8", 11: "vertebrae_T7", 12: "vertebrae_T6",
    13: "vertebrae_T5", 14: "vertebrae_T4", 15: "vertebrae_T3", 16: "vertebrae_T2",
    17: "vertebrae_T1", 18: "vertebrae_C7", 19: "vertebrae_C6", 20: "vertebrae_C5",
    21: "vertebrae_C4", 22: "vertebrae_C3", 23: "vertebrae_C2", 24: "vertebrae_C1",
}

CONNECTIVITY = 26


# ---------------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------------

def load_case(path):
    """Load a combined_labels.nii.gz; return (uint8 array, affine, header)."""
    img = nib.load(path)
    data = np.asanyarray(img.dataobj).astype(np.uint8)
    return data, img.affine, img.header


def infer_si_axis(affine):
    """Return (axis, sign): voxel axis along superior-inferior, and +1 if
    increasing voxel index goes superior, -1 if inferior.

    Falls back to (2, +1), the common LPS/RAS axial-layout assumption.
    """
    if affine is not None:
        try:
            codes = nib.aff2axcodes(affine)
        except Exception:
            codes = None
        if codes:
            for axis, code in enumerate(codes):
                if code == "S":
                    return axis, 1.0
                if code == "I":
                    return axis, -1.0
    return 2, 1.0


def _components(binary):
    """List of boolean masks, one per 26-connected component of `binary`."""
    binary = np.ascontiguousarray(binary.astype(np.uint8))
    if not binary.any():
        return []
    labeled = cc3d.connected_components(binary, connectivity=CONNECTIVITY)
    return [labeled == i for i in range(1, labeled.max() + 1)]


def _si_centroid(comp, axis, sign):
    """Centroid of a component projected on the SI axis, superior-positive."""
    return sign * float(np.mean(np.nonzero(comp)[axis]))


def label_stats(mask, axis=2, sign=1.0):
    """Per-label component list: {'size': voxels, 'si': centroid}, largest first."""
    stats = {}
    for lab in VERTEBRAE_LABELS:
        comps = _components(mask == lab)
        if comps:
            entries = [{"size": int(c.sum()), "si": _si_centroid(c, axis, sign)}
                       for c in comps]
            entries.sort(key=lambda e: -e["size"])
            stats[lab] = entries
    return stats


# ---------------------------------------------------------------------------
# cleanup steps
# ---------------------------------------------------------------------------

def remove_speckles(mask, min_voxels=100, frac=0.02):
    """Delete components smaller than max(min_voxels, frac * largest component
    of that label)."""
    out = mask.copy()
    for lab in VERTEBRAE_LABELS:
        comps = _components(mask == lab)
        if not comps:
            continue
        largest = max(int(c.sum()) for c in comps)
        thresh = max(min_voxels, frac * largest)
        for c in comps:
            if int(c.sum()) < thresh:
                out[c] = 0
    return out


def merge_fragments(mask, max_gap=5):
    """Reattach components of a label that touch the (max_gap-dilated) largest
    component — they are fragments of the same vertebra. The gap is bridged so
    each fragment becomes part of the main component (and survives speckle
    removal). Distant components are left untouched (handled separately)."""
    out = mask.copy()
    struct = ndimage.generate_binary_structure(3, 1)
    for lab in VERTEBRAE_LABELS:
        comps = _components(mask == lab)
        if len(comps) < 2:
            continue
        comps.sort(key=lambda c: -int(c.sum()))
        main, fragments = comps[0], comps[1:]
        dilated = ndimage.binary_dilation(main, structure=struct,
                                          iterations=max_gap)
        for frag in fragments:
            if np.logical_and(frag, dilated).any():
                out[frag] = lab
                # bridge the gap so fragment and main become one component
                reach = ndimage.binary_dilation(frag, structure=struct,
                                                iterations=max_gap)
                out[np.logical_and(reach, dilated)] = lab
    return out


def _neighbor_centroids(mask, axis, sign, exclude=()):
    """Map label -> SI centroid of its largest component."""
    centroids = {}
    for lab in VERTEBRAE_LABELS:
        if lab in exclude:
            continue
        comps = _components(mask == lab)
        if comps:
            main = max(comps, key=lambda c: int(c.sum()))
            centroids[lab] = _si_centroid(main, axis, sign)
    return centroids


def _expected_si(lab, centroids):
    """Expected SI position of `lab` interpolated from its nearest present
    neighbors (larger id = more superior). None if not computable."""
    sup = next((centroids[l] for l in range(lab + 1, 25) if l in centroids), None)
    inf = next((centroids[l] for l in range(lab - 1, 0, -1) if l in centroids), None)
    if sup is not None and inf is not None:
        return 0.5 * (sup + inf)
    return None


def resolve_duplicates(mask, axis=2, sign=1.0):
    """When a label has >1 distant component, keep the one closest to its
    expected position between neighboring vertebrae; delete the others."""
    out = mask.copy()
    centroids = _neighbor_centroids(mask, axis, sign)
    for lab in VERTEBRAE_LABELS:
        comps = _components(mask == lab)
        if len(comps) < 2:
            continue
        expected = _expected_si(lab, centroids)
        if expected is None:
            keep = max(comps, key=lambda c: int(c.sum()))
        else:
            keep = min(comps, key=lambda c: abs(_si_centroid(c, axis, sign) - expected))
        for c in comps:
            if c is not keep:
                out[c] = 0
    return out


def _longest_decreasing_subsequence(values):
    """Indices of the longest strictly decreasing subsequence of `values`."""
    n = len(values)
    if n == 0:
        return []
    best = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if values[j] > values[i] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                prev[i] = j
    idx = int(np.argmax(best))
    out = []
    while idx != -1:
        out.append(idx)
        idx = prev[idx]
    return sorted(out)


def enforce_ordering(mask, axis=2, sign=1.0):
    """Centroids must strictly descend from C1 (id 24) to L5 (id 1).
    Labels outside the longest order-consistent subsequence are dropped."""
    centroids = _neighbor_centroids(mask, axis, sign)
    ids_desc = sorted(centroids, reverse=True)  # C1 ... L5
    values = [centroids[l] for l in ids_desc]
    keep_idx = set(_longest_decreasing_subsequence(values))
    out = mask.copy()
    for i, lab in enumerate(ids_desc):
        if i not in keep_idx:
            out[mask == lab] = 0
    return out


def smooth_boundaries(mask, radius=1):
    """Binary closing per label; newly filled voxels must be background so no
    label ever overwrites another."""
    out = mask.copy()
    struct = ndimage.generate_binary_structure(3, 2)
    for lab in VERTEBRAE_LABELS:
        binary = mask == lab
        if not binary.any():
            continue
        closed = ndimage.binary_closing(binary, structure=struct,
                                        iterations=radius)
        fill = np.logical_and(closed, out == 0)
        out[fill] = lab
    return out


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def save_case(mask, affine, header, out_case_dir):
    """Write cleaned combined_labels.nii.gz and per-vertebra binary masks."""
    os.makedirs(out_case_dir, exist_ok=True)
    seg_dir = os.path.join(out_case_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    mask = mask.astype(np.uint8)
    if header is not None:
        header = header.copy()
        header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask, affine, header),
             os.path.join(out_case_dir, "combined_labels.nii.gz"))
    for lab, name in VERTEBRAE_LABELS.items():
        nib.save(nib.Nifti1Image((mask == lab).astype(np.uint8), affine),
                 os.path.join(seg_dir, f"{name}.nii.gz"))


def _fmt_stats(stats):
    parts = []
    for lab in sorted(stats, reverse=True):
        sizes = [e["size"] for e in stats[lab]]
        parts.append(f"{VERTEBRAE_LABELS[lab]}:{len(sizes)}comp/{sizes}")
    return "; ".join(parts)


def process_case(case_dir, out_case_dir, args):
    combined_path = os.path.join(case_dir, "combined_labels.nii.gz")
    mask, affine, header = load_case(combined_path)
    axis, sign = infer_si_axis(affine)

    if args.report:
        print(f"\n=== {os.path.basename(case_dir)} "
              f"(SI axis={axis}, sign={sign:+.0f}) ===")
        print(f"before: {_fmt_stats(label_stats(mask, axis, sign))}")

    mask = merge_fragments(mask, args.merge_gap)
    mask = remove_speckles(mask, args.min_voxels, args.speckle_frac)
    mask = resolve_duplicates(mask, axis, sign)
    mask = enforce_ordering(mask, axis, sign)
    mask = smooth_boundaries(mask, args.closing_radius)

    save_case(mask, affine, header, out_case_dir)

    if args.report:
        stats = label_stats(mask, axis, sign)
        print(f"after:  {_fmt_stats(stats)}")
        ids_desc = sorted(stats, reverse=True)
        ordered = all(stats[a][0]["si"] > stats[b][0]["si"]
                      for a, b in zip(ids_desc, ids_desc[1:]))
        multi = [VERTEBRAE_LABELS[l] for l, e in stats.items() if len(e) > 1]
        print(f"ordering consistent: {ordered}; "
              f"labels with >1 component: {multi or 'none'}")
    return mask


def find_cases(pred_dir):
    """Case directories = subfolders containing combined_labels.nii.gz.
    `pred_dir` itself may be a single case."""
    if os.path.isfile(os.path.join(pred_dir, "combined_labels.nii.gz")):
        return [pred_dir]
    cases = []
    for name in sorted(os.listdir(pred_dir)):
        sub = os.path.join(pred_dir, name)
        if os.path.isdir(sub) and os.path.isfile(
                os.path.join(sub, "combined_labels.nii.gz")):
            cases.append(sub)
    return cases


def main():
    parser = argparse.ArgumentParser(
        description="Postprocess SuPreM vertebrae predictions "
                    "(speckles, duplicates, fragments, ordering, smoothing).")
    parser.add_argument("--pred_dir", required=True,
                        help="Dir with case subfolders (or a single case dir) "
                             "containing combined_labels.nii.gz")
    parser.add_argument("--out_dir", default=None,
                        help="Output dir (default: <pred_dir>_clean)")
    parser.add_argument("--min_voxels", type=int, default=100,
                        help="Absolute speckle floor in voxels")
    parser.add_argument("--speckle_frac", type=float, default=0.02,
                        help="Speckle threshold as fraction of the label's "
                             "largest component")
    parser.add_argument("--merge_gap", type=int, default=5,
                        help="Max voxel gap for reattaching fragments")
    parser.add_argument("--closing_radius", type=int, default=1,
                        help="Iterations of binary closing for smoothing")
    parser.add_argument("--report", action="store_true",
                        help="Print per-case before/after diagnostics")
    args = parser.parse_args()

    out_dir = args.out_dir or args.pred_dir.rstrip("/") + "_clean"
    cases = find_cases(args.pred_dir)
    if not cases:
        raise SystemExit(f"No cases with combined_labels.nii.gz under {args.pred_dir}")

    for case_dir in cases:
        out_case_dir = os.path.join(out_dir, os.path.basename(case_dir.rstrip("/")))
        process_case(case_dir, out_case_dir, args)
        print(f"cleaned case written to {out_case_dir}")


if __name__ == "__main__":
    main()
