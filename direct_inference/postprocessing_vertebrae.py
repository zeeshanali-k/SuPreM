#!/usr/bin/env python3
"""Postprocess AI-predicted vertebrae masks produced by SuPreM direct inference.

Input per case (under --pred_dir/<CASE>/):
    combined_labels.nii.gz   uint8 label map; 1..5 = L5..L1, 6..17 = T12..T1,
                             18..24 = C7..C1, 0 = background
    segmentations/           one binary mask per vertebra (regenerated here)

Design follows the related work listed in vertebrae.md:

  * ShapeKit (Liu et al., 2025): shape-error taxonomy — speckles, false
    positives, fragmented structures, redundant lobes — fixed with fast
    morphological/topological operators, no retraining.
  * Meng et al., 2023: vertebra *identification* is a global problem.
    Consecutive vertebrae must carry consecutive labels; a shortest-path
    search over candidate components enforces this and RELABELS misplaced
    components instead of deleting them (their graph optimization lifted
    identification accuracy from 84.6% to 97.4%).
  * Jaus et al., 2024 (DAP-Atlas post-processing): fragments of one
    vertebra embedded in a neighbor are reassigned to that neighbor
    (spine_adjacent_pairs), and anatomy-based rules (body-region limits)
    suppress implausible voxels.

Cleanup steps (in order):
    1. bone gating (optional, needs --ct_dir)
                             drop predicted voxels whose CT intensity is below
                             --hu_threshold (interspinous soft tissue, disc and
                             muscle bleed), then refill enclosed holes
    2. rib tail trim         thoracic vertebrae must stay within
                             --rib_radius_mm of the spine centerline; rib
                             segments absorbed into vertebra labels are cut
    3. speckle removal       tiny distant components are deleted
    4. stack identification  dynamic-programming shortest path (Meng-style
                             graph optimization) assigns every candidate
                             component an anatomical index; misplaced labels
                             are RELABELED, never dropped; components that
                             cannot fit the stack are reassigned to the
                             nearest neighboring vertebra (Jaus-style) or, if
                             isolated and tiny, set to background
    5. superior overflow trim
                             spinous processes point DOWN, so a vertebra may
                             extend inferiorly toward the next level but not
                             superiorly past its upper neighbor; voxels of a
                             label lying above (midpoint + margin) of the gap
                             to the upper neighbor are moved to that neighbor
                             when they touch its mass, kept otherwise (they
                             are usually the vertebra's own superior facets)
    6. boundary smoothing    binary closing fills holes, claiming background
                             only (never steals voxels from another label)

Usage:
    python postprocessing_vertebrae.py \
        --pred_dir AbdomenAtlasDemoPredict \
        --ct_dir AbdomenAtlasDemo \
        --out_dir AbdomenAtlasDemoPredictClean --report
"""

import argparse
import os

import numpy as np
import cc3d
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree

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

# Thoracic labels (ribs articulate only with thoracic vertebrae).
THORACIC_IDS = set(range(6, 18))

# Loose plausible full-vertebra volume ranges in cm^3, used only for
# report warnings (never for automatic deletion).
PLAUSIBLE_CM3 = {
    "C": (5.0, 35.0),   # C1..C7  (ids 18..24)
    "T": (10.0, 50.0),  # T1..T12 (ids 6..17)
    "L": (22.0, 95.0),  # L1..L5  (ids 1..5)
}

# 26-connectivity: voxels touching at a face, edge, or corner count as one
# component. This is the standard choice for 3D label cleanup.
CONNECTIVITY = 26


# ---------------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------------

def load_case(case_dir, ct_dir=None):
    """Load the predicted label map (and optionally the CT) of one case.

    Args:
        case_dir: Directory containing ``combined_labels.nii.gz``.
        ct_dir: Optional root with ``<CASE>/ct.nii.gz``; when given and the
            file exists, the CT is loaded for bone gating.

    Returns:
        tuple ``(data, affine, header, ct, case_name)``. ``data`` is a uint8
        numpy array of class ids (0 = background, 1..24 = L5..C1). ``ct`` is
        a float32 array of Hounsfield units or ``None``. The affine and
        header are carried through unchanged so the cleaned output stays
        aligned with the original CT in viewers such as ITK-SNAP.
    """
    combined_path = os.path.join(case_dir, "combined_labels.nii.gz")
    img = nib.load(combined_path)
    data = np.asanyarray(img.dataobj).astype(np.uint8)
    case_name = os.path.basename(case_dir.rstrip("/"))
    ct = None
    if ct_dir:
        ct_path = os.path.join(ct_dir, case_name, "ct.nii.gz")
        if os.path.isfile(ct_path):
            ct = np.asanyarray(nib.load(ct_path).dataobj).astype(np.float32)
            if ct.shape != data.shape:
                print(f"  WARNING: CT shape {ct.shape} != label shape "
                      f"{data.shape} for {case_name}; bone gating skipped")
                ct = None
        else:
            print(f"  WARNING: no CT at {ct_path}; bone gating skipped")
    return data, img.affine, img.header, ct, case_name


def infer_si_axis(affine):
    """Find the superior-inferior (head-to-feet) voxel axis and its direction.

    Args:
        affine: 4x4 voxel-to-world matrix, or ``None`` if unavailable.

    Returns:
        tuple ``(axis, sign)``: ``axis`` is the voxel axis index along which
        the spine is stacked; ``sign`` is ``+1.0`` when increasing voxel
        index goes superior (toward the head) and ``-1.0`` when it goes
        inferior. Falls back to ``(2, +1.0)`` — the common axial CT layout.
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


def get_spacing(header, affine):
    """Voxel spacing in mm as a length-3 array (from header, else affine)."""
    if header is not None:
        try:
            sp = np.array(header.get_zooms()[:3], dtype=float)
            if np.all(sp > 0):
                return sp
        except Exception:
            pass
    if affine is not None:
        return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return np.ones(3)


def _crop_slices(binary, pad=3):
    """Bounding-box slices of a boolean array, enlarged by ``pad`` voxels."""
    idx = np.nonzero(binary)
    lo = [max(int(i.min()) - pad, 0) for i in idx]
    hi = [min(int(i.max()) + pad + 1, binary.shape[d])
          for d, i in enumerate(idx)]
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def _component_coords(binary):
    """Split a binary mask into its 26-connected components.

    Works on the mask's bounding box so even hundreds of components on a
    large volume stay cheap in memory (no full-size array per component).

    Returns:
        List of (N, 3) int32 arrays of absolute voxel coordinates, one per
        connected component.
    """
    binary = np.ascontiguousarray(binary.astype(np.uint8))
    if not binary.any():
        return []
    sl = _crop_slices(binary)
    labeled = cc3d.connected_components(binary[sl], connectivity=CONNECTIVITY)
    origin = np.array([s.start for s in sl], dtype=np.int32)
    out = []
    for i in range(1, labeled.max() + 1):
        coords = np.stack(np.nonzero(labeled == i), axis=1) + origin
        out.append(coords.astype(np.int32))
    return out


def _si_centroid_coords(coords, axis, sign):
    """Superior-positive SI centroid from an (N, 3) voxel-coordinate array."""
    return sign * float(coords[:, axis].mean())


def _label_centroid(mask, lab, axis, sign):
    """Superior-positive SI centroid of a label's voxels, or None."""
    coords = np.nonzero(mask == lab)
    if len(coords[0]) == 0:
        return None
    return sign * float(coords[axis].mean())


# ---------------------------------------------------------------------------
# step 1: bone gating (CT-guided)
# ---------------------------------------------------------------------------

def gate_to_bone(mask, ct, hu_threshold=130, reattach_mm=3.0, spacing=None):
    """Restrict each vertebra label to bony CT voxels.

    The network sometimes paints interspinous soft tissue, discs, or muscle
    into a vertebra mask (ShapeKit "false positives"). Those tissues sit
    below ~130 HU while cortical and trabecular bone stay above it, so an
    intensity gate removes them. Enclosed interior holes (fatty marrow can
    dip below the threshold) are refilled afterwards, and small components
    that remain within ``reattach_mm`` of the main component are kept
    (thin cortical shells can fragment after gating).

    Only voxels OUTSIDE the bone gate are removed; no voxels are added
    beyond hole filling, so recall on the true vertebra is preserved.

    Args:
        mask: uint8 3D label map.
        ct: float32 3D CT array in Hounsfield units, same shape as ``mask``.
        hu_threshold: Intensity floor in HU (default 130).
        reattach_mm: Components of a label within this distance of the
            label's largest component are kept.
        spacing: Voxel spacing in mm.

    Returns:
        Gated copy of ``mask``.
    """
    out = np.zeros_like(mask)
    it = max(1, int(round(reattach_mm / float(np.min(spacing))))) \
        if spacing is not None else 2
    for lab in VERTEBRAE_LABELS:
        binary = mask == lab
        if not binary.any():
            continue
        gated = np.logical_and(binary, ct >= hu_threshold)
        if not gated.any():
            out[binary] = lab  # gate removed everything: keep original
            continue
        sl = _crop_slices(gated)
        crop = ndimage.binary_fill_holes(gated[sl])
        cc, n = ndimage.label(crop)
        if n > 1:
            sizes = ndimage.sum(crop, cc, range(1, n + 1))
            main = int(np.argmax(sizes)) + 1
            dil = ndimage.binary_dilation(cc == main, iterations=it)
            keep = np.zeros_like(crop)
            for i in range(1, n + 1):
                if i == main or np.logical_and(dil, cc == i).any():
                    keep |= cc == i
            crop = keep
        out[sl][crop] = lab
    return out


# ---------------------------------------------------------------------------
# step 2: rib tail trim
# ---------------------------------------------------------------------------

def trim_rib_tails(mask, axis, sign, spacing, radius_mm=40.0):
    """Cut rib segments absorbed into thoracic vertebra labels.

    Ribs articulate with the thoracic vertebrae, and the model regularly
    drags vertebra labels along the rib bone (ShapeKit "redundant
    structures": extensions into anatomically implausible regions). A
    thoracic vertebra including transverse processes stays within ~35 mm of
    the spine centerline, so label voxels farther than ``radius_mm`` from
    the interpolated centerline are removed. Only thoracic labels
    (T1..T12) are trimmed: cervical transverse foramina and lumbar
    transverse processes are legitimate lateral extensions, and ribs exist
    only at thoracic levels.

    The centerline is estimated per case from the (x, y) positions of every
    label's centroid, linearly interpolated along the SI axis — robust to
    scoliosis and off-center patient positioning.

    Args:
        mask: uint8 3D label map.
        axis, sign: SI axis and direction from ``infer_si_axis``.
        spacing: Voxel spacing in mm (length 3).
        radius_mm: Maximum allowed radial distance from the centerline.

    Returns:
        Trimmed copy of ``mask``.
    """
    if radius_mm <= 0:
        return mask
    centroids = {}  # lab -> (si, [x, y] in mm)
    lat_axes = [a for a in range(3) if a != axis]
    for lab in VERTEBRAE_LABELS:
        coords = np.nonzero(mask == lab)
        if len(coords[0]) == 0:
            continue
        pts = np.stack(coords, axis=1).astype(float)
        centroids[lab] = (sign * pts[:, axis].mean(),
                          (pts[:, lat_axes].mean(axis=0))
                          * spacing[lat_axes])
    if len(centroids) < 2:
        return mask
    labs = sorted(centroids, key=lambda l: centroids[l][0])
    cl_si = np.array([centroids[l][0] for l in labs])
    cl_xy = np.stack([centroids[l][1] for l in labs])  # (n, 2) in mm

    out = mask.copy()
    for lab in THORACIC_IDS:
        coords = np.nonzero(mask == lab)
        if len(coords[0]) == 0:
            continue
        pts = np.stack(coords, axis=1)
        si_vox = sign * pts[:, axis].astype(float)
        cx = np.interp(si_vox, cl_si, cl_xy[:, 0])
        cy = np.interp(si_vox, cl_si, cl_xy[:, 1])
        r = np.sqrt((pts[:, lat_axes[0]] * spacing[lat_axes[0]] - cx) ** 2 +
                    (pts[:, lat_axes[1]] * spacing[lat_axes[1]] - cy) ** 2)
        cut = r > radius_mm
        if cut.any():
            out[tuple(pts[cut, d] for d in range(3))] = 0
    return out


# ---------------------------------------------------------------------------
# step 3: speckle removal
# ---------------------------------------------------------------------------

def remove_speckles(mask, min_voxels=100, frac=0.02):
    """Delete implausibly small components (ShapeKit "artifacts/speckles").

    The threshold is per label: the larger of an absolute floor and a
    fraction of that label's largest component (scales the rule from small
    cervical to large lumbar vertebrae).

    Args:
        mask: uint8 3D label map.
        min_voxels: Absolute component-size floor, in voxels.
        frac: Extra relative floor — components smaller than ``frac`` times
            the label's largest component are also deleted.

    Returns:
        Cleaned copy of ``mask``; small components set to 0 (background).
    """
    out = mask.copy()
    for lab in VERTEBRAE_LABELS:
        comps = _component_coords(mask == lab)
        if not comps:
            continue
        largest = max(len(c) for c in comps)
        thresh = max(min_voxels, frac * largest)
        for c in comps:
            if len(c) < thresh:
                out[tuple(c.T)] = 0
    return out


# ---------------------------------------------------------------------------
# step 4: global stack identification (Meng et al. graph optimization)
# ---------------------------------------------------------------------------

class Unit:
    """One candidate vertebra component (or a merged fragment group).

    Attributes:
        pred: Predicted label id (1..24) from the input mask.
        coords: (N, 3) int32 array of voxel coordinates.
        size: Voxel count.
        si: Superior-positive SI centroid (float, voxel units).
    """

    __slots__ = ("pred", "coords", "size", "si")

    def __init__(self, pred, coords, axis, sign):
        self.pred = pred
        self.coords = coords
        self.size = int(coords.shape[0])
        self.si = _si_centroid_coords(coords, axis, sign)


def build_units(mask, axis, sign, spacing, merge_gap_mm=8.0):
    """Collect candidate vertebra components for global identification.

    Per label, components within ``merge_gap_mm`` of the largest component
    are merged into one logical unit (a vertebra split into pieces —
    ShapeKit "fragmented structures" — should be identified as a whole;
    no gap filling is performed, the grouping is logical only). Distant
    components become separate units so the optimizer can reject or
    relabel them independently.

    Returns:
        List of ``Unit`` sorted by ascending SI centroid (feet -> head).
    """
    max_it = int(np.max(np.maximum(
        1, np.round(merge_gap_mm / spacing).astype(int))))
    struct = ndimage.generate_binary_structure(3, 1)
    units = []
    for lab in VERTEBRAE_LABELS:
        comps = _component_coords(mask == lab)
        if not comps:
            continue
        comps.sort(key=len, reverse=True)
        main, rest = comps[0], comps[1:]
        # "near" = any fragment voxel lands inside the dilation of main
        lo = np.maximum(main.min(axis=0) - (max_it + 1), 0)
        hi = np.minimum(main.max(axis=0) + max_it + 2, mask.shape)
        dil = np.zeros(tuple(hi - lo), dtype=bool)
        main_local = tuple((main - lo).T)
        dil[main_local] = True
        dil = ndimage.binary_dilation(dil, structure=struct,
                                      iterations=max_it)
        merged = [main]
        for frag in rest:
            inside = np.all((frag >= lo) & (frag < hi), axis=1)
            if inside.any() and dil[tuple((frag[inside] - lo).T)].any():
                merged.append(frag)
            else:
                units.append(Unit(lab, frag, axis, sign))
        units.append(Unit(lab, np.concatenate(merged), axis, sign))
    units.sort(key=lambda u: u.si)
    return units


def _estimate_level_height(units):
    """Typical SI distance between consecutive vertebrae (voxel units).

    Uses the median gap between consecutive candidate components; robust to
    a few merged/missing levels.
    """
    if len(units) < 2:
        return 1.0
    gaps = np.diff([u.si for u in units])
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return 1.0
    med = np.median(gaps)
    core = gaps[gaps < 2.0 * med]
    return float(np.median(core)) if len(core) else float(med)


def identify_stack(units, level_height, w_relabel=1.0, w_gap=1.5,
                   w_skip=2.0, w_spacing=0.5):
    """Assign each candidate component an anatomical index (1..24) or skip it.

    Meng et al. (2023) cast vertebra identification as a shortest-path
    search: components are naturally ordered along the spine, and
    consecutive vertebrae must carry consecutive labels. We solve the same
    problem with dynamic programming. Costs:

      * unary   : assigning unit ``i`` to an index different from its
                  predicted label costs ``w_relabel * confidence`` (big
                  components trust their predicted label more);
      * gap     : skipping an anatomical index (a vertebra the model never
                  segmented) costs ``w_gap`` per missing level;
      * skip    : leaving a component unassigned (false positive) costs
                  ``w_skip * confidence``;
      * spacing : assigned neighbors should sit ``level_height`` apart per
                  skipped index; deviations cost ``w_spacing`` times the
                  relative error.

    A perfectly consistent input (all labels present, ordered, evenly
    spaced) has total cost zero, so the identity assignment always wins —
    the optimizer is a NO-OP on well-formed predictions and only repairs
    shifted, duplicated, or gapped stacks.

    Args:
        units: List of ``Unit`` sorted by ascending SI centroid.
        level_height: Typical inter-vertebra SI distance (voxel units).

    Returns:
        List of assignments, one per unit: an anatomical index 1..24, or
        ``None`` for skipped (false-positive) units.
    """
    n = len(units)
    if n == 0:
        return []
    if n == 1:
        return [units[0].pred]
    conf = [min(1.0, u.size / 5000.0) for u in units]
    unary = [[0.0 if j == u.pred else w_relabel * conf[i]
              for j in range(25)] for i, u in enumerate(units)]
    skip = [w_skip * c for c in conf]
    skip_prefix = np.concatenate([[0.0], np.cumsum(skip)])
    h = max(level_height, 1e-6)

    INF = float("inf")
    # dp[i][j] = min cost over assignments of units[0..i] with unit i -> j
    dp = [[INF] * 25 for _ in range(n)]
    parent = [[None] * 25 for _ in range(n)]
    for i in range(n):
        for j in range(1, 25):
            # option 1: unit i is the first (most inferior) assigned unit
            best = skip_prefix[i] + unary[i][j]
            # option 2: previous assigned unit k at index jp < j
            for k in range(i):
                skipped_units = skip_prefix[i] - skip_prefix[k + 1]
                for jp in range(1, j):
                    if dp[k][jp] == INF:
                        continue
                    idx_gap = j - jp
                    missing = w_gap * (idx_gap - 1)
                    expect = idx_gap * h
                    sp_err = abs((units[i].si - units[k].si) - expect) / expect
                    cost = (dp[k][jp] + skipped_units + missing +
                            w_spacing * sp_err + unary[i][j])
                    if cost < best:
                        best = cost
                        parent[i][j] = (k, jp)
            dp[i][j] = best

    # the last assigned unit may be any unit; trailing units are skipped
    best_cost, best_end = INF, None
    for i in range(n):
        trailing = skip_prefix[n] - skip_prefix[i + 1]
        for j in range(1, 25):
            total = dp[i][j] + trailing
            if total < best_cost:
                best_cost, best_end = total, (i, j)

    assign = [None] * n
    node = best_end
    while node is not None:
        i, j = node
        assign[i] = j
        node = parent[i][j]
    return assign


# ---------------------------------------------------------------------------
# step 4b: apply the assignment, reassign orphans (Jaus-style)
# ---------------------------------------------------------------------------

def apply_assignment(mask, units, assign, spacing, orphan_max_mm=8.0,
                     min_voxels=100):
    """Build the relabeled label map from the stack assignment.

    Assigned units are written with their (possibly corrected) anatomical
    index. Skipped units ("orphans") are handled like the false-positive
    reassignment of ShapeKit and the adjacent-vertebra consolidation of
    Jaus et al.: an orphan whose voxels lie within ``orphan_max_mm`` of an
    assigned vertebra's mass is absorbed into that label (it is almost
    always a detached fragment or a duplicated prediction of the same
    vertebra); a distant orphan smaller than ``min_voxels`` is deleted,
    and a distant LARGE orphan is left assigned to its nearest neighbor
    anyway but reported loudly (it should essentially never happen).

    Args:
        mask: Input uint8 label map (used only for shape).
        units: List of ``Unit``.
        assign: Per-unit anatomical index or ``None`` (orphan).
        spacing: Voxel spacing in mm.
        orphan_max_mm: Distance at which an orphan is absorbed.

    Returns:
        tuple ``(out, log)``: the relabeled uint8 map and a list of
        human-readable action strings for the report.
    """
    out = np.zeros_like(mask)
    assigned = [(u, j) for u, j in zip(units, assign) if j is not None]
    orphans = [u for u, j in zip(units, assign) if j is None]
    log = []

    for u, j in assigned:
        out[tuple(u.coords.T)] = j
        if j != u.pred:
            log.append(f"relabeled {VERTEBRAE_LABELS[u.pred]} -> "
                       f"{VERTEBRAE_LABELS[j]} "
                       f"({u.size} vox at SI {u.si:.0f})")

    if not orphans:
        return out, log

    # KD-trees of each assigned label's mass (subsampled), in mm
    trees = {}
    for u, j in assigned:
        if j not in trees:
            pts = u.coords.astype(float) * spacing
            step = max(1, len(pts) // 20000)
            trees[j] = (cKDTree(pts[::step]), pts)
    if not trees:  # everything was orphaned (degenerate): keep predictions
        for u in orphans:
            out[tuple(u.coords.T)] = u.pred
        log.append("WARNING: no consistent stack found; kept raw labels")
        return out, log

    for u in orphans:
        pts = u.coords.astype(float) * spacing
        step = max(1, len(pts) // 2000)
        probe = pts[::step]
        best_lab, best_dist = None, float("inf")
        for j, (tree, _) in trees.items():
            d, _ = tree.query(probe)
            med = float(np.median(d))
            if med < best_dist:
                best_dist, best_lab = med, j
        if best_dist <= orphan_max_mm or u.size >= min_voxels:
            out[tuple(u.coords.T)] = best_lab
            log.append(f"reassigned orphan of {VERTEBRAE_LABELS[u.pred]} -> "
                       f"{VERTEBRAE_LABELS[best_lab]} ({u.size} vox, "
                       f"median distance {best_dist:.1f} mm)"
                       + ("  [REVIEW: large distant orphan]"
                          if best_dist > orphan_max_mm else ""))
        else:
            log.append(f"deleted orphan of {VERTEBRAE_LABELS[u.pred]} "
                       f"({u.size} vox, median distance "
                       f"{best_dist:.1f} mm)")
    return out, log


# ---------------------------------------------------------------------------
# step 5: superior overflow trim
# ---------------------------------------------------------------------------

def trim_superior_overflow(mask, axis, sign, margin=0.35, touch_it=3):
    """Remove label voxels that climb past the upper neighbor's territory.

    Spinous processes point INFERIORLY, so a vertebra mask legitimately
    extends downward toward (or past) the level below, but its superior
    reach is short. A label whose voxels cross far above the midpoint of
    the gap to its upper neighbor has bled into that neighbor (observed as
    two or three labels stacked in the same axial slices). Such voxels are
    given to the upper neighbor when they touch its mass (they are almost
    always that vertebra's own bone mislabeled downward — the Jaus et al.
    adjacent-vertebra consolidation case). Overflow voxels NOT touching the
    upper neighbor are kept: disconnected superior reach is usually the
    vertebra's own superior articular facets, which do rise toward the
    upper level, and deleting them costs real recall. (Interspinous
    soft-tissue bleed between levels is removed earlier by bone gating,
    and small leftovers by the final speckle pass.)

    The margin is expressed as a fraction of the inter-centroid gap above
    the midpoint: the cut is at ``midpoint + margin * gap`` (margin 0.35
    keeps the superior articular facets, which reach a few mm above the
    midpoint). The C2->C1 pair is exempt: the C2 dens legitimately rises
    into the C1 ring.

    Args:
        mask: uint8 3D label map.
        axis, sign: SI axis and direction.
        margin: Fraction of the inter-centroid gap above the midpoint.
        touch_it: Dilation iterations defining "touches the upper label".

    Returns:
        Trimmed copy of ``mask``.
    """
    centroids = {lab: _label_centroid(mask, lab, axis, sign)
                 for lab in VERTEBRAE_LABELS}
    centroids = {l: c for l, c in centroids.items() if c is not None}
    out = mask.copy()
    struct = ndimage.generate_binary_structure(3, 1)
    present = sorted(centroids)  # ascending id = inferior -> superior
    for lo, hi in zip(present, present[1:]):
        if (lo, hi) == (23, 24):
            continue  # C2 dens rises into the C1 ring
        gap = centroids[hi] - centroids[lo]
        if gap <= 0:
            continue  # ordering already broken; identification handles it
        cut = centroids[lo] + (0.5 + margin) * gap
        lo_mask = mask == lo
        if not lo_mask.any():
            continue
        coords = np.nonzero(lo_mask)
        overflow = sign * coords[axis].astype(float) > cut
        if not overflow.any():
            continue
        ov = np.stack([c[overflow] for c in coords], axis=1)
        # only voxels touching the upper neighbor's mass are moved to it;
        # disconnected overflow (own superior facets) is kept
        hi_mask = mask == hi
        if not hi_mask.any():
            continue
        hi_pts = np.nonzero(hi_mask)
        lo_b = np.minimum(ov.min(axis=0),
                          [int(p.min()) for p in hi_pts]) - (touch_it + 1)
        hi_b = np.maximum(ov.max(axis=0),
                          [int(p.max()) for p in hi_pts]) + (touch_it + 2)
        lo_b = np.maximum(lo_b, 0)
        hi_b = np.minimum(hi_b, mask.shape)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo_b, hi_b))
        dil = ndimage.binary_dilation(hi_mask[sl], structure=struct,
                                      iterations=touch_it)
        origin = np.array([s.start for s in sl])
        to_hi = dil[tuple((ov - origin).T)]
        if to_hi.any():
            out[tuple(ov[to_hi].T)] = hi
    return out


# ---------------------------------------------------------------------------
# step 6: boundary smoothing
# ---------------------------------------------------------------------------

def smooth_boundaries(mask, radius=1):
    """Fill small holes and ragged edges inside each vertebra.

    Binary closing seals small internal holes and notches (ShapeKit
    "broken boundaries"). Newly filled voxels are only claimed from
    BACKGROUND, so one label can never steal voxels from a neighboring
    vertebra and the label map stays a partition.

    Args:
        mask: uint8 3D label map.
        radius: Closing iterations (18-connected structuring element).

    Returns:
        Copy of ``mask`` with holes filled; existing labels never
        overwritten.
    """
    out = mask.copy()
    struct = ndimage.generate_binary_structure(3, 2)
    for lab in VERTEBRAE_LABELS:
        binary = mask == lab
        if not binary.any():
            continue
        sl = _crop_slices(binary, pad=radius + 1)
        closed = ndimage.binary_closing(binary[sl], structure=struct,
                                        iterations=radius)
        fill = np.logical_and(closed, out[sl] == 0)
        out[sl][fill] = lab
    return out


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def save_case(mask, affine, header, out_case_dir):
    """Write the cleaned label map and per-vertebra binary masks for a case.

    The output layout mirrors the inference output so downstream tooling and
    ITK-SNAP workflows keep working:

    ::

        <out_case_dir>/
            combined_labels.nii.gz          cleaned uint8 label map
            segmentations/
                vertebrae_L5.nii.gz         binary mask per label (all 24
                ...                         files written, even when empty)

    Args:
        mask: Cleaned uint8 3D label map.
        affine: 4x4 voxel-to-world matrix from the source file.
        header: NIfTI header from the source file, or ``None``.
        out_case_dir: Output directory for this case (created if needed).
    """
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


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _plausible_range(lab):
    name = VERTEBRAE_LABELS[lab]
    return PLAUSIBLE_CM3[name.split("_")[1][0]]


def report_case(mask, axis, sign, spacing, ct, title):
    """Print a per-label QC table for one case.

    Columns: voxel count, volume in cm^3 (with a warning mark outside the
    plausible range), number of connected components, superior-positive SI
    centroid, SI span in mm, and (when the CT is available) the fraction of
    voxels lying on bone (HU >= 130). Also checks that centroid ordering is
    anatomically consistent and lists missing levels.
    """
    vx = float(np.prod(spacing))
    print(f"--- {title} ---")
    print("label        vox    cm3     comps  si_c   span_mm  on_bone")
    stats = {}
    for lab in sorted(VERTEBRAE_LABELS, reverse=True):
        binary = mask == lab
        n = int(binary.sum())
        name = VERTEBRAE_LABELS[lab].replace("vertebrae_", "")
        if n == 0:
            print(f"{name:<6}  ABSENT")
            continue
        coords = np.nonzero(binary)
        si = sign * coords[axis].astype(float)
        span = (si.max() - si.min() + 1) * spacing[axis]
        comps = len(_component_coords(binary))
        vol = n * vx / 1000.0
        lo_p, hi_p = _plausible_range(lab)
        flag = "" if lo_p <= vol <= hi_p else "  <-- volume unusual"
        bone = ""
        if ct is not None:
            bone = f"  {(ct[binary] >= 130).mean():6.1%}"
        print(f"{name:<6} {n:8d} {vol:7.1f}  {comps:4d}  {si.mean():6.0f} "
              f"{span:7.1f}{bone}{flag}")
        stats[lab] = si.mean()
    ids_desc = sorted(stats, reverse=True)
    ordered = all(stats[a] > stats[b] for a, b in zip(ids_desc, ids_desc[1:]))
    missing = [VERTEBRAE_LABELS[l].replace("vertebrae_", "")
               for l in range(min(ids_desc, default=1),
                              max(ids_desc, default=1) + 1)
               if l not in stats] if stats else []
    print(f"ordering consistent: {ordered}; "
          f"missing levels in stack: {missing or 'none'}")


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def process_case(case_dir, out_case_dir, args):
    """Run the full cleanup pipeline on one case and write the result.

    Pipeline order (order matters — see individual docstrings):
    ``gate_to_bone`` -> ``trim_rib_tails`` -> ``remove_speckles`` ->
    ``identify_stack`` + ``apply_assignment`` -> ``trim_superior_overflow``
    -> ``remove_speckles`` (small pass) -> ``smooth_boundaries``.
    """
    mask, affine, header, ct, case_name = load_case(case_dir, args.ct_dir)
    axis, sign = infer_si_axis(affine)
    spacing = get_spacing(header, affine)

    if args.report:
        print(f"\n=== {case_name} (SI axis={axis}, sign={sign:+.0f}, "
              f"spacing={np.round(spacing, 2)} mm) ===")
        report_case(mask, axis, sign, spacing, ct, "input")

    if ct is not None:
        mask = gate_to_bone(mask, ct, args.hu_threshold, spacing=spacing)
    mask = trim_rib_tails(mask, axis, sign, spacing, args.rib_radius_mm)
    mask = remove_speckles(mask, args.min_voxels, args.speckle_frac)

    units = build_units(mask, axis, sign, spacing, args.merge_gap_mm)
    height = _estimate_level_height(units)
    assign = identify_stack(units, height, args.w_relabel, args.w_gap,
                            args.w_skip, args.w_spacing)
    mask, log = apply_assignment(mask, units, assign, spacing,
                                 args.orphan_max_mm, args.min_voxels)

    if not args.no_overflow_trim:
        mask = trim_superior_overflow(mask, axis, sign,
                                      args.overflow_margin)
        mask = remove_speckles(mask, args.min_voxels, args.speckle_frac)
    mask = smooth_boundaries(mask, args.closing_radius)

    save_case(mask, affine, header, out_case_dir)

    if args.report:
        if log:
            print("identification actions:")
            for line in log:
                print(f"  {line}")
        else:
            print("identification actions: none (stack already consistent)")
        report_case(mask, axis, sign, spacing, ct, "output")
    return mask


def find_cases(pred_dir):
    """Discover case directories under a prediction root.

    A "case" is any directory containing a ``combined_labels.nii.gz``. For
    convenience, ``pred_dir`` itself may also be a single case directory.
    """
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
    """CLI entry point: parse arguments and clean every discovered case."""
    parser = argparse.ArgumentParser(
        description="Postprocess SuPreM vertebrae predictions (bone gating, "
                    "rib trim, speckles, global stack identification, "
                    "overflow trim, smoothing).")
    parser.add_argument("--pred_dir", required=True,
                        help="Dir with case subfolders (or a single case dir) "
                             "containing combined_labels.nii.gz")
    parser.add_argument("--out_dir", default=None,
                        help="Output dir (default: <pred_dir>_clean)")
    parser.add_argument("--ct_dir", default=None,
                        help="Optional dir with <CASE>/ct.nii.gz; enables "
                             "CT bone gating")
    parser.add_argument("--hu_threshold", type=float, default=130,
                        help="HU floor for bone gating")
    parser.add_argument("--rib_radius_mm", type=float, default=40.0,
                        help="Max radial distance of thoracic labels from "
                             "the spine centerline, in mm (0 disables)")
    parser.add_argument("--min_voxels", type=int, default=100,
                        help="Absolute speckle floor in voxels")
    parser.add_argument("--speckle_frac", type=float, default=0.02,
                        help="Speckle threshold as fraction of the label's "
                             "largest component")
    parser.add_argument("--merge_gap_mm", type=float, default=8.0,
                        help="Max gap for grouping fragments of one vertebra "
                             "into a single identification unit, in mm")
    parser.add_argument("--orphan_max_mm", type=float, default=8.0,
                        help="Orphan components within this distance of a "
                             "vertebra are absorbed into it, in mm")
    parser.add_argument("--overflow_margin", type=float, default=0.35,
                        help="Superior-overflow cut: midpoint + margin x "
                             "inter-centroid gap")
    parser.add_argument("--no_overflow_trim", action="store_true",
                        help="Disable the superior overflow trim")
    parser.add_argument("--closing_radius", type=int, default=1,
                        help="Iterations of binary closing for smoothing")
    parser.add_argument("--w_relabel", type=float, default=1.0,
                        help="Identification cost of relabeling a component")
    parser.add_argument("--w_gap", type=float, default=1.5,
                        help="Identification cost per missing level")
    parser.add_argument("--w_skip", type=float, default=2.0,
                        help="Identification cost of skipping a component")
    parser.add_argument("--w_spacing", type=float, default=0.5,
                        help="Identification weight of spacing regularity")
    parser.add_argument("--report", action="store_true",
                        help="Print per-case before/after QC tables")
    args = parser.parse_args()

    out_dir = args.out_dir or args.pred_dir.rstrip("/") + "_clean"
    cases = find_cases(args.pred_dir)
    if not cases:
        raise SystemExit(f"No cases with combined_labels.nii.gz under "
                         f"{args.pred_dir}")

    for case_dir in cases:
        out_case_dir = os.path.join(out_dir,
                                    os.path.basename(case_dir.rstrip("/")))
        process_case(case_dir, out_case_dir, args)
        print(f"cleaned case written to {out_case_dir}")


if __name__ == "__main__":
    main()
