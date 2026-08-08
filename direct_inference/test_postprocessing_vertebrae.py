"""Unit tests for postprocessing_vertebrae.py using synthetic 3D masks.

Run from the repo root:
    python -m pytest SuPreM/direct_inference/test_postprocessing_vertebrae.py -v

Synthetic layout: shape (12, 12, 120), SI axis = 2 with increasing z going
superior. Labels 1..10 (L5..T7) are 5x5x5 cubes (125 voxels) at z centers
15, 23, ..., 87 — strictly ascending with label id, mimicking the real
L5(bottom) -> C1(top) ordering.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import postprocessing_vertebrae as pp


def z_center(label_id):
    return 15 + (label_id - 1) * 8


def make_stack(ids=None):
    mask = np.zeros((12, 12, 120), dtype=np.uint8)
    for i in (ids or range(1, 11)):
        z = z_center(i)
        mask[3:8, 3:8, z - 2:z + 3] = i  # 125 voxels
    return mask


# ---------------------------------------------------------------------------
# infer_si_axis
# ---------------------------------------------------------------------------

def test_infer_si_axis_standard():
    assert pp.infer_si_axis(np.eye(4)) == (2, 1.0)  # RAS -> z is Superior


def test_infer_si_axis_flipped():
    affine = np.diag([1.0, 1.0, -1.0, 1.0])
    assert pp.infer_si_axis(affine) == (2, -1.0)


def test_infer_si_axis_fallback():
    assert pp.infer_si_axis(None) == (2, 1.0)


# ---------------------------------------------------------------------------
# speckles
# ---------------------------------------------------------------------------

def test_remove_speckles_drops_tiny_distant_blob():
    mask = make_stack()
    mask[0:3, 0:3, 100:103] = 5  # 27-voxel speckle far from vertebrae_L1
    out = pp.remove_speckles(mask, min_voxels=100, frac=0.02)
    assert (out == 5).sum() == 125          # main blob untouched
    assert out[0:3, 0:3, 100:103].sum() == 0
    assert (out == 1).sum() == 125          # other labels untouched


def test_remove_speckles_keeps_real_vertebrae():
    mask = make_stack()
    out = pp.remove_speckles(mask, min_voxels=100, frac=0.02)
    assert np.array_equal(out, mask)


# ---------------------------------------------------------------------------
# fragments
# ---------------------------------------------------------------------------

def test_merge_fragments_reattaches_near_fragment():
    mask = make_stack()
    # 27-voxel fragment of label 7 (z center 63, cube z 61..65, x 3..7)
    # placed 1 voxel away in x -> within merge gap
    mask[9:12, 3:6, 61:64] = 7
    out = pp.merge_fragments(mask, max_gap=3)
    # fragment survived AND is now one connected component with the main blob
    assert len(pp._components(out == 7)) == 1
    assert (out == 7).sum() > 125


def test_pipeline_removes_distant_fragment_as_speckle():
    mask = make_stack()
    mask[9:12, 3:6, 61:64] = 7   # near fragment -> merged
    mask[3:6, 3:6, 95:98] = 7    # distant fragment -> speckle, removed
    out = pp.merge_fragments(mask, max_gap=3)
    out = pp.remove_speckles(out, min_voxels=100, frac=0.02)
    assert len(pp._components(out == 7)) == 1
    assert out[3:6, 3:6, 95:98].sum() == 0


# ---------------------------------------------------------------------------
# duplicates
# ---------------------------------------------------------------------------

def test_resolve_duplicates_keeps_positionally_consistent_blob():
    mask = make_stack()
    mask[3:8, 3:8, 96:101] = 5  # second large "vertebrae_L1" near the top
    out = pp.resolve_duplicates(mask, axis=2, sign=1.0)
    comps = pp._components(out == 5)
    assert len(comps) == 1
    assert abs(pp._si_centroid(comps[0], 2, 1.0) - z_center(5)) < 2


def test_resolve_duplicates_without_neighbors_keeps_largest():
    mask = np.zeros((12, 12, 120), dtype=np.uint8)
    mask[3:8, 3:8, 10:15] = 5   # 125 voxels
    mask[3:6, 3:6, 90:93] = 5   # 27 voxels, no other labels present
    out = pp.resolve_duplicates(mask, axis=2, sign=1.0)
    comps = pp._components(out == 5)
    assert len(comps) == 1
    assert int(comps[0].sum()) == 125


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------

def test_enforce_ordering_drops_out_of_order_label():
    mask = make_stack()
    mask[mask == 3] = 0
    mask[3:8, 3:8, 98:103] = 3  # "vertebrae_L3" above T7 -> impossible
    out = pp.enforce_ordering(mask, axis=2, sign=1.0)
    assert (out == 3).sum() == 0
    for i in (1, 2, 4, 5, 6, 7, 8, 9, 10):
        assert (out == i).sum() == 125


def test_enforce_ordering_leaves_consistent_stack_unchanged():
    mask = make_stack()
    out = pp.enforce_ordering(mask, axis=2, sign=1.0)
    assert np.array_equal(out, mask)


# ---------------------------------------------------------------------------
# smoothing
# ---------------------------------------------------------------------------

def test_smooth_boundaries_fills_internal_hole():
    mask = make_stack()
    mask[5, 5, z_center(1)] = 0  # single-voxel hole inside vertebrae_L5
    out = pp.smooth_boundaries(mask, radius=1)
    assert out[5, 5, z_center(1)] == 1


def test_smooth_boundaries_never_overwrites_other_label():
    mask = make_stack()
    before = {i: (mask == i).sum() for i in range(1, 11)}
    out = pp.smooth_boundaries(mask, radius=1)
    for i, n in before.items():
        assert (out == i).sum() >= n  # labels only grow into background


# ---------------------------------------------------------------------------
# io round-trip
# ---------------------------------------------------------------------------

def test_save_case_round_trip(tmp_path):
    mask = make_stack()
    affine = np.eye(4)
    pp.save_case(mask, affine, None, str(tmp_path))

    import nibabel as nib
    combined = nib.load(str(tmp_path / "combined_labels.nii.gz"))
    assert np.array_equal(np.asanyarray(combined.dataobj), mask)
    assert np.allclose(combined.affine, affine)

    l5 = nib.load(str(tmp_path / "segmentations" / "vertebrae_L5.nii.gz"))
    assert np.array_equal(np.asanyarray(l5.dataobj), (mask == 1).astype(np.uint8))
    # all 24 per-vertebra files are written, like the inference output
    assert len(list((tmp_path / "segmentations").glob("vertebrae_*.nii.gz"))) == 24


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_help_exits_zero():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "postprocessing_vertebrae.py")
    result = subprocess.run([sys.executable, script, "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--pred_dir" in result.stdout


def test_cli_end_to_end_on_synthetic_case(tmp_path):
    case = tmp_path / "pred" / "BDMAP_00000001"
    case.mkdir(parents=True)
    mask = make_stack()
    mask[0:3, 0:3, 100:103] = 5  # speckle that should be cleaned
    import nibabel as nib
    nib.save(nib.Nifti1Image(mask, np.eye(4)),
             str(case / "combined_labels.nii.gz"))

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "postprocessing_vertebrae.py")
    out_dir = tmp_path / "clean"
    result = subprocess.run(
        [sys.executable, script, "--pred_dir", str(tmp_path / "pred"),
         "--out_dir", str(out_dir), "--report"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    cleaned = nib.load(str(out_dir / "BDMAP_00000001" / "combined_labels.nii.gz"))
    cleaned = np.asanyarray(cleaned.dataobj)
    assert cleaned[0:3, 0:3, 100:103].sum() == 0
    assert (cleaned == 5).sum() >= 125
