# coding: utf-8
"""head_geometry_from_mask: mask + organized head point cloud → base-frame footprint (mm).

Verifies NaN-hole exclusion, the metres→mm ×1000 + tf_base_cam transform, the min-valid-points
degrade signal (None), and uniform mask→cloud downsampling."""

import numpy as np

from scripts.cruzr._head_cloud import HeadGeom, head_geometry_from_mask


def _cloud_4x4_with_hole():
    """A 4×4 organized cloud (metres, optical frame). A 2×2 block at rows/cols 1-2 is the target;
    one of its four cells is a NaN hole. All other cells are far-away finite filler that must NOT
    leak in (they're outside the mask)."""
    cloud = np.full((4, 4, 3), 9.0, dtype=np.float32)     # finite filler outside the mask
    cloud[1, 1] = [1.0, 0.0, 0.5]
    cloud[1, 2] = [1.1, 0.1, 0.6]
    cloud[2, 1] = [1.2, -0.1, 0.55]
    cloud[2, 2] = [np.nan, np.nan, np.nan]                # untextured hole → dropped
    return cloud


def _mask_4x4():
    m = np.zeros((4, 4), dtype=bool)
    m[1:3, 1:3] = True                                    # the 2×2 target block
    return m


def test_head_geometry_from_mask_basic():
    # identity tf → base coords == metres×1000; NaN cell excluded → n_valid=3; footprint from the
    # 3 finite points p1(1000,0,500) p2(1100,100,600) p3(1200,-100,550).
    g = head_geometry_from_mask(_mask_4x4(), _cloud_4x4_with_hole(), np.eye(4), min_valid_points=3)
    assert isinstance(g, HeadGeom)
    assert g.n_valid == 3                                 # the NaN hole was dropped
    assert np.isclose(g.front_x, 1000.0) and np.isclose(g.back_x, 1200.0)
    assert np.isclose(g.center_x, 1100.0)                 # mean X of the 3 points
    assert np.isclose(g.center_y, 0.0)                    # mean Y (0 + 100 - 100)/3
    assert np.isclose(g.center_z, 550.0)                  # mean Z
    assert np.isclose(g.half_width, 100.0)                # (100 - (-100)) / 2
    assert np.isclose(g.top_z, 600.0)                     # max Z


def test_head_geometry_min_valid_returns_none():
    # only 3 finite in-mask points but require 4 → degrade signal (None).
    g = head_geometry_from_mask(_mask_4x4(), _cloud_4x4_with_hole(), np.eye(4), min_valid_points=4)
    assert g is None


def test_head_geometry_none_when_cloud_or_tf_missing():
    assert head_geometry_from_mask(_mask_4x4(), None, np.eye(4), min_valid_points=3) is None
    assert head_geometry_from_mask(_mask_4x4(), _cloud_4x4_with_hole(), None, min_valid_points=3) is None


def test_head_geometry_applies_tf_translation():
    # tf translating +200mm X, -50mm Z shifts every base coord by that offset (confirms mm scale).
    tf = np.eye(4)
    tf[0, 3] = 200.0
    tf[2, 3] = -50.0
    g = head_geometry_from_mask(_mask_4x4(), _cloud_4x4_with_hole(), tf, min_valid_points=3)
    assert np.isclose(g.front_x, 1200.0) and np.isclose(g.back_x, 1400.0)
    assert np.isclose(g.center_x, 1300.0)
    assert np.isclose(g.top_z, 550.0)                     # 600 - 50
    assert np.isclose(g.center_z, 500.0)                  # 550 - 50


def test_head_geometry_downsamples_full_res_mask():
    # cloud 4×4 but mask 8×8: cloud cell (r,c) samples mask[2r,2c]. Marking mask[2,4]/[4] True
    # selects the same 4 cloud cells as the basic case → identical geometry.
    cloud = _cloud_4x4_with_hole()
    mask = np.zeros((8, 8), dtype=bool)
    mask[2, 2] = mask[2, 4] = mask[4, 2] = mask[4, 4] = True
    g = head_geometry_from_mask(mask, cloud, np.eye(4), min_valid_points=3)
    assert g is not None and g.n_valid == 3
    assert np.isclose(g.front_x, 1000.0) and np.isclose(g.back_x, 1200.0)
    assert np.isclose(g.center_x, 1100.0)
