# coding: utf-8
import numpy as np

from jiuwensymbiosis.perception.object_geometry import object_geometry_from_mask


def _identity_tf():
    return np.eye(4)


def _pinhole(fx=345.0, fy=345.0, cx=320.0, cy=180.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def test_planar_patch_gives_zero_depth_extent_and_correct_width():
    # 40x40 px mask centered, constant depth 0.5 m, identity extrinsics.
    h, w = 360, 640
    depth = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    depth[160:200, 300:340] = 0.5
    mask[160:200, 300:340] = True
    out = object_geometry_from_mask(mask, depth, _pinhole(), _identity_tf())
    assert out.ok
    # base==cam (identity): z_mm = 500; front_x_mm is min X (cam x = (u-cx)*z/fx)
    assert abs(out.center_mm[2] - 500.0) < 1.0
    assert out.width_mm > 0.0          # base-Y extent from the patch's row span
    assert out.height_mm < 1.0         # base-Z (depth) extent ~0 for a fronto-parallel patch
    assert out.n_points > 1000


def test_too_few_points_returns_not_ok():
    h, w = 360, 640
    depth = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    depth[10, 10] = 0.5
    mask[10, 10] = True
    out = object_geometry_from_mask(mask, depth, _pinhole(), _identity_tf(), min_points=50)
    assert not out.ok
    assert out.reason == "too_few_points"


def test_mask_rescaled_to_depth_resolution():
    # mask at half resolution should still localize the same region.
    h, w = 360, 640
    depth = np.zeros((h, w), dtype=np.float32)
    depth[160:200, 300:340] = 0.5
    mask_small = np.zeros((h // 2, w // 2), dtype=bool)
    mask_small[80:100, 150:170] = True
    out = object_geometry_from_mask(mask_small, depth, _pinhole(), _identity_tf())
    assert out.ok
    assert out.n_points > 500


def test_back_x_gives_depth_extent():
    # a slab spanning a depth range -> back_x_mm > front_x_mm; midpoint usable.
    h, w = 360, 640
    depth = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    # vary depth across columns so the masked region has an X (and depth) extent
    for j, dval in zip(range(300, 360), np.linspace(0.5, 0.7, 60)):
        depth[160:200, j] = dval
        mask[160:200, j] = True
    out = object_geometry_from_mask(mask, depth, _pinhole(), _identity_tf())
    assert out.ok
    assert out.back_x_mm > out.front_x_mm
    mid = (out.front_x_mm + out.back_x_mm) / 2.0
    assert out.front_x_mm < mid < out.back_x_mm
