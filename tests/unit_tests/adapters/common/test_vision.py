# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.vision."""

from __future__ import annotations

import numpy as np
import pytest

from jiuwensymbiosis.adapters._common.vision import (
    DETECTION_REASONS,
    GraspFailure,
    GraspResult,
    _mask_centroid,
    _median_depth_window,
    apply_xy_correction,
    detect_and_centroid,
)


class TestTypeContract:
    def test_detection_reasons_is_the_documented_set(self):
        assert DETECTION_REASONS == frozenset(
            {
                "no_camera",
                "no_detection",
                "empty_mask",
                "no_valid_depth",
                "detector_unavailable",
            }
        )

    def test_typeddicts_importable(self):
        # Importing the TypedDicts must not error; they exist as module attrs.
        assert GraspResult is not None
        assert GraspFailure is not None


class TestMaskCentroid:
    def test_center_mask(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[45:55, 95:105] = True
        result = _mask_centroid({"mask": mask}, img_w=200, img_h=100, log_prefix="[test]")
        assert abs(result["u"] - 100) < 2
        assert abs(result["v"] - 50) < 2

    def test_corner_mask(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[0:10, 0:10] = True
        result = _mask_centroid({"mask": mask}, img_w=200, img_h=100, log_prefix="[test]")
        assert result["u"] < 15
        assert result["v"] < 10


class TestMedianDepthWindow:
    def test_valid_depth(self):
        depth = np.ones((480, 640), dtype=np.float32) * 0.5
        result = _median_depth_window(depth, 320.0, 240.0, "[test]")
        assert result is not None
        assert abs(result - 0.5) < 0.01

    def test_zero_depth_returns_none(self):
        depth = np.zeros((480, 640), dtype=np.float32)
        result = _median_depth_window(depth, 320.0, 240.0, "[test]")
        assert result is None


class TestApplyXyCorrection:
    def test_no_correction(self):
        xyz = np.array([100.0, 200.0, 300.0])
        result, desc = apply_xy_correction(xyz)
        np.testing.assert_array_almost_equal(result, xyz)
        assert desc == "none"

    def test_translation_correction(self):
        xyz = np.array([100.0, 200.0, 300.0])
        result, desc = apply_xy_correction(xyz, xy_correction_mm=[5.0, -10.0])
        np.testing.assert_array_almost_equal(result[0], 105.0)
        np.testing.assert_array_almost_equal(result[1], 190.0)
        np.testing.assert_array_almost_equal(result[2], 300.0)

    def test_affine_correction(self):
        xyz = np.array([100.0, 200.0, 300.0])
        A = np.eye(2)
        b = np.array([5.0, -10.0])
        result, desc = apply_xy_correction(
            xyz,
            xy_transform={"A": A.tolist(), "b": b.tolist(), "method": "affine", "n_samples": 3, "rms_residual_mm": 1.2},
        )
        np.testing.assert_array_almost_equal(result[0], 105.0)
        np.testing.assert_array_almost_equal(result[1], 190.0)

    def test_affine_priority_over_translation(self):
        xyz = np.array([100.0, 200.0, 300.0])
        A = np.array([[1.1, 0.0], [0.0, 0.9]])
        b = np.array([5.0, -10.0])
        result, desc = apply_xy_correction(
            xyz,
            xy_transform={"A": A.tolist(), "b": b.tolist(), "method": "affine", "n_samples": 3, "rms_residual_mm": 1.0},
            xy_correction_mm=[999.0, 999.0],
        )
        assert "xy_transform" in desc
        assert "affine" in desc


class TestDetectAndCentroidReasonContract:
    """Every failure reason emitted by detect_and_centroid must be in
    DETECTION_REASONS — the TypedDict contract the LLM/adapter relies on."""

    def _frame(self):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        depth = np.ones((100, 200), dtype=np.float32) * 0.5
        return rgb, depth

    def test_no_seg_fn_yields_detector_unavailable(self):
        rgb, depth = self._frame()
        det = detect_and_centroid(
            rgb=rgb,
            depth_img_m=depth,
            seg_fn=None,
            object_name="box",
            tcp_at_grab=type("P", (), {"x": 0, "y": 0, "z": 0, "r": 0})(),
        )
        assert det["ok"] is False
        assert det["reason"] in DETECTION_REASONS

    def test_no_detection_reason(self):
        rgb, depth = self._frame()

        def _seg(rgb, text_prompt):
            return []  # nothing detected

        det = detect_and_centroid(
            rgb=rgb,
            depth_img_m=depth,
            seg_fn=_seg,
            object_name="box",
            tcp_at_grab=type("P", (), {"x": 0, "y": 0, "z": 0, "r": 0})(),
        )
        assert det["ok"] is False
        assert det["reason"] == "no_detection"
        assert det["reason"] in DETECTION_REASONS


class _MockLowLevel:
    """Minimal low_level satisfying the VisionDriver surface for the eye-in-hand
    default helpers: tf_flange_cam / intrinsics / grab_frames / calibration."""

    def __init__(self, rgb, depth, tf_flange_cam, intrinsics, calibration=None):
        self._rgb = rgb
        self._depth = depth
        self._tf = tf_flange_cam
        self._K = intrinsics
        self.calibration = calibration

    @property
    def tf_flange_cam(self):
        return self._tf

    @property
    def intrinsics(self):
        return self._K

    def grab_frames(self):
        return (self._rgb, self._depth)


class _MockEnv:
    """Stand-in env: reports a fixed flange pose + a low_level driver."""

    def __init__(self, low_level, pose, z_min_safe=0.0):
        self.low_level = low_level
        self._pose = pose
        self._z_min_safe = z_min_safe

    def get_flange_pose(self):
        return self._pose

    @property
    def z_min_safe(self):
        return self._z_min_safe


class _StubApi:
    """Minimal api-like object: exposes ``env`` + the geometry constants the
    eye-in-hand helpers read, plus a detector seg_fn slot."""

    def __init__(self, env, seg_fn, *, z_correction_mm=0.0, grasp_z_offset_mm=-25.0, chip_thickness_mm=75.0):
        self.env = env
        self._seg_fn = seg_fn
        self._z_correction_mm = z_correction_mm
        self._grasp_z_offset_mm = grasp_z_offset_mm
        self._chip_thickness_mm = chip_thickness_mm


def _identity_pose_to_tf(pose):
    # The default helpers only need a 4x4 base←flange transform; for tests we
    # use identity so the projected point equals the camera-frame point.
    return np.eye(4)


class TestDefaultEyeInHandHelpers:
    """default_get_grasp_info_simple / default_pixel_to_base_xyz factor out the
    ~130 lines of detect→centroid→project→correct→geometry that every eye-in-hand
    camera robot duplicates."""

    def _setup(self, *, depth_m=0.5, z_min_safe=0.0):
        from tests.mocks.mock_detector import make_mock_seg_fn

        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), float(depth_m), dtype=np.float32)
        tf_flange_cam = np.eye(4)  # camera at flange origin
        intrinsics = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        ll = _MockLowLevel(rgb, depth, tf_flange_cam, intrinsics)
        pose = type("P", (), {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0, "ry": 0, "rz": 0})()
        env = _MockEnv(ll, pose, z_min_safe=z_min_safe)
        api = _StubApi(env, make_mock_seg_fn(score=0.8))
        return api

    def test_get_grasp_info_simple_success_shape(self):
        from jiuwensymbiosis.adapters._common.vision import default_get_grasp_info_simple

        api = self._setup()
        result = default_get_grasp_info_simple(
            api,
            "box",
            seg_fn=api._seg_fn,
            pose_to_tf=_identity_pose_to_tf,
        )
        assert result["ok"] is True
        assert result["object"] == "box"
        for key in (
            "position",
            "grasp_z",
            "grasp_position",
            "place_z",
            "place_position",
            "score",
            "pixel_uv",
            "depth_m",
        ):
            assert key in result
        assert result["position"][2] == result["depth_m"] * 1000.0  # identity tf, mm

    def test_grasp_z_clamped_to_z_min_safe(self):
        from jiuwensymbiosis.adapters._common.vision import default_get_grasp_info_simple

        # detected top at 500mm; z_min_safe 600mm → grasp_z clamped up to 600.
        api = self._setup(depth_m=0.5, z_min_safe=600.0)
        result = default_get_grasp_info_simple(
            api,
            "box",
            seg_fn=api._seg_fn,
            pose_to_tf=_identity_pose_to_tf,
        )
        assert result["ok"] is True
        assert result["grasp_z"] >= 600.0

    def test_get_grasp_info_simple_no_camera(self):
        from jiuwensymbiosis.adapters._common.vision import default_get_grasp_info_simple

        api = self._setup()
        # Make grab_frames return None.
        api.env.low_level._rgb = None
        api.env.low_level.grab_frames = lambda: None
        result = default_get_grasp_info_simple(
            api,
            "box",
            seg_fn=api._seg_fn,
            pose_to_tf=_identity_pose_to_tf,
        )
        assert result["ok"] is False
        assert result["reason"] == "no_camera"

    def test_get_grasp_info_simple_propagates_detection_failure(self):
        from jiuwensymbiosis.adapters._common.vision import default_get_grasp_info_simple
        from tests.mocks.mock_detector import make_mock_seg_fn

        api = self._setup()
        # Empty detection → detect_and_centroid returns ok=False reason=no_detection.
        api._seg_fn = make_mock_seg_fn(returns_empty=True)
        result = default_get_grasp_info_simple(
            api,
            "box",
            seg_fn=api._seg_fn,
            pose_to_tf=_identity_pose_to_tf,
        )
        assert result["ok"] is False
        assert result["reason"] == "no_detection"

    def test_get_grasp_info_simple_no_calibration_raises(self):
        from jiuwensymbiosis.adapters._common.vision import default_get_grasp_info_simple

        api = self._setup()
        api.env.low_level._tf = None
        api.env.low_level._K = None
        with pytest.raises(RuntimeError):
            default_get_grasp_info_simple(
                api,
                "box",
                seg_fn=api._seg_fn,
                pose_to_tf=_identity_pose_to_tf,
            )

    def test_pixel_to_base_xyz_returns_xyz(self):
        from jiuwensymbiosis.adapters._common.vision import default_pixel_to_base_xyz

        api = self._setup(depth_m=0.5)
        result = default_pixel_to_base_xyz(api, 320.0, 240.0, 0.5, pose_to_tf=_identity_pose_to_tf)
        assert set(result.keys()) == {"x", "y", "z"}
        # Principal point (320,240) with identity tf → x=0, y=0, z=depth*1000.
        assert abs(result["x"]) < 1e-6
        assert abs(result["y"]) < 1e-6
        assert abs(result["z"] - 500.0) < 1e-6
