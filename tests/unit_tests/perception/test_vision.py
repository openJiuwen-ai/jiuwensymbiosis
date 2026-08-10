# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.perception.vision."""

from __future__ import annotations

import numpy as np
import pytest

from jiuwensymbiosis.api.mixins import VisionMixin
from jiuwensymbiosis.perception.vision import (
    DETECTION_REASONS,
    GraspFailure,
    GraspResult,
    _mask_centroid,
    _median_depth_window,
    apply_xy_correction,
    build_grasp_result,
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


class _MixinStub(VisionMixin):
    """Minimal VisionMixin subclass exercising the shared grasp pipeline.

    The projection seam is an identity eye-in-hand back-projection (camera at the
    base origin) so the projected base XYZ equals the camera-frame point; the
    geometry constants the mixin reads are supplied directly. Tool emission /
    capability gating are out of scope here.
    """

    def __init__(
        self,
        env,
        seg_fn,
        *,
        z_correction_mm=0.0,
        grasp_z_offset_mm=-25.0,
        place_z_offset_mm=75.0,
        floor_margin_mm=0.0,
    ):
        self.env = env
        self._seg_fn = seg_fn
        self._z_correction_mm = z_correction_mm
        self._grasp_z_offset_mm = grasp_z_offset_mm
        self._place_z_offset_mm = place_z_offset_mm
        self._floor_margin_mm = floor_margin_mm

    def _project_pixel_to_base_raw(self, u, v, depth_m):
        from jiuwensymbiosis.utils.geometry import pixel_and_depth_to_camera_xyz

        ll = self.env.low_level
        calib = getattr(ll, "calibration", None)
        intrinsics = calib.get("intrinsics") if calib is not None else ll.intrinsics
        return pixel_and_depth_to_camera_xyz((u, v), depth_m, intrinsics)

    def _grasp_debug_tcp(self):
        from types import SimpleNamespace

        return SimpleNamespace(x=0.0, y=0.0, z=0.0, r=0.0)


class TestBuildGraspResult:
    """build_grasp_result owns xy/z correction + grasp/place geometry, shared by
    every adapter through VisionMixin."""

    def _call(
        self,
        *,
        xyz_raw,
        calib=None,
        z_correction_mm=0.0,
        grasp_z_offset_mm=-25.0,
        place_z_offset_mm=75.0,
        z_floor=None,
        floor_margin_mm=0.0,
    ):
        return build_grasp_result(
            object_name="box",
            best={"score": 0.9},
            u=10.0,
            v=20.0,
            depth_m=0.5,
            xyz_raw=np.asarray(xyz_raw, dtype=np.float64),
            calib=calib,
            z_correction_mm=z_correction_mm,
            grasp_z_offset_mm=grasp_z_offset_mm,
            place_z_offset_mm=place_z_offset_mm,
            z_floor=z_floor,
            floor_margin_mm=floor_margin_mm,
        )

    def test_no_correction_shape_and_geometry(self):
        result, xyz_final = self._call(xyz_raw=[100.0, 200.0, 500.0])
        assert result["ok"] is True
        assert result["object"] == "box"
        assert set(result) == {
            "ok",
            "object",
            "position",
            "grasp_z",
            "grasp_position",
            "place_z",
            "place_position",
            "score",
            "pixel_uv",
            "depth_m",
        }
        assert result["position"] == [100.0, 200.0, 500.0]
        assert result["grasp_z"] == 475.0
        assert result["place_z"] == 575.0
        assert result["pixel_uv"] == [10.0, 20.0]
        assert list(xyz_final) == [100.0, 200.0, 500.0]

    def test_xy_correction_mm_applied(self):
        result, _ = self._call(xyz_raw=[100.0, 200.0, 500.0], calib={"xy_correction_mm": [5.0, -3.0]})
        assert result["position"][0] == 105.0
        assert result["position"][1] == 197.0

    def test_xy_transform_takes_priority_over_correction_mm(self):
        calib = {
            "xy_transform": {
                "A": [[2.0, 0.0], [0.0, 2.0]],
                "b": [0.0, 0.0],
                "method": "sim",
                "n_samples": 3,
                "rms_residual_mm": 0.1,
            },
            "xy_correction_mm": [999.0, 999.0],
        }
        result, _ = self._call(xyz_raw=[100.0, 200.0, 500.0], calib=calib)
        assert result["position"][0] == 200.0
        assert result["position"][1] == 400.0

    def test_z_correction_added(self):
        result, _ = self._call(xyz_raw=[0.0, 0.0, 500.0], z_correction_mm=-57.0)
        assert result["position"][2] == pytest.approx(443.0)

    def test_grasp_z_clamped_to_floor(self):
        result, _ = self._call(xyz_raw=[0.0, 0.0, 500.0], z_floor=600.0)
        assert result["grasp_z"] == 600.0

    def test_grasp_z_clamped_to_floor_plus_margin(self):
        result, _ = self._call(xyz_raw=[0.0, 0.0, 500.0], z_floor=600.0, floor_margin_mm=8.0)
        assert result["grasp_z"] == 608.0


class TestVisionMixinPipeline:
    """VisionMixin.get_grasp_info_simple / pixel_to_base_xyz drive the shared
    pipeline, delegating only the projection to the per-adapter seam."""

    @pytest.fixture(autouse=True)
    def _no_debug_dump(self, monkeypatch):
        # The shared pipeline dumps debug artifacts best-effort; silence it here.
        monkeypatch.setattr("jiuwensymbiosis.api.mixins.dump_grasp_debug", lambda **_kwargs: None)

    def _setup(self, *, depth_m=0.5, z_min_safe=0.0, seg_fn=None):
        from tests.mocks.mock_detector import make_mock_seg_fn

        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), float(depth_m), dtype=np.float32)
        intrinsics = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        ll = _MockLowLevel(rgb, depth, np.eye(4), intrinsics)
        pose = type("P", (), {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0, "ry": 0, "rz": 0})()
        env = _MockEnv(ll, pose, z_min_safe=z_min_safe)
        return _MixinStub(env, seg_fn or make_mock_seg_fn(score=0.8))

    def test_success_shape(self):
        api = self._setup()
        result = api.get_grasp_info_simple("box")
        assert result["ok"] is True
        assert result["object"] == "box"
        assert set(result) == {
            "ok",
            "object",
            "position",
            "grasp_z",
            "grasp_position",
            "place_z",
            "place_position",
            "score",
            "pixel_uv",
            "depth_m",
        }
        assert result["position"][2] == pytest.approx(result["depth_m"] * 1000.0)

    def test_grasp_z_clamped_to_z_min_safe(self):
        api = self._setup(depth_m=0.5, z_min_safe=600.0)
        result = api.get_grasp_info_simple("box")
        assert result["ok"] is True
        assert result["grasp_z"] >= 600.0

    def test_no_camera(self):
        api = self._setup()
        api.env.low_level.grab_frames = lambda: None
        result = api.get_grasp_info_simple("box")
        assert result["ok"] is False
        assert result["reason"] == "no_camera"

    def test_propagates_detection_failure(self):
        from tests.mocks.mock_detector import make_mock_seg_fn

        api = self._setup(seg_fn=make_mock_seg_fn(returns_empty=True))
        result = api.get_grasp_info_simple("box")
        assert result["ok"] is False
        assert result["reason"] == "no_detection"

    def test_pixel_to_base_xyz_returns_xyz(self):
        api = self._setup(depth_m=0.5)
        result = api.pixel_to_base_xyz(320.0, 240.0, 0.5)
        assert set(result) == {"x", "y", "z"}
        # Principal point (320,240), identity projection → x=0, y=0, z=depth*1000.
        assert abs(result["x"]) < 1e-6
        assert abs(result["y"]) < 1e-6
        assert abs(result["z"] - 500.0) < 1e-6

    def test_projection_seam_is_invoked(self):
        api = self._setup()
        calls = []
        original = api._project_pixel_to_base_raw

        def _wrapped(u, v, depth_m):
            calls.append((u, v, depth_m))
            return original(u, v, depth_m)

        api._project_pixel_to_base_raw = _wrapped
        api.get_grasp_info_simple("box")
        assert len(calls) == 1
