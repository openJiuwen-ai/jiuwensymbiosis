# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reject invalid effective workspace configuration before even constructing CAN."""

import sys
from types import SimpleNamespace

import pytest

from jiuwensymbiosis.adapters.piper import lowlevel


def test_can_precheck_precedes_sdk_construction_and_enable(piper_factory):
    calls = []
    piper_factory(sdk_calls=calls)
    assert calls[:4] == ["can_precheck", "construct", "connect", "enable"]


@pytest.mark.parametrize("calibration", ["none", "anchorless"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_tip_floor_is_rejected_before_can_precheck(piper_factory, calibration, value):
    calls = []
    with pytest.raises(ValueError, match="z_min_safe must be finite"):
        piper_factory(calibration=calibration, z_min_safe_mm=value, sdk_calls=calls)
    assert calls == []


@pytest.mark.parametrize("field", range(6))
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_home_coordinate_or_angle_is_rejected_before_can(piper_factory, field, value):
    home = [200, 20, 300, 180, 30, 0]
    home[field] = value
    calls = []
    with pytest.raises(ValueError, match="home pose must contain finite"):
        piper_factory(home_pose_xyzrxryrz_mm_deg=home, sdk_calls=calls)
    assert calls == []


@pytest.mark.parametrize(
    "config, message",
    [
        ({"home_pose_xyzrxryrz_mm_deg": None}, "set home_pose_xyzrxryrz_mm_deg"),
        ({"home_pose_xyzrxryrz_mm_deg": [200, 20, 300]}, "set home_pose_xyzrxryrz_mm_deg"),
        (
            {"calib_object_xyzrxryrz_mm_deg": [0, 0, float("nan"), 0, 0, 0]},
            "calibration object pose must contain finite",
        ),
        ({"calib_object_xyzrxryrz_mm_deg": [1, 2, 3]}, "must be 6-tuple"),
        ({"tool_offset_mm": float("nan")}, "tool_offset_mm must be finite"),
        ({"tool_offset_mm": -1}, "non-negative"),
        ({"x_max_mm": float("inf")}, "workspace bounds must be finite"),
    ],
)
def test_invalid_workspace_configuration_never_touches_can(piper_factory, config, message):
    calls = []
    with pytest.raises(ValueError, match=message):
        piper_factory(sdk_calls=calls, **config)
    assert calls == []


@pytest.mark.parametrize("axis", ["x", "y"])
def test_reversed_workspace_bounds_are_rejected_before_can(piper_factory, axis):
    calls = []
    config = {f"{axis}_min_mm": 700, f"{axis}_max_mm": 0}
    with pytest.raises(ValueError, match=f"{axis}_min_mm must be <= {axis}_max_mm; got 700 > 0"):
        piper_factory(sdk_calls=calls, **config)
    assert calls == []


@pytest.mark.parametrize(
    "bounds",
    [
        {"x_min_mm": 200, "x_max_mm": 200, "y_min_mm": 20, "y_max_mm": 20},
        {"x_min_mm": None},
        {"x_max_mm": None},
        {"y_min_mm": None},
        {"y_max_mm": None},
    ],
)
def test_equal_or_one_sided_bounds_remain_commandable(piper_factory, bounds):
    robot = piper_factory(tool_offset_mm=0, **bounds)
    robot.env.servo_to_flange({"x": 200, "y": 20, "z": 300, "rx": 180, "ry": 30, "rz": 0})
    assert robot.arm.commands == [(200000, 20000, 300000, 180000, 30000, 0)]


@pytest.mark.parametrize(
    "config, message",
    [
        ({"anchor": (float("nan"), 20, 60)}, "pose must contain finite"),
        ({"anchor": (200, 20, float("inf"))}, "z_min_safe must be finite"),
        ({"anchor": (200, 20)}, "must contain 3 coordinates"),
        ({"home_lift_mm": float("nan")}, "home pose must contain finite"),
        ({"home_lift_mm": float("inf")}, "home pose must contain finite"),
        ({"z_safe_margin_mm": float("nan")}, "z_min_safe must be finite"),
        ({"z_safe_margin_mm": 10}, "calibration object z .*below"),
    ],
)
def test_anchor_derived_values_are_validated_before_can(piper_factory, config, message):
    calls = []
    with pytest.raises(ValueError, match=message):
        piper_factory(calibration="anchored", sdk_calls=calls, **config)
    assert calls == []


def test_bad_calibration_file_fails_before_can(tmp_path, monkeypatch):
    def must_not_call(*args):
        pytest.fail("invalid calibration must be rejected before CAN access")

    monkeypatch.setenv("JIUWEN_PIPER_CMD_LOG", "0")
    monkeypatch.setattr(lowlevel, "_require_can_interface", must_not_call)
    monkeypatch.setitem(sys.modules, "piper_sdk", SimpleNamespace(C_PiperInterface_V2=must_not_call))
    with pytest.raises(FileNotFoundError):
        lowlevel.PiperLowLevel(calib_path=tmp_path / "absent-calibration.json")


def test_anchor_ignores_unused_config_fallbacks(piper_factory):
    robot = piper_factory(
        calibration="anchored", tool_offset_mm=95, z_min_safe_mm=float("nan"), home_pose_xyzrxryrz_mm_deg=None
    )
    assert robot.env.z_min_safe == 50
    assert robot.env.home_pose.as_tuple() == (210, 30, 405, 180, 30, 0)


@pytest.mark.parametrize("calibration", ["none", "anchorless", "anchored"])
def test_startup_home_ignores_unused_home_configuration(piper_factory, calibration):
    robot = piper_factory(
        calibration=calibration, home_use_init_pose=True, home_pose_xyzrxryrz_mm_deg=None, home_lift_mm=float("nan")
    )
    assert robot.env.home_pose.as_tuple() == (200, 20, 400, 180, 30, 0)
