# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Exercise target safety through real Piper motion paths, with only CAN faked."""

from math import cos, radians, sqrt

import numpy as np
import pytest

from jiuwensymbiosis.adapters.piper import api as api_module
from jiuwensymbiosis.adapters.piper.geometry import FlangePose
from jiuwensymbiosis.agent.fast.realtime.binding import ServoBinding
from jiuwensymbiosis.agent.fast.realtime.servo import ServoConfig, ServoController
from jiuwensymbiosis.errors import SafetyViolationError
from jiuwensymbiosis.rails.safety import SafetyRail


def _send_flange(robot, pose, mode):
    if mode == "blocking":
        robot.env.move_to_flange(FlangePose(*pose))
    else:
        robot.env.servo_to_flange(dict(zip(("x", "y", "z", "rx", "ry", "rz"), pose, strict=True)))


@pytest.mark.parametrize("mode", ["blocking", "servo"])
@pytest.mark.parametrize("angles, accepted", [((0, 0, 0), True), ((0, 90, 0), False), ((180, 0, 0), False)])
def test_floor_constrains_target_tip_instead_of_flange_height(piper_factory, mode, angles, accepted):
    robot = piper_factory(tool_offset_mm=135.8)
    target = (200, 20, 40, *angles)
    if accepted:
        _send_flange(robot, target, mode)
        assert robot.arm.commands[-1][2] == 40000
        assert robot.api.get_pose()["z"] == pytest.approx(175.8)
    else:
        with pytest.raises(SafetyViolationError, match="TIP z=.*below"):
            _send_flange(robot, target, mode)
        assert robot.arm.commands == []


@pytest.mark.parametrize("calibration", ["none", "anchorless", "anchored"])
@pytest.mark.parametrize("mode", ["blocking", "servo"])
def test_flange_floor_cannot_ignore_tool_extension(piper_factory, calibration, mode):
    robot = piper_factory(calibration=calibration, tool_offset_mm=95)
    assert robot.env.z_min_safe == 50
    assert robot.env.low_level.flange_z_min_safe == 145
    with pytest.raises(SafetyViolationError, match="TIP z=.*below") as exc:
        _send_flange(robot, (200, 20, 50, 180, 30, 0), mode)
    assert exc.value.code == "safety_rejected"
    assert robot.arm.commands == []


@pytest.mark.parametrize(
    "calibration, mode, length",
    [
        pytest.param("none", "blocking", 95, id="config-floor"),
        pytest.param("anchorless", "blocking", 95, id="anchorless-floor"),
        pytest.param("anchored", "blocking", 95, id="anchor-floor"),
        pytest.param("none", "servo", 135.8, id="servo-fractional-offset"),
        pytest.param("none", "blocking", 0, id="zero-offset"),
    ],
)
def test_tip_floor_is_commandable(piper_factory, calibration, mode, length):
    robot = piper_factory(calibration=calibration, tool_offset_mm=length)
    target = {"x": 200, "y": 20, "z": 50, "r": 90}
    SafetyRail(robot).validate_motion("goto_xyzr", target)
    if mode == "blocking":
        robot.api.goto_xyzr(**target)
    else:
        ServoBinding(robot).servo_to(target)
    x, y, z, rx, ry, rz = (v / 1000 for v in robot.arm.commands[-1])
    assert (x, y, rx, ry, rz) == pytest.approx((200, 20 + length / 2, 180, 30, 90))
    # Independently evaluate the encoded command: rounding may lift by < 1 micron.
    actual_tip_z = z - sqrt(3) * length / 2
    assert 50 <= actual_tip_z < 50.001
    assert robot.api.get_pose()["z"] == pytest.approx(actual_tip_z)


@pytest.mark.parametrize("mode", ["blocking", "servo"])
def test_floor_uses_target_orientation_instead_of_live_pose(piper_factory, mode):
    robot = piper_factory(tool_offset_mm=95, initial_pose=(200, 20, 400, 180, 30, 0))
    # A target pointing up is safe with the flange at the TIP floor.
    _send_flange(robot, (200, 20, 50, 0, 0, 0), mode)
    assert len(robot.arm.commands) == 1
    # The live pose now points up; turning down at the same height is unsafe.
    with pytest.raises(SafetyViolationError, match="TIP z=.*below"):
        _send_flange(robot, (200, 20, 50, 180, 30, 0), mode)
    assert len(robot.arm.commands) == 1


def test_servo_missing_orientation_uses_live_pose_for_safety(piper_factory):
    robot = piper_factory(tool_offset_mm=95)
    with pytest.raises(SafetyViolationError, match="TIP z=.*below"):
        robot.env.servo_to_flange({"x": 200, "y": 20, "z": 50})
    assert robot.arm.commands == []


def test_servo_cannot_invent_missing_orientation_after_read_failure(piper_factory, monkeypatch):
    robot = piper_factory(tool_offset_mm=95)

    def fail_read():
        raise OSError("CAN read failed")

    monkeypatch.setattr(robot.arm, "GetArmEndPoseMsgs", fail_read)
    with pytest.raises(RuntimeError, match="failed to read end pose"):
        robot.env.servo_to_flange({"x": 200, "y": 20, "z": 50})
    assert robot.arm.commands == []
    # A complete safe target has all geometry needed even without readback.
    robot.env.servo_to_flange({"x": 200, "y": 20, "z": 50, "rx": 0, "ry": 0, "r": 90})
    assert robot.arm.commands == [(200000, 20000, 50000, 0, 0, 90000)]


@pytest.mark.parametrize("mode", ["blocking", "servo"])
@pytest.mark.parametrize("yaw, y", [(90, 480), (-90, -480)])
def test_yaw_offset_outside_flange_y_is_rejected_without_clamping(piper_factory, mode, yaw, y):
    robot = piper_factory(tool_offset_mm=95)
    target = {"x": 200, "y": y, "z": 100, "r": yaw}
    SafetyRail(robot).validate_motion("goto_xyzr", target)  # TIP itself is in bounds.
    with pytest.raises(SafetyViolationError, match="FLANGE y=.*out of bounds"):
        if mode == "blocking":
            robot.api.goto_xyzr(**target)
        else:
            ServoBinding(robot).servo_to(target)
    assert robot.arm.commands == []


def test_servo_reports_boundary_failure_instead_of_timing_out(piper_factory):
    robot = piper_factory(tool_offset_mm=100, initial_pose=(200, 500, 300, 180, 30, 90))
    binding = ServoBinding(robot)
    target = {**binding.read_pose(), "y": 480}
    result = ServoController(
        binding.read_pose,
        binding.servo_to,
        lambda: target,
        config=ServoConfig(max_lin_step_mm=6, timeout_s=1, absolute_timeout_s=2),
    ).run()
    assert not result.ok
    assert result.reason == "stopped"
    assert result.error_code == "safety_rejected"
    assert robot.arm.commands == []


@pytest.mark.parametrize("mode", ["blocking", "servo"])
@pytest.mark.parametrize(
    "pose, message",
    [
        ((10, 0, 200, 180, 30, 0), "TIP x=.*out of bounds"),
        ((200, -480, 200, 180, 30, 90), "TIP y=.*out of bounds"),
        ((701, 0, 200, 180, 30, 0), "FLANGE x=.*out of bounds"),
        ((200, 0, 801, 180, 30, 0), "FLANGE z=.*above"),
    ],
)
def test_driver_checks_both_frames_and_flange_ceiling(piper_factory, mode, pose, message):
    robot = piper_factory(tool_offset_mm=95)
    with pytest.raises(SafetyViolationError, match=message):
        _send_flange(robot, pose, mode)
    assert robot.arm.commands == []


@pytest.mark.parametrize("home_use_init_pose", [False, True])
def test_home_cannot_bypass_tip_floor(piper_factory, home_use_init_pose):
    robot = piper_factory(
        tool_offset_mm=95,
        home_use_init_pose=home_use_init_pose,
        initial_pose=(200, 20, 50, 180, 30, 0),
        home_pose_xyzrxryrz_mm_deg=[200, 20, 50, 180, 30, 0],
    )
    with pytest.raises(SafetyViolationError, match="TIP z=.*below"):
        robot.api.home()
    assert robot.arm.commands == []


@pytest.mark.parametrize("mode", ["blocking", "servo"])
@pytest.mark.parametrize("field", range(6))
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_pose_never_reaches_can(piper_factory, mode, field, value):
    robot = piper_factory(tool_offset_mm=95)
    pose = [200, 20, 300, 180, 30, 0]
    pose[field] = value
    with pytest.raises(SafetyViolationError, match="non-finite"):
        _send_flange(robot, pose, mode)
    assert robot.arm.commands == []


@pytest.mark.parametrize("mode", ["blocking", "servo"])
def test_encoded_orientation_is_used_when_rounding_z_at_floor(piper_factory, mode):
    robot = piper_factory(tool_offset_mm=135.8)
    ry = 30.00049  # rounds down, making the tool extend slightly farther downward
    z = 50 + 135.8 * cos(radians(ry))
    _send_flange(robot, (200, 20, z, 180, ry, 45), mode)
    encoded = robot.arm.commands[-1]
    assert encoded[4] == 30000
    actual_tip_z = encoded[2] / 1000 - 135.8 * sqrt(3) / 2
    assert 50 <= actual_tip_z < 50.001


def test_rounding_cannot_hide_an_unsafe_request(piper_factory):
    robot = piper_factory(tool_offset_mm=95)
    with pytest.raises(SafetyViolationError, match="TIP z=.*below"):
        robot.api.goto_xyzr(200, 20, 49.9999, 0)
    assert robot.arm.commands == []


def test_floor_rounding_that_exceeds_ceiling_is_rejected(piper_factory):
    robot = piper_factory(tool_offset_mm=95, z_max_mm=132.2725)
    # Raw legal flange Z = 132.272413...; no SDK Z can satisfy both boundaries.
    with pytest.raises(SafetyViolationError, match="FLANGE z=.*above"):
        robot.api.goto_xyzr(200, 20, 50, 0)
    assert robot.arm.commands == []


def test_quantized_xy_is_checked_before_sending(piper_factory):
    robot = piper_factory(tool_offset_mm=0, x_max_mm=200.0008)
    with pytest.raises(SafetyViolationError, match="FLANGE x=.*out of bounds"):
        robot.api.goto_xyzr(200.0007, 20, 100, 0)
    assert robot.arm.commands == []


def test_disabled_workspace_sides_allow_the_requested_target(piper_factory):
    robot = piper_factory(tool_offset_mm=95, y_max_mm=None, z_max_mm=None)
    robot.api.goto_xyzr(200, 600, 900, 90)
    assert robot.arm.commands[-1][1] == 647500
    assert robot.api.get_pose()["y"] == pytest.approx(600)


def test_grasp_height_clamped_to_floor_can_be_commanded(piper_factory, monkeypatch):
    robot = piper_factory(calibration="anchored", tool_offset_mm=95)
    monkeypatch.setattr(robot.env.low_level, "grab_frames", lambda: (np.zeros((2, 2, 3)), np.ones((2, 2))))
    monkeypatch.setattr(robot.api, "_ensure_detector", lambda: None)
    monkeypatch.setattr(
        api_module,
        "detect_and_centroid",
        lambda **kwargs: {
            "ok": True,
            "u": 0,
            "v": 0,
            "depth_m": 0.1,
            "best": {"score": 0.9},
            "img_shape": (2, 2),
            "mask_shape": (2, 2),
        },
    )
    monkeypatch.setattr(api_module, "pixel_and_depth_to_base_xyz", lambda *args: np.array([200.0, 20.0, 60.0]))
    monkeypatch.setattr(api_module, "dump_grasp_debug", lambda **kwargs: None)
    grasp = robot.api.get_grasp_info_simple("block")
    assert grasp["grasp_z"] == 50  # detected top 60 minus grasp offset 25, floored at 50
    robot.api.goto_xyzr(*grasp["grasp_position"], r=0)
    assert 50 <= robot.api.get_pose()["z"] < 50.001
