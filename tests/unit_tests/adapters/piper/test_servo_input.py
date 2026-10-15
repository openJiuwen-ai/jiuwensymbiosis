# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared pose carriers and distinct blocking/servo completion policies."""

from types import MappingProxyType, SimpleNamespace

import pytest

from jiuwensymbiosis.adapters.piper.geometry import FlangePose
from jiuwensymbiosis.agent.fast.realtime.binding import ServoBinding
from jiuwensymbiosis.errors import SafetyViolationError


@pytest.mark.parametrize(
    "pose",
    [
        pytest.param({"x": 250, "y": 30, "z": 300, "rx": 180, "ry": 30, "rz": 90}, id="dict"),
        pytest.param(
            MappingProxyType({"x": 250, "y": 30, "z": 300, "rx": 180, "ry": 30, "rz": 90}), id="readonly-mapping"
        ),
        pytest.param(SimpleNamespace(x=250, y=30, z=300, rx=180, ry=30, rz=90), id="attributes"),
        pytest.param(FlangePose(250, 30, 300, 180, 30, 90), id="native-flange"),
    ],
)
@pytest.mark.parametrize("mode", ["blocking", "servo"])
def test_complete_target_uses_the_same_command_in_both_modes(piper_factory, pose, mode):
    robot = piper_factory()
    robot.arm.calls.clear()

    if mode == "blocking":
        robot.env.move_to_flange(pose)
    else:
        robot.env.servo_to_flange(pose)

    assert robot.arm.commands == [(250000, 30000, 300000, 180000, 30000, 90000)]
    # Blocking reads only to poll completion, after dispatch; servo never reads.
    assert robot.arm.calls[:2] == ["motion_mode", "end_pose"]
    if mode == "servo":
        assert robot.arm.calls == ["motion_mode", "end_pose"]


@pytest.mark.parametrize("mode", ["blocking", "servo"])
@pytest.mark.parametrize("explicit_rz", [False, True])
def test_yaw_alias_and_rz_precedence_are_shared(piper_factory, mode, explicit_rz):
    values = {"x": "250", "y": 30, "z": 300, "rx": 180, "ry": 30, "r": -90}
    if explicit_rz:
        values["rz"] = 90
    robot = piper_factory()
    send = robot.env.move_to_flange if mode == "blocking" else robot.env.servo_to_flange
    send(MappingProxyType(values))
    assert robot.arm.commands == [(250000, 30000, 300000, 180000, 30000, 90000 if explicit_rz else -90000)]


@pytest.mark.parametrize("frame", ["tip", "flange"])
@pytest.mark.parametrize(
    "yaw_fields, expected_yaw, readback",
    [
        ({"r": 45, "rz": 90}, 90, False),
        ({"r": 45, "rz": 0}, 0, False),
        ({"r": 45}, 45, False),
        ({"rz": 90}, 90, False),
        ({"r": 45, "rz": None}, 45, False),
        ({"r": None, "rz": 90}, 90, False),
        ({}, -90, True),
        ({"r": None, "rz": None}, -90, True),
    ],
)
def test_tip_and_flange_servo_share_yaw_precedence_and_readback_policy(
    piper_factory, frame, yaw_fields, expected_yaw, readback
):
    robot = piper_factory(initial_pose=(200, 20, 400, 180, 30, -90))
    robot.arm.calls.clear()
    send = robot.api.servo_to_tip if frame == "tip" else robot.env.servo_to_flange
    send({"x": 200, "y": 20, "z": 300, "rx": 180, "ry": 30, **yaw_fields})

    # Positions use different frames; both inputs must select the same orientation.
    assert robot.arm.commands[-1][3:] == (180000, 30000, expected_yaw * 1000)
    assert robot.arm.calls == (["read_pose"] if readback else []) + ["motion_mode", "end_pose"]


def test_blocking_mapping_requires_orientation_without_readback(piper_factory):
    robot = piper_factory()
    robot.arm.calls.clear()
    with pytest.raises(TypeError, match="missing required fields: rx, ry"):
        robot.env.move_to_flange(MappingProxyType({"x": 250, "y": 30, "z": 300, "r": 90}))
    assert robot.arm.calls == []


@pytest.mark.parametrize("carrier", [MappingProxyType, lambda values: SimpleNamespace(**values)])
def test_servo_fills_only_missing_orientation_and_preserves_yaw_alias(piper_factory, carrier):
    robot = piper_factory(initial_pose=(200, 20, 400, 180, 30, -90))
    robot.arm.calls.clear()

    robot.env.servo_to_flange(carrier({"x": 250, "y": 30, "z": 300, "rx": 170, "r": 45}))

    assert robot.arm.commands == [(250000, 30000, 300000, 170000, 30000, 45000)]
    assert robot.arm.calls == ["read_pose", "motion_mode", "end_pose"]


@pytest.mark.parametrize(
    "pose, missing",
    [
        pytest.param(MappingProxyType({"y": 30, "z": 300}), "x", id="missing-x"),
        pytest.param(SimpleNamespace(x=250, z=300), "y", id="missing-y"),
        pytest.param({"x": 250, "y": 30}, "z", id="missing-z"),
        pytest.param({"x": 250, "y": 30, "z": None}, "z", id="null-z"),
    ],
)
def test_invalid_servo_input_fails_before_readback_or_dispatch(piper_factory, pose, missing):
    robot = piper_factory()
    robot.arm.calls.clear()

    with pytest.raises(TypeError, match=f"missing required fields: {missing}"):
        robot.env.servo_to_flange(pose)

    assert robot.arm.commands == []
    assert robot.arm.calls == []


@pytest.mark.parametrize("field", ["x", "y", "z"])
@pytest.mark.parametrize("null_value", [False, True])
def test_tip_binding_rejects_missing_coordinates_before_readback_or_dispatch(piper_factory, field, null_value):
    robot = piper_factory()
    binding = ServoBinding(robot)
    pose = {"x": 200, "y": 20, "z": 300}
    if null_value:
        pose[field] = None
    else:
        del pose[field]
    robot.arm.calls.clear()

    with pytest.raises(TypeError, match=f"Piper TIP pose missing required fields: {field}"):
        binding.servo_to(pose)

    assert robot.arm.calls == []


def test_tip_servo_rejects_unsupported_carrier_before_readback_or_dispatch(piper_factory):
    robot = piper_factory()
    robot.arm.calls.clear()
    with pytest.raises(TypeError, match="Piper TIP pose must be a mapping"):
        robot.api.servo_to_tip(None)
    assert robot.arm.calls == []


@pytest.mark.parametrize("mode", ["blocking", "servo"])
@pytest.mark.parametrize(
    "pose, error, message",
    [
        (object(), TypeError, "unsupported type object"),
        ({"x": "bad", "y": 20, "z": 300, "rx": 180, "ry": 30, "rz": 0}, ValueError, "field 'x' must be numeric"),
    ],
)
def test_shared_input_diagnostics_precede_readback_or_dispatch(piper_factory, mode, pose, error, message):
    robot = piper_factory()
    robot.arm.calls.clear()
    send = robot.env.move_to_flange if mode == "blocking" else robot.env.servo_to_flange
    with pytest.raises(error, match=message):
        send(pose)
    assert robot.arm.calls == []


@pytest.mark.parametrize(
    "pose",
    [
        MappingProxyType({"x": 250, "y": 30, "z": 50, "rx": 180, "ry": 30, "rz": 90}),
        FlangePose(250, 30, 50, 180, 30, 90),
    ],
)
def test_normalized_servo_input_still_enforces_tip_floor(piper_factory, pose):
    robot = piper_factory(tool_offset_mm=95)
    robot.arm.calls.clear()

    with pytest.raises(SafetyViolationError, match="TIP z=.*below"):
        robot.env.servo_to_flange(pose)

    assert robot.arm.commands == []
    assert robot.arm.calls == []
