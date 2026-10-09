# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Relative public actions must reach the real driver in the correct frame."""

from math import sqrt

import pytest

from jiuwensymbiosis.errors import SafetyViolationError


@pytest.mark.parametrize(
    "direction, delta",
    [
        ("forward", (10, 0, 0)),
        ("back", (-10, 0, 0)),
        ("left", (0, 10, 0)),
        ("right", (0, -10, 0)),
        ("up", (0, 0, 10)),
        ("down", (0, 0, -10)),
    ],
)
@pytest.mark.parametrize(
    "angles, offset", [((180, 30, 90), (0, -50, -50 * sqrt(3))), ((0, 90, 45), (50 * sqrt(2), 50 * sqrt(2), 0))]
)
def test_relative_action_preserves_orientation_and_reports_tip(piper_factory, direction, delta, angles, offset):
    robot = piper_factory(tool_offset_mm=100, initial_pose=(200, 20, 300, *angles))
    robot.arm.calls.clear()

    result = robot.api.move_direction(direction, 10)

    flange_xyz = tuple(start + step for start, step in zip((200, 20, 300), delta, strict=True))
    expected_tip = {axis: coord + shift for axis, coord, shift in zip(("x", "y", "z"), flange_xyz, offset, strict=True)}
    assert robot.arm.commands == [tuple(round(value * 1000) for value in (*flange_xyz, *angles))]
    assert result["pose"] == pytest.approx(expected_tip)
    assert result["ok"] is True
    assert result["direction"] == direction
    assert result["distance_mm"] == 10
    # Read exactly one initial sample; later reads are driver completion polling.
    assert robot.arm.calls[: robot.arm.calls.index("end_pose")].count("read_pose") == 1
    assert {key: robot.api.get_pose()[key] for key in expected_tip} == pytest.approx(expected_tip)


@pytest.mark.parametrize(
    "initial_pose, direction, distance, message",
    [
        ((200, 20, 200, 180, 30, 0), "down", 40, "below z_min_safe"),
        ((75, 20, 300, 180, 30, 0), "back", 10, "x=.*out of"),
        ((200, -420, 300, 180, 30, 90), "right", 20, "y=.*out of"),
    ],
)
def test_tip_precheck_rejects_while_flange_is_still_in_bounds(
    piper_factory, initial_pose, direction, distance, message
):
    robot = piper_factory(tool_offset_mm=135.8, initial_pose=initial_pose)
    robot.arm.calls.clear()
    with pytest.raises(ValueError, match=message):
        robot.api.move_direction(direction, distance)
    assert robot.arm.commands == []
    assert robot.arm.calls == ["read_pose"]


def test_tip_precheck_allows_safe_flange_below_tip_floor(piper_factory):
    robot = piper_factory(tool_offset_mm=100, initial_pose=(200, 20, 60, 0, 0, 90))
    result = robot.api.move_direction("down", 20)
    assert robot.arm.commands[-1] == (200000, 20000, 40000, 0, 0, 90000)
    assert result["pose"] == pytest.approx({"x": 200, "y": 20, "z": 140})


def test_driver_still_rejects_flange_outside_bounds_when_tip_passes(piper_factory):
    robot = piper_factory(tool_offset_mm=100, initial_pose=(200, 490, 300, 180, 30, 90))
    with pytest.raises(SafetyViolationError, match="FLANGE y=.*out of bounds"):
        robot.api.move_direction("left", 20)
    assert robot.arm.commands == []
