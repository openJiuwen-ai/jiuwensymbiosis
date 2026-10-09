# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Public observation, planning, and tracing must describe the controlled TIP."""

from math import sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.adapters.piper import env as env_module
from jiuwensymbiosis.agent.trace import _observation_snapshot
from jiuwensymbiosis.api.world_state import WorldState
from jiuwensymbiosis.calibration.adapters.piper import PiperCalibrationDevice


@pytest.mark.parametrize(
    "angles, offset",
    [((180, 30, 0), (-50, 0, -50 * sqrt(3))), ((180, 30, 90), (0, -50, -50 * sqrt(3))), ((0, 90, 0), (100, 0, 0))],
)
def test_observation_planning_and_trace_share_tip_coordinates(piper_factory, angles, offset):
    robot = piper_factory(tool_offset_mm=100, initial_pose=(200, 20, 300, *angles))
    robot.arm.calls.clear()
    obs = robot.env.get_observation()
    assert robot.arm.calls == ["read_pose"]
    tip = {key: coord + shift for key, coord, shift in zip(("x", "y", "z"), (200, 20, 300), offset, strict=True)}
    tip.update(zip(("rx", "ry", "rz"), angles, strict=True))
    flange = dict(zip(("x", "y", "z", "rx", "ry", "rz"), (200, 20, 300, *angles), strict=True))
    assert obs.pose == pytest.approx(tip)
    assert obs.extra["flange_pose"] == flange
    assert robot.api.get_pose() == pytest.approx(tip)
    world = WorldState.snapshot(robot)
    assert world.pose == pytest.approx(tip)
    assert world.describe()["pose"] == pytest.approx(tip)  # state CLI's payload
    trace = _observation_snapshot(robot.env, observation=obs)
    assert trace["pose"] == pytest.approx(tip)
    assert trace["extra"]["flange_pose"] == flange

    # Calibration keeps using raw flange geometry, independently of observations.
    tf = PiperCalibrationDevice(robot.env).get_flange_transform_mm()
    assert tf[:3, 3] == pytest.approx((200, 20, 300))
    assert robot.env.get_flange_pose().as_tuple() == (200, 20, 300, *angles)


def test_missing_pose_sample_does_not_report_stale_coordinates(piper_factory, monkeypatch):
    robot = piper_factory()

    def failed_read():
        raise OSError("CAN read failed")

    monkeypatch.setattr(robot.arm, "GetArmEndPoseMsgs", failed_read)
    obs = robot.env.get_observation()
    assert obs.pose is None
    assert obs.extra["flange_pose"] is None


def test_tip_conversion_failure_publishes_neither_pose_but_keeps_other_sensing(piper_factory, monkeypatch, caplog):
    robot = piper_factory()
    rgb, depth = np.zeros((2, 2, 3)), np.ones((2, 2))
    monkeypatch.setattr(robot.env.low_level, "grab_frames", lambda: (rgb, depth))
    monkeypatch.setattr(robot.env.low_level, "get_angles", lambda: SimpleNamespace(as_tuple=lambda: (1, 2, 3, 4, 5, 6)))

    def fail_conversion(*args):
        raise ValueError("TIP conversion failed")

    monkeypatch.setattr(env_module, "flange_pose_to_tip", fail_conversion)
    robot.arm.calls.clear()
    with caplog.at_level("DEBUG", logger=env_module.__name__):
        obs = robot.env.get_observation()
    assert robot.arm.calls == ["read_pose"]
    assert obs.pose is None
    assert obs.extra["flange_pose"] is None
    assert obs.rgb is rgb
    assert obs.depth is depth
    assert obs.joints == [1, 2, 3, 4, 5, 6]
    assert "TIP conversion failed" in caplog.text
