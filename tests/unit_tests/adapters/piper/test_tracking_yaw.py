# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Exercise runner target generation, real servo slew, and Piper CAN dispatch."""

from math import cos, radians, sin

import pytest

from jiuwensymbiosis.agent.fast import runner
from jiuwensymbiosis.agent.fast.realtime.servo import ServoConfig


@pytest.mark.parametrize("yaw_key", ["rz", "r"])
def test_track_detect_restores_entry_yaw_with_slew_and_completion_checks(piper_factory, monkeypatch, yaw_key):
    robot = piper_factory(initial_pose=(200, 20, 400, 180, 30, 45))
    native_read = robot.api.get_pose
    entry = native_read()
    if yaw_key == "r":
        # Exercise the generic runner's four-DoF readback convention as well.
        def read_four_dof():
            sample = native_read()
            return {"x": sample["x"], "y": sample["y"], "z": sample["z"], "r": sample["rz"]}

        monkeypatch.setattr(robot.api, "get_pose", read_four_dof)

    detection = {"x": 250.0, "y": 30.0, "z": 60.0}
    stopped = []

    class StaticTracker:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            # Disturb yaw AFTER the runner samples its entry pose. Keep TIP XYZ
            # fixed so a lost angular target cannot hide behind XY completion.
            horizontal_offset = robot.env.tool_offset_mm / 2
            drifted_flange = (
                entry["x"] + horizontal_offset * cos(radians(12.5)),
                entry["y"] + horizontal_offset * sin(radians(12.5)),
                400,
                180,
                30,
                12.5,
            )
            robot.arm.pose = tuple(round(value * 1000) for value in drifted_flange)

        def wait_first(self, *args, **kwargs):
            return True

        def latest_target(self):
            return dict(detection)

        def target_is_live(self, **kwargs):
            return True

        def stop(self):
            stopped.append(True)

        def raise_if_failed(self):
            pass

    monkeypatch.setattr(runner, "BackgroundTracker", StaticTracker)
    cfg = runner.TrackingConfig(
        servo=ServoConfig(
            max_ang_step_deg=5,
            pos_tol_mm=0.01,
            ang_tol_deg=0.01,
            settle_ticks=1,
            timeout_s=1,
            absolute_timeout_s=2,
        )
    )

    result = runner._track_detect(robot, "block", cfg, {}, occluded=False)

    assert result == detection
    assert stopped == [True]
    assert [command[-1] / 1000 for command in robot.arm.commands] == [17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 45]
    actual = native_read()
    assert actual["rz"] == 45
    assert [actual[key] for key in ("x", "y", "z")] == pytest.approx([entry[key] for key in ("x", "y", "z")], abs=0.001)
