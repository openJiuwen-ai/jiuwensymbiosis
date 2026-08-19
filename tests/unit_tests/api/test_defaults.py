# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.api.defaults — the reusable action implementations.

They are plain functions taking the api first, so they are tested directly: no body
has to be assembled to exercise the generic behaviour, which is the point of the shape.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jiuwensymbiosis.api import defaults


def _rec_env(**extra):
    calls: list[tuple] = []
    env = SimpleNamespace(
        navigate_relative=lambda dx, dy, dyaw: calls.append(("nav", dx, dy, dyaw)) or {"ok": True},
        navigate_arc=lambda r, dyaw: calls.append(("arc", r, dyaw)) or {"ok": True},
        set_lifter=lambda q: calls.append(("lift", q)) or {"ok": True},
        turn_waist=lambda d: calls.append(("waist", d)) or {"ok": True},
        set_end_effector=lambda on: calls.append(("ee", on)),
        move_joint=lambda q: calls.append(("joint", list(q))),
        home=lambda: calls.append(("home",)),
        **extra,
    )
    return SimpleNamespace(env=env), calls


class TestEnvDelegation:
    """Each generic action is exactly one Env verb — that is why it needs no adapter code."""

    def test_base_motion(self):
        api, calls = _rec_env()
        defaults.navigate_relative(api, 0.3, 0.0, 0.1)
        defaults.rotate_base(api, 0.2)
        assert ("nav", 0.3, 0.0, 0.1) in calls
        assert ("nav", 0.0, 0.0, 0.2) in calls  # rotate_base cannot translate — that IS its contract

    def test_drive_arc(self):
        api, calls = _rec_env()
        defaults.drive_arc(api, 0.8, -0.5)
        assert calls == [("arc", 0.8, -0.5)]

    def test_lift_and_waist(self):
        api, calls = _rec_env()
        defaults.set_lift_pose(api, {"lifter_pitch_1_joint": 0.1})
        defaults.turn_waist(api, 0.5)
        assert calls == [("lift", {"lifter_pitch_1_joint": 0.1}), ("waist", 0.5)]

    def test_home_and_joint(self):
        api, calls = _rec_env()
        defaults.home(api)
        defaults.move_joint(api, [0.0, 1.0])
        assert calls == [("home",), ("joint", [0.0, 1.0])]

    @pytest.mark.parametrize(
        ("fn", "arg", "expected_state"),
        [
            (defaults.activate_suction, None, True),
            (defaults.deactivate_suction, None, False),
            (defaults.close_gripper, None, True),
            (defaults.open_gripper, 80.0, False),
        ],
    )
    def test_end_effector(self, fn, arg, expected_state):
        api, calls = _rec_env()
        res = fn(api) if arg is None else fn(api, arg)
        assert res["ok"] is True
        assert calls == [("ee", expected_state)]


class _FakeMotionEnv:
    """Minimal env exposing just what move_direction needs."""

    z_min_safe = 20.0
    workspace_bounds = (-300.0, -300.0, 300.0, 300.0)

    def __init__(self):
        self._pose = SimpleNamespace(x=100.0, y=0.0, z=200.0, rx=180.0, ry=0.0, rz=0.0)
        self.moved_to = None

    def get_flange_pose(self):
        return self._pose

    def move_to_flange(self, pose):
        self.moved_to = pose
        self._pose = pose


class TestMoveDirection:
    """move_direction is outside SafetyRail's watch set, so it does its own checking."""

    def test_left_moves_plus_y(self):
        env = _FakeMotionEnv()
        res = defaults.move_direction(SimpleNamespace(env=env), "left", 20)
        assert res["ok"] is True
        assert env.moved_to.y == 20.0  # left = +y
        assert env.moved_to.x == 100.0
        assert env.moved_to.z == 200.0

    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="out of"):
            defaults.move_direction(SimpleNamespace(env=_FakeMotionEnv()), "right", 10_000)

    def test_unknown_direction_raises(self):
        with pytest.raises(ValueError, match="unknown direction"):
            defaults.move_direction(SimpleNamespace(env=_FakeMotionEnv()), "sideways", 10)

    @pytest.mark.parametrize("distance", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_distance_never_reaches_driver(self, distance):
        env = _FakeMotionEnv()
        env.workspace_bounds = None

        with pytest.raises(ValueError, match="finite and positive"):
            defaults.move_direction(SimpleNamespace(env=env), "up", distance)

        assert env.moved_to is None

    def test_non_finite_target_never_reaches_driver(self):
        env = _FakeMotionEnv()
        env.workspace_bounds = None
        env._pose.x = float("nan")

        with pytest.raises(ValueError, match="target coordinates must be finite"):
            defaults.move_direction(SimpleNamespace(env=env), "up", 10)

        assert env.moved_to is None


class TestPoseHelpers:
    def test_pose_to_dict_emits_only_the_fields_the_pose_has(self):
        assert defaults.pose_to_dict(SimpleNamespace(x=1, y=2, z=3, r=4)) == {"x": 1.0, "y": 2.0, "z": 3.0, "r": 4.0}

    def test_goto_xyzr_keeps_the_current_yaw_when_r_is_omitted(self):
        env = _FakeMotionEnv()
        env._pose.rz = 42.0
        defaults.goto_xyzr(SimpleNamespace(env=env), 150.0, 0.0, 100.0)
        assert env.moved_to.rz == 42.0
        assert (env.moved_to.rx, env.moved_to.ry) == (180.0, 0.0)
