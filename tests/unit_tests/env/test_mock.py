# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.env.mock."""

from __future__ import annotations

import numpy as np
import pytest

from jiuwensymbiosis.env.mock import MockArmEnv


class TestMockArmEnvLifecycle:
    def test_connect_disconnect(self):
        env = MockArmEnv()
        assert env._connected is False
        env.connect()
        assert env._connected is True
        env.disconnect()
        assert env._connected is False

    def test_connect_idempotent(self):
        env = MockArmEnv()
        env.connect()
        env.connect()
        assert env._connected is True


class TestMockArmEnvMove:
    def test_move_updates_pose(self):
        env = MockArmEnv()
        env.move(100.0, 50.0, 300.0, 45.0)
        assert env._pose["x"] == pytest.approx(100.0)
        assert env._pose["y"] == pytest.approx(50.0)
        assert env._pose["z"] == pytest.approx(300.0)
        assert env._pose["r"] == pytest.approx(45.0)

    def test_move_without_r_preserves_old_r(self):
        env = MockArmEnv()
        env.move(100.0, 50.0, 300.0, 45.0)
        env.move(110.0, 60.0, 310.0)
        assert env._pose["r"] == pytest.approx(45.0)

    def test_move_appends_log(self):
        env = MockArmEnv()
        env.move(100.0, 50.0, 300.0)
        assert len(env._move_log) == 1

    def test_z_below_floor_raises(self):
        env = MockArmEnv(z_min_safe=50.0)
        with pytest.raises(RuntimeError, match="z="):
            env.move(100.0, 50.0, 30.0)


class TestMockArmEnvHome:
    def test_home_resets_pose(self):
        env = MockArmEnv(home_pose={"x": 200, "y": 0, "z": 250, "r": 0})
        env.move(300.0, 100.0, 350.0)
        env.home()
        assert env._pose["x"] == pytest.approx(200)

    def test_home_pose_returns_copy(self):
        env = MockArmEnv()
        hp = env.home_pose
        assert hp["x"] == pytest.approx(env._home["x"])


class TestMockArmEnvSuction:
    def test_set_suction(self):
        env = MockArmEnv()
        env.set_suction(True)
        assert env._suction is True
        env.set_suction(False)
        assert env._suction is False


class TestMockArmEnvSafetyContract:
    def test_z_min_safe_property(self):
        env = MockArmEnv(z_min_safe=50.0)
        assert env.z_min_safe == pytest.approx(50.0)

    def test_workspace_bounds_default_none(self):
        assert MockArmEnv().workspace_bounds is None

    def test_workspace_bounds_property(self):
        env = MockArmEnv(workspace_bounds=(0.0, -300.0, 500.0, 300.0))
        assert env.workspace_bounds == (0.0, -300.0, 500.0, 300.0)


class TestMockArmEnvObservation:
    def test_get_observation_shape(self):
        env = MockArmEnv(image_hw=(240, 320))
        env.connect()
        obs = env.get_observation()
        assert obs.rgb is not None
        assert obs.rgb.shape == (240, 320, 3)
        assert obs.rgb.dtype == np.uint8
        assert obs.pose is not None
        assert "suction" in obs.extra

    def test_reset(self):
        env = MockArmEnv()
        env.move(100, 50, 300)
        env.set_suction(True)
        env.reset()
        assert env._pose["x"] == pytest.approx(env._home["x"])
        assert env._suction is False
        assert len(env._move_log) == 0
