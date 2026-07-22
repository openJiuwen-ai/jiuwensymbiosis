# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.env.base."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.env.base import KNOWN_CAPABILITIES, BaseRobotEnv, RobotObservation


class TestKnownCapabilities:
    def test_completeness(self):
        expected = {
            "motion.cartesian",
            "motion.joint",
            "motion.servo",
            "grasp.suction",
            "grasp.parallel",
            "vision.camera",
            "vision.depth",
            "vision.detection",
            "vision.eye_to_hand",
            "sorting.command",
            "speech.tts",
        }
        assert KNOWN_CAPABILITIES == expected

    def test_is_frozenset(self):
        assert isinstance(KNOWN_CAPABILITIES, frozenset)


class TestRobotObservation:
    def test_defaults(self):
        obs = RobotObservation()
        assert obs.pose is None
        assert obs.joints is None
        assert obs.rgb is None
        assert obs.depth is None
        assert obs.extra == {}

    def test_with_pose(self):
        obs = RobotObservation(pose={"x": 1, "y": 2, "z": 3, "r": 0})
        assert obs.pose == {"x": 1, "y": 2, "z": 3, "r": 0}


class TestBaseRobotEnvSubclass:
    def test_valid_capabilities(self):
        class GoodEnv(BaseRobotEnv):
            capabilities = frozenset({"motion.cartesian", "grasp.parallel"})
            name = "good"

            def connect(self):
                pass

            def disconnect(self):
                pass

            def get_observation(self):
                return RobotObservation()

        assert GoodEnv.capabilities == frozenset({"motion.cartesian", "grasp.parallel"})

    def test_invalid_capabilities_raises(self):
        with pytest.raises(ValueError, match="unknown capabilities"):

            class BadEnv(BaseRobotEnv):
                capabilities = frozenset({"telekinesis"})
                name = "bad"

                def connect(self):
                    pass

                def disconnect(self):
                    pass

                def get_observation(self):
                    return RobotObservation()

    def test_has_method(self):
        class ValidEnv(BaseRobotEnv):
            capabilities = frozenset({"motion.cartesian"})
            name = "valid"

            def connect(self):
                pass

            def disconnect(self):
                pass

            def get_observation(self):
                return RobotObservation()

        env = ValidEnv()
        assert env.has("motion.cartesian") is True
        assert env.has("grasp.suction") is False

    def test_context_manager_protocol(self):
        class ConEnv(BaseRobotEnv):
            capabilities = frozenset()
            name = "con"
            _connected = False

            def connect(self):
                self._connected = True

            def disconnect(self):
                self._connected = False

            def get_observation(self):
                return RobotObservation()

        env = ConEnv()
        with env:
            assert env._connected is True
        assert env._connected is False


class TestOptionalHardwareContract:
    """Default optional contract: low_level / z_min_safe / workspace_bounds → None."""

    def _make_env(self):
        class PlainEnv(BaseRobotEnv):
            capabilities = frozenset({"motion.cartesian"})
            name = "plain"

            def connect(self):
                pass

            def disconnect(self):
                pass

            def get_observation(self):
                return RobotObservation()

        return PlainEnv()

    def test_low_level_defaults_none(self):
        assert self._make_env().low_level is None

    def test_z_min_safe_defaults_none(self):
        assert self._make_env().z_min_safe is None

    def test_workspace_bounds_defaults_none(self):
        assert self._make_env().workspace_bounds is None
