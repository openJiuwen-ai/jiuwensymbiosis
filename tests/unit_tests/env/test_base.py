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
            # End-effector axis: what the body can HOLD. Split out of the old
            # "grasp.dual_arm", which had been standing in for two axes at once.
            "grasp.paddle",
            "vision.camera",
            "vision.depth",
            "vision.detection",
            "vision.eye_to_hand",
            "sorting.command",
            "speech.tts",
            # Mobility capabilities (P1, 2026-07-15)
            "motion.base",
            "motion.jump",
            "motion.base_servo",
            "motion.lift",
            "motion.waist",
            "motion.goal",
            # Topology axis: what the body can MOVE. Decides which action to call; what the
            # arms hold is the grasp.* axis above.
            "motion.dual_arm",
            # Planning-time URDF reachability (2026-07-30)
            "planning.reachability",
            # The body can aim a camera by turning something (head / waist / base), so it
            # can look around for a target instead of only seeing what is in front of it.
            "vision.search",
            "simulation.truth",
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

            def home(self):
                pass

        assert GoodEnv.capabilities == frozenset({"motion.cartesian", "grasp.parallel"})

    def test_home_is_abstract(self):
        """home() must be implemented by every subclass — it's an unconditional action."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class EnvWithoutHome(BaseRobotEnv):
                capabilities = frozenset({"motion.joint"})
                name = "no_home"

                def connect(self):
                    pass

                def disconnect(self):
                    pass

                def get_observation(self):
                    return RobotObservation()

            EnvWithoutHome()

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

                def home(self):
                    pass

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

            def home(self):
                pass

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

            def home(self):
                pass

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

            def home(self):
                pass

        return PlainEnv()

    def test_low_level_defaults_none(self):
        assert self._make_env().low_level is None

    def test_z_min_safe_defaults_none(self):
        assert self._make_env().z_min_safe is None

    def test_workspace_bounds_defaults_none(self):
        assert self._make_env().workspace_bounds is None

    @pytest.mark.parametrize(
        ("verb", "args"),
        [
            ("navigate_arc", (0.8, 0.5)),
            ("start_base_drive", ()),
            ("base_drive_running", ("h",)),
            ("steer_base_drive", ("h", 0.1)),
            ("hold_base_drive", ("h",)),
            ("stop_base_drive", ("h",)),
            ("grab_calibrated_frame", ()),
        ],
    )
    def test_optional_verbs_name_the_missing_capability(self, verb, args):
        """A body that skipped a verb must learn which capability it belongs to, not just that it is missing."""
        env = self._make_env()
        with pytest.raises(NotImplementedError, match=r"declare/implement '[a-z._]+'"):
            getattr(env, verb)(*args)


class TestReachabilityIsDerivedNotDeclared:
    """``planning.reachability`` used to be a hand-written marker on both the Api and the Env.
    so101 declared it in both places while its Env exposed no URDF at all, so its judge could
    only ever answer "unknown" — and nothing noticed, because a declaration is not checked
    against anything. It is now derived from what each side actually has: a judge on the Api,
    the model that judge reads on the Env. Only the intersection is true."""

    def test_env_derives_it_from_shipping_a_model(self):
        from jiuwensymbiosis.env.base import BaseRobotEnv

        class _Env(BaseRobotEnv):
            capabilities = frozenset({"motion.joint"})

            def connect(self) -> None: ...
            def disconnect(self) -> None: ...
            def get_observation(self): ...
            def home(self) -> None: ...

        env = _Env()
        assert "planning.reachability" not in env.effective_capabilities
        env.urdf_path = "/tmp/robot.urdf"
        assert "planning.reachability" not in env.effective_capabilities, "a URDF alone is not enough"
        env.arm_chains = {"left": ("base_link", "tool")}
        assert "planning.reachability" in env.effective_capabilities

    def test_a_judge_without_a_model_does_not_add_up_to_the_capability(self):
        """The RULE, deliberately not pinned to any shipped body: holding the judge is not
        enough without the model it reads. Asserting "so101 has no URDF" here would freeze
        today's fact as if it were the rule — and the day so101 ships one the capability
        SHOULD come true, not break a test."""
        from jiuwensymbiosis.env.base import BaseRobotEnv
        from jiuwensymbiosis.tools.builder import _effective_capabilities

        class _Env(BaseRobotEnv):
            capabilities = frozenset({"motion.cartesian"})

            def connect(self) -> None: ...
            def disconnect(self) -> None: ...
            def get_observation(self): ...
            def home(self) -> None: ...

        class _ApiWithJudge:
            capabilities = frozenset({"motion.cartesian", "planning.reachability"})

        env = _Env()
        assert "planning.reachability" not in _effective_capabilities(_ApiWithJudge(), env)

    def test_so101_would_gain_it_the_moment_its_env_ships_a_model(self):
        """so101 holds the generic judge already; what it lacks is the model. Wiring a URDF
        into its Env must be all it takes — that is what "derived, not declared" buys, and
        this stays true both before and after someone does it."""
        import importlib

        from jiuwensymbiosis.tools.builder import _effective_capabilities

        session = importlib.import_module("jiuwensymbiosis.adapters.so101").build_so101_session.from_yaml(
            "configs/so101/so101.yaml"
        )
        assert "planning.reachability" in session.api.capabilities, "it does hold a judge"
        session.env.urdf_path = "/tmp/so101.urdf"
        session.env.arm_chains = {"arm": ("base_link", "gripper_link")}
        assert "planning.reachability" in _effective_capabilities(session.api, session.env)

    def test_a_body_with_both_gets_it_without_declaring_it(self):
        import importlib

        from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
        from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv
        from jiuwensymbiosis.tools.builder import _effective_capabilities

        assert "planning.reachability" not in (getattr(CruzrApi, "capability", None) or set())
        assert "planning.reachability" not in CruzrEnv.capabilities
        session = importlib.import_module("jiuwensymbiosis.adapters.cruzr").build_cruzr_session.from_yaml(
            "configs/cruzr/cruzr.yaml"
        )
        assert "planning.reachability" in _effective_capabilities(session.api, session.env)
