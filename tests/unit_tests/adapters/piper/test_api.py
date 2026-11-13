# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.piper.api — structural and config-level tests.

Hardware-dependent paths (actual motion, detection) are tested via mock injection.
"""

from __future__ import annotations

from math import sqrt
from types import SimpleNamespace

import pytest


class TestPiperApiStructure:
    def test_api_has_action_methods(self):
        from jiuwensymbiosis.adapters.piper.api import PiperApi

        expected_methods = [
            "home",
            "get_pose",
            "get_home_pose",
            "goto_xyzr",
            "close_gripper",
            "open_gripper",
            "get_grasp_info_simple",
            "pixel_to_base_xyz",
            "get_image",
            "analyze_scene",
        ]
        for name in expected_methods:
            method = getattr(PiperApi, name, None)
            assert method is not None, f"PiperApi.{name} not found"
            assert hasattr(method, "__tool_meta__"), f"PiperApi.{name} missing @implements"

    def test_api_capabilities(self):
        from jiuwensymbiosis.adapters.piper.api import PiperApi
        from jiuwensymbiosis.adapters.piper.config import PiperConfig
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        api = PiperApi(PiperEnv(PiperConfig()))
        assert api.capabilities == frozenset(
            {
                "motion.cartesian",
                "motion.joint",
                "grasp.parallel",
                "vision.detection",
                # Derived from the actions this api implements: get_image /
                # pixel_to_base_xyz are gated on vision.camera, not on the detector.
                "vision.camera",
                # Marker capabilities the api claims explicitly, because no ACTION carries
                # them and the agent gates on api ∩ env: an ability the hardware has and
                # the api omits is switched off in silence. motion.servo is what allows
                # the fast path to FOLLOW a moving target.
                "motion.servo",
                "vision.depth",
            }
        )


class _SpyDriver:
    """Records driver calls; satisfies what PiperApi/PiperEnv delegate to."""

    def __init__(self):
        self.log: list = []
        self.home_pose = type("P", (), {"x": 200.0, "y": 0.0, "z": 400.0, "rx": 0.0, "ry": 90.0, "rz": 0.0})()
        self.tool_offset_mm = 135.8
        self.z_min_safe = 50.0

    def home(self):
        self.log.append("home")

    def get_pose(self):
        return type("P", (), {"x": 1.0, "y": 2.0, "z": 3.0, "rx": 0.0, "ry": 0.0, "rz": 7.0})()

    def move_to_pose_blocking(self, pose):
        self.log.append(("move", pose))

    def move_joint_blocking(self, q, *, timeout_s=30.0):
        self.log.append(("joint", list(q)))

    def set_gripper(self, on):
        self.log.append(("gripper", on))


class _PoseDriver(_SpyDriver):
    """Instantly applies flange commands and exposes their raw hardware readback."""

    def __init__(self):
        super().__init__()
        self.pose = super().get_pose()
        self.tool_offset_mm = 100.0

    def get_pose(self):
        return self.pose

    def move_to_pose_blocking(self, pose):
        super().move_to_pose_blocking(pose)
        self.pose = SimpleNamespace(
            x=pose.x_mm, y=pose.y_mm, z=pose.z_mm, rx=pose.rx_deg, ry=pose.ry_deg, rz=pose.rz_deg
        )

    def servo_to_pose(self, pose):
        self.log.append(("servo", pose))
        self.pose = SimpleNamespace(**pose)


class TestPiperTipCoordinates:
    def _build(self):
        from jiuwensymbiosis.adapters.piper.api import PiperApi
        from jiuwensymbiosis.adapters.piper.config import PiperConfig
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        env = PiperEnv(PiperConfig())
        driver = _PoseDriver()
        env._inner = driver
        return PiperApi(env), env, driver

    @pytest.mark.parametrize(
        ("angles", "tip_offset"),
        [
            ((0, 0, 0), (0, 0, 100)),
            ((180, 0, 0), (0, 0, -100)),
            ((180, 30, 0), (-50, 0, -50 * sqrt(3))),
            ((180, 30, 90), (0, -50, -50 * sqrt(3))),
            ((90, 0, 0), (0, -100, 0)),
            ((0, 90, 0), (100, 0, 0)),
        ],
    )
    def test_readback_rotates_offset_by_live_orientation(self, angles, tip_offset):
        api, _env, driver = self._build()
        driver.pose = SimpleNamespace(x=200, y=20, z=300, rx=angles[0], ry=angles[1], rz=angles[2])

        pose = api.get_pose()

        assert [pose[k] for k in ("x", "y", "z")] == pytest.approx(
            [200 + tip_offset[0], 20 + tip_offset[1], 300 + tip_offset[2]]
        )
        assert [pose[k] for k in ("rx", "ry", "rz")] == list(angles)

    @pytest.mark.parametrize("command", ["blocking", "servo_r", "servo_rz"])
    @pytest.mark.parametrize(
        ("yaw", "flange_xy"), [(0, (250, 20)), (90, (200, 70)), (180, (150, 20)), (-90, (200, -30))]
    )
    def test_command_and_readback_agree_at_different_yaws(self, command, yaw, flange_xy):
        api, _env, driver = self._build()
        if command == "blocking":
            api.goto_xyzr(200, 20, 100, yaw)
        else:
            yaw_key = "r" if command == "servo_r" else "rz"
            api.servo_to_tip({"x": 200, "y": 20, "z": 100, yaw_key: yaw})

        # Independent flange expectations prevent two wrong inverse transforms from passing.
        assert (driver.pose.x, driver.pose.y, driver.pose.z) == pytest.approx((*flange_xy, 100 + 50 * sqrt(3)))
        assert api.get_pose() == pytest.approx({"x": 200, "y": 20, "z": 100, "rx": 180, "ry": 30, "rz": yaw})

    @pytest.mark.parametrize("command", ["blocking", "servo"])
    def test_missing_yaw_preserves_live_yaw(self, command):
        api, _env, driver = self._build()
        driver.pose.rz = 90
        if command == "blocking":
            api.goto_xyzr(200, 20, 100)
        else:
            api.servo_to_tip({"x": 200, "y": 20, "z": 100})

        assert (driver.pose.x, driver.pose.y, driver.pose.rz) == pytest.approx((200, 70, 90))
        assert api.get_pose()["z"] == pytest.approx(100)

    def test_zero_tool_extension_keeps_tip_at_flange(self):
        api, _env, driver = self._build()
        driver.tool_offset_mm = 0
        api.goto_xyzr(200, 20, 100, 45)
        assert (driver.pose.x, driver.pose.y, driver.pose.z) == pytest.approx((200, 20, 100))
        assert api.get_pose() == pytest.approx({"x": 200, "y": 20, "z": 100, "rx": 180, "ry": 30, "rz": 45})

    def test_servo_binding_reaches_target_with_flange_readback(self):
        from jiuwensymbiosis.agent.fast.realtime.binding import ServoBinding
        from jiuwensymbiosis.agent.fast.realtime.servo import ServoConfig, ServoController

        api, env, _driver = self._build()
        api.goto_xyzr(180, 0, 120, 45)
        binding = ServoBinding(SimpleNamespace(api=api, env=env))
        target = {"x": 200, "y": 0, "z": 120, "rz": 45}
        result = ServoController(
            binding.read_pose,
            binding.servo_to,
            lambda: target,
            config=ServoConfig(
                control_hz=200, max_lin_step_mm=6, pos_tol_mm=0.001, settle_ticks=1, timeout_s=1, absolute_timeout_s=2
            ),
        ).run()

        assert result.ok, result
        assert result.reason == "reached"
        assert {k: binding.read_pose()[k] for k in target} == pytest.approx(target)


class TestPiperApiDelegatesThroughEnv:
    """Motion/gripper route api -> env public method -> driver (not via _ll)."""

    def _build(self):
        from jiuwensymbiosis.adapters.piper.api import PiperApi
        from jiuwensymbiosis.adapters.piper.config import PiperConfig
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        env = PiperEnv(PiperConfig())
        driver = _SpyDriver()
        env._inner = driver
        return PiperApi(env), env, driver

    def test_home_reaches_driver_through_env(self):
        api, _env, driver = self._build()
        api.home()
        assert "home" in driver.log

    def test_move_joint_reaches_driver_through_env(self):
        """The action names joints; the Piper SDK takes a vector. The Env converts using
        ``joint_names`` (j1..j6, read off PiperJointAngles)."""
        api, _env, driver = self._build()
        api.move_joint({"j1": 0.0, "j2": 1.0, "j3": 2.0, "j4": 3.0, "j5": 4.0, "j6": 5.0})
        assert ("joint", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) in driver.log

    def test_a_partial_command_holds_the_joints_it_does_not_name(self):
        """Naming one joint must not zero the others: the Env fills the rest from the
        current reading before handing the indexed driver its vector."""
        api, _env, driver = self._build()
        api.move_joint({"j2": 1.5})
        sent = [entry for entry in driver.log if entry[0] == "joint"][-1][1]
        assert sent[1] == 1.5
        assert len(sent) == 6

    def test_an_unknown_joint_name_is_refused(self):
        api, _env, _driver = self._build()
        with pytest.raises(ValueError, match="unknown joint"):
            api.move_joint({"elbow": 1.0})

    def test_the_flange_frame_is_reached_through_the_env_not_the_api(self):
        """Where the flange sits depends on this robot's tool length, so it is no action.
        Bring-up and calibration speak to the Env verb directly; the api surface is TIP-only."""
        from jiuwensymbiosis.adapters.piper.api import PiperApi
        from jiuwensymbiosis.adapters.piper.geometry import FlangePose
        from jiuwensymbiosis.api.actions import ACTIONS

        api, env, driver = self._build()
        env.move_to_flange(FlangePose(1, 2, 3, 180, 0, 0))
        assert any(c[0] == "move" for c in driver.log)
        assert "goto_flange_pose" not in ACTIONS
        assert not hasattr(PiperApi, "goto_flange_pose")
        assert not hasattr(PiperApi, "get_flange_pose")

    def test_goto_xyzr_reaches_driver_through_env(self):
        api, _env, driver = self._build()
        api.goto_xyzr(100.0, 0.0, 200.0, 0.0)
        assert any(c[0] == "move" for c in driver.log)

    def test_close_gripper_calls_env_set_end_effector(self):
        from unittest.mock import MagicMock

        api, env, _driver = self._build()
        env.set_end_effector = MagicMock()
        api.close_gripper()
        env.set_end_effector.assert_called_once_with(True)

    def test_open_gripper_engages_driver_via_env(self):
        api, _env, driver = self._build()
        api.open_gripper()
        assert ("gripper", False) in driver.log

    def test_grasp_z_floor_reads_env_property(self):
        # env.z_min_safe (formal contract) is used, not getattr on the driver.
        api, env, _driver = self._build()
        assert env.z_min_safe == 50.0  # comes from the spy driver via PiperEnv property
