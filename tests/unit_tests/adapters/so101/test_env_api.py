# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.so101.env and api (no LeRobot).

Covers:
- So101Env capabilities, read-only property setters (AttributeError), joint_limits
  ordering over ARM_JOINT_ORDER, observation extra.
- So101Api structure: @robot_tool methods present, no VisionMixin tools.
- So101Api delegates: open/close_gripper -> set_end_effector, goto_pose(pose) ->
  move_to_flange(So101Pose), goto_xyzr preserves r.
- build_robot_tools gating: SO-101 tools emitted only for the milestone-A caps.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from jiuwensymbiosis.adapters.so101.api import So101Api
from jiuwensymbiosis.adapters.so101.config import So101Config
from jiuwensymbiosis.adapters.so101.env import So101Env
from jiuwensymbiosis.adapters.so101.geometry import So101Pose
from jiuwensymbiosis.adapters.so101.lowlevel import ARM_JOINT_ORDER

_ARM_LIMITS = {
    "shoulder_pan": (-90.0, 90.0),
    "shoulder_lift": (-90.0, 90.0),
    "elbow_flex": (-90.0, 90.0),
    "wrist_flex": (-90.0, 90.0),
    "wrist_roll": (-180.0, 180.0),
}


def _make_env(*, camera_serial: str | None = None) -> So101Env:
    return So101Env(
        So101Config(
            port="/dev/fake",
            home_joints_deg=[0.0, 0.0, 0.0, 0.0, 0.0],
            joint_limits=_ARM_LIMITS,
            safety_validated=True,
            camera_serial=camera_serial,
        )
    )


class _SpyDriver:
    """Satisfies what So101Api/So101Env delegate to via the public verbs."""

    def __init__(self) -> None:
        self.log: list = []
        self.z_min_safe = 30.0
        self.tool_offset_mm = 0.0
        self.home_pose = So101Pose(10.0, 20.0, 30.0, 0.0, 0.0, 0.0)
        # Eye-to-hand vision surface (milestone B): a fake constant T_base_cam +
        # intrinsics + calibration so vision tools run without real hardware.
        self.tf_base_cam = np.eye(4, dtype=np.float64)
        self.intrinsics = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.calibration: dict | None = None

    def grab_frames(self):
        """Return a tiny (rgb, depth_m) pair so vision tools have a frame."""
        return (
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8), 0.5, dtype=np.float32),
        )

    def home(self) -> None:
        self.log.append("home")

    def get_pose(self) -> So101Pose:
        return So101Pose(1.0, 2.0, 3.0, 180.0, 0.0, 7.0)

    def move_to_pose_blocking(self, pose, *args, **kwargs) -> None:
        self.log.append(("move", pose))

    def move_joint_blocking(self, q, *, timeout_s=30.0) -> None:
        self.log.append(("joint", list(q)))

    def set_gripper(self, on: bool) -> None:
        self.log.append(("gripper", on))

    def get_angles(self) -> list[float]:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    def get_gripper_position(self) -> float:
        return 50.0


def _build_api():
    # The API/tool emission tests exercise the milestone-B surface explicitly.
    env = _make_env(camera_serial="test-camera")
    driver = _SpyDriver()
    env._inner = driver  # bind without LeRobot connect
    return So101Api(env), env, driver


# ====================================================================== ENV
class TestSo101EnvCapabilities:
    def test_capability_set(self):
        env = _make_env()
        assert env.capabilities == frozenset(
            {
                "motion.cartesian",
                "motion.joint",
                "grasp.parallel",
                "motion.servo",
            }
        )

    def test_vision_capabilities_present(self):
        env = _make_env(camera_serial="test-camera")
        for cap in ("vision.camera", "vision.depth", "vision.detection", "vision.eye_to_hand"):
            assert cap in env.capabilities

    def test_failed_camera_start_removes_vision_capabilities(self):
        env = _make_env(camera_serial="test-camera")
        env.capabilities = env._capabilities_for_driver(SimpleNamespace(camera_available=False))
        assert not any(cap.startswith("vision.") for cap in env.capabilities)


class TestSo101EnvReadOnlyProperties:
    """Each read-only property setter must raise AttributeError (no either/or)."""

    @pytest.mark.parametrize(
        "prop,val",
        [
            ("low_level", None),
            ("z_min_safe", 50.0),
            ("workspace_bounds", (0.0, 0.0, 100.0, 100.0)),
            ("joint_limits", None),
            ("home_pose", So101Pose(1, 2, 3, 0, 0, 0)),
            ("tool_offset_mm", 10.0),
        ],
    )
    def test_setter_raises_attribute_error(self, prop, val):
        env = _make_env()
        with pytest.raises(AttributeError, match="read-only"):
            setattr(env, prop, val)


class TestSo101EnvJointLimits:
    def test_joint_limits_keyed_over_arm_order(self):
        env = _make_env()
        limits = env.joint_limits
        assert list(limits.keys()) == list(ARM_JOINT_ORDER)
        assert len(limits) == 5

    def test_joint_limits_resists_dict_mutation(self):
        env = _make_env()
        limits1 = env.joint_limits
        limits2 = env.joint_limits
        # Fresh dict each access (stable indexing even if source dict order drifts).
        assert limits1 == limits2
        assert limits1 is not limits2


class TestSo101EnvObservation:
    def test_extra_contains_gripper_and_z_floor(self):
        env = _make_env()
        env._inner = _SpyDriver()
        obs = env.get_observation()
        assert obs.extra is not None
        assert obs.extra["z_min_safe"] == 30.0
        assert obs.extra["gripper_state"] == 50.0

    def test_observation_pose_is_mm_deg(self):
        env = _make_env()
        env._inner = _SpyDriver()
        obs = env.get_observation()
        assert obs.pose == {"x": 1.0, "y": 2.0, "z": 3.0, "rx": 180.0, "ry": 0.0, "rz": 7.0}

    def test_observation_rgb_depth_none_without_camera(self):
        """A driver without grab_frames yields rgb/depth=None (camera read is best-effort)."""

        class _NoCamDriver(_SpyDriver):
            # Hide the camera method so the env's best-effort read falls back.
            grab_frames = None

        env = _make_env()
        env._inner = _NoCamDriver()
        obs = env.get_observation()
        assert obs.rgb is None
        assert obs.depth is None

    def test_observation_rgb_depth_from_camera(self):
        """A driver exposing grab_frames feeds rgb/depth into the observation."""
        env = _make_env()
        env._inner = _SpyDriver()
        obs = env.get_observation()
        assert obs.rgb is not None
        assert obs.depth is not None


# ====================================================================== API
class TestSo101ApiStructure:
    def test_api_has_robot_tool_methods(self):
        expected = [
            "home",
            "get_pose",
            "get_home_pose",
            "goto_xyzr",
            "goto_pose",
            "close_gripper",
            "open_gripper",
            "move_joint",
        ]
        for name in expected:
            method = getattr(So101Api, name, None)
            assert method is not None, f"So101Api.{name} not found"
            assert hasattr(method, "__robot_tool__"), f"So101Api.{name} missing @robot_tool"

    def test_vision_methods_present(self):
        """VisionMixin IS mixed in (milestone B): its tools are on So101Api."""
        for name in ("get_grasp_info_simple", "pixel_to_base_xyz", "get_image", "analyze_scene"):
            assert hasattr(So101Api, name), f"So101Api missing vision method {name}"

    def test_api_capabilities(self):
        # api.capabilities is the union of mixin `capability` strings across the
        # MRO (motion*2 + grasp + vision.detection); vision.camera/depth are
        # ENV hardware capabilities, not Api mixin capabilities.
        api, _env, _driver = _build_api()
        assert api.capabilities == frozenset(
            {
                "motion.cartesian",
                "motion.joint",
                "grasp.parallel",
                "vision.detection",
            }
        )

    def test_open_gripper_has_no_input_params(self):
        meta = So101Api.open_gripper.__robot_tool__
        assert meta.input_params == {"type": "object", "properties": {}}

    def test_close_gripper_has_no_input_params(self):
        meta = So101Api.close_gripper.__robot_tool__
        assert meta.input_params == {"type": "object", "properties": {}}

    def test_goto_pose_input_params_exposes_nested_pose(self):
        meta = So101Api.goto_pose.__robot_tool__
        top = meta.input_params
        assert top.get("type") == "object"
        assert top.get("required") == ["pose"]
        pose_schema = top.get("properties", {}).get("pose")
        assert isinstance(pose_schema, dict)
        assert pose_schema.get("type") == "object"
        pose_props = pose_schema.get("properties", {})
        for key in ("x", "y", "z", "rx", "ry", "rz"):
            assert key in pose_props, f"goto_pose pose.properties missing {key}"
        assert set(pose_schema.get("required", [])) == {"x", "y", "z", "rx", "ry", "rz"}


class TestSo101ApiDelegates:
    def test_open_gripper_calls_set_end_effector_false(self):
        api, env, _driver = _build_api()
        env.set_end_effector = MagicMock()
        result = api.open_gripper()
        env.set_end_effector.assert_called_once_with(False)
        assert result["state"] == "open"

    def test_close_gripper_calls_set_end_effector_true(self):
        api, env, _driver = _build_api()
        env.set_end_effector = MagicMock()
        result = api.close_gripper()
        env.set_end_effector.assert_called_once_with(True)
        assert result["state"] == "closed"

    def test_open_gripper_ignores_width_mm(self):
        api, env, _driver = _build_api()
        env.set_end_effector = MagicMock()
        # width_mm accepted for parity, ignored — no unit conversion happens.
        api.open_gripper(width_mm=999.0)
        env.set_end_effector.assert_called_once_with(False)

    def test_close_gripper_ignores_force_n(self):
        api, env, _driver = _build_api()
        env.set_end_effector = MagicMock()
        api.close_gripper(force_n=42.0)
        env.set_end_effector.assert_called_once_with(True)

    def test_goto_pose_routes_to_move_to_flange_so101pose(self):
        api, _env, driver = _build_api()
        api.goto_pose(So101Pose(100.0, 200.0, 300.0, 180.0, 0.0, 45.0))
        assert any(c[0] == "move" for c in driver.log)
        move = [c for c in driver.log if c[0] == "move"][0]
        pose = move[1]
        assert isinstance(pose, So101Pose)
        assert pose.x == 100.0 and pose.z == 300.0 and pose.rz == 45.0

    def test_goto_pose_accepts_dict_from_llm_json_object(self):
        # The LLM / RobotControlTool delivers pose as a JSON object (dict at
        # runtime); goto_pose must coerce it to So101Pose before delegating.
        api, _env, driver = _build_api()
        api.goto_pose({"x": 100.0, "y": 200.0, "z": 300.0, "rx": 180.0, "ry": 0.0, "rz": 45.0})
        move = [c for c in driver.log if c[0] == "move"][0]
        pose = move[1]
        assert isinstance(pose, So101Pose)
        assert pose.x == 100.0 and pose.z == 300.0 and pose.rz == 45.0

    def test_goto_xyzr_preserves_r_when_omitted(self):
        api, _env, driver = _build_api()
        # get_flange_pose returns rz=7.0; goto_xyzr with r=None must reuse it.
        api.goto_xyzr(10.0, 20.0, 30.0)
        move = [c for c in driver.log if c[0] == "move"][0]
        pose = move[1]
        assert pose.rz == 7.0
        assert pose.rx == 180.0  # top-down default
        assert pose.ry == 0.0

    def test_goto_xyzr_explicit_r_overrides(self):
        api, _env, driver = _build_api()
        api.goto_xyzr(10.0, 20.0, 30.0, 45.0)
        pose = [c for c in driver.log if c[0] == "move"][0][1]
        assert pose.rz == 45.0

    def test_home_reaches_driver(self):
        api, _env, driver = _build_api()
        api.home()
        assert "home" in driver.log

    def test_move_joint_reaches_driver(self):
        api, _env, driver = _build_api()
        api.move_joint([1.0, 2.0, 3.0, 4.0, 5.0])
        assert ("joint", [1.0, 2.0, 3.0, 4.0, 5.0]) in driver.log

    def test_move_direction_routes_so101pose_not_namespace(self):
        """The generic MotionMixin.move_direction hands a SimpleNamespace to
        env.move_to_flange; So101Env must normalize it to a So101Pose so the real
        driver (which requires So101Pose) doesn't raise TypeError. The spy driver
        accepts any object, so this test pins the normalization explicitly."""
        api, _env, driver = _build_api()
        # get_flange_pose returns So101Pose(1, 2, 3, rx=180, ry=0, rz=7).
        api.move_direction("up", 50.0)
        move = [c for c in driver.log if c[0] == "move"][0]
        pose = move[1]
        # Must be a So101Pose (not a SimpleNamespace) — the real driver enforces this.
        assert isinstance(pose, So101Pose), f"expected So101Pose, got {type(pose).__name__}"
        # up = +z, so z went 3.0 -> 53.0; orientation preserved from current.
        assert pose.z == pytest.approx(53.0, abs=1e-9)
        assert pose.rx == 180.0 and pose.ry == 0.0 and pose.rz == 7.0

    @pytest.mark.parametrize(
        "pose",
        [
            SimpleNamespace(z=50.0, rx=180.0, ry=0.0, rz=0.0),
            {"z": 50.0, "rx": 180.0, "ry": 0.0, "rz": 0.0},
        ],
    )
    def test_move_to_flange_rejects_missing_coordinates(self, pose):
        _api, env, driver = _build_api()
        with pytest.raises(TypeError, match="missing required fields.*x.*y"):
            env.move_to_flange(pose)
        assert not any(entry[0] == "move" for entry in driver.log if isinstance(entry, tuple))

    def test_move_to_flange_accepts_complete_mapping_and_r_alias(self):
        _api, env, driver = _build_api()
        env.move_to_flange({"x": 10, "y": 20, "z": 50, "rx": 180, "ry": 0, "r": 15})
        pose = [entry[1] for entry in driver.log if entry[0] == "move"][0]
        assert pose == So101Pose(10.0, 20.0, 50.0, 180.0, 0.0, 15.0)


# ============================================================ TOOL EMISSION
class TestToolEmission:
    def test_tools_gated_by_capabilities(self):
        """build_robot_tools emits motion + grasp + vision tools (capabilities intersect)."""
        from jiuwensymbiosis.tools.builder import build_robot_tools

        api, env, _driver = _build_api()
        tools = build_robot_tools(api, env=env)
        names = {t.card.name for t in tools}
        # Motion + grasp tools present.
        assert "goto_xyzr" in names
        assert "goto_pose" in names
        assert "open_gripper" in names
        assert "close_gripper" in names
        assert "home" in names
        # Vision tools present (env declares vision.* capabilities).
        assert "get_grasp_info_simple" in names
        assert "pixel_to_base_xyz" in names
        assert "analyze_scene" in names
