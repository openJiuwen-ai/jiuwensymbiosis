# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for Cruzr env/api/session structure."""

from __future__ import annotations

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv
from jiuwensymbiosis.adapters.cruzr.session import build_cruzr_session
from jiuwensymbiosis.tools.builder import list_tool_meta


class _FakeLowLevel:
    def __init__(self):
        self.calls = []
        self.joints = {"L_shoulder_pitch_joint": 0.0}

    def raise_arm_blocking(self, **kwargs):
        self.calls.append(("raise_arm_blocking", kwargs))
        return {"ok": True, **kwargs}

    def home(self, **kwargs):
        self.calls.append(("home", kwargs))
        return {"ok": True, **kwargs}

    def move_joint_blocking(self, *args, **kwargs):
        self.calls.append(("move_joint_blocking", args, kwargs))
        return {"ok": True}

    def get_joint_positions(self):
        return dict(self.joints)

    def close(self):
        self.calls.append(("close",))


def _connected_env() -> CruzrEnv:
    env = CruzrEnv(CruzrConfig())
    env._inner = _FakeLowLevel()
    env._connected = True
    return env


class TestCruzrEnv:
    def test_capabilities(self):
        env = CruzrEnv(CruzrConfig())
        assert "motion.joint" in env.capabilities
        assert "motion.cartesian" not in env.capabilities

    def test_observation_uses_joint_state(self):
        env = _connected_env()
        obs = env.get_observation()
        assert obs.extra["joint_positions"]["L_shoulder_pitch_joint"] == 0.0


class TestCruzrApi:
    def test_tool_methods_are_exposed(self):
        api = CruzrApi(_connected_env())
        names = {m["name"] for m in list_tool_meta(api)}
        assert "raise_left_arm" in names
        assert "raise_right_arm" in names
        assert "lower_left_arm" in names
        assert "lower_right_arm" in names
        assert "raise_arm" in names
        # 不再暴露 move_joint：共享动作指整组关节向量，而本体驱动只接受单个肩关节值。
        # 单关节需求由 move_named_joint 承担（说清楚它做什么）。
        assert "move_joint" not in names
        assert "move_named_joint" in names

    def test_raise_left_arm_dispatches_to_lowlevel(self):
        env = _connected_env()
        api = CruzrApi(env)
        result = api.raise_left_arm()
        assert result["ok"] is True
        assert env.low_level.calls[0] == ("raise_arm_blocking", {"arm": "left", "return_home": True})

    def test_move_named_joint_dispatches(self):
        env = _connected_env()
        api = CruzrApi(env)
        api.move_named_joint("L_shoulder_pitch_joint", 0.5)
        assert env.low_level.calls[0][0] == "move_joint_blocking"
        assert env.low_level.calls[0][1][0] == {"L_shoulder_pitch_joint": 0.5}


class TestCruzrSession:
    def test_builder_from_dict(self):
        session = build_cruzr_session.from_dict({"name": "cruzr_test"})
        assert session.name == "cruzr_test"
        assert isinstance(session.env, CruzrEnv)
        assert isinstance(session.api, CruzrApi)
        assert session.extra_globals["cruzr_cfg"].name == "cruzr_test"


class TestCruzrVisionWiring:
    def test_vision_capabilities(self):
        env = CruzrEnv(CruzrConfig())
        assert "vision.detection" in env.capabilities
        assert "vision.camera" in env.capabilities
        assert "vision.depth" in env.capabilities

    def test_detect_tool_exposed_via_session(self):
        session = build_cruzr_session.from_dict({"name": "cruzr_vis"})
        names = {m["name"] for m in list_tool_meta(session.api)}
        assert "detect" in names

    def test_api_gets_calib_path_from_cfg(self):
        session = build_cruzr_session.from_dict({"camera_calib_path": "/tmp/c.json"})
        assert session.api._camera_calib_path == "/tmp/c.json"
