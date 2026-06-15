# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.builder."""

from __future__ import annotations

import yaml

from jiuwensymbiosis.env.mock import MockArmEnv
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters.piper.config import PiperConfig
from tests.mocks.mock_api import MockApi


class _TestEnv(MockArmEnv):
    def __init__(self, cfg=None):
        super().__init__()


class _TestApi(MockApi):
    def __init__(self, env, **kwargs):
        super().__init__(env)


class TestMakeBuilder:
    def test_direct_call(self):
        builder = make_builder(PiperConfig, _TestEnv, _TestApi)
        cfg = PiperConfig()
        session = builder(cfg)
        assert isinstance(session, RobotSession)
        assert session._connected is False

    def test_from_dict(self):
        builder = make_builder(PiperConfig, _TestEnv, _TestApi)
        session = builder.from_dict({"can_port": "can_left"})
        assert isinstance(session, RobotSession)

    def test_from_yaml(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"can_port": "can_left"}), encoding="utf-8")
        builder = make_builder(PiperConfig, _TestEnv, _TestApi)
        session = builder.from_yaml(p)
        assert isinstance(session, RobotSession)
