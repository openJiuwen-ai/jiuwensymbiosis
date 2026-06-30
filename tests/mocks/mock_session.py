# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Mock RobotSession for testing builder and rails without a real robot."""

from __future__ import annotations

from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.env.mock import MockArmEnv
from tests.mocks.mock_api import MockApi


def make_mock_session(**api_kwargs) -> RobotSession:
    """Build a RobotSession with MockArmEnv + MockApi."""
    env = MockArmEnv()
    api = MockApi(env, **api_kwargs)
    return RobotSession(env=env, api=api, name="test_mock")
