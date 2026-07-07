# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Mock RobotSession for testing builder and rails without a real robot."""

from __future__ import annotations

from jiuwensymbiosis.agent.session import RobotSession
from tests.helpers import make_mock_session as _make_mock_session


def make_mock_session(**api_kwargs) -> RobotSession:
    """Build a RobotSession with MockArmEnv + MockApi."""
    return _make_mock_session(name="test_mock", api_kwargs=api_kwargs)
