# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Global fixtures for jiuwensymbiosis tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def mock_env():
    from jiuwensymbiosis.env.mock import MockArmEnv

    return MockArmEnv()


@pytest.fixture
def mock_api(mock_env):
    from tests.mocks.mock_api import MockApi

    return MockApi(mock_env)


@pytest.fixture
def mock_seg_fn():
    from tests.mocks.mock_detector import make_mock_seg_fn

    return make_mock_seg_fn()
