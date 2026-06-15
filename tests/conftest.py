# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Global fixtures for jiuwensymbiosis tests."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.utils.proxy import clear_proxy_env

clear_proxy_env()

from jiuwensymbiosis.env.mock import MockArmEnv
from tests.mocks.mock_api import MockApi
from tests.mocks.mock_detector import make_mock_seg_fn


@pytest.fixture
def mock_env():
    return MockArmEnv()


@pytest.fixture
def mock_api(mock_env):
    return MockApi(mock_env)


@pytest.fixture
def mock_seg_fn():
    return make_mock_seg_fn()
