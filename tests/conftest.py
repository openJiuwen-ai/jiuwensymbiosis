# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Global fixtures for jiuwensymbiosis tests."""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def caplog(caplog):
    """Capture framework records at their namespace roots, like Runtime/Trace."""
    loggers = [logging.getLogger(name) for name in ("jiuwensymbiosis", "jiuwensymbiosis_gui")]
    propagation = [logger.propagate for logger in loggers]
    for logger in loggers:
        logger.addHandler(caplog.handler)
        logger.propagate = False
    yield caplog
    for logger, propagate in zip(loggers, propagation, strict=True):
        logger.removeHandler(caplog.handler)
        logger.propagate = propagate


@pytest.fixture
def isolated_framework_logging():
    """Keep logging configuration tests from closing other tests' outputs."""
    from openjiuwen.core.common.logging.log_config import configure_log_config, get_log_config_snapshot

    from jiuwensymbiosis import _openjiuwen_logging
    from jiuwensymbiosis.utils import logging as log_module

    upstream_config = get_log_config_snapshot()
    pending_closes = log_module._pending_handler_closes.copy()
    root = logging.getLogger()
    root_handlers, root_level = root.handlers[:], root.level
    namespaces = ("jiuwensymbiosis", "jiuwensymbiosis_gui", "openjiuwen")
    for name in namespaces:
        logging.getLogger(name)
    states = {
        name: (logger.level, logger.handlers[:], logger.propagate)
        for name, logger in root.manager.loggerDict.items()
        if isinstance(logger, logging.Logger) and any(name == n or name.startswith(n + ".") for n in namespaces)
    }
    for name in namespaces:
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            if getattr(handler, "_jiuwensymbiosis_owned", False) or isinstance(
                handler, _openjiuwen_logging._PlainConsoleHandler
            ):
                logger.removeHandler(handler)
    yield
    if get_log_config_snapshot() != upstream_config:
        configure_log_config(upstream_config)
    to_close = set()
    for name, logger in list(root.manager.loggerDict.items()):
        if not isinstance(logger, logging.Logger) or not any(name == n or name.startswith(n + ".") for n in namespaces):
            continue
        level, handlers, propagate = states.get(name, (logging.NOTSET, [], True))
        to_close.update(h for h in logger.handlers if h not in handlers)
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate
    for handler in to_close:
        handler.close()
    for handler in log_module._pending_handler_closes - pending_closes:
        handler.close()
    log_module._pending_handler_closes.clear()
    log_module._pending_handler_closes.update(pending_closes)
    root.handlers[:] = root_handlers
    root.setLevel(root_level)


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
