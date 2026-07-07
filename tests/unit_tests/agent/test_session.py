# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.agent.session."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.api.mixins import JointMotionMixin
from jiuwensymbiosis.env.mock import MockArmEnv
from tests.mocks.mock_api import MockApi


class _ApiWithJointMixin(JointMotionMixin, MockApi):
    """MockApi + joint motion: declares motion.joint, which MockArmEnv lacks.

    This makes ``api_only == {"motion.joint"}`` — a clear config error to test
    the strict_capabilities gate.
    """


class TestRobotSessionConstruction:
    def test_basic(self):
        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="test")
        assert s.name == "test"
        assert s._connected is False

    def test_default_name(self):
        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api)
        assert s.name == "robot"


class TestRobotSessionLifecycle:
    def test_connect_sets_connected(self):
        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="t")
        s.connect()
        assert s._connected is True

    def test_disconnect_clears(self):
        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="t")
        s.connect()
        s.disconnect()
        assert s._connected is False

    def test_context_manager(self):
        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="t")
        with s:
            assert s._connected is True
        assert s._connected is False

    def test_connect_idempotent(self):
        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="t")
        s.connect()
        s.connect()
        assert s._connected is True


class TestRobotSessionGlobals:
    def test_globals_provider_contains_env_api_np(self):
        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="t")
        g = s.globals_provider()
        assert "env" in g
        assert "api" in g
        assert "np" in g
        assert g["env"] is env
        assert g["api"] is api

    def test_extra_globals(self):
        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="t", extra_globals={"custom": 42})
        g = s.globals_provider()
        assert g["custom"] == 42


class TestRobotSessionDescribe:
    def test_describe(self):
        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="mybot")
        desc = s.describe()
        assert desc["name"] == "mybot"
        assert "env_capabilities" in desc
        assert "api_capabilities" in desc

    def test_describe_effective_capabilities_is_intersection(self):
        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="mybot")
        desc = s.describe()
        expected = sorted(set(env.capabilities) & set(api.capabilities))
        assert desc["effective_capabilities"] == expected


class TestRobotSessionSidecars:
    def test_sidecar_started_on_connect(self):
        env = MockArmEnv()
        api = MockApi(env)
        started = []

        class FakeSidecar:
            def __enter__(self):
                started.append(True)
                return self

            def __exit__(self, *a):
                pass

        def starter():
            return FakeSidecar()

        s = RobotSession(env=env, api=api, name="t", sidecar_starters=[starter])
        s.connect()
        assert len(started) == 1
        s.disconnect()


class TestStrictCapabilities:
    def test_strict_raises_on_api_only(self):
        # api declares motion.joint but env does not — a config error.
        env = MockArmEnv()
        api = _ApiWithJointMixin(env)
        assert "motion.joint" in api.capabilities
        assert "motion.joint" not in env.capabilities

        s = RobotSession(env=env, api=api, name="t", strict_capabilities=True)
        with pytest.raises(ValueError) as exc_info:
            s.connect()
        assert "motion.joint" in str(exc_info.value)
        # Did not transition to connected.
        assert s._connected is False

    def test_strict_disabled_warns_on_api_only(self, caplog):
        import logging

        env = MockArmEnv()
        api = _ApiWithJointMixin(env)
        s = RobotSession(env=env, api=api, name="t", strict_capabilities=False)
        with caplog.at_level(logging.WARNING):
            s.connect()  # must NOT raise
        assert s._connected is True
        assert any("motion.joint" in r.getMessage() for r in caplog.records)
        s.disconnect()

    def test_strict_passes_when_aligned(self):
        env = MockArmEnv()
        api = MockApi(env)  # capabilities ⊆ env.capabilities
        s = RobotSession(env=env, api=api, name="t", strict_capabilities=True)
        s.connect()
        assert s._connected is True
        s.disconnect()

    def test_env_only_never_raises_even_strict(self):
        # env has a capability the api doesn't declare — that's a feature not
        # surfaced as a tool, not necessarily an error. strict must not fire.
        env = MockArmEnv()
        api = MockApi(env)
        # MockArmEnv has vision.camera; MockApi does not declare it.
        assert "vision.camera" in env.capabilities
        assert "vision.camera" not in api.capabilities
        s = RobotSession(env=env, api=api, name="t", strict_capabilities=True)
        s.connect()
        assert s._connected is True
        s.disconnect()


class TestTraceFinalizeOnDisconnect:
    """disconnect() fully tears down the TraceRail."""

    def test_disconnect_calls_close(self):
        from types import SimpleNamespace

        from tests.mocks.mock_api import MockApi

        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="t")
        calls = []
        fake = SimpleNamespace(close=lambda: calls.append(1))
        s._trace_rail = fake
        s.connect()
        s.disconnect()
        assert calls == [1]
        # disconnect clears the reference so it isn't left dangling.
        assert s._trace_rail is None

    def test_disconnect_without_trace_rail_ok(self):
        from tests.mocks.mock_api import MockApi

        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="t")
        s._trace_rail = None
        s.connect()
        s.disconnect()  # must not raise

    def test_disconnect_idempotent_after_close(self):
        from types import SimpleNamespace

        from tests.mocks.mock_api import MockApi

        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="t")
        calls = []
        fake = SimpleNamespace(close=lambda: calls.append(1))
        s._trace_rail = fake
        s.connect()
        s.disconnect()
        s.disconnect()  # second call: trace_rail already None, must be no-op
        assert calls == [1]
