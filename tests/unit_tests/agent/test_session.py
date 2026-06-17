# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.agent.session."""

from __future__ import annotations

from jiuwensymbiosis.env.mock import MockArmEnv
from jiuwensymbiosis.agent.session import RobotSession

from tests.mocks.mock_api import MockApi


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

        starter = lambda: FakeSidecar()
        s = RobotSession(env=env, api=api, name="t", sidecar_starters=[starter])
        s.connect()
        assert len(started) == 1
        s.disconnect()
