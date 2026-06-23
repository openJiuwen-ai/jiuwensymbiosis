# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the adapter smoke test harness (scripts/smoke_test_adapter.py).

The smoke harness drives an adapter's Api with a MockEnv and asserts every
emitted tool can be called without crashing and returns a JSON-serializable
result — catching field-name typos and runtime shape errors at adapter-onboard
time rather than on first real-hardware run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from jiuwensymbiosis.env.mock import MockArmEnv
from tests.mocks.mock_api import MockApi


def _load_smoke_module():
    """Load scripts/smoke_test_adapter.py as a module (scripts/ has no __init__)."""
    repo_root = Path(__file__).resolve().parents[4]
    path = repo_root / "scripts" / "smoke_test_adapter.py"
    spec = importlib.util.spec_from_file_location("smoke_test_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoke_test_adapter"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def smoke():
    return _load_smoke_module()


class TestSmokeTestApi:
    def test_passes_on_mock_api(self, smoke):
        env = MockArmEnv()
        api = MockApi(env)
        results = smoke.smoke_test_api(api, env=env)
        # Every emitted tool must have a status (pass/fail/skip), no crash.
        assert results, "expected at least one tool result"
        names = {r["name"] for r in results}
        assert "home" in names
        assert "get_grasp_info_simple" in names
        # No tool should have errored on the mock api.
        failures = [r for r in results if r["status"] == "fail"]
        assert failures == [], f"unexpected failures: {failures}"

    def test_results_are_json_serializable(self, smoke):
        env = MockArmEnv()
        api = MockApi(env)
        results = smoke.smoke_test_api(api, env=env)
        # The whole report (incl. each tool's return value) must serialize.
        json.dumps(results)

    def test_return_value_captured_on_pass(self, smoke):
        env = MockArmEnv()
        api = MockApi(env)
        results = smoke.smoke_test_api(api, env=env)
        get_pose = next(r for r in results if r["name"] == "get_pose")
        assert get_pose["status"] == "pass"
        assert get_pose["return"] is not None or get_pose.get("returns_none") is True

    def test_unfillable_required_param_skipped(self, smoke):
        # A tool with a required param we have no heuristic for → skipped, not crashed.
        from jiuwensymbiosis.api.decorators import robot_tool
        from jiuwensymbiosis.api.base import BaseRobotApi

        class _AwkwardApi(BaseRobotApi):
            @robot_tool
            def needs_unknown(self, mysterious_param) -> dict:
                return {"echo": mysterious_param}

        env = MockArmEnv()
        api = _AwkwardApi(env)
        results = smoke.smoke_test_api(api, env=env)
        r = next(r for r in results if r["name"] == "needs_unknown")
        assert r["status"] == "skip"
        assert "mysterious_param" in r["reason"]

    def test_crashing_tool_recorded_as_fail(self, smoke):
        from jiuwensymbiosis.api.decorators import robot_tool
        from jiuwensymbiosis.api.base import BaseRobotApi

        class _CrashingApi(BaseRobotApi):
            @robot_tool
            def boom(self) -> None:
                raise RuntimeError("intentional crash")

        env = MockArmEnv()
        api = _CrashingApi(env)
        results = smoke.smoke_test_api(api, env=env)
        r = next(r for r in results if r["name"] == "boom")
        assert r["status"] == "fail"
        assert "intentional crash" in r["error"]
