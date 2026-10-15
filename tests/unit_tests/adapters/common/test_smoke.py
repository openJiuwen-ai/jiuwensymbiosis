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
from types import SimpleNamespace
from unittest.mock import Mock

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


@pytest.mark.parametrize(
    "keys, physical_id, accepted",
    [
        ((), None, False),
        (("can:can0",), None, True),
        ((), "robot-1", True),
        ("can:can0", None, False),
    ],
)
def test_admission_is_checked_before_session_build(smoke, tmp_path, keys, physical_id, accepted):
    source = tmp_path / "robot.yaml"
    source.write_text(json.dumps({"physical_device_id": physical_id}), encoding="utf-8")
    cfg = SimpleNamespace()
    builder = Mock()
    builder.config_factory = SimpleNamespace(from_dict=lambda _: cfg)
    builder.resource_keys = lambda _: keys
    if accepted:
        smoke._build_session(builder, "example.robot", str(source))
        builder.assert_called_once_with(cfg)
    else:
        with pytest.raises((TypeError, ValueError)):
            smoke._build_session(builder, "example.robot", str(source))
        builder.assert_not_called()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "adapter, fields, extra",
    [
        ("piper", ("calib_path",), {}),
        (
            "so101",
            ("calib_path", "urdf_path", "calibration_dir"),
            {
                "port": "/dev/test-arm",
                "home_use_init_pose": True,
                "joint_limits": {
                    name: [-90, 90]
                    for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
                },
            },
        ),
        (
            "cruzr",
            ("camera_calib_path", "urdf_path", "urdf_package_dir", "ros_workspace", "camera_fastdds_profiles_file"),
            {},
        ),
    ],
)
def test_yaml_runtime_and_smoke_share_source_relative_paths(
    smoke, tmp_path, monkeypatch, adapter, fields, extra, nested
):
    """Every admission/validation path must use the config directory, not cwd."""
    from jiuwensymbiosis.runtime.bindings import prepare_binding

    monkeypatch.delenv("CAMERA_SERIAL", raising=False)
    source_dir = tmp_path / "config"
    source_dir.mkdir()
    work_dir = tmp_path / "elsewhere"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    low_level = {**extra, **{name: f"missing/{name}" for name in fields}}
    data = {"adapter": adapter, **({"env": {"cfg": {"low_level": low_level}}} if nested else low_level)}
    source = source_dir / "robot.yaml"
    source.write_text(json.dumps(data), encoding="utf-8")
    module = importlib.import_module(f"jiuwensymbiosis.adapters.{adapter}")
    builder = getattr(module, f"build_{adapter}_session")
    sessions = [
        builder.from_yaml(source, include_sidecars=False),
        prepare_binding(source, include_sidecars=False).build_session(),
        smoke._build_session(builder, module.__name__, str(source)),
    ]
    for session in sessions:
        for name in fields:
            assert getattr(session.env.cfg, name) == str(source_dir / "missing" / name)
        assert session.cleanup_report().released


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
        from jiuwensymbiosis.api.actions import ActionSpec, implements
        from jiuwensymbiosis.api.base import BaseRobotApi

        class _AwkwardApi(BaseRobotApi):
            @implements(ActionSpec(name="needs_unknown", description="takes an un-inferable param"))
            def needs_unknown(self, mysterious_param) -> dict:
                return {"echo": mysterious_param}

        env = MockArmEnv()
        api = _AwkwardApi(env)
        results = smoke.smoke_test_api(api, env=env)
        r = next(r for r in results if r["name"] == "needs_unknown")
        assert r["status"] == "skip"
        assert "mysterious_param" in r["reason"]

    def test_crashing_tool_recorded_as_fail(self, smoke):
        from jiuwensymbiosis.api.actions import ActionSpec, implements
        from jiuwensymbiosis.api.base import BaseRobotApi

        class _CrashingApi(BaseRobotApi):
            @implements(ActionSpec(name="boom", description="always raises"))
            def boom(self) -> None:
                raise RuntimeError("intentional crash")

        env = MockArmEnv()
        api = _CrashingApi(env)
        results = smoke.smoke_test_api(api, env=env)
        r = next(r for r in results if r["name"] == "boom")
        assert r["status"] == "fail"
        assert "intentional crash" in r["error"]
