# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for scripts/validate_adapter.py — the A-13 capability-tag check.

Guards against the silent-failure regression where the check read a different
attribute name (``_tool_meta``) than the decorator sets (``__tool_meta__``),
which made the check always return an empty list.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.validate_adapter as va
from jiuwensymbiosis.api import defaults
from jiuwensymbiosis.api.actions import GET_HOME_POSE, GOTO_XYZR, ActionSpec, implements
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.env.base import KNOWN_CAPABILITIES as BASE_KNOWN_CAPABILITIES


class _CartesianApi(BaseRobotApi):
    """The two Cartesian actions the checks below need: one gated, one ungated (`home`)."""

    @implements(GOTO_XYZR)
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None,
                  orientation_policy: str = "top_down") -> None:
        return defaults.goto_xyzr(self, x, y, z, r)

    @implements(GET_HOME_POSE)
    def get_home_pose(self) -> dict:
        return defaults.get_home_pose(self)


class _ApiWithBadCapability(_CartesianApi):

    @implements(ActionSpec(name="do_special", description="a tool that claims a capability the env lacks",
                           capability="grasp.suction"))
    def do_special(self) -> None:
        return None


class _ApiWithAlignedCapability(_CartesianApi):
    @implements(ActionSpec(name="do_aligned", description="a tool whose capability matches the env",
                           capability="motion.cartesian"))
    def do_aligned(self) -> None:
        return None


class TestCheckToolTags:
    def test_detects_capability_not_in_env(self):
        # env declares only motion.cartesian; the tool claims grasp.suction.
        env_caps = {"motion.cartesian"}
        warnings = va._check_tool_tags(_ApiWithBadCapability, env_caps)
        assert any("do_special" in w and "grasp.suction" in w for w in warnings)

    def test_clean_when_capability_aligned(self):
        env_caps = {"motion.cartesian"}
        warnings = va._check_tool_tags(_ApiWithAlignedCapability, env_caps)
        assert warnings == []

    def test_tools_without_explicit_capability_are_not_flagged(self):
        # ``home`` carries no capability at all (every body owes a safe posture), so it
        # must never be flagged, whatever the env declares.
        env_caps = set()
        warnings = va._check_tool_tags(_ApiWithAlignedCapability, env_caps)
        # do_aligned IS flagged (its explicit motion.cartesian not in empty env_caps),
        # but the ungated `home` must not appear. Match the quoted tool name, not a
        # substring — `get_home_pose` also contains "home" and IS legitimately flagged.
        assert any("'do_aligned'" in w for w in warnings)
        assert all("'home'" not in w for w in warnings)


@pytest.mark.parametrize("attribute", ["resource_keys", "config_factory"])
def test_missing_admission_metadata_is_reported(attribute, monkeypatch):
    from jiuwensymbiosis.adapters.piper import build_piper_session

    monkeypatch.setattr(build_piper_session, attribute, None)
    results = va.run_checks("jiuwensymbiosis.adapters.piper")
    assert any(code == "S-17" and severity == "ERROR" for code, severity, _ in results)


class TestKnownCapabilitiesSingleSource:
    def test_validate_adapter_known_capabilities_matches_base(self):
        assert va.KNOWN_CAPABILITIES == BASE_KNOWN_CAPABILITIES


class TestCalibrationIntegrationCheck:
    _MODULE = "jiuwensymbiosis.adapters.new_arm"
    _WRAPPER = "jiuwensymbiosis.calibration.adapters.new_arm"

    def test_missing_wrapper_is_optional_info(self, monkeypatch):
        def _missing(name: str):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

        monkeypatch.setattr(va.importlib, "import_module", _missing)

        results = va._check_calibration_integration(self._MODULE)

        assert len(results) == 1
        assert results[0][0:2] == ("C-16", va._SEVERITY_INFO)

    def test_wrapper_dependency_failure_is_error(self, monkeypatch):
        def _dependency_missing(_name: str):
            raise ModuleNotFoundError("optional vendor SDK", name="optional_vendor_sdk")

        monkeypatch.setattr(va.importlib, "import_module", _dependency_missing)

        results = va._check_calibration_integration(self._MODULE)

        assert results[0][0:2] == ("C-16", va._SEVERITY_ERROR)
        assert "optional_vendor_sdk" in results[0][2]

    def test_wrapper_runtime_failure_is_error(self, monkeypatch):
        def _broken(_name: str):
            raise RuntimeError("spec construction failed")

        monkeypatch.setattr(va.importlib, "import_module", _broken)

        results = va._check_calibration_integration(self._MODULE)

        assert results[0][0:2] == ("C-16", va._SEVERITY_ERROR)
        assert "spec construction failed" in results[0][2]

    def test_malformed_spec_is_error(self, monkeypatch):
        monkeypatch.setattr(
            va.importlib,
            "import_module",
            lambda _name: SimpleNamespace(CALIBRATION_ADAPTER_SPEC=object()),
        )

        results = va._check_calibration_integration(self._MODULE)

        assert results[0][0:2] == ("C-16", va._SEVERITY_ERROR)
