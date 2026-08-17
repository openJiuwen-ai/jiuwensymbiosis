# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for scripts/validate_adapter.py — the A-13 capability-tag check.

Guards against the silent-failure regression where the check read a different
attribute name (``_tool_meta``) than the decorator sets (``__robot_tool__``),
which made the check always return an empty list.
"""

from __future__ import annotations

import scripts.validate_adapter as va
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api import defaults
from jiuwensymbiosis.api.actions import GET_HOME_POSE, GOTO_XYZR, implements
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.env.base import KNOWN_CAPABILITIES as BASE_KNOWN_CAPABILITIES


class _CartesianApi(BaseRobotApi):
    """The two Cartesian actions the checks below need: one gated, one ungated (`home`)."""

    @implements(GOTO_XYZR)
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
        return defaults.goto_xyzr(self, x, y, z, r)

    @implements(GET_HOME_POSE)
    def get_home_pose(self) -> dict:
        return defaults.get_home_pose(self)


class _ApiWithBadCapability(_CartesianApi):

    @robot_tool(desc="a tool that claims a capability the env does not have", capability="myvendor.special")
    def do_special(self) -> None:
        return None


class _ApiWithAlignedCapability(_CartesianApi):
    @robot_tool(desc="a tool whose capability matches the env", capability="motion.cartesian")
    def do_aligned(self) -> None:
        return None


class TestCheckToolTags:
    def test_detects_capability_not_in_env(self):
        # env declares only motion.cartesian; the tool claims myvendor.special.
        env_caps = {"motion.cartesian"}
        warnings = va._check_tool_tags(_ApiWithBadCapability, env_caps)
        assert any("do_special" in w and "myvendor.special" in w for w in warnings)

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


class TestKnownCapabilitiesSingleSource:
    def test_validate_adapter_known_capabilities_matches_base(self):
        assert va.KNOWN_CAPABILITIES == BASE_KNOWN_CAPABILITIES
