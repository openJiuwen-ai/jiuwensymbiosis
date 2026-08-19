# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SafetyRail is wired to Cruzr — the pre-flight check a gripperless dual-arm body
used to skip entirely (``enable_safety: false``) because the rail only knew three
single-arm tool names. It now derives its policy from declared capabilities, so
every verb Cruzr can move with is checked."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv
from jiuwensymbiosis.adapters.cruzr.geometry import LIFTER_LIMITS
from jiuwensymbiosis.rails.safety import SafetyRail
from tests.helpers import FakeCtx


class _Session:
    def __init__(self, env):
        self.env = env
        self.api = None


@pytest.fixture
def rail():
    return SafetyRail(_Session(CruzrEnv(CruzrConfig())))


def test_config_enables_the_rail():
    root = Path(__file__).resolve().parents[4]
    cfg = yaml.safe_load((root / "configs" / "cruzr" / "cruzr.yaml").read_text(encoding="utf-8"))
    assert cfg["agent"]["enable_safety"] is True


def test_capability_gate_lets_the_rail_attach():
    from jiuwensymbiosis.agent.builder import _resolve_rails

    rails = _resolve_rails(_Session(CruzrEnv(CruzrConfig())), False, True, False, None)
    assert [type(r).__name__ for r in rails] == ["SafetyRail"]


def test_watches_every_verb_cruzr_can_move_with(rail):
    assert {
        "move_joint",
        "navigate_relative",
        "rotate_base",
        "drive_arc",
        "set_lift_pose",
        "turn_waist",
    } <= rail.watch_tools


def test_lift_limits_are_the_urdf_range():
    assert CruzrEnv(CruzrConfig()).lift_limits == LIFTER_LIMITS


class TestLiftPolicy:
    """The lifter is the one envelope Cruzr can state exactly: the grasp/place planner
    already filters its own candidates against ``LIFTER_LIMITS``, so the rail rejects
    only targets that planner would never emit."""

    async def test_planner_reachable_target_passes(self, rail):
        q = {name: 0.5 * lo + 0.5 * hi for name, (lo, hi) in LIFTER_LIMITS.items()}
        await rail.before_tool_call(FakeCtx(tool_name="set_lift_pose", tool_args={"q_lifter": q}))

    async def test_out_of_urdf_range_is_rejected(self, rail):
        joint, (_, hi) = next(iter(LIFTER_LIMITS.items()))
        ctx = FakeCtx(tool_name="set_lift_pose", tool_args={"q_lifter": {joint: hi + 1.0}})
        with pytest.raises(ValueError, match="out of limits"):
            await rail.before_tool_call(ctx)


class TestUnconstrainedVerbsStillDispatch:
    """Cruzr declares no base / waist envelope, and ``joint_limits`` stays unset.
    Absent limits mean "no range check" — the rail must not turn that into a refusal,
    or enabling it would break real-hardware behaviour."""

    @pytest.mark.parametrize(
        ("tool_name", "tool_args"),
        [
            ("move_joint", {"q": [0.4]}),
            ("navigate_relative", {"dx_m": 1.5, "dy_m": 0.0, "dyaw_rad": 0.3}),
            ("rotate_base", {"dyaw_rad": 3.0}),
            ("drive_arc", {"radius_m": 0.8, "dyaw_rad": 1.1}),
            ("turn_waist", {"delta_rad": -1.2}),
            ("dual_arm_grasp", {}),
        ],
    )
    async def test_passes(self, rail, tool_name, tool_args):
        await rail.before_tool_call(FakeCtx(tool_name=tool_name, tool_args=tool_args))
