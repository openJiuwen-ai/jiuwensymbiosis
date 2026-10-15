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
from tests.unit_tests.adapters.cruzr import description


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
    assert cfg["agent"]["modules"]["safety"]["enabled"] is True


def test_capability_gate_lets_the_rail_attach():
    from jiuwensymbiosis.agent.builder import _resolve_rails
    from jiuwensymbiosis.agent.config import ModulesConfig, SwitchConfig

    rails = _resolve_rails(_Session(CruzrEnv(CruzrConfig())), ModulesConfig(recovery=SwitchConfig(False)), None)
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


class TestNamedJointRangeIsEnforced:
    """``move_named_joint`` is how a plan says "raise the arm", so it is the one joint verb
    Cruzr actually exposes — and it is only guarded if the body states a range. A bug that
    left ``joint_limits`` unset disabled this check silently, which is what these tests
    exist to catch.

    The range is injected rather than read off the URDF: what is under test here is the RAIL
    WIRING, which must be verifiable on a machine that has no Cruzr description checked out.
    That the URDF itself yields limits is ``test_urdf_limits_are_read`` below, and that
    ``Chain.limits()`` returns a mapping at all is covered deterministically in
    ``tests/unit_tests/kinematics/test_urdf_chain.py``.
    """

    @pytest.fixture
    def rail_with_range(self):
        env = CruzrEnv(CruzrConfig())
        env.joint_limits = {"waist_yaw_joint": (-1.5, 1.5)}
        return SafetyRail(_Session(env))

    async def test_in_range_target_passes(self, rail_with_range):
        args = {"joint_name": "waist_yaw_joint", "position_rad": 0.0}
        await rail_with_range.before_tool_call(FakeCtx(tool_name="move_named_joint", tool_args=args))

    async def test_out_of_range_is_rejected(self, rail_with_range):
        args = {"joint_name": "waist_yaw_joint", "position_rad": 2.5}
        with pytest.raises(ValueError, match="out of limits"):
            await rail_with_range.before_tool_call(FakeCtx(tool_name="move_named_joint", tool_args=args))

    async def test_joint_the_body_states_no_limit_for_passes(self, rail_with_range):
        """Same "no range stated → no range check" rule the rest of the rail follows."""
        args = {"joint_name": "not_a_cruzr_joint", "position_rad": 99.0}
        await rail_with_range.before_tool_call(FakeCtx(tool_name="move_named_joint", tool_args=args))

    @description.requires_description
    def test_urdf_limits_are_read(self):
        """Cruzr's own description really does state the ranges the rail above enforces.

        Needs the description checked out, so it skips elsewhere — the config no longer
        carries a default path, because a wrong one fails more quietly than none.
        """
        limits = CruzrEnv(CruzrConfig(urdf_path=description.URDF)).joint_limits
        assert limits, "no joint_limits → move_named_joint would dispatch unchecked"
        assert "waist_yaw_joint" in limits


class TestUnconstrainedVerbsStillDispatch:
    """Cruzr declares no base / waist envelope. Absent limits mean "no range check" — the
    rail must not turn that into a refusal, or enabling it would break real-hardware
    behaviour. (``move_joint`` is not on this list: Cruzr both states joint limits and never
    implements that action — a bare list has no meaning across two arms plus a waist and a
    lifter, which is why the body exposes ``move_named_joint`` instead.)"""

    @pytest.mark.parametrize(
        ("tool_name", "tool_args"),
        [
            ("navigate_relative", {"dx_m": 1.5, "dy_m": 0.0, "dyaw_rad": 0.3}),
            ("rotate_base", {"dyaw_rad": 3.0}),
            ("drive_arc", {"radius_m": 0.8, "dyaw_rad": 1.1}),
            ("turn_waist", {"delta_rad": -1.2}),
            ("dual_arm_grasp", {}),
        ],
    )
    async def test_passes(self, rail, tool_name, tool_args):
        await rail.before_tool_call(FakeCtx(tool_name=tool_name, tool_args=tool_args))
