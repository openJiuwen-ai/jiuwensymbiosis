# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SafetyRail integration with the SO-101 adapter (no LeRobot).

The critical contract: ``goto_pose`` ships its coordinates inside a nested
``pose={...}`` object (one Cartesian pose as a value object), and
``SafetyRail.before_tool_call`` unpacks that nested object to read
``x``/``y``/``z`` for Z-floor + XY-bound checks. ``goto_xyzr`` keeps its
coordinates top-level and is covered the same way. ``move_joint`` is also
watched for soft-limit pre-checks.
"""

from __future__ import annotations

import pytest

from jiuwensymbiosis.adapters.so101.api import So101Api
from jiuwensymbiosis.adapters.so101.config import So101Config
from jiuwensymbiosis.adapters.so101.env import So101Env
from jiuwensymbiosis.adapters.so101.lowlevel import ARM_JOINT_ORDER
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.rails.safety import SafetyRail
from tests.helpers import FakeCtx

_ARM_LIMITS = {
    "shoulder_pan": (-90.0, 90.0),
    "shoulder_lift": (-90.0, 90.0),
    "elbow_flex": (-90.0, 90.0),
    "wrist_flex": (-90.0, 90.0),
    "wrist_roll": (-180.0, 180.0),
}


def _make_session(
    *,
    z_floor_mm: float = 50.0,
    xy_bounds: tuple[float, float, float, float] | None = None,
) -> RobotSession:
    env = So101Env(
        So101Config(
            port="/dev/fake",
            home_joints_deg=[0.0, 0.0, 0.0, 0.0, 0.0],
            joint_limits=_ARM_LIMITS,
            z_min_safe_mm=z_floor_mm,
            workspace_bounds=xy_bounds,
        )
    )
    return RobotSession(env=env, api=So101Api(env), name="so101-test")  # api bound but unused: rail reads env only


class TestSafetyRailGotoPoseNested:
    @pytest.mark.asyncio
    async def test_above_z_floor_passes(self):
        session = _make_session(z_floor_mm=50.0)
        rail = SafetyRail(session, z_floor_mm=50.0)
        ctx = FakeCtx(
            tool_name="goto_pose",
            tool_args={"pose": {"x": 100, "y": 0, "z": 200, "rx": 180, "ry": 0, "rz": 0}},
        )
        await rail.before_tool_call(ctx)  # must not raise

    @pytest.mark.asyncio
    async def test_below_z_floor_raises(self):
        session = _make_session(z_floor_mm=50.0)
        rail = SafetyRail(session, z_floor_mm=50.0)
        ctx = FakeCtx(
            tool_name="goto_pose",
            tool_args={"pose": {"x": 100, "y": 0, "z": 30, "rx": 180, "ry": 0, "rz": 0}},
        )
        with pytest.raises(ValueError, match="below z_floor"):
            await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_out_of_xy_bounds_raises(self):
        session = _make_session(
            z_floor_mm=50.0,
            xy_bounds=(0.0, -300.0, 500.0, 300.0),
        )
        rail = SafetyRail(session, xy_bounds_mm=(0.0, -300.0, 500.0, 300.0))
        ctx = FakeCtx(
            tool_name="goto_pose",
            tool_args={"pose": {"x": 600, "y": 0, "z": 200, "rx": 180, "ry": 0, "rz": 0}},
        )
        with pytest.raises(ValueError, match="out of bounds"):
            await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_goto_xyzr_flat_still_covered(self):
        # Regression: goto_xyzr keeps top-level x/y/z; the nested-pose flatten
        # logic must not change its observable behaviour. A low-Z goto_xyzr
        # call is still rejected.
        session = _make_session(z_floor_mm=50.0)
        rail = SafetyRail(session, z_floor_mm=50.0)
        ctx = FakeCtx(
            tool_name="goto_xyzr",
            tool_args={"x": 100, "y": 0, "z": 30, "r": 0},
        )
        with pytest.raises(ValueError, match="below z_floor"):
            await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_goto_pose_is_in_watch_tools(self):
        # Structural: goto_pose must be one of SafetyRail's watched tools, else
        # its Z/XY check never fires regardless of signature shape.
        session = _make_session()
        rail = SafetyRail(session, z_floor_mm=50.0)
        assert "goto_pose" in rail.watch_tools


class TestSafetyRailMoveJointSoftLimits:
    @pytest.mark.asyncio
    async def test_in_limits_passes(self):
        session = _make_session()
        rail = SafetyRail(session, z_floor_mm=50.0)
        # 5 joints, all within ARM_JOINT_ORDER limits.
        ctx = FakeCtx(
            tool_name="move_joint",
            tool_args={"q": [0.0, 0.0, 0.0, 0.0, 0.0]},
        )
        await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_out_of_soft_limit_raises(self):
        session = _make_session()
        rail = SafetyRail(session, z_floor_mm=50.0)
        # shoulder_pan limit is [-90, 90]; 95 exceeds it.
        ctx = FakeCtx(
            tool_name="move_joint",
            tool_args={"q": [95.0, 0.0, 0.0, 0.0, 0.0]},
        )
        with pytest.raises(ValueError, match="soft limit|out of range|joint"):
            await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_wrong_joint_count_rejected(self):
        """joint_limits has exactly 5 entries (ARM_JOINT_ORDER); q must match."""
        session = _make_session()
        rail = SafetyRail(session, z_floor_mm=50.0)
        ctx = FakeCtx(
            tool_name="move_joint",
            tool_args={"q": [0.0, 0.0, 0.0]},  # only 3
        )
        with pytest.raises((ValueError, TypeError)):
            await rail.before_tool_call(ctx)


class TestJointLimitsOrderStable:
    """SafetyRail's len(q) == len(names) check requires Env to expose joint_limits
    keyed over exactly ARM_JOINT_ORDER (5 items)."""

    def test_env_joint_limits_has_five_arm_keys(self):
        session = _make_session()
        limits = session.env.joint_limits
        assert limits is not None
        assert len(limits) == len(ARM_JOINT_ORDER)
        assert list(limits.keys()) == list(ARM_JOINT_ORDER)
