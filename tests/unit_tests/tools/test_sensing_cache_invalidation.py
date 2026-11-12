# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The api's sensing cache and ``memory.locations`` go stale together.

Two stores hold one fact: ``ExecutionMemory`` owns whether a reading is still valid,
the api cache owns the payload a grasp/place consumes. If only the memory is cleared
when the base moves, the planner sees "no location" while the acting step still reads
base-frame coordinates taken from a standpoint the body has left — and drives to them.
"""

from __future__ import annotations

import asyncio

import pytest

from jiuwensymbiosis.agent.fast.runner import direct_executor
from jiuwensymbiosis.api.actions import APPROACH_FOR_GRASP, LOCATE_FOR_GRASP, NAVIGATE_RELATIVE, implements
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.env.mock import MockArmEnv
from jiuwensymbiosis.tools.builder import build_robot_tools
from jiuwensymbiosis.tools.robot_control_tool import RobotControlTool, begin_action, record_action


class CacheApi(BaseRobotApi):
    """Three real contracts: sense, move, and move-while-sensing."""

    @implements(LOCATE_FOR_GRASP)
    def locate_for_grasp(self, object_name: str = "box", reference: str | None = None, relation: str = "on") -> dict:
        result = {"ok": True, "object": object_name, "position": [100.0, 0.0, 50.0]}
        self.last_detection = result
        return result

    @implements(NAVIGATE_RELATIVE)
    def navigate_relative(self, dx_m: float, dy_m: float = 0.0, dyaw_rad: float = 0.0) -> dict:
        if self.move_error:
            raise RuntimeError("timeout after moving")
        return {"ok": self.move_ok}

    @implements(APPROACH_FOR_GRASP)
    def approach_for_grasp(self, object_name: str = "box", reference: str | None = None, relation: str = "on") -> dict:
        # Mirrors the real approach loops: the move and the re-sense happen inside one
        # action, so the cache it leaves behind was measured from the NEW standpoint.
        fresh = {"ok": True, "object": object_name, "position": [200.0, 0.0, 50.0]}
        self.last_detection = fresh
        return fresh

    move_ok = True
    move_error = False


def _sensed_api() -> CacheApi:
    api = CacheApi(MockArmEnv())
    record_action(api, api.locate_for_grasp, {"object_name": "box"}, api.locate_for_grasp("box"))
    return api


class TestBaseMotionStalesBothStores:
    def test_sensing_fills_both(self):
        api = _sensed_api()
        assert api.last_detection is not None
        assert api.memory.get("box") is not None

    def test_navigate_clears_both(self):
        api = _sensed_api()
        record_action(api, api.navigate_relative, {"dx_m": 1.0}, api.navigate_relative(1.0))
        assert api.memory.locations == {}
        assert api.last_detection is None, "stale base-frame geometry survived a base move"
        assert api.last_surface is None

    def test_a_failed_move_invalidates_both_stores(self):
        api = _sensed_api()
        api.move_ok = False
        record_action(api, api.navigate_relative, {"dx_m": 1.0}, api.navigate_relative(1.0))
        assert api.memory.get("box") is None
        assert api.last_detection is None


class TestProducingActionsKeepWhatTheyMeasured:
    def test_approach_keeps_its_own_fresh_reading(self):
        api = _sensed_api()
        record_action(api, api.approach_for_grasp, {"object_name": "box"}, api.approach_for_grasp("box"))
        assert api.last_detection is not None, "the reading the action just took was thrown away"
        assert api.last_detection["position"] == [200.0, 0.0, 50.0]
        assert api.memory.get("box") is not None

    def test_begin_action_preserves_the_input_cache_for_producing_actions(self):
        # Approach consumes the cached detection as its input (the real loops skip a
        # full search when one is fresh); clearing it pre-dispatch would force a
        # re-search on every approach. Failure clearing stays post-hoc.
        api = _sensed_api()
        begin_action(api, api.approach_for_grasp)
        assert api.last_detection is not None, "approach lost the detection it consumes as input"

    def test_begin_action_clears_the_cache_for_pure_motions(self):
        api = _sensed_api()
        begin_action(api, api.navigate_relative)
        assert api.last_detection is None

    async def test_approach_consumes_the_detection_locate_just_produced(self):
        seen = {}

        class ApproachingApi(CacheApi):
            @implements(APPROACH_FOR_GRASP)
            def approach_for_grasp(self, object_name="box", reference=None, relation="on"):
                seen["input"] = self.last_detection
                return {"ok": True}

        api = ApproachingApi(MockArmEnv())
        api.last_detection = {"ok": True, "position": [100.0, 0.0, 50.0]}
        await RobotControlTool(api).invoke({"action": "approach_for_grasp", "params": {}})
        assert seen["input"] is not None, "dispatch cleared the cache the approach consumes"


def test_home_stales_sensed_positions_like_any_motion():
    # Homing may move the body (a mobile base drives home), so positions sensed from a
    # departed standpoint do not survive it — same rule as any other motion.
    api = _sensed_api()
    record_action(api, api.home, {}, None)
    assert api.last_detection is None
    assert api.memory.get("box") is None


class TestEveryDispatchPathAgrees:
    def test_direct_executor_stales_the_cache_too(self):
        api = _sensed_api()
        run = direct_executor(api)
        assert run("navigate_relative", {"dx_m": 1.0})["ok"] is True
        assert api.memory.locations == {}
        assert api.last_detection is None, "direct_executor left the cache the tool path clears"

    def test_direct_executor_records_a_sensing(self):
        api = CacheApi(MockArmEnv())
        run = direct_executor(api)
        run("locate_for_grasp", {"object_name": "box"})
        assert api.memory.get("box") is not None


@pytest.mark.parametrize("strategy", ["direct", "dispatcher", "separate"])
@pytest.mark.parametrize("raises", [False, True])
async def test_failed_motion_invalidates_through_every_tool_strategy(strategy, raises):
    api = _sensed_api()
    api.move_ok = False
    api.move_error = raises
    if strategy == "direct":
        direct_executor(api)("navigate_relative", {"dx_m": 0.4})
    elif strategy == "dispatcher":
        await RobotControlTool(api).invoke({"action": "navigate_relative", "params": {"dx_m": 0.4}})
    else:
        tool = next(t for t in build_robot_tools(api) if t.card.name == "navigate_relative")
        try:
            await tool.invoke({"dx_m": 0.4})
        except RuntimeError:
            pass
    assert api.memory.latest() is None
    assert api.last_detection is None


async def test_cancelling_async_motion_drops_even_an_intermediate_reading():
    started = asyncio.Event()

    class AsyncApi(CacheApi):
        @implements(APPROACH_FOR_GRASP)
        async def approach_for_grasp(self, object_name="box", reference=None, relation="on"):
            self.last_detection = {"ok": True, "position": [200, 0, 50]}
            started.set()
            await asyncio.Event().wait()

    api = AsyncApi(MockArmEnv())
    tool = RobotControlTool(api)
    task = asyncio.create_task(tool.invoke({"action": "approach_for_grasp", "params": {}}))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert api.last_detection is None
    assert api.memory.latest() is None
