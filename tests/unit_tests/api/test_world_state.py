# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""World state must track what executing actions actually established.

The planner reads this to decide where it is starting from, so the two failure
modes that matter are opposite: claiming something holds when it does not (a
sensing that failed, a reading taken before the body moved) would send the robot
to stale coordinates, while dropping something that does hold would make it
re-sense pointlessly.
"""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.api.world_state import WorldState
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation
from jiuwensymbiosis.tools.builder import build_robot_tools
from jiuwensymbiosis.tools.robot_control_tool import RobotControlTool


class _Env(BaseRobotEnv):
    """A mobile body that reports pose, joints and (optionally) payload state."""

    capabilities = frozenset({"motion.base", "vision.detection", "grasp.parallel"})
    name = "fake"

    def __init__(self) -> None:
        self.holding_payload: bool | None = None

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_observation(self) -> RobotObservation:
        return RobotObservation(pose={"x": 1.0, "y": 2.0, "z": 3.0}, joints=[0.1, 0.2], extra={"battery": 88})


class _Api(BaseRobotApi):
    # Declared as a set so ``BaseRobotApi.capabilities`` picks all three up; the
    # tool builder gates on api ∩ env, so without this the body emits no tools.
    capability = {"motion.base", "vision.detection", "grasp.parallel"}

    def __init__(self, env: _Env) -> None:
        super().__init__(env)
        self.detection_ok = True

    @robot_tool(desc="sense", produces_location=True, capability="vision.detection")
    def detect(self, object_name: str) -> dict:
        if not self.detection_ok:
            return {"ok": False, "reason": "not_in_view", "object": object_name}
        return {"ok": True, "object": object_name, "position": [100.0, 0.0, 50.0]}

    @robot_tool(desc="drive", invalidates_locations=True, capability="motion.base", tags=["motion"])
    def drive(self, dx_m: float) -> dict:
        return {"ok": True}

    @robot_tool(
        desc="grip", requires=["payload.clear"], provides=["payload.held"], capability="grasp.parallel", tags=["grasp"]
    )
    def grip(self) -> dict:
        return {"ok": True}

    @robot_tool(desc="release", provides=["payload.clear"], capability="grasp.parallel", tags=["grasp"])
    def release(self) -> dict:
        return {"ok": True}


class _Session:
    def __init__(self) -> None:
        self.env = _Env()
        self.api = _Api(self.env)


def _fresh() -> _Session:
    return _Session()


# --------------------------------------------------------------------------- #
# ExecutionMemory
# --------------------------------------------------------------------------- #
def test_a_successful_sensing_is_remembered():
    s = _fresh()
    s.api.memory.observe(_Api.detect.__robot_tool__, {"object_name": "crate"}, s.api.detect("crate"))
    record = s.api.memory.get("crate")
    assert record is not None and record.op == "detect"
    assert record.result["position"] == [100.0, 0.0, 50.0]


def test_a_failed_sensing_establishes_nothing():
    # This is how "the target was not in view" recovers: no location is recorded,
    # so a later step that needs one finds none and the caller can re-plan into a
    # search instead of driving to coordinates that were never produced.
    s = _fresh()
    s.api.detection_ok = False
    s.api.memory.observe(_Api.detect.__robot_tool__, {"object_name": "crate"}, s.api.detect("crate"))
    assert s.api.memory.get("crate") is None
    assert s.api.memory.latest() is None


def test_moving_the_base_drops_every_location():
    s = _fresh()
    s.api.memory.observe(_Api.detect.__robot_tool__, {"object_name": "crate"}, s.api.detect("crate"))
    s.api.memory.observe(_Api.drive.__robot_tool__, {"dx_m": 1.0}, s.api.drive(1.0))
    assert s.api.memory.locations == {}


def test_effects_advance_the_believed_self_state():
    s = _fresh()
    s.api.memory.observe(_Api.grip.__robot_tool__, {}, s.api.grip())
    assert "payload.held" in s.api.memory.self_state
    s.api.memory.observe(_Api.release.__robot_tool__, {}, s.api.release())
    assert s.api.memory.self_state == frozenset({"payload.clear"})  # mutually exclusive


# --------------------------------------------------------------------------- #
# WorldState
# --------------------------------------------------------------------------- #
def test_snapshot_reports_belief_when_the_env_cannot_measure_it():
    s = _fresh()
    s.api.memory.observe(_Api.grip.__robot_tool__, {}, s.api.grip())
    assert "payload.held" in WorldState.snapshot(s).tokens


def test_the_env_overrides_a_contradicting_belief():
    s = _fresh()
    s.api.memory.observe(_Api.grip.__robot_tool__, {}, s.api.grip())  # believes it is holding
    s.env.holding_payload = False  # the hardware says otherwise
    tokens = WorldState.snapshot(s).tokens
    assert "payload.clear" in tokens and "payload.held" not in tokens


def test_snapshot_carries_proprioception():
    state = WorldState.snapshot(_fresh())
    assert state.pose == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert state.joints == [0.1, 0.2]
    assert state.extra["battery"] == 88


def test_snapshot_survives_a_body_that_cannot_report():
    class _Broken(_Env):
        def get_observation(self):
            raise RuntimeError("sensor down")

    s = _fresh()
    s.env = _Broken()
    s.api.env = s.env
    assert WorldState.snapshot(s).pose is None  # degraded, not crashed


def test_prompt_block_is_empty_when_nothing_is_known():
    assert WorldState().as_prompt_block() == ""


def test_prompt_block_names_what_is_known():
    s = _fresh()
    s.api.memory.observe(_Api.detect.__robot_tool__, {"object_name": "crate"}, s.api.detect("crate"))
    block = WorldState.snapshot(s).as_prompt_block()
    assert "【当前状态】" in block and "crate" in block


# --------------------------------------------------------------------------- #
# Both dispatch paths must record identically
# --------------------------------------------------------------------------- #
async def test_robot_control_dispatch_records():
    s = _fresh()
    tool = RobotControlTool(s.api, env=s.env)
    await tool.invoke({"action": "detect", "params": {"object_name": "crate"}})
    assert s.api.memory.get("crate") is not None
    await tool.invoke({"action": "drive", "params": {"dx_m": 1.0}})
    assert s.api.memory.locations == {}


async def test_separate_tool_dispatch_records():
    s = _fresh()
    by_name = {t.card.name: t for t in build_robot_tools(s.api, env=s.env)}
    await by_name["detect"].invoke({"object_name": "crate"})
    assert s.api.memory.get("crate") is not None
    await by_name["drive"].invoke({"dx_m": 1.0})
    assert s.api.memory.locations == {}


def test_separate_tool_dispatch_preserves_tool_metadata():
    # Rails and the tool builder read __robot_tool__ back off the callable.
    s = _fresh()
    wrapped: Any = build_robot_tools(s.api, env=s.env)
    by_name = {t.card.name: t for t in wrapped}
    assert by_name["grip"].card.description == "grip"
    meta = getattr(by_name["grip"]._func, "__robot_tool__", None)  # noqa: SLF001 - asserting the wrapper is transparent
    assert meta is not None and meta.tags == ["grasp"]
