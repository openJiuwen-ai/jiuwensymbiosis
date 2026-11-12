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

import pytest

from jiuwensymbiosis.api.actions import ActionSpec, implements
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.world_state import WorldState, current_tokens
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

    def home(self) -> None: ...

    def get_observation(self) -> RobotObservation:
        return RobotObservation(pose={"x": 1.0, "y": 2.0, "z": 3.0}, joints=[0.1, 0.2], extra={"battery": 88})


class _Api(BaseRobotApi):
    # Declared as a set so ``BaseRobotApi.capabilities`` picks all three up; the
    # tool builder gates on api ∩ env, so without this the body emits no tools.
    capability = {"motion.base", "vision.detection", "grasp.parallel"}

    def __init__(self, env: _Env) -> None:
        super().__init__(env)
        self.detection_ok = True

    @implements(ActionSpec(name="detect", description="sense", produces_location=True, capability="vision.detection"))
    def detect(self, object_name: str) -> dict:
        if not self.detection_ok:
            return {"ok": False, "reason": "not_in_view", "object": object_name}
        return {"ok": True, "object": object_name, "position": [100.0, 0.0, 50.0]}

    @implements(
        ActionSpec(
            name="drive", description="drive", invalidates_locations=True, capability="motion.base", tags=("motion",)
        )
    )
    def drive(self, dx_m: float) -> dict:
        return {"ok": True}

    @implements(
        ActionSpec(
            name="grip",
            description="grip",
            requires=("payload.clear",),
            provides=("payload.held",),
            capability="grasp.parallel",
            tags=("grasp",),
        )
    )
    def grip(self) -> dict:
        return {"ok": True, "grasp_confirmed": True}

    @implements(
        ActionSpec(
            name="stow",
            description="raise the held payload to travel height",
            requires=("payload.held",),
            provides=("payload.stowed",),
            capability="motion.base",
            tags=("motion",),
        )
    )
    def stow(self) -> dict:
        return {"ok": True}

    @implements(
        ActionSpec(
            name="release",
            description="release",
            provides=("payload.clear",),
            capability="grasp.parallel",
            tags=("grasp",),
        )
    )
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
    s.api.memory.observe(_Api.detect.__tool_meta__, {"object_name": "crate"}, s.api.detect("crate"))
    record = s.api.memory.get("crate")
    assert record is not None and record.op == "detect"
    assert record.result["position"] == [100.0, 0.0, 50.0]


def test_a_failed_sensing_establishes_nothing():
    # This is how "the target was not in view" recovers: no location is recorded,
    # so a later step that needs one finds none and the caller can re-plan into a
    # search instead of driving to coordinates that were never produced.
    s = _fresh()
    s.api.detection_ok = False
    s.api.memory.observe(_Api.detect.__tool_meta__, {"object_name": "crate"}, s.api.detect("crate"))
    assert s.api.memory.get("crate") is None
    assert s.api.memory.latest() is None


def test_moving_the_base_drops_every_location():
    s = _fresh()
    s.api.memory.observe(_Api.detect.__tool_meta__, {"object_name": "crate"}, s.api.detect("crate"))
    s.api.memory.observe(_Api.drive.__tool_meta__, {"dx_m": 1.0}, s.api.drive(1.0))
    assert s.api.memory.locations == {}


def test_effects_advance_the_believed_self_state():
    s = _fresh()
    s.api.memory.observe(_Api.grip.__tool_meta__, {}, s.api.grip())
    assert "payload.held" in s.api.memory.self_state
    s.api.memory.observe(_Api.release.__tool_meta__, {}, s.api.release())
    assert s.api.memory.self_state == frozenset({"payload.clear"})  # mutually exclusive


# --------------------------------------------------------------------------- #
# WorldState
# --------------------------------------------------------------------------- #
def test_snapshot_reports_belief_when_the_env_cannot_measure_it():
    s = _fresh()
    s.api.memory.observe(_Api.grip.__tool_meta__, {}, s.api.grip())
    assert "payload.held" in WorldState.snapshot(s).tokens


def test_the_env_overrides_a_contradicting_belief():
    s = _fresh()
    s.api.memory.observe(_Api.grip.__tool_meta__, {}, s.api.grip())  # believes it is holding
    s.env.holding_payload = False  # the hardware says otherwise
    tokens = WorldState.snapshot(s).tokens
    assert "payload.clear" in tokens and "payload.held" not in tokens


def test_observation_keeps_the_belief_it_cannot_speak_to():
    # ``holding_payload`` is a bool: it can rule out "empty", never "not yet stowed".
    # Dropping every payload.* belief whenever it answers loses the one token the
    # belief layer exists for, and the planner then re-stows a payload already stowed.
    s = _fresh()
    s.api.memory.observe(_Api.grip.__tool_meta__, {}, s.api.grip())
    s.api.memory.observe(_Api.stow.__tool_meta__, {}, s.api.stow())
    s.env.holding_payload = True
    tokens = current_tokens(s)
    assert "payload.stowed" in tokens and "payload.held" in tokens


def test_observation_still_drops_the_belief_it_does_contradict():
    s = _fresh()
    s.api.memory.observe(_Api.grip.__tool_meta__, {}, s.api.grip())
    s.api.memory.observe(_Api.stow.__tool_meta__, {}, s.api.stow())
    s.env.holding_payload = False  # an empty end effector cannot be carrying a stowed payload
    tokens = current_tokens(s)
    assert tokens == frozenset({"payload.clear"})


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
    s.api.memory.observe(_Api.detect.__tool_meta__, {"object_name": "crate"}, s.api.detect("crate"))
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
    # Rails and the tool builder read __tool_meta__ back off the callable.
    s = _fresh()
    wrapped: Any = build_robot_tools(s.api, env=s.env)
    by_name = {t.card.name: t for t in wrapped}
    assert by_name["grip"].card.description == "grip"
    meta = getattr(by_name["grip"]._func, "__tool_meta__", None)  # noqa: SLF001 - asserting the wrapper is transparent
    assert meta is not None and meta.tags == ["grasp"]


class TestJointUnitsReachThePrompt:
    """Joint numbers used to render into the planner prompt bare: ``关节：1.50``. 1.5 is a small
    nudge in degrees and 86 degrees in radians, and nothing can infer which — piper and so101
    are degrees, cruzr is radians. The unit therefore travels with the numbers, and an
    unstated one says so rather than being guessed."""

    def test_the_declared_unit_is_rendered(self):
        assert "关节(rad)：" in WorldState(joints=[1.5], joint_units="rad").as_prompt_block()
        assert "关节(deg)：" in WorldState(joints=[1.5], joint_units="deg").as_prompt_block()

    def test_an_unstated_unit_is_named_as_unknown_not_guessed(self):
        block = WorldState(joints=[1.5]).as_prompt_block()
        assert "单位未声明" in block
        assert "关节(rad)" not in block and "关节(deg)" not in block

    def test_describe_carries_the_unit_for_the_state_cli(self):
        assert WorldState(joints=[1.5], joint_units="rad").describe()["joint_units"] == "rad"

    def test_every_shipped_body_states_its_unit(self):
        """An unstated unit is legal but weak; the three real bodies must not rely on it."""
        from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv
        from jiuwensymbiosis.adapters.piper.env import PiperEnv
        from jiuwensymbiosis.adapters.so101.env import So101Env

        assert PiperEnv._joint_units == "deg"
        assert So101Env._joint_units == "deg"
        assert CruzrEnv._joint_units == "rad"


class TestDefaultOrientationPolicyReachesThePrompt:
    """``goto_xyzr``'s schema can only say ``default: null`` — ``implements()`` runs at class
    definition time, before any config exists. So the planner could see THAT there is a
    default but not WHICH, and on so101 (``preserve``) that is the difference between
    approaching a grasp top-down and approaching it sideways, with no error raised."""

    def test_the_policy_and_what_it_means_are_both_rendered(self):
        block = WorldState(default_orientation_policy="preserve").as_prompt_block()
        assert "preserve" in block
        assert "不会自动朝下" in block, "the name alone does not warn a planner"

    def test_a_body_without_a_cartesian_default_says_nothing(self):
        assert WorldState(default_orientation_policy=None).as_prompt_block() == ""

    def test_describe_carries_it_for_the_state_cli(self):
        state = WorldState(default_orientation_policy="top_down")
        assert state.describe()["default_orientation_policy"] == "top_down"

    def test_a_body_reports_a_policy_exactly_when_it_has_cartesian_motion(self):
        """The rule, not each body's current value: goto_xyzr is what the policy is FOR, so a
        body that has the action must say which default it applies and a body that has not
        must stay silent rather than report a policy nothing will read."""
        import importlib

        for name in ("piper", "so101", "cruzr"):
            mod = importlib.import_module(f"jiuwensymbiosis.adapters.{name}")
            session = getattr(mod, f"build_{name}_session").from_yaml(f"configs/{name}/{name}.yaml")
            has_cartesian = "motion.cartesian" in session.env.capabilities
            stated = session.env.default_orientation_policy
            assert bool(stated) == has_cartesian, name
            if stated:
                assert stated in ("preserve", "top_down", "grasp"), name

    def test_so101_tracks_its_config_rather_than_a_hard_coded_constant(self):
        """It is a per-workcell setting; a constant here would drift the moment the YAML changed."""
        import importlib

        mod = importlib.import_module("jiuwensymbiosis.adapters.so101")
        session = mod.build_so101_session.from_yaml("configs/so101/so101.yaml")
        session.env.cfg.cartesian_orientation_policy = "top_down"
        assert session.env.default_orientation_policy == "top_down"


class TestReachEnvelopeReachesThePlanner:
    """The body's working range used to exist only as a SafetyRail rejection: a plan could be
    compiled, validated and started before anyone mentioned the arm cannot go there. Stating it
    at plan time is the difference between planning inside the envelope and being bounced off
    it one step at a time."""

    def test_config_bounds_are_stated_with_what_to_do_about_them(self):
        block = WorldState(workspace_bounds=(0.0, -500.0, 700.0, 500.0), z_min_safe=50.0).as_prompt_block()
        assert "x∈[0,700]" in block and "y∈[-500,500]" in block
        assert "Z 下限 50mm" in block
        assert "请在范围内规划" in block

    def test_a_body_that_states_no_envelope_says_nothing(self):
        """A mobile body has no fixed Cartesian box; inventing one would be worse than silence."""
        assert WorldState().as_prompt_block() == ""

    def test_describe_carries_the_envelope_for_the_state_cli(self):
        d = WorldState(workspace_bounds=(0.0, -1.0, 2.0, 3.0), z_min_safe=5.0, reach_prior={"r_mm": 700}).describe()
        assert d["workspace_bounds"] == [0.0, -1.0, 2.0, 3.0]
        assert d["z_min_safe"] == 5.0 and d["reach_prior"] == {"r_mm": 700}

    def test_a_config_box_and_a_urdf_reach_model_are_independent(self):
        """The rule: these are two different ways to know a working range, and a body may have
        either without the other. piper is the case that matters — a fixed arm with no URDF
        still has a box, and that box is what keeps it from planning outside itself. Not
        asserting piper HAS no URDF: the day it ships one, both should be present."""
        import importlib

        session = importlib.import_module("jiuwensymbiosis.adapters.piper").build_piper_session.from_yaml(
            "configs/piper/piper.yaml"
        )
        state = WorldState.snapshot(session)
        assert state.workspace_bounds is not None, "a config box does not depend on a URDF"
        assert "工作范围" in state.as_prompt_block()


class TestSensedLocationsCarryReachability:
    """The two channels used to be split: the pre-run scene had reach annotations but went stale,
    and these locations stayed fresh but carried none. A re-plan reads THIS one."""

    def test_unknown_reach_is_omitted_not_rendered_as_out_of_reach(self):
        block = WorldState(
            locations=[
                {"referent": "box", "sensed_by": "locate_for_grasp", "age_s": 1.0, "position_mm": [100.0, 0.0, 50.0]}
            ]
        ).as_prompt_block()
        assert "够不着" not in block and "够得着" not in block

    def test_out_of_reach_says_what_to_do(self):
        block = WorldState(
            locations=[
                {
                    "referent": "box",
                    "sensed_by": "locate_for_grasp",
                    "age_s": 1.0,
                    "position_mm": [100.0, 0.0, 50.0],
                    "reachable": False,
                }
            ]
        ).as_prompt_block()
        assert "够不着，需先移动过去" in block


def test_empty_close_never_establishes_payload_held():
    from jiuwensymbiosis.api.actions import CLOSE_GRIPPER
    from jiuwensymbiosis.api.decorators import ToolMeta

    s = _fresh()
    s.api.memory.observe(ToolMeta(CLOSE_GRIPPER, {}), {}, {"ok": True, "contact": False})
    assert "payload.held" not in WorldState.snapshot(s).tokens


def test_unmeasured_close_never_establishes_payload_held():
    from jiuwensymbiosis.api.actions import CLOSE_GRIPPER
    from jiuwensymbiosis.api.decorators import ToolMeta

    s = _fresh()
    s.api.memory.observe(ToolMeta(CLOSE_GRIPPER, {}), {}, {"ok": True, "state": "closed"})
    assert "payload.held" not in WorldState.snapshot(s).tokens


@pytest.mark.parametrize(
    "result,hook,trusted,expected",
    [
        # No evidence and no declaration: unknown fails closed.
        ({"ok": True, "state": "closed"}, None, False, None),
        # A declared body accepts a successful command; a failed one it does not.
        ({"ok": True, "state": "closed"}, None, True, True),
        ({"ok": False, "reason": "timeout"}, None, True, None),
        # ``contact`` counts in both directions (same judgement as so101's hook).
        ({"ok": True, "contact": True}, None, False, True),
        # Explicit result evidence.
        ({"grasp_confirmed": True}, None, False, True),
        # Explicit negative evidence beats the hook AND beats a trust declaration.
        ({"contact": False}, True, False, False),
        ({"contact": False}, True, True, False),
        # The hook alone is measured evidence.
        ({}, True, False, True),
        # A truthy string is not evidence.
        ({"grasp_confirmed": "true"}, None, False, None),
    ],
)
def test_grasp_confirmation_requires_evidence_and_respects_contradictions(result, hook, trusted, expected):
    from types import SimpleNamespace

    from jiuwensymbiosis.tools.robot_control_tool import read_grasp_confirmation

    api = SimpleNamespace(trust_command_grasp=trusted)
    if hook is not None:
        api.is_grasp_confirmed = lambda result: hook
    assert read_grasp_confirmation(api, result) is expected
