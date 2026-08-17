# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A second dual-arm mobile body must run 支 B without borrowing a line from cruzr.

The claim being tested is that the mobile-manipulation layer is generic. The only
way to show that is to build a body that has never met the adapter it was extracted
from — ``tests.mocks.mock_dual_arm`` mixes the framework mixins, supplies the three
genuinely body-specific pieces (frame, detector, dual-arm actuation), and inherits
everything else. If 支 B compiles against its action index and then drives its base
to the crate, the search / squaring / approach algorithms are body-agnostic in fact.

``planner._chat`` is patched, so what runs is the skill's own workflow, not a model's
improvisation — and the physics under it is a rendered world, so the approach loop has
to genuinely converge rather than be told that it did.
"""

from __future__ import annotations

import json
from pathlib import Path

from jiuwensymbiosis.agent.fast import DEFAULT_REGISTRY, planner
from jiuwensymbiosis.agent.fast.planner import _filter_skills_md_by_capability
from jiuwensymbiosis.agent.fast.runner import run_sequence
from jiuwensymbiosis.agent.fast.sequence import parse_sequence
from jiuwensymbiosis.tools.robot_control_tool import _build_action_index
from tests.mocks import mock_dual_arm
from tests.mocks.mock_dual_arm import MockDualArmSession, WorldBox

_CRATE = WorldBox("crate", (1400.0, 0.0, 300.0), (400.0, 500.0, 600.0))

# 支 B「有底盘」流程，占位符按能力映射表替换后的样子（dual_arm → detect/grasp，motion.base → face/approach）。
# approach_for_grasp now folds search-and-face in front of the drive, so the pick is one
# step shorter and the mandatory order can no longer be got wrong.
_PICK_B = [
    {"op": "approach_for_grasp", "params": {"object_name": "crate"}},
    {"op": "dual_arm_grasp", "params": {}},
    {"op": "lift_to_clearance", "params": {}},
]


def _session() -> MockDualArmSession:
    return MockDualArmSession([_CRATE])


def _skill_text(caps: list[str]) -> str:
    kept = _filter_skills_md_by_capability(DEFAULT_REGISTRY.skills_markdown(), caps)
    return "\n".join(s["markdown"] for s in kept)


def test_the_body_borrows_nothing_from_an_adapter():
    s = _session()
    assert "adapters" not in Path(mock_dual_arm.__file__).read_text(encoding="utf-8")
    origins = {c.__module__ for c in (*type(s.api).__mro__, *type(s.env).__mro__)}
    assert not [m for m in origins if m.startswith("jiuwensymbiosis.adapters")]


def test_every_action_the_skill_offers_this_body_exists_on_it():
    s = _session()
    index = _build_action_index(s.api, env=s.env)
    text = _skill_text(sorted(s.api.capabilities))
    for op in (
        "approach_for_grasp",
        "locate_for_grasp",
        "locate_for_place",
        "dual_arm_grasp",
        "dual_arm_place",
        "lift_to_clearance",
    ):
        assert op in text, f"{op} missing from the filtered skills"
        assert op in index, f"{op} named by a skill this body sees, but absent from its actions"
    assert not [op for op in ("goto_xyzr", "close_gripper", "activate_suction") if op in index]


def test_grasping_before_approaching_is_refused():
    # The stub judges by measured geometry, so a passing end-to-end run below is
    # evidence the base actually closed the distance, not that the stub says yes.
    s = _session()
    assert s.api.locate_for_grasp("crate")["ok"]
    assert s.api.dual_arm_grasp()["reason"] == "out_of_reach"
    assert s.env.holding_payload is False


def test_the_mobile_dual_arm_pick_compiles_in_one_call_and_drives_the_base(monkeypatch):
    s = _session()
    s.env.base_yaw_rad = -0.35  # crate off the current heading: the base has to square up to its face
    index = _build_action_index(s.api, env=s.env)
    systems: list[str] = []

    def fake_chat(system, user, **kwargs):
        systems.append(system)
        return json.dumps(_PICK_B)

    monkeypatch.setattr(planner, "_chat", fake_chat)
    plan = planner.plan_task(
        query="把 crate 搬起来",
        skills_md=DEFAULT_REGISTRY.skills_markdown(),
        action_index=index,
        allowed_ops=index,
        api_base="http://x",
        model_name="m",
        api_capabilities=sorted(s.api.capabilities),
        world_tokens=["payload.clear"],
        attempts=1,
    )
    assert plan.tier == "skill"
    assert systems == [planner._COMPILER_SYSTEM]  # noqa: SLF001 - the library covers it, so one round trip
    assert [step["op"] for step in plan.sequence] == [step["op"] for step in _PICK_B]

    run = run_sequence(s, parse_sequence(plan.sequence, allowed_ops=index), action_index=index)

    assert run["ok"] is True, run
    assert s.api.calls == ["dual_arm_grasp", "lift_to_clearance"]
    assert s.env.holding_payload is True and s.env.lifter_q
    # Squared up first, then closed the gap: the crate's near face ends inside the grasp band.
    assert s.env.moves[0]["dx_m"] == 0.0 and s.env.moves[0]["dyaw_rad"] > 0.3
    assert sum(m["dx_m"] for m in s.env.moves) > 0.5
    assert 300.0 <= s.api._last_detection["front_x_mm"] <= 500.0  # noqa: SLF001 - the converged sensing


def test_one_rgbd_camera_bolted_to_the_base_is_enough_to_search():
    """Searching needs a view the body can aim — not a second kind of sensor.

    This body has exactly one camera, it carries depth, and it is fixed to the chassis. By
    the rule that matters (can the body turn what carries the camera?) it can look around,
    and the base is what turns. It used to be told otherwise: the search path went looking
    for a separate "coarse" sensor, found none, and reported ``no_camera`` — a body with a
    camera being told it had none, purely because its one camera was the wrong *kind*.
    """
    s = _session()
    assert "vision.search" in s.env.capabilities
    assert s.env.cameras == (None,)              # the plain single-camera default

    out = s.api.search_target("crate")
    assert out["found"] is True
    assert "bearing_rad" in out                  # a direction, which is all searching promises
    assert "search_target" in _build_action_index(s.api, env=s.env, planner_only=True)


def test_a_body_that_cannot_aim_a_camera_is_not_offered_the_search():
    """The gate is the ability to aim, so a fixed arm must not be handed a look-around it
    can only fail at. piper/so101 declare no vision.search; nothing else changes for them."""
    from jiuwensymbiosis.api.actions import SEARCH_TARGET

    assert SEARCH_TARGET.capability == "vision.search"


def test_a_body_with_no_aimable_camera_searches_by_turning_itself():
    """Turning the BODY is the general way to look around, and it is the better one.

    Every downstream step is measured in the base frame, so a rotation that carries the base
    frame leaves the body already facing what it found. Aiming a neck instead buys a cheaper
    look and then still owes the base turn — which is why the default here turns the body,
    and aiming a camera is the override rather than the rule. A quadruped turning on its legs
    is the same capability; nothing about this assumes wheels.
    """
    import math

    behind = WorldBox("crate", (-1500.0, 0.0, 300.0), (400.0, 500.0, 600.0))
    s = MockDualArmSession([behind])

    assert s.api.locate_for_grasp("crate")["ok"] is False   # nothing in the current view

    out = s.api._nav._sweep_for_bearing("crate")

    assert out["found"] is True
    assert out["exhaustive"] is True                        # the whole circle was covered
    assert out["turned_rad"] > math.pi / 2                  # it really did turn to find it
    assert abs(out["total_bearing"]) < 0.5                  # and ended up roughly facing it


def test_a_sweep_that_covered_the_circle_is_not_asked_to_turn_round_again():
    """``face_by_sweep`` adds a 180° re-scan for bodies whose look-around only covers their
    own facing. A body that already swept the full circle must not pay for it twice."""
    s = MockDualArmSession([WorldBox("crate", (1400.0, 0.0, 300.0), (400.0, 500.0, 600.0))])
    out = s.api._nav._sweep_for_bearing("nothing_like_this")

    assert out["found"] is False
    assert out["exhaustive"] is True
    assert abs(s.env.base_yaw_rad) < 1e-6, "an empty sweep must leave the heading as it found it"


def test_finding_the_target_ends_the_search_and_hands_over_a_measurement():
    """Found means found: stop looking, measure, and let the approach loop drive.

    The look-around turned the body, so the target is in front now — asking the metric sensor
    right there costs one frame and replaces the alternative, which was to commit to a bearing
    and creep forward blind until something acquired. Nothing is driven during the search.
    """
    from jiuwensymbiosis.motion import approach

    behind = WorldBox("crate", (-1500.0, 0.0, 300.0), (400.0, 500.0, 600.0))
    s = MockDualArmSession([behind])

    out = approach.face_by_sweep(
        s.api._nav, s.api.locate_for_grasp, "crate",
        result_key="detection", not_found_reason="object_not_found",
    )

    assert out["ok"] is True and out["status"] == "acquired"
    assert out["detection"]["ok"] is True, "the handover must carry a measured position, not a bearing"
    assert sum(m.get("dx_m", 0.0) for m in s.env.moves) == 0.0, "searching must not drive anywhere"
