# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Access — the third part of the planning contract (``api/state.py`` §3).

A door, a drawer, a lid, a lock, a crate stacked on top: one situation, many nouns,
and the framework knows none of them. What it knows is that an action declared it
*clears* what it acts on, that a step *reaches* (it changes what the hand holds),
and that the task or the camera said something was in the way. These tests pin the
mechanism on all of those, and — as importantly — pin what it must NOT reject:
sensing a shut thing, driving up to it, and the plan that deals with an obstacle by
picking it up rather than opening it.
"""

from __future__ import annotations

import pytest

from jiuwensymbiosis.agent.fast.sequence import SequenceError, parse_sequence
from jiuwensymbiosis.agent.run import _blocked_access
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation
from jiuwensymbiosis.tools.robot_control_tool import _build_action_index


class _Env(BaseRobotEnv):
    capabilities = frozenset({"vision.detection", "grasp.parallel", "motion.base"})
    name = "fake"

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_observation(self) -> RobotObservation:
        return RobotObservation(pose={}, joints=[])


class _Api(BaseRobotApi):
    """A body that can look, drive, grasp, place — and open things."""

    capability = {"vision.detection", "grasp.parallel", "motion.base"}

    @robot_tool(desc="measure a thing", capability="vision.detection", produces_location=True,
                planner_visible=True)
    def locate(self, object_name: str, reference: str | None = None, relation: str = "on") -> dict:
        return {"ok": True, "position": [0.0, 0.0, 0.0]}

    @robot_tool(desc="drive up to a thing", capability="motion.base", invalidates_locations=True,
                planner_visible=True)
    def approach(self, object_name: str, reference: str | None = None, relation: str = "on") -> dict:
        return {"ok": True}

    @robot_tool(desc="take hold of it", capability="grasp.parallel", requires=["payload.clear"],
                provides=["payload.held"], planner_visible=True)
    def grasp(self) -> dict:
        return {"ok": True}

    @robot_tool(desc="put it down", capability="grasp.parallel", requires=["payload.held"],
                provides=["payload.clear"], planner_visible=True)
    def place(self) -> dict:
        return {"ok": True}

    @robot_tool(desc="pull it open", capability="motion.base", requires=["payload.clear"],
                opens_access=True, planner_visible=True)
    def pull(self, object_name: str) -> dict:
        return {"ok": True}

    @robot_tool(desc="push it shut", capability="motion.base", requires=["payload.clear"],
                closes_access=True, planner_visible=True)
    def push(self, object_name: str) -> dict:
        return {"ok": True}


class _NoOpenerApi(_Api):
    """The same body minus any way to open something — a plain arm (piper / so101)."""

    pull = None  # type: ignore[assignment]
    push = None  # type: ignore[assignment]


@pytest.fixture
def index():
    return _build_action_index(_Api(_Env()), planner_only=True)


@pytest.fixture
def index_no_opener():
    idx = _build_action_index(_Api(_Env()), planner_only=True)
    return {k: v for k, v in idx.items() if k not in ("pull", "push")}


def _parse(seq, index, **kw):
    return parse_sequence(seq, allowed_ops=index, initial_state=["payload.clear"], **kw)


# --------------------------------------------------------------------------- #
# The plan's own testimony — no world knowledge needed
# --------------------------------------------------------------------------- #
def test_reaching_into_something_this_plan_only_opens_later_is_rejected(index):
    with pytest.raises(SequenceError, match="only clears"):
        _parse([
            {"op": "approach", "params": {"object_name": "carton", "reference": "cabinet", "relation": "in"}},
            {"op": "grasp"},
            {"op": "pull", "params": {"object_name": "cabinet"}},
        ], index)


def test_placing_at_something_this_plan_only_opens_later_is_rejected(index):
    """The C6 shape: put a box *into* the cabinet, then open the cabinet."""
    with pytest.raises(SequenceError, match="only clears"):
        _parse([
            {"op": "approach", "params": {"object_name": "crate"}},
            {"op": "grasp"},
            {"op": "approach", "params": {"object_name": "cabinet"}},
            {"op": "place"},
            {"op": "pull", "params": {"object_name": "cabinet"}},
        ], index)


def test_opening_first_is_accepted(index):
    steps = _parse([
        {"op": "pull", "params": {"object_name": "cabinet"}},
        {"op": "approach", "params": {"object_name": "carton", "reference": "cabinet", "relation": "in"}},
        {"op": "grasp"},
    ], index)
    assert [s.op for s in steps] == ["pull", "approach", "grasp"]


def test_a_barrier_named_at_a_different_grain_still_matches(index):
    """``pull("cabinet door")`` is what clears ``"cabinet"`` — nobody says it twice the same way."""
    with pytest.raises(SequenceError, match="only clears"):
        _parse([
            {"op": "approach", "params": {"object_name": "carton", "reference": "cabinet", "relation": "in"}},
            {"op": "grasp"},
            {"op": "pull", "params": {"object_name": "cabinet door"}},
        ], index)


def test_reaching_through_something_the_plan_shut_again_is_rejected(index):
    with pytest.raises(SequenceError, match="closed again"):
        _parse([
            {"op": "pull", "params": {"object_name": "cabinet"}},
            {"op": "push", "params": {"object_name": "cabinet"}},
            {"op": "approach", "params": {"object_name": "carton", "reference": "cabinet", "relation": "in"}},
            {"op": "grasp"},
        ], index)


def test_closing_up_after_the_work_is_done_is_accepted(index):
    steps = _parse([
        {"op": "pull", "params": {"object_name": "cabinet"}},
        {"op": "approach", "params": {"object_name": "carton", "reference": "cabinet", "relation": "in"}},
        {"op": "grasp"},
        {"op": "approach", "params": {"object_name": "table"}},
        {"op": "place"},
        {"op": "push", "params": {"object_name": "cabinet"}},
    ], index)
    assert [s.op for s in steps][-1] == "push"


# --------------------------------------------------------------------------- #
# What perception saw
# --------------------------------------------------------------------------- #
def test_a_barrier_only_the_camera_knows_about_blocks_the_reach(index):
    """Nobody said the crate was on the box — the task never mentions it. The camera did."""
    with pytest.raises(SequenceError, match="in the way"):
        _parse([
            {"op": "approach", "params": {"object_name": "carton"}},
            {"op": "grasp"},
        ], index, blocked_access={"carton": "crate"})


def test_moving_the_obstacle_away_counts_as_clearing_it(index):
    """There is nothing to *open* on a crate sitting on top — you pick it up and move it.

    Requiring an ``opens_access`` action here would reject the only sensible plan.
    """
    steps = _parse([
        {"op": "approach", "params": {"object_name": "crate"}},
        {"op": "grasp"},
        {"op": "approach", "params": {"object_name": "floor"}},
        {"op": "place"},
        {"op": "approach", "params": {"object_name": "carton"}},
        {"op": "grasp"},
    ], index, blocked_access={"carton": "crate"})
    assert len(steps) == 6


def test_looking_at_and_driving_up_to_a_blocked_thing_is_fine(index):
    """Sensing a shut cabinet is how you find out it is shut. Only reaching in is an error."""
    steps = _parse([
        {"op": "locate", "params": {"object_name": "carton", "reference": "cabinet", "relation": "in"}},
        {"op": "approach", "params": {"object_name": "carton", "reference": "cabinet", "relation": "in"}},
    ], index, blocked_access={"carton": "cabinet"})
    assert len(steps) == 2


def test_a_body_that_cannot_open_anything_is_not_blocked_by_the_inference(index_no_opener):
    """"Not seen" is weak evidence. A body with an opener loses one step to a false
    positive; a body without one would lose the only plan it had — so it is not asked."""
    steps = parse_sequence([
        {"op": "approach", "params": {"object_name": "carton", "reference": "cabinet", "relation": "in"}},
        {"op": "grasp"},
    ], allowed_ops=index_no_opener, initial_state=["payload.clear"],
        blocked_access={"carton": "cabinet"})
    assert len(steps) == 2


def test_an_unobserved_barrier_is_never_invented(index):
    """No evidence and no self-contradiction → nothing to reject. Unknown is not false."""
    steps = _parse([
        {"op": "approach", "params": {"object_name": "carton", "reference": "cabinet", "relation": "in"}},
        {"op": "grasp"},
    ], index)
    assert len(steps) == 2


def test_locating_relations_that_enclose_nothing_do_not_block(index):
    """"the box ON the table" / "beside the hat" name no barrier, so nothing is required."""
    for relation in ("on", "beside", "near"):
        steps = _parse([
            {"op": "approach", "params": {"object_name": "carton", "reference": "table", "relation": relation}},
            {"op": "grasp"},
            {"op": "place"},
            {"op": "pull", "params": {"object_name": "table"}},
        ], index)
        assert len(steps) == 4, relation


# --------------------------------------------------------------------------- #
# A loop is not a way out
# --------------------------------------------------------------------------- #
def _loop(body):
    return {"loop": {"detect": {"op": "locate", "params": {"object_name": "carton"}},
                     "bind": "target", "body": body}}


def test_wrapping_the_reach_in_a_loop_does_not_open_the_barrier_any_earlier(index):
    """A loop body is validated on its own — so access state has to cross the boundary.

    Left alone, this is the escape hatch: put the reach inside a loop, leave the opening
    outside and after it, and neither half sees the contradiction.
    """
    with pytest.raises(SequenceError, match="only clears AFTER this loop"):
        _parse([
            _loop([
                {"op": "approach", "params": {"object_name": "carton", "reference": "cabinet",
                                              "relation": "in"}},
                {"op": "grasp"},
                {"op": "place"},
            ]),
            {"op": "pull", "params": {"object_name": "cabinet"}},
        ], index)


def test_a_loop_after_the_barrier_is_open_is_accepted(index):
    steps = _parse([
        {"op": "pull", "params": {"object_name": "cabinet"}},
        _loop([
            {"op": "approach", "params": {"object_name": "carton", "reference": "cabinet",
                                          "relation": "in"}},
            {"op": "grasp"},
            {"op": "place"},
        ]),
    ], index)
    assert len(steps) == 2


def test_perception_evidence_reaches_inside_a_loop_body(index):
    with pytest.raises(SequenceError, match="in the way"):
        _parse([
            _loop([
                {"op": "approach", "params": {"object_name": "carton"}},
                {"op": "grasp"},
                {"op": "place"},
            ]),
        ], index, blocked_access={"carton": "crate"})


# --------------------------------------------------------------------------- #
# Deriving the evidence from the pre-plan look
# --------------------------------------------------------------------------- #
def _box(name, cx, cy, cz, w=200.0, d=200.0, h=200.0):
    return {
        "object": name, "center_mm": [cx, cy, cz],
        "front_x_mm": cx - d / 2, "back_x_mm": cx + d / 2,
        "width_mm": w, "height_mm": h, "top_z_mm": cz + h / 2,
        "distance_mm": (cx**2 + cy**2) ** 0.5, "forward_mm": cx, "n_points": 500,
    }


def test_evidence_from_the_task_qualifier():
    scene = {"count": 0, "objects": [], "references": [_box("cabinet", 1500, 0, 600)], "missing": ["carton"]}
    grounding = {"carton": {"reference": "cabinet", "relation": "in"}}
    assert _blocked_access(scene, grounding) == {"carton": "cabinet"}


def test_no_evidence_when_the_container_was_not_seen_either():
    scene = {"count": 0, "objects": [], "missing": ["carton", "cabinet"]}
    grounding = {"carton": {"reference": "cabinet", "relation": "in"}}
    assert _blocked_access(scene, grounding) == {}


def test_evidence_from_geometry_alone_with_no_qualifier():
    """A crate measured sitting on the box. The task said nothing about it."""
    scene = {"count": 2, "objects": [_box("carton", 800, 0, 300), _box("crate", 800, 0, 500)]}
    assert _blocked_access(scene, {}) == {"carton": "crate"}


def test_things_standing_side_by_side_block_nothing():
    scene = {"count": 2, "objects": [_box("carton", 800, 0, 300), _box("crate", 800, 400, 300)]}
    assert _blocked_access(scene, {}) == {}
