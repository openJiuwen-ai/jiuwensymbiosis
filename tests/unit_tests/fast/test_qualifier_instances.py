# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""One name, several qualified instances.

"把香蕉旁的箱子放到紫桌上，把柜子里的箱子放到白桌上" is two boxes. Both are called
box; where each one is, is the only thing telling them apart. Keying one qualifier
per name drops one of them — and then the step that handles the dropped one gets
rejected for carrying "the wrong" qualifier, which is how a correct plan fails.
"""

from __future__ import annotations

import json

import pytest

from jiuwensymbiosis.agent.fast import planner
from jiuwensymbiosis.agent.fast.sequence import SequenceError, parse_sequence, qualifiers_for
from jiuwensymbiosis.agent.run import _blocked_access
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation
from jiuwensymbiosis.tools.robot_control_tool import _build_action_index

_TWO = {"carton": [{"reference": "banana", "relation": "beside"},
                   {"reference": "cabinet", "relation": "in"}]}


class _Env(BaseRobotEnv):
    capabilities = frozenset({"vision.detection", "grasp.parallel"})
    name = "fake"

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_observation(self) -> RobotObservation:
        return RobotObservation(pose={}, joints=[])


class _Api(BaseRobotApi):
    capability = {"vision.detection", "grasp.parallel"}

    @robot_tool(desc="measure a thing", capability="vision.detection", produces_location=True,
                planner_visible=True)
    def locate(self, object_name: str, reference: str | None = None, relation: str = "on") -> dict:
        return {"ok": True, "position": [0.0, 0.0, 0.0]}

    @robot_tool(desc="take hold", capability="grasp.parallel", requires=["payload.clear"],
                provides=["payload.held"], planner_visible=True)
    def grasp(self) -> dict:
        return {"ok": True}

    @robot_tool(desc="put down", capability="grasp.parallel", requires=["payload.held"],
                provides=["payload.clear"], planner_visible=True)
    def place(self) -> dict:
        return {"ok": True}


@pytest.fixture
def index():
    return _build_action_index(_Api(_Env()), planner_only=True)


def _step(**params):
    return {"op": "locate", "params": {"object_name": "carton", **params}}


def test_both_qualified_instances_are_kept(index):
    assert qualifiers_for(_TWO, "carton") == (
        {"reference": "banana", "relation": "beside"},
        {"reference": "cabinet", "relation": "in"},
    )


def test_either_qualifier_is_accepted(index):
    """The task said which boxes exist, not which order to take them in."""
    steps = parse_sequence([
        _step(reference="banana", relation="beside"),
        {"op": "grasp"}, {"op": "place"},
        _step(reference="cabinet", relation="in"),
        {"op": "grasp"}, {"op": "place"},
    ], allowed_ops=index, initial_state=["payload.clear"], grounding=_TWO)
    assert len(steps) == 6


def test_a_qualifier_the_task_never_stated_is_still_rejected(index):
    with pytest.raises(SequenceError, match="beside 'banana' / in 'cabinet'"):
        parse_sequence([_step(reference="hat", relation="on")],
                       allowed_ops=index, initial_state=["payload.clear"], grounding=_TWO)


def test_dropping_the_qualifier_entirely_is_still_rejected(index):
    with pytest.raises(SequenceError, match="but it passes neither"):
        parse_sequence([_step()], allowed_ops=index, initial_state=["payload.clear"], grounding=_TWO)


def test_the_single_qualifier_shorthand_still_works(index):
    one = {"carton": {"reference": "cabinet", "relation": "in"}}
    assert qualifiers_for(one, "carton") == ({"reference": "cabinet", "relation": "in"},)
    with pytest.raises(SequenceError):
        parse_sequence([_step(reference="banana", relation="beside")],
                       allowed_ops=index, initial_state=["payload.clear"], grounding=one)


def test_the_parser_keeps_every_qualifier_a_task_states(monkeypatch):
    monkeypatch.setattr(planner, "_chat", lambda *a, **k: json.dumps({
        "targets": ["carton", "carton"],
        "references": ["banana", "cabinet"],
        "grounding": {"carton": [{"reference": "banana", "relation": "beside"},
                                 {"reference": "cabinet", "relation": "in"}]},
        "mode": "multi",
    }))
    out = planner.parse_task("把香蕉旁的箱子和柜子里的箱子都搬走", api_base="x", model_name="m")
    assert out["grounding"]["carton"] == [
        {"reference": "banana", "relation": "beside"},
        {"reference": "cabinet", "relation": "in"},
    ]


def test_the_parser_still_accepts_a_single_dict(monkeypatch):
    monkeypatch.setattr(planner, "_chat", lambda *a, **k: json.dumps({
        "targets": ["apple"], "references": ["drawer"],
        "grounding": {"apple": {"reference": "drawer", "relation": "in"}},
    }))
    out = planner.parse_task("拿抽屉里的苹果", api_base="x", model_name="m")
    assert out["grounding"]["apple"] == [{"reference": "drawer", "relation": "in"}]


def test_blocked_access_reads_every_qualifier():
    """Only the cabinet one is a containment relation, and only that box was unseen."""
    scene = {"count": 0, "objects": [],
             "references": [{"object": "cabinet", "center_mm": [1500, 0, 600], "front_x_mm": 1400,
                             "back_x_mm": 1600, "width_mm": 800, "height_mm": 1200, "top_z_mm": 1200,
                             "n_points": 500}],
             "missing": ["carton"]}
    assert _blocked_access(scene, _TWO) == {"carton": "cabinet"}
