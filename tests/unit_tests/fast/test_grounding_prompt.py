# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The spatial qualifier is decided once, then shown — not derived twice.

LLM① reads "the table the banana is on" off the task and every step is validated
against what it decided. Leaving that out of LLM②'s prompt made LLM② derive the
same thing again from the same sentence, and the only way it learned its answer
differed was by being rejected — a round trip per disagreement, on a fact nobody
had restated. These tests pin that both tiers are now told.
"""

from __future__ import annotations

import json

import pytest

from jiuwensymbiosis.agent.fast import planner
from jiuwensymbiosis.agent.fast.planner import _format_grounding

_G = {"white table": [{"reference": "banana", "relation": "under"}],
      "carton": [{"reference": "banana", "relation": "beside"},
                 {"reference": "cabinet", "relation": "in"}]}


def test_every_qualifier_of_every_name_is_rendered():
    block = _format_grounding(_G)
    assert "white table：under banana" in block
    assert "carton：beside banana" in block and "carton：in cabinet" in block


def test_the_reading_direction_is_stated_with_it():
    """A list of pairs is ambiguous without saying which way round it reads."""
    assert "目标 关系 参照物" in _format_grounding(_G)


def test_nothing_is_rendered_when_the_task_qualified_nothing():
    assert _format_grounding({}) == "" and _format_grounding(None) == ""
    assert _format_grounding({"carton": []}) == ""


def _capture(monkeypatch, reply):
    seen: list[str] = []

    def fake_chat(system, user, **kwargs):
        seen.append(user)
        return json.dumps(reply)

    monkeypatch.setattr(planner, "_chat", fake_chat)
    return seen


_SEQ = [{"op": "locate", "params": {"object_name": "carton", "reference": "cabinet", "relation": "in"}}]


@pytest.mark.parametrize("tier", ["compile", "compose"])
def test_both_tiers_are_shown_the_qualifiers(monkeypatch, tier):
    from jiuwensymbiosis.api.base import BaseRobotApi
    from jiuwensymbiosis.api.decorators import robot_tool
    from jiuwensymbiosis.tools.robot_control_tool import _build_action_index

    class _Api(BaseRobotApi):
        capability = {"vision.detection"}

        @robot_tool(desc="measure", capability="vision.detection", produces_location=True,
                    planner_visible=True)
        def locate(self, object_name: str, reference: str | None = None, relation: str = "on") -> dict:
            return {"ok": True}

    index = _build_action_index(_Api(None), planner_only=True)
    seen = _capture(monkeypatch, _SEQ)
    grounding = {"carton": [{"reference": "cabinet", "relation": "in"}]}
    common = {"action_index": index, "allowed_ops": index, "api_base": "http://x",
              "model_name": "m", "grounding": grounding, "attempts": 1}
    if tier == "compile":
        planner.compile_sequence("拿柜子里的箱子", skills_md=[{"name": "s", "markdown": "# s"}],
                                 action_vocab=sorted(index), **common)
    else:
        planner.compose_actions("拿柜子里的箱子", **common)
    assert "【任务的空间限定】" in seen[0]
    assert "carton：in cabinet" in seen[0]


def test_the_parser_is_told_which_way_a_qualifier_reads():
    """The direction was never stated, and the one worked example (a thing IN a container)
    cannot go the wrong way round — so nothing taught the case that does."""
    system = planner._PARSER_SYSTEM  # noqa: SLF001 - the prompt is the unit under test
    assert "object_name <relation> reference" in system
    assert "under" in system and "'on' 就成了" in system  # the inversion trap, spelled out
