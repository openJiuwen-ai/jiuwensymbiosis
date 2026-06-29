# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for the C1 sequence compiler (planner.compile_sequence).

The LLM HTTP call (``_chat``) is monkeypatched, so these run offline and assert
the compiler's own behavior: prompt assembly, JSON extraction, schema validation,
and retry/raise — not the model.
"""

from __future__ import annotations

import json

import pytest

from jiuwensymbiosis.agent.fast import planner

_VOCAB = ["home", "goto_xyzr", "open_gripper", "close_gripper"]
_SKILLS_MD = [
    {"name": "visual_pick", "markdown": "# visual_pick\n抓取 workflow ..."},
    {"name": "visual_place", "markdown": "# visual_place\n放置 workflow ..."},
]

_GOOD_SEQUENCE = [
    {"op": "home"},
    {"op": "track_detect", "params": {"object_name": "黑盒子"}, "bind": "pick"},
    {"op": "goto_xyzr", "params": {"x": "pick.x", "y": "pick.y", "z": "pick.grasp_z"}},
    {"op": "close_gripper"},
]


def _patch_chat(monkeypatch, reply):
    captured = {}

    def fake_chat(system, user, **kwargs):
        captured["system"] = system
        captured["user"] = user
        return reply if isinstance(reply, str) else reply.pop(0)

    monkeypatch.setattr(planner, "_chat", fake_chat)
    return captured


def _compile(**overrides):
    kwargs = {
        "query": "把黑盒子抓起来",
        "skills_md": _SKILLS_MD,
        "action_vocab": _VOCAB,
        "allowed_ops": set(_VOCAB),
        "api_base": "http://x",
        "model_name": "m",
    }
    kwargs.update(overrides)
    return planner.compile_sequence(**kwargs)


def test_compile_returns_validated_sequence(monkeypatch):
    _patch_chat(monkeypatch, json.dumps(_GOOD_SEQUENCE))
    out = _compile()
    assert [s["op"] for s in out] == ["home", "track_detect", "goto_xyzr", "close_gripper"]


def test_compile_prompt_includes_skill_md_and_vocab(monkeypatch):
    cap = _patch_chat(monkeypatch, json.dumps(_GOOD_SEQUENCE))
    _compile()
    # full SKILL.md text + vocab + track_detect are all in the prompt
    assert "visual_pick" in cap["user"] and "抓取 workflow" in cap["user"]
    assert "goto_xyzr" in cap["user"] and "track_detect" in cap["user"]


def test_compile_tolerates_code_fence(monkeypatch):
    fenced = "```json\n" + json.dumps(_GOOD_SEQUENCE) + "\n```"
    _patch_chat(monkeypatch, fenced)
    out = _compile()
    assert out[0]["op"] == "home"


def test_compile_retries_on_invalid_then_raises(monkeypatch):
    bad = json.dumps([{"op": "fly_to_moon"}])  # op not in vocab → SequenceError
    _patch_chat(monkeypatch, bad)
    with pytest.raises(RuntimeError, match="no valid sequence"):
        _compile(attempts=2)


def test_compile_recovers_on_second_attempt(monkeypatch):
    replies = [json.dumps([{"op": "nonsense"}]), json.dumps(_GOOD_SEQUENCE)]
    _patch_chat(monkeypatch, replies)
    out = _compile(attempts=3)
    assert [s["op"] for s in out] == ["home", "track_detect", "goto_xyzr", "close_gripper"]


def test_compile_raises_when_no_json(monkeypatch):
    _patch_chat(monkeypatch, "对不起我不知道怎么做")
    with pytest.raises(RuntimeError):
        _compile(attempts=1)
