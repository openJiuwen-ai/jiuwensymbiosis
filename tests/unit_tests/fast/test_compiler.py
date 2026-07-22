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

_GOOD_TRACK_GRASP_SEQUENCE = [
    {"op": "home"},
    {"op": "track_grasp", "params": {"object_name": "banana", "approach_mm": 40}, "bind": "banana"},
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
        "special_ops": frozenset({"track_detect"}),
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


def test_compile_uses_runtime_special_ops(monkeypatch):
    cap = _patch_chat(monkeypatch, json.dumps(_GOOD_TRACK_GRASP_SEQUENCE))
    out = _compile(special_ops=frozenset({"track_grasp"}))
    assert [s["op"] for s in out] == ["home", "track_grasp", "close_gripper"]
    assert "track_grasp" in cap["user"]
    assert "track_detect" not in cap["user"].split("【特殊动作】：", 1)[1].split("\n", 1)[0]
    # The generic planner advertises availability only. Skill-specific workflow
    # rewrites belong to SKILL.md, not Python conditionals in planner.py.
    assert "必须将标准 workflow" not in cap["user"]
    assert "absolute approach+descend" not in cap["user"]


def test_compile_takes_special_op_policy_from_skill_markdown(monkeypatch):
    skill_rule = "若 track_grasp 可用，用它替换本 skill 的检测和下降步骤。"
    cap = _patch_chat(monkeypatch, json.dumps(_GOOD_TRACK_GRASP_SEQUENCE))
    _compile(
        skills_md=[{"name": "custom_pick", "markdown": f"# custom_pick\n{skill_rule}"}],
        special_ops=frozenset({"track_grasp"}),
    )
    assert skill_rule in cap["user"]


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


def test_compile_retries_on_chat_timeout_then_recovers(monkeypatch):
    # A transient LLM/HTTP failure (e.g. read timeout) makes _chat raise; the
    # compiler must count it as one attempt and retry, not abort the whole compile.
    calls = {"n": 0}

    def fake_chat(system, user, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("planner LLM call failed after 1 attempts: The read operation timed out")
        return json.dumps(_GOOD_SEQUENCE)

    monkeypatch.setattr(planner, "_chat", fake_chat)
    out = _compile(attempts=3)
    assert calls["n"] == 2
    assert [s["op"] for s in out] == ["home", "track_detect", "goto_xyzr", "close_gripper"]


def test_compile_raises_after_all_chat_timeouts(monkeypatch):
    def always_timeout(system, user, **kwargs):
        raise RuntimeError("The read operation timed out")

    monkeypatch.setattr(planner, "_chat", always_timeout)
    with pytest.raises(RuntimeError, match="no valid sequence"):
        _compile(attempts=2)


def test_compile_feeds_validation_error_back_and_recovers(monkeypatch):
    # First reply drifts: object_name 'black box' but no bind matching the later
    # 'black_box.*' reference (the production bug). The compiler must reject it,
    # feed the error into the next prompt, and accept the corrected retry.
    bad = json.dumps(
        [
            {"op": "track_detect", "params": {"object_name": "black box"}, "bind": "pick"},
            {"op": "goto_xyzr", "params": {"x": "black_box.position[0]"}},
        ]
    )
    good = json.dumps(
        [
            {"op": "track_detect", "params": {"object_name": "black box"}, "bind": "black_box"},
            {"op": "goto_xyzr", "params": {"x": "black_box.position[0]"}},
        ]
    )
    prompts: list[str] = []

    def fake_chat(system, user, **kwargs):
        prompts.append(user)
        return bad if len(prompts) == 1 else good

    monkeypatch.setattr(planner, "_chat", fake_chat)
    out = _compile(attempts=3)
    assert out[0]["bind"] == "black_box"
    # the retry prompt carries the corrective feedback, not just a re-sample
    assert "unbound" in prompts[1] or "bind" in prompts[1]
