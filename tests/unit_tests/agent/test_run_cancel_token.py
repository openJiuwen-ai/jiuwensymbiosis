# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""run_robot_task: the public cancel token reaches both the fast path and rails."""

from __future__ import annotations

from types import SimpleNamespace

from jiuwensymbiosis.agent import run as run_module
from jiuwensymbiosis.agent.config import RobotAgentConfig


def test_run_robot_task_exposes_the_token_on_the_session(monkeypatch):
    session = SimpleNamespace(cancel_token=None)
    token = object()
    captured: dict[str, object] = {}

    def fake_run(session_arg, query, config, *, conversation_id, cancel_token):
        captured["token"] = cancel_token
        return {"ok": True}

    monkeypatch.setattr(run_module, "run_fast_task", fake_run)

    result = run_module.run_robot_task(
        session,
        "go home",
        RobotAgentConfig(execution={"mode": "fastagent"}),
        cancel_token=token,
    )

    assert result == {"ok": True}
    assert session.cancel_token is token
    assert captured["token"] is token


def test_run_robot_task_leaves_the_session_untouched_without_a_token(monkeypatch):
    session = SimpleNamespace(cancel_token=None)
    monkeypatch.setattr(run_module, "run_fast_task", lambda *args, **kwargs: None)

    run_module.run_robot_task(
        session,
        "go home",
        RobotAgentConfig(execution={"mode": "fastagent"}),
    )

    assert session.cancel_token is None
