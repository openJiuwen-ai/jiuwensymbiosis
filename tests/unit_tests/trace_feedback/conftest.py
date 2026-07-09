# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared trace fixtures for trace_feedback tests — see design §7.2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def make_entry(
    step: int,
    *,
    tool_name: str = "goto_xyzr",
    params: dict[str, Any] | None = None,
    success: bool = True,
    error: str | None = None,
    rail_events: list[dict] | None = None,
    frame_path: str | None = None,
) -> dict:
    return {
        "step": step,
        "tool_name": tool_name,
        "input_params": params or {},
        "success": success,
        "error": error,
        "duration_s": 0.1,
        "observation": None,
        "frame_path": frame_path,
        "output_summary": "",
        "rail_events": rail_events or [],
        "log_events": [],
    }


def make_trace(
    entries: list[dict],
    *,
    conversation_id: str = "conv-1",
    query: str = "pick",
) -> dict:
    return {
        "conversation_id": conversation_id,
        "robot_name": "piper",
        "query": query,
        "started_at": 0,
        "entries": entries,
        "trace_log": [],
        "workspace": "/tmp",
        "initial_frame_path": None,
    }


def safety_reject(reason: str, *, tool_name: str = "goto_xyzr") -> dict:
    return {
        "rail_name": "SafetyRail",
        "kind": "reject",
        "detail": {"tool_name": tool_name, "reason": reason},
        "success": False,
    }


def recovery_event(*, home_ok: bool = True, released_ok: bool = True, tool_name: str = "goto_xyzr") -> dict:
    return {
        "rail_name": "RecoveryRail",
        "kind": "recover",
        "detail": {"tool_name": tool_name, "home_ok": home_ok, "released_ok": released_ok},
        "success": home_ok,
    }


@pytest.fixture
def write_trace(tmp_path: Path):
    """Factory: write a trace dict to a tmp file, return its Path."""

    def _write(trace: dict, name: str = "t.json") -> Path:
        p = tmp_path / name
        p.write_text(json.dumps(trace), encoding="utf-8")
        return p

    return _write
