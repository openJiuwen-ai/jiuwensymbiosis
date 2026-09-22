# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Normalize wrapped task results to runtime terminal phase names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

__all__ = ["RunPhase", "normalize_result"]

RunPhase = Literal["succeeded", "failed", "incomplete", "cancelled"]


def normalize_result(wrapped_result: Any) -> RunPhase:
    """Map a run-engine result wrapper to its machine-readable terminal phase.

    An outer failure means execution raised. The fast planner can instead
    complete without raising while reporting an unsuccessful action sequence;
    that is ``incomplete``. A stopped result is a cancellation, and an agent
    error result (for example, exhausted iterations) is incomplete rather than
    successful.
    """
    if not isinstance(wrapped_result, Mapping) or not wrapped_result.get("ok"):
        return "failed"

    payload = wrapped_result.get("result")
    if isinstance(payload, Mapping) and "steps_done" in payload:
        return "succeeded" if payload.get("ok") else "incomplete"

    result_type = payload.get("result_type") if isinstance(payload, Mapping) else ""
    if result_type == "stopped":
        return "cancelled"
    if result_type == "error":
        return "incomplete"
    return "succeeded"
