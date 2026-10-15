# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for wrapped run-result normalization."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.runtime.results import normalize_result


@pytest.mark.parametrize(
    ("result", "phase"),
    [
        ({"ok": False, "error": "connection failed"}, "failed"),
        ({"error": "missing success marker"}, "failed"),
        ({"ok": True, "result": {"result_type": "answer", "output": "done"}}, "succeeded"),
        ({"ok": True, "result": {"result_type": "answer", "output": None}}, "succeeded"),
        ({"ok": True, "result": {"result_type": "stopped", "output": "stopped"}}, "cancelled"),
        ({"ok": True, "result": {"result_type": "error", "output": "max iterations"}}, "incomplete"),
        ({"ok": True, "result": {"ok": True, "steps_done": 2, "steps": [{"ok": True}]}}, "succeeded"),
        (
            {
                "ok": True,
                "result": {
                    "ok": False,
                    "steps_done": 1,
                    "steps": [{"op": "locate_for_grasp", "ok": False, "reason": "no depth"}],
                },
            },
            "incomplete",
        ),
        ({"ok": True, "result": {"ok": False, "steps_done": 0, "steps": []}}, "incomplete"),
        ({"ok": True, "result": "plain output"}, "succeeded"),
        (None, "failed"),
    ],
)
def test_normalize_result_matches_workbench_status_cases(result, phase):
    assert normalize_result(result) == phase
