# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for the C1 fast-path generic action-sequence runner.

Detection here uses the direct ``get_grasp_info_simple`` op (bound via ``bind``)
rather than the threaded ``track_detect``, so the core execution loop — param
resolution, detection binding, gripper-occlusion bookkeeping, failure retreat —
is tested deterministically. ``track_detect`` end-to-end (servo threads) is
covered by the mock smoke script.

A custom ``action_index`` is passed so no ``@robot_tool`` plumbing is needed; the
runner is task-agnostic, so the ops are just whatever the index provides.
"""

from __future__ import annotations

import types

from jiuwensymbiosis.agent.fast.runner import run_sequence
from jiuwensymbiosis.agent.fast.sequence import parse_sequence


class _FakeApi:
    """Records arm calls; returns canned detections."""

    def __init__(self, objects, fail_goto_at=None):
        self.calls = []
        self.objects = objects
        self.fail_goto_at = fail_goto_at  # raise on the Nth goto (1-based) if set
        self._n_goto = 0

    def home(self):
        self.calls.append(("home",))

    def goto_xyzr(self, x, y, z, r=None):
        self._n_goto += 1
        if self.fail_goto_at is not None and self._n_goto == self.fail_goto_at:
            raise RuntimeError("EXCEEDS_LIMIT")
        self.calls.append(("goto", round(x, 1), round(y, 1), round(z, 1)))

    def open_gripper(self):
        self.calls.append(("open",))
        return {"ok": True}

    def close_gripper(self):
        self.calls.append(("close",))
        return {"ok": True}

    def get_grasp_info_simple(self, object_name):
        return self.objects.get(object_name, {"ok": False, "reason": "not_found"})


def _session(api):
    return types.SimpleNamespace(api=api, env=None)


def _index(api):
    return {
        "home": api.home,
        "goto_xyzr": api.goto_xyzr,
        "open_gripper": api.open_gripper,
        "close_gripper": api.close_gripper,
        "get_grasp_info_simple": api.get_grasp_info_simple,
    }


_GRASP_OBJ = {"box": {"ok": True, "position": [250.0, 90.0, 70.0], "grasp_z": 50.0, "place_z": 80.0, "score": 0.9}}


def test_runner_executes_grasp_like_sequence_descends_to_grasp_z():
    api = _FakeApi(_GRASP_OBJ)
    raw = [
        {"op": "home"},
        {"op": "open_gripper"},
        {"op": "get_grasp_info_simple", "params": {"object_name": "box"}, "bind": "b"},
        {"op": "goto_xyzr", "params": {"x": "b.x", "y": "b.y", "z": "b.grasp_z"}},  # direct, no offset
        {"op": "close_gripper"},
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)))
    res = run_sequence(_session(api), steps, action_index=_index(api))

    assert res["ok"] is True and res["steps_done"] == 5
    assert ("goto", 250.0, 90.0, 50.0) in api.calls  # straight to grasp_z=50, no approach/lift
    assert api.calls.count(("close",)) == 1


def test_runner_literal_offset_still_resolves():
    # No named constants exist, but a literal numeric offset in an expression
    # still evaluates — so a skill that DOES want a small clearance can write one.
    api = _FakeApi(_GRASP_OBJ)
    raw = [
        {"op": "get_grasp_info_simple", "params": {"object_name": "box"}, "bind": "b"},
        {"op": "goto_xyzr", "params": {"x": "b.x", "y": "b.y", "z": "b.grasp_z + 30"}},
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)))
    res = run_sequence(_session(api), steps, action_index=_index(api))
    assert res["ok"] is True
    assert ("goto", 250.0, 90.0, 80.0) in api.calls  # 50 + 30 literal


def test_runner_is_task_agnostic_position_only():
    # A detection with NO grasp_z/place_z — a generic "go to the object" task.
    api = _FakeApi({"thing": {"ok": True, "position": [100.0, 0.0, 30.0], "score": 0.8}})
    raw = [
        {"op": "get_grasp_info_simple", "params": {"object_name": "thing"}, "bind": "t"},
        {"op": "goto_xyzr", "params": {"x": "t.position[0]", "y": "t.position[1]", "z": "t.z"}},
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)))
    res = run_sequence(_session(api), steps, action_index=_index(api))
    assert res["ok"] is True
    assert ("goto", 100.0, 0.0, 30.0) in api.calls  # straight to detected z


def test_runner_stops_and_retreats_on_failure():
    api = _FakeApi(_GRASP_OBJ, fail_goto_at=1)  # first goto raises
    raw = [
        {"op": "get_grasp_info_simple", "params": {"object_name": "box"}, "bind": "b"},
        {"op": "goto_xyzr", "params": {"x": "b.x", "y": "b.y", "z": "b.grasp_z"}},
        {"op": "close_gripper"},  # must NOT run after the failure
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)))
    res = run_sequence(_session(api), steps, action_index=_index(api))

    assert res["ok"] is False
    assert res["steps"][-1]["op"] == "goto_xyzr" and not res["steps"][-1]["ok"]
    assert "EXCEEDS_LIMIT" in res["steps"][-1]["reason"]
    assert ("close",) not in api.calls  # stopped before close
    assert ("home",) in api.calls  # best-effort safe retreat ran


def test_runner_reports_unknown_op_on_robot():
    api = _FakeApi(_GRASP_OBJ)
    # 'wave' is allowed by schema (vocab) but not in the runtime action_index.
    steps = parse_sequence([{"op": "wave"}], allowed_ops={"wave"})
    res = run_sequence(_session(api), steps, action_index=_index(api))
    assert res["ok"] is False and "not available" in res["steps"][-1]["reason"]


def test_runner_missing_detection_fails_cleanly():
    api = _FakeApi(_GRASP_OBJ)
    raw = [
        {"op": "get_grasp_info_simple", "params": {"object_name": "ghost"}, "bind": "g"},
        {"op": "goto_xyzr", "params": {"x": "g.x", "y": "g.y", "z": "g.z"}},
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)))
    res = run_sequence(_session(api), steps, action_index=_index(api))
    # detection returns ok=False → not bound → the goto referencing g.x fails clean
    assert res["ok"] is False
