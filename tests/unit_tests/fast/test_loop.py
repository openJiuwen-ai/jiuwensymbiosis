# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Phase D: perception-terminated multi-object loop (detect-one → process → repeat)."""

from types import SimpleNamespace

from jiuwensymbiosis.agent.fast.runner import run_sequence
from jiuwensymbiosis.agent.fast.sequence import LoopStep, parse_sequence

_ALLOWED = {"locate_for_grasp", "dual_arm_grasp", "dual_arm_place"}


def _loop_seq():
    return [
        {
            "loop": {
                "detect": {"op": "locate_for_grasp", "params": {"object_name": "box"}},
                "bind": "t",
                "body": [{"op": "dual_arm_grasp", "params": {}}, {"op": "dual_arm_place", "params": {}}],
            }
        }
    ]


def test_parse_loop():
    steps = parse_sequence(_loop_seq(), allowed_ops=_ALLOWED)
    assert len(steps) == 1
    loop = steps[0]
    assert isinstance(loop, LoopStep)
    assert loop.detect_op == "locate_for_grasp"
    assert loop.bind == "t"
    assert len(loop.body) == 2


def _executor_for(n_targets, calls):
    def executor(op, params):
        if op == "locate_for_grasp":
            k = calls["detect"]
            calls["detect"] += 1
            if k < n_targets:
                return {"ok": True, "result": {"ok": True, "center_mm": [500.0 + k, 0.0, 300.0]}}
            return {"ok": True, "result": {"ok": False, "reason": "no_detection"}}
        if op == "dual_arm_grasp":
            calls["grasp"] += 1
            return {"ok": True, "result": {"ok": True}}
        if op == "dual_arm_place":
            calls["place"] += 1
            return {"ok": True, "result": {"ok": True}}
        return {"ok": True, "result": {"ok": True}}

    return executor


def _run(n_targets):
    calls = {"detect": 0, "grasp": 0, "place": 0}
    steps = parse_sequence(_loop_seq(), allowed_ops=_ALLOWED)
    session = SimpleNamespace(api=SimpleNamespace(home=lambda: None))
    res = run_sequence(session, steps, executor=_executor_for(n_targets, calls))
    return res, calls


def test_loop_processes_three_targets():
    res, calls = _run(3)
    assert res["ok"]
    assert calls["grasp"] == 3 and calls["place"] == 3  # processed all 3
    assert calls["detect"] == 4  # 3 targets + 1 empty (terminates)


def test_loop_processes_one_target():
    res, calls = _run(1)
    assert res["ok"]
    assert calls["grasp"] == 1 and calls["place"] == 1
    assert calls["detect"] == 2  # 1 + terminating empty


def test_loop_zero_targets_terminates_immediately():
    res, calls = _run(0)
    assert res["ok"]
    assert calls["grasp"] == 0 and calls["place"] == 0
    assert calls["detect"] == 1  # one empty detection → stop
