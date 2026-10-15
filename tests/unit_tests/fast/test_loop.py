# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Phase D: perception-terminated multi-object loop (detect-one → process → repeat)."""

from types import SimpleNamespace

import pytest

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.agent.fast.runner import run_sequence
from jiuwensymbiosis.agent.fast.sequence import LoopStep, parse_sequence

_ALLOWED = {"locate_for_grasp", "dual_arm_grasp", "dual_arm_place"}


@pytest.mark.parametrize("outcome", ["exception", "failure", "empty", "success"])
def test_loop_cancellation_wins_over_detection_outcome(outcome):
    token = CancelToken()
    motions = []
    session = SimpleNamespace(
        api=SimpleNamespace(home=lambda: motions.append("home"), open_gripper=lambda: motions.append("release")),
        env=SimpleNamespace(holding_payload=False),
        cancel_token=token,
    )

    def detect(*args):
        token.set()
        if outcome == "exception":
            raise RuntimeError("interrupted detection")
        if outcome == "failure":
            return {"ok": False, "result": {"ok": False, "error_code": "inference_timeout"}}
        if outcome == "empty":
            return {"ok": True, "result": {"ok": False, "reason": "no_detection"}}
        return {"ok": True, "result": {"ok": True, "center_mm": [100.0, 0.0, 300.0]}}

    with pytest.raises(RunCancelled):
        run_sequence(session, parse_sequence(_loop_seq(), allowed_ops=_ALLOWED), executor=detect)
    assert motions == []


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


def test_service_failure_cannot_terminate_as_scene_clear():
    steps = parse_sequence(_loop_seq(), allowed_ops=_ALLOWED)
    session = SimpleNamespace(api=SimpleNamespace(home=lambda: None))
    result = run_sequence(
        session,
        steps,
        executor=lambda op, params: {
            "ok": True,
            "result": {"ok": False, "reason": "detector_unavailable", "error_code": "inference_timeout"},
        },
    )
    assert result["ok"] is False
    assert result["steps"][-1]["error_code"] == "inference_timeout"


def test_direct_executor_normal_empty_detection_terminates():
    steps = parse_sequence(_loop_seq(), allowed_ops=_ALLOWED)
    session = SimpleNamespace(api=SimpleNamespace(home=lambda: None))
    # Both real executor implementations mirror result.ok at the outer level.
    result = run_sequence(
        session, steps, executor=lambda op, params: {"ok": False, "result": {"ok": False, "reason": "no_detection"}}
    )
    assert result["ok"]
    assert result["steps"][-1]["result"]["terminated"] == "no_more_target"
