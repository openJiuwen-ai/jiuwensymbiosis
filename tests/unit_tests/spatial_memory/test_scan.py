"""Observation loop contracts: persistence, partial outcomes and safe stopping."""

import json
import threading

import pytest

from jiuwensymbiosis.spatial_memory import SpatialMemory
from jiuwensymbiosis.spatial_memory.completion import assess_completion, observed_scope
from jiuwensymbiosis.spatial_memory.scan import ScanBudget, run_scan
from tests.unit_tests.spatial_memory.test_memory import sample


class Port:
    def __init__(self, *, blocked=False, stopped=True):
        self.frames = 0
        self.moves = 0
        self.blocked = blocked
        self.stopped = stopped

    def capture(self):
        obs, _ = sample(self.frames)
        self.frames += 1
        return obs

    def execute(self, choice, observation):
        self.moves += 1
        return {"ok": not self.blocked, "motion_executed": not self.blocked, "stopped": self.stopped}

    def confirm_stopped(self):
        return self.stopped


class Detector:
    def detect(self, rgb):
        return sample()[1]


def test_frame_budget_commits_final_observation_and_keeps_incomplete(tmp_path):
    port = Port()
    out = tmp_path / "scan"
    result = run_scan(port, Detector(), out, budget=ScanBudget(max_frames=3))
    assert (port.frames, port.moves) == (3, 2)
    assert result["stop_reason"] == "budget_exhausted"
    assert result["committed_frames"] == 3
    assert result["scan_complete"] is None
    assert result["inventory_complete"] is False
    with SpatialMemory(out / "memory.sqlite", readonly=True) as memory:
        assert memory.get_status()["scan_assessment"]["frames"] == 3
    assert len(json.loads((out / "capture/sequence.json").read_text())["frames"]) == 3


def test_admission_failure_is_blocked_not_complete(tmp_path):
    result = run_scan(Port(blocked=True), Detector(), tmp_path / "s", budget=ScanBudget())
    assert result["stop_reason"] == "blocked"
    assert result["executed_actions"] == 0
    assert result["scan_complete"] is None


def test_inference_failure_retains_capture_and_does_not_write_empty_success(tmp_path):
    class Broken:
        def detect(self, rgb):
            raise RuntimeError("detector unavailable")

    port = Port()
    result = run_scan(port, Broken(), tmp_path / "s", budget=ScanBudget())
    assert result["stop_reason"] == "failed"
    assert result["frames"] == 1 and result["committed_frames"] == 0
    assert port.moves == 0
    assert "detector unavailable" in result["error"]


def test_cancelled_before_capture_and_unconfirmed_stop_are_not_success(tmp_path):
    token = threading.Event()
    token.set()
    result = run_scan(Port(), Detector(), tmp_path / "cancel", budget=ScanBudget(), cancelled=token)
    assert result["stop_reason"] == "cancelled" and result["frames"] == 0
    result = run_scan(Port(stopped=False), Detector(), tmp_path / "stop", budget=ScanBudget(max_frames=1))
    assert result["stop_reason"] == "stop_unconfirmed" and not result["stopped"]


def test_unknown_boundary_budget_and_deferred_tasks_prevent_completion():
    closed = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    scope = observed_scope((0, 0), {(0, 0)}, closed)
    kwargs = {
        "reason": "no_candidates",
        "scope": scope,
        "frontier_cells": 0,
        "gains": [0, 0, 0],
        "tasks": [],
        "stopped": True,
    }
    assert assess_completion(**kwargs)["scan_complete"] is True
    assert assess_completion(**{**kwargs, "reason": "budget_exhausted"})["scan_complete"] is None
    assert assess_completion(**{**kwargs, "tasks": [{"state": "deferred"}]})["scan_complete"] is None
    unknown = observed_scope((0, 0), {(0, 0)}, closed - {(1, 0)})
    assert unknown["kind"] == "unknown"
    assert assess_completion(**{**kwargs, "scope": unknown})["scan_complete"] is None
    assert assess_completion(**kwargs)["inventory_complete"] is False


def test_new_evidence_invalidates_prior_scan_assessment(tmp_path):
    with SpatialMemory(tmp_path / "m") as memory:
        memory.ingest(*sample())
        memory.record_scan_report({"stop_reason": "budget_exhausted"})
        assert memory.get_status()["scan_assessment"] is not None
        memory.ingest(*sample(1))
        assert memory.get_status()["scan_assessment"] is None


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_frame_budget(value):
    with pytest.raises(ValueError):
        ScanBudget(max_frames=value)


@pytest.mark.parametrize("refuse_revisit", [False, True])
def test_available_revisit_is_attempted_before_more_frontier_gain(tmp_path, monkeypatch, refuse_revisit):
    from jiuwensymbiosis.spatial_memory.exploration import OnlineMap

    choices = [
        {"kind": "advance", "utility": 10.0, "signature": [0, 0, 0]},
        {"kind": "reobserve", "utility": 0.1, "signature": [0, 1, 0], "memory_task_id": "observed-gap"},
    ]
    monkeypatch.setattr(OnlineMap, "candidates", lambda *_: (choices, {"frontier_cells": 5}))
    attempted = []

    class RecordingPort(Port):
        def execute(self, choice, observation):
            attempted.append(choice["kind"])
            executed = not (refuse_revisit and choice["kind"] == "reobserve")
            return {"ok": executed, "motion_executed": executed, "stopped": True}

    result = run_scan(RecordingPort(), Detector(), tmp_path / "scan", budget=ScanBudget(max_frames=2))
    assert attempted == (["reobserve", "advance"] if refuse_revisit else ["reobserve"])
    assert result["executed_actions"] == 1
