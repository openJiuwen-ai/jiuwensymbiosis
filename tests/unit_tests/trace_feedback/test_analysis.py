# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for trace_feedback.analysis — see design §7.2."""

from __future__ import annotations

import json

from jiuwensymbiosis.trace_feedback import (
    build_failure_signature,
    cluster_failures,
    extract_failure_evidence,
    load_trace_corpus,
)
from tests.unit_tests.trace_feedback.conftest import (
    make_entry,
    make_trace,
    recovery_event,
    safety_reject,
)


# --------------------------------------------------------------------------- load
class TestLoadCorpus:
    def test_loads_valid_traces(self, write_trace):
        p = write_trace(make_trace([make_entry(1)]), "a.json")
        corpus = load_trace_corpus([p])
        assert len(corpus.traces) == 1
        assert corpus.traces[0].path == p
        assert corpus.traces[0].data["conversation_id"] == "conv-1"

    def test_bad_json_skipped_not_raised(self, tmp_path, write_trace):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        good = write_trace(make_trace([make_entry(1)]), "good.json")
        corpus = load_trace_corpus([bad, good])
        assert len(corpus.traces) == 1
        assert corpus.traces[0].path == good

    def test_non_utf8_file_skipped_not_raised(self, tmp_path, write_trace):
        # A non-UTF-8 / binary .json must not crash the batch (UnicodeDecodeError
        # is a ValueError, not OSError/JSONDecodeError) — skip + warning, per contract.
        bad = tmp_path / "binary.json"
        bad.write_bytes(b"\xff\xfe\x00{bad utf-8}")
        good = write_trace(make_trace([make_entry(1)]), "good.json")
        corpus = load_trace_corpus([bad, good])
        assert len(corpus.traces) == 1
        assert corpus.traces[0].path == good

    def test_non_dict_skipped(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        corpus = load_trace_corpus([p])
        assert corpus.traces == []


# ------------------------------------------------------------------------ extract
class TestExtractEvidence:
    def test_failed_step_becomes_evidence(self, write_trace):
        p = write_trace(
            make_trace(
                [make_entry(1, success=False, error="boom", rail_events=[safety_reject("z=-5 below z_floor=10")])]
            )
        )
        corpus = load_trace_corpus([p])
        evs = extract_failure_evidence(corpus)
        assert len(evs) == 1
        ev = evs[0]
        assert ev.step == 1
        assert ev.tool_name == "goto_xyzr"
        assert ev.error == "boom"
        assert ev.trace_path == p
        assert ev.conversation_id == "conv-1"

    def test_success_true_with_error_also_failed(self, write_trace):
        # Defensive: TraceRail backfills success=False, but if a fixture has
        # success=True yet error set, treat as failed.
        p = write_trace(make_trace([make_entry(1, success=True, error="weird")]))
        evs = extract_failure_evidence(load_trace_corpus([p]))
        assert len(evs) == 1

    def test_missing_success_defaults_true_not_failed(self, write_trace):
        # A stale/hand-written trace without `success` must not be misread as failed.
        trace = make_trace([{"step": 1, "tool_name": "goto_xyzr", "input_params": {}}])
        p = write_trace(trace)
        evs = extract_failure_evidence(load_trace_corpus([p]))
        assert evs == []

    def test_context_steps_boundaries(self, write_trace):
        entries = [make_entry(i, success=(i != 2)) for i in range(1, 5)]
        # step 2 fails; before=[step1], after=[step3] with context_steps=2 (but
        # only 1 before exists).
        entries[1] = make_entry(2, success=False, error="fail")
        p = write_trace(make_trace(entries))
        evs = extract_failure_evidence(load_trace_corpus([p]), context_steps=2)
        assert len(evs) == 1
        assert [e["step"] for e in evs[0].before_context] == [1]
        assert [e["step"] for e in evs[0].after_context] == [3, 4]

    def test_frame_path_to_path_or_none(self, write_trace):
        p = write_trace(make_trace([make_entry(1, success=False, error="x", frame_path="/tmp/frames/step_001.jpg")]))
        evs = extract_failure_evidence(load_trace_corpus([p]))
        assert evs[0].frame_path is not None
        assert str(evs[0].frame_path) == "/tmp/frames/step_001.jpg"

    def test_null_or_non_numeric_step_does_not_crash(self, write_trace):
        # A hand-written fixture with "step": null or "step": "abc" must not
        # crash the batch (int(None) / int("abc") raise); it falls back to 0.
        entries = [
            {"step": None, "tool_name": "goto_xyzr", "input_params": {}, "success": False, "error": "x"},
            {"step": "abc", "tool_name": "goto_xyzr", "input_params": {}, "success": False, "error": "y"},
        ]
        p = write_trace(make_trace(entries))
        evs = extract_failure_evidence(load_trace_corpus([p]))
        assert [e.step for e in evs] == [0, 0]

    def test_non_list_entries_skipped(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"conversation_id": "c", "entries": "not a list"}), encoding="utf-8")
        evs = extract_failure_evidence(load_trace_corpus([p]))
        assert evs == []


# ---------------------------------------------------------------------- signature
class TestBuildSignature:
    def test_same_reason_different_numbers_collide(self):
        ev = _failed_evidence(error="z=-50 below z_floor=10", rail=[safety_reject("z=-50 below z_floor=10")])
        ev2 = _failed_evidence(error="z=-99 below z_floor=10", rail=[safety_reject("z=-99 below z_floor=10")])
        assert build_failure_signature(ev) == build_failure_signature(ev2)

    def test_different_rail_kind_not_equal(self):
        ev = _failed_evidence(rail=[safety_reject("z=-5 below floor")])
        ev2 = _failed_evidence(rail=[recovery_event()])
        # recovery_event success=True → not the failing rail; signature rail_name=None
        sig1 = build_failure_signature(ev)
        sig2 = build_failure_signature(ev2)
        assert sig1.rail_name == "SafetyRail"
        assert sig2.rail_name is None

    def test_failed_recovery_not_treated_as_root_cause(self):
        # RecoveryRail/recover with home_ok=False is a failed *remedy*, not the
        # original cause. The cause lives in entry.error; rail_name must be None
        # so the cluster isn't mislabeled as a recovery problem.
        ev = _failed_evidence(
            error="AbilityExecutionError: target out of reach",
            rail=[recovery_event(home_ok=False)],
        )
        sig = build_failure_signature(ev)
        assert sig.rail_name is None
        assert sig.reason_norm == "abilityexecutionerror: target out of reach"

    def test_z_magnitude_and_sign_bucketed(self):
        ev = _failed_evidence(params={"z": -50})
        sig = build_failure_signature(ev)
        z_bucket = dict(sig.param_bucket)["z"]
        assert z_bucket == "neg/10-100"

    def test_missing_none_nonfinite_markers(self):
        ev_missing = _failed_evidence(params={"x": None})  # x present but None
        ev_nan = _failed_evidence(params={"x": float("nan")})
        sig_missing = build_failure_signature(ev_missing)
        sig_nan = build_failure_signature(ev_nan)
        assert dict(sig_missing.param_bucket)["x"] == "<none>"
        assert dict(sig_nan.param_bucket)["x"] == "<nan>"

    def test_param_bucket_is_hashable(self):
        ev = _failed_evidence(params={"x": 1, "y": 2, "z": 3, "r": 0, "q": [1, 2, 3]})
        sig = build_failure_signature(ev)
        s = {sig}  # should not raise
        assert sig in s

    def test_q_bucket_by_length_and_non_finite(self):
        ev = _failed_evidence(params={"q": [1, 2, float("inf")]})
        sig = build_failure_signature(ev)
        assert dict(sig.param_bucket)["q"] == "len=3/non-finite"

    def test_long_string_bucket_is_deterministic(self):
        # Builtin hash() is per-process-randomized; the bucket must use sha256
        # so the same long prompt produces the same cluster key across runs.
        # The expected hex is sha256("a very long object…threshold")[:8] — a
        # fixed expectation proves cross-process stability (any process, any
        # PYTHONHASHSEED) more rigorously than a same-process re-computation.
        long_prompt = "a very long object description that exceeds the forty char inline threshold"
        ev = _failed_evidence(params={"object_name": long_prompt})
        sig = build_failure_signature(ev)
        assert dict(sig.param_bucket)["object_name"] == "long/sha=46222d86"

    def test_skill_hint_none_first_pass(self):
        ev = _failed_evidence()
        assert build_failure_signature(ev).skill_hint is None


# ------------------------------------------------------------------------ cluster
class TestClusterFailures:
    def test_three_same_signature_one_cluster(self):
        evs = [
            _failed_evidence(error=f"z=-{i}0 below z_floor=10", rail=[safety_reject(f"z=-{i}0 below z_floor=10")])
            for i in range(1, 4)
        ]
        clusters = cluster_failures(evs, min_size=2)
        assert len(clusters) == 1
        assert clusters[0].count == 3

    def test_min_size_filters_small_groups(self):
        evs = [_failed_evidence(error="z=-10 below z_floor=10", rail=[safety_reject("z=-10 below z_floor=10")])]
        clusters = cluster_failures(evs, min_size=2)
        assert clusters == []

    def test_different_kind_not_misclustered(self):
        ev_safety = _failed_evidence(error="z=-5 below floor", rail=[safety_reject("z=-5 below floor")])
        ev_tool = _failed_evidence(error="ValueError: gripper timeout", rail=[])
        clusters = cluster_failures([ev_safety, ev_safety, ev_tool, ev_tool], min_size=2)
        assert len(clusters) == 2

    def test_examples_capped_at_three(self):
        evs = [_failed_evidence(error="same", rail=[safety_reject("same")]) for _ in range(5)]
        clusters = cluster_failures(evs, min_size=2)
        assert len(clusters[0].examples) == 3

    def test_affected_conversations_deduped(self):
        evs = [
            _failed_evidence(error="same", rail=[safety_reject("same")], cid="c1"),
            _failed_evidence(error="same", rail=[safety_reject("same")], cid="c1"),
            _failed_evidence(error="same", rail=[safety_reject("same")], cid="c2"),
        ]
        clusters = cluster_failures(evs, min_size=2)
        assert clusters[0].affected_conversations == {"c1", "c2"}

    def test_sorted_by_count_descending(self):
        evs_a = [_failed_evidence(error="a", rail=[safety_reject("a")]) for _ in range(5)]
        evs_b = [_failed_evidence(error="b", rail=[safety_reject("b")]) for _ in range(3)]
        clusters = cluster_failures(evs_a + evs_b, min_size=2)
        assert clusters[0].count >= clusters[1].count


def _failed_evidence(
    *,
    error: str = "boom",
    rail: list[dict] | None = None,
    params: dict | None = None,
    cid: str = "conv-1",
):
    from pathlib import Path

    from jiuwensymbiosis.trace_feedback import FailureEvidence

    return FailureEvidence(
        trace_path=Path("/tmp/t.json"),
        conversation_id=cid,
        step=1,
        tool_name="goto_xyzr",
        input_params=params or {},
        error=error,
        output_summary="",
        rail_events=rail or [],
        log_events=[],
        frame_path=None,
        before_context=[],
        after_context=[],
    )
