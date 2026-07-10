# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for trace_feedback.report — see design §7.2."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from jiuwensymbiosis.trace_feedback.analysis import (
    FailureCluster,
    FailureEvidence,
    FailureSignature,
    TraceCorpus,
    TraceRecord,
)
from jiuwensymbiosis.trace_feedback.report import (
    render_clusters_json,
    render_failure_report,
)


def _sig(**over) -> FailureSignature:
    base = FailureSignature(
        skill_hint=None,
        tool_name="goto_xyzr",
        rail_name="SafetyRail",
        kind="reject",
        reason_norm="z=<num> below z_floor=<num>",
        param_bucket=(("z", "neg/10-100"),),
    )
    return dataclasses.replace(base, **over) if over else base


def _ev(step: int = 1, *, error: str = "boom", cid: str = "conv-1") -> FailureEvidence:
    return FailureEvidence(
        trace_path=Path("/tmp/t.json"),
        conversation_id=cid,
        step=step,
        tool_name="goto_xyzr",
        input_params={},
        error=error,
        output_summary="",
        rail_events=[],
        log_events=[],
        frame_path=None,
        before_context=[],
        after_context=[],
    )


def _cluster(**over) -> FailureCluster:
    base = FailureCluster(
        signature=_sig(),
        count=3,
        examples=[_ev(1), _ev(2)],
        affected_conversations={"c1", "c2"},
    )
    return dataclasses.replace(base, **over) if over else base


class TestRenderClustersJson:
    def test_round_trips_to_dict(self):
        s = render_clusters_json([_cluster()])
        d = json.loads(s)
        assert "clusters" in d
        c = d["clusters"][0]
        assert c["count"] == 3
        assert c["signature"]["tool_name"] == "goto_xyzr"
        assert c["signature"]["param_bucket"] == [["z", "neg/10-100"]]
        assert c["affected_conversations"] == ["c1", "c2"]

    def test_empty_clusters(self):
        s = render_clusters_json([])
        assert json.loads(s) == {"clusters": []}


class TestRenderFailureReport:
    def test_contains_cluster_id_count_trace_step(self):
        md = render_failure_report([_cluster()])
        assert "## Cluster 1" in md
        assert "count: **3**" in md
        assert "t.json:step 1" in md
        assert "goto_xyzr" in md

    def test_empty_clusters_renders_overview(self):
        md = render_failure_report([])
        assert "# Trace Failure Report" in md
        assert "clusters: 0" in md
        assert "no recurring failure clusters" in md

    def test_corpus_trace_count_in_overview(self):
        corpus = TraceCorpus(root=Path("/tmp"), traces=[TraceRecord(Path("/tmp/a.json"), {})])
        md = render_failure_report([_cluster()], corpus=corpus)
        assert "traces analyzed: 1" in md

    def test_rail_and_kind_in_header(self):
        md = render_failure_report([_cluster()])
        assert "goto_xyzr / SafetyRail / reject" in md

    def test_tool_failure_when_no_rail(self):
        c = _cluster(signature=_sig(rail_name=None, kind=None))
        md = render_failure_report([c])
        assert "tool failure" in md
