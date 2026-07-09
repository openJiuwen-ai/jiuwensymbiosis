# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for scripts/analyze_traces.py CLI — see design §4.3 / §7.2."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_traces import run
from tests.unit_tests.trace_feedback.conftest import make_entry, make_trace, safety_reject


def _write_failed_trace(path: Path, *, z: int, cid: str) -> None:
    trace = make_trace(
        [
            make_entry(
                1,
                params={"x": 150, "y": 0, "z": z, "r": 0},
                success=False,
                error=f"SafetyRail: z={z} below z_floor=10",
                rail_events=[safety_reject(f"z={z} below z_floor=10")],
            )
        ],
        conversation_id=cid,
    )
    path.write_text(json.dumps(trace), encoding="utf-8")


class TestRunP2:
    def test_three_same_reject_produces_cluster(self, tmp_path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        for i, z in enumerate([-50, -99, -20]):
            _write_failed_trace(traces_dir / f"t{i}.json", z=z, cid=f"c{i}")
        out_dir = tmp_path / "out"

        code = run(sorted(traces_dir.glob("*.json")), out_dir=out_dir, min_cluster_size=3)

        assert code == 0
        assert (out_dir / "failure_clusters.json").is_file()
        report = (out_dir / "failure_report.md").read_text(encoding="utf-8")
        assert "## Cluster 1" in report
        assert "goto_xyzr" in report
        clusters = json.loads((out_dir / "failure_clusters.json").read_text(encoding="utf-8"))
        assert len(clusters["clusters"]) == 1
        assert clusters["clusters"][0]["count"] == 3

    def test_single_trace_flag(self, tmp_path):
        # --trace single-file path: pass a one-element list.
        t = tmp_path / "one.json"
        _write_failed_trace(t, z=-5, cid="c1")
        out_dir = tmp_path / "out"
        assert run([t], out_dir=out_dir, min_cluster_size=1) == 0
        assert (out_dir / "failure_report.md").is_file()

    def test_empty_paths_exit_1(self, tmp_path):
        assert run([], out_dir=tmp_path / "out") == 1

    def test_all_bad_json_exit_1(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert run([bad], out_dir=tmp_path / "out") == 1

    def test_valid_trace_no_failures_exit_0_empty_report(self, tmp_path):
        t = tmp_path / "ok.json"
        t.write_text(json.dumps(make_trace([make_entry(1, success=True)])), encoding="utf-8")
        out_dir = tmp_path / "out"
        code = run([t], out_dir=out_dir, min_cluster_size=2)
        assert code == 0
        report = (out_dir / "failure_report.md").read_text(encoding="utf-8")
        assert "clusters: 0" in report
        assert "no recurring failure clusters" in report


class TestRunP3:
    def test_produces_patch_proposals_md(self, tmp_path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        for i, z in enumerate([-50, -99, -20, -30, -40]):
            _write_failed_trace(traces_dir / f"t{i}.json", z=z, cid=f"c{i}")
        out_dir = tmp_path / "out"

        code = run(sorted(traces_dir.glob("*.json")), out_dir=out_dir, min_cluster_size=3)

        assert code == 0
        proposals_path = out_dir / "skill_patch_proposals.md"
        assert proposals_path.is_file()
        text = proposals_path.read_text(encoding="utf-8")
        assert "target:" in text
        assert "<unresolved>" in text
        assert "z_min_safe" in text  # SafetyRail z-floor template fired
        assert "confidence" in text

    def test_three_outputs_when_failures_exist(self, tmp_path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        for i, z in enumerate([-50, -99, -20]):
            _write_failed_trace(traces_dir / f"t{i}.json", z=z, cid=f"c{i}")
        out_dir = tmp_path / "out"
        run(sorted(traces_dir.glob("*.json")), out_dir=out_dir, min_cluster_size=2)
        assert {p.name for p in out_dir.iterdir()} == {
            "failure_clusters.json",
            "failure_report.md",
            "skill_patch_proposals.md",
        }
