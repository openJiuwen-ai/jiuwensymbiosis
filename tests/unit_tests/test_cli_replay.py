# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.cli.replay / replay_html."""

from __future__ import annotations

import io
import json

from jiuwensymbiosis.cli import replay, replay_html


def _sample_trace() -> dict:
    return {
        "conversation_id": "conv-1",
        "robot_name": "test_robot",
        "query": "pick the red box",
        "entries": [
            {
                "step": 1,
                "tool_name": "goto_xyzr",
                "input_params": {"x": 150, "y": 0, "z": 80},
                "success": True,
                "duration_s": 0.8,
                "error": None,
                "observation": {"pose": {"x": 150, "y": 0, "z": 80}},
                "frame_path": None,
                "rail_events": [],
                "log_events": [],
            },
            {
                "step": 2,
                "tool_name": "close_gripper",
                "input_params": {"force_n": 10},
                "success": False,
                "duration_s": 1.2,
                "error": "ValueError: gripper timeout",
                "observation": None,
                "frame_path": None,
                "rail_events": [
                    {"rail_name": "RecoveryRail", "kind": "recover", "detail": {"home_ok": True}, "success": True},
                ],
                "log_events": [
                    {
                        "logger": "jiuwensymbiosis.rails.recovery",
                        "level": "WARNING",
                        "msg": "home() retried",
                        "ts": 0.0,
                    },
                ],
            },
        ],
        "trace_log": [
            {"logger": "jiuwensymbiosis.detector", "level": "WARNING", "msg": "detector unreachable", "ts": 0.0},
        ],
    }


class TestReplay:
    def test_renders_timeline(self, tmp_path):
        p = tmp_path / "conv-1.json"
        p.write_text(json.dumps(_sample_trace()), encoding="utf-8")
        out = io.StringIO()
        rc = replay(str(p), out=out)
        text = out.getvalue()
        assert rc == 0
        assert "conv-1" in text
        assert "pick the red box" in text
        assert "goto_xyzr" in text
        assert "close_gripper" in text
        assert "✅" in text
        assert "❌" in text
        assert "RecoveryRail" in text
        assert "detector unreachable" in text
        assert "2 step(s)" in text

    def test_missing_file_returns_1(self, tmp_path):
        rc = replay(str(tmp_path / "nope.json"))
        assert rc == 1

    def test_invalid_json_returns_1(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        rc = replay(str(p))
        assert rc == 1

    def test_empty_entries(self, tmp_path):
        t = _sample_trace()
        t["entries"] = []
        p = tmp_path / "empty.json"
        p.write_text(json.dumps(t), encoding="utf-8")
        out = io.StringIO()
        rc = replay(str(p), out=out)
        assert rc == 0
        assert "no tool-call steps" in out.getvalue()


class TestReplayHtml:
    def test_writes_html_and_prints_path(self, tmp_path):
        # Default: write HTML + print path.
        p = tmp_path / "conv-1.json"
        p.write_text(json.dumps(_sample_trace()), encoding="utf-8")
        out = io.StringIO()
        rc = replay_html(str(p), out=out)
        assert rc == 0
        html_path = p.with_suffix(".html")
        assert html_path.is_file()
        text = out.getvalue()
        assert f"wrote {html_path}" in text

    def test_html_contains_step_content(self, tmp_path):
        p = tmp_path / "conv-1.json"
        p.write_text(json.dumps(_sample_trace()), encoding="utf-8")
        out = io.StringIO()
        rc = replay_html(str(p), out=out)
        assert rc == 0
        html = p.with_suffix(".html").read_text(encoding="utf-8")
        assert "goto_xyzr" in html
        assert "close_gripper" in html
        assert "pick the red box" in html

    def test_missing_file_returns_1(self, tmp_path):
        rc = replay_html(str(tmp_path / "nope.json"))
        assert rc == 1

    def test_invalid_json_returns_1(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        rc = replay_html(str(p))
        assert rc == 1
