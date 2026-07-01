# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.agent.trace_html.render_trace_html."""

from __future__ import annotations

import base64
import html as html_module
import re

from jiuwensymbiosis.agent import trace_html
from jiuwensymbiosis.agent.trace_html import render_trace_html


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
                "output_summary": '{"ok": true}',
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


class TestRenderTraceHtml:
    def test_renders_key_content(self):
        html = render_trace_html(_sample_trace())
        assert "test_robot" in html
        assert "conv-1" in html
        assert "pick the red box" in html
        assert "goto_xyzr" in html
        assert "close_gripper" in html
        assert "✅" in html
        assert "❌" in html
        assert "RecoveryRail" in html
        assert "recover" in html
        assert "home() retried" in html
        assert "detector unreachable" in html
        assert "2 step(s) recorded" in html

    def test_inlines_frame_as_base64(self, tmp_path):
        # A real (tiny) JPEG bytes so the file reads and base64-encodes.
        frame = tmp_path / "step_001.jpg"
        frame.write_bytes(b"\xff\xd8\xff\xe0dummy")
        t = _sample_trace()
        t["entries"][0]["frame_path"] = str(frame)
        html = render_trace_html(t)
        expected = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff\xe0dummy").decode()
        assert expected in html

    def test_missing_frame_shows_placeholder(self, tmp_path):
        t = _sample_trace()
        t["entries"][0]["frame_path"] = str(tmp_path / "does_not_exist.jpg")
        html = render_trace_html(t)
        assert "data:image/jpeg;base64," not in html
        assert "frame missing" in html

    def test_relative_frame_resolves_against_trace_dir(self, tmp_path):
        # Frame lives under tmp_path; trace JSON also under tmp_path; frame_path
        # is relative so it must resolve against trace_path.parent.
        frame = tmp_path / "step_001.jpg"
        frame.write_bytes(b"img-bytes")
        t = _sample_trace()
        t["entries"][0]["frame_path"] = "step_001.jpg"
        trace_json = tmp_path / "trace.json"
        html = render_trace_html(t, trace_path=trace_json)
        expected = "data:image/jpeg;base64," + base64.b64encode(b"img-bytes").decode()
        assert expected in html

    def test_workspace_relative_frame_resolves_against_trace_parent(self, tmp_path):
        # Reproduces the `trace_dir: ./traces` case: frame_path carries its own
        # `traces/` segment (workspace/cwd-relative), and the JSON lives in
        # <base>/traces/. The frame must resolve against trace_dir.parent, not
        # trace_dir (which would double the prefix → traces/traces/...).
        traces_dir = tmp_path / "traces"
        frame = traces_dir / "frames" / "run-1" / "step_001.jpg"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b"ws-rel-bytes")
        t = _sample_trace()
        t["entries"][0]["frame_path"] = "traces/frames/run-1/step_001.jpg"
        trace_json = traces_dir / "trace.json"
        html = render_trace_html(t, trace_path=trace_json)
        expected = "data:image/jpeg;base64," + base64.b64encode(b"ws-rel-bytes").decode()
        assert expected in html
        assert "frame missing" not in html

    def test_relative_frame_without_trace_dir_no_crash(self):
        # Relative frame but no trace_path → can't resolve → placeholder, no raise.
        t = _sample_trace()
        t["entries"][0]["frame_path"] = "step_001.jpg"
        html = render_trace_html(t)  # no trace_path
        assert "data:image/jpeg;base64," not in html
        assert "frame missing" in html

    def test_html_escapes_angle_brackets(self):
        t = _sample_trace()
        # Error only renders on a failed step (new layout: error is a timeline
        # FAIL row + callout), so mark step 1 failed to surface the error text.
        t["entries"][0]["success"] = False
        t["entries"][0]["error"] = "ValueError: <script>bad & stuff"
        t["trace_log"][0]["msg"] = "a < b & c"
        html = render_trace_html(t)
        assert "<script>bad" not in html  # raw <script> must not survive
        assert "&lt;script&gt;bad" in html
        assert "a &lt; b &amp; c" in html

    def test_empty_entries(self):
        t = _sample_trace()
        t["entries"] = []
        html = render_trace_html(t)
        assert "0 step(s) recorded" in html
        assert "✅ 0" in html
        assert "❌ 0" in html

    def test_output_summary_collapsed(self):
        t = _sample_trace()
        t["entries"][0]["output_summary"] = '{"ok": true, "n": 3}'
        html = render_trace_html(t)
        assert "<details><summary>output</summary>" in html
        assert "{&quot;ok&quot;: true, &quot;n&quot;: 3}" in html

    def test_failed_step_error_callout_and_timeline(self):
        # A failed step surfaces the error as BOTH a highlighted callout AND a
        # FAIL row inside the unified timeline, so "what went wrong" is loud and
        # also part of the causal chain with rail/log events.
        t = _sample_trace()
        # step 2 is already failed with an error + RecoveryRail recover (ok).
        html = render_trace_html(t)
        assert 'class="error-callout"' in html
        assert "gripper timeout" in html
        # Timeline rows: FAIL (step) + ok (RecoveryRail) + WARNING (log) —
        # uniform badge/source/content shape, ordered.
        assert html.count('class="timeline"') >= 1
        assert '<span class="badge fail">FAIL</span>' in html
        assert '<span class="badge ok">ok</span>' in html
        # The ok RecoveryRail row is still green (recovery succeeded) — that's
        # correct: it's the rail's own success, distinct from the step's failure.
        assert "RecoveryRail/recover" in html

    def test_output_pulled_out_of_timeline(self):
        # output_summary lives in its own .row.output block, not inline among
        # the timeline rows, so it doesn't masquerade as an event.
        t = _sample_trace()
        t["entries"][0]["output_summary"] = '{"ok": true}'
        html = render_trace_html(t)
        assert 'class="row output"' in html

    def test_robot_emphasized_in_header(self):
        t = _sample_trace()
        html = render_trace_html(t)
        assert 'class="robot"' in html
        assert "test_robot" in html

    def test_brand_mark_in_header(self):
        # The JiuwenSymbiosis wordmark + logo ship in every page header. The
        # logo is the local PNG inlined as a base64 data URI (no external
        # requests); an inline SVG fallback is wired via onerror so a missing
        # image never yields a broken-image icon.
        html = render_trace_html(_sample_trace())
        assert "JiuwenSymbiosis" in html
        assert 'class="brandname"' in html
        assert 'class="brandmark"' in html
        assert "data:image/png;base64," in html  # local PNG inlined
        assert "cdn-img.gitcode.com" not in html  # no external CDN dependency
        # Assert the onerror JS is *executable* (decode like the browser, then
        # check the '-literal is balanced and carries a real <svg>…</svg>) — a
        # plain text-existence check once passed while the JS had a SyntaxError.
        m = re.search(r'onerror="([^"]*)"', html)
        assert m, "onerror attribute not found"
        js = html_module.unescape(m.group(1))
        assert js.startswith("this.outerHTML='") and js.endswith("'"), (
            f"onerror JS must be this.outerHTML='<svg…>', got: {js[:60]!r}"
        )
        payload = js[len("this.outerHTML='") : -1]
        assert "'" not in payload, (
            "SVG payload must be single-quote-free or the JS '-literal closes "
            f"early (the original bug); got a bare ' in: {payload[:60]!r}"
        )
        assert payload.startswith("<svg") and payload.endswith("</svg>"), (
            f"onerror payload must be a full <svg>…</svg>, got: {payload[:60]!r}"
        )

    def test_logo_fallback_onerror_is_valid_js_when_png_missing(self, monkeypatch):
        # Missing PNG → _LOGO_SRC="" → <img src=""> fires onerror. Verify the
        # fallback JS is balanced (no bare ' closing the '-literal early — the
        # original SyntaxError bug) and carries a real <svg>…</svg>.
        monkeypatch.setattr(trace_html, "_LOGO_SRC", "")
        html = render_trace_html(_sample_trace())
        assert 'src=""' in html
        m = re.search(r'onerror="([^"]*)"', html)
        assert m, "onerror attribute not found"
        js = html_module.unescape(m.group(1))
        assert js.startswith("this.outerHTML='") and js.endswith("'")
        payload = js[len("this.outerHTML='") : -1]
        assert "'" not in payload, f"single quote in onerror SVG payload would break the JS literal: {payload[:60]!r}"
        assert payload.startswith("<svg") and payload.endswith("</svg>")
        assert 'class="logo"' in payload

    def test_before_after_frames_side_by_side(self, tmp_path):
        # initial_frame_path + two steps each with a frame → step1's before is
        # the initial frame, step2's before is step1's after-frame. Each step
        # card renders two <img> (before + after), both base64-inlined.
        init_frame = tmp_path / "step_000.jpg"
        init_frame.write_bytes(b"INITJPEG")
        f1 = tmp_path / "step_001.jpg"
        f1.write_bytes(b"FRAME1JPEG")
        f2 = tmp_path / "step_002.jpg"
        f2.write_bytes(b"FRAME2JPEG")
        t = {
            "conversation_id": "conv-x",
            "robot_name": "piper",
            "query": "q",
            "initial_frame_path": str(init_frame),
            "entries": [
                {"step": 1, "tool_name": "goto_xyzr", "input_params": {"x": 1}, "success": True, "frame_path": str(f1)},
                {"step": 2, "tool_name": "close_gripper", "input_params": {}, "success": True, "frame_path": str(f2)},
            ],
        }
        html = render_trace_html(t)
        import base64 as _b64

        assert "data:image/jpeg;base64," + _b64.b64encode(b"INITJPEG").decode() in html
        assert "data:image/jpeg;base64," + _b64.b64encode(b"FRAME1JPEG").decode() in html
        assert "data:image/jpeg;base64," + _b64.b64encode(b"FRAME2JPEG").decode() in html
        # Step 2's before-frame is step 1's after-frame (FRAME1) — already
        # asserted above; the initial frame only appears in step 1's before.
        # Each step card has a before + after label.
        assert html.count('class="frame-label">before') == 2
        assert html.count('class="frame-label">after') == 2

    def test_no_initial_frame_step1_before_missing(self, tmp_path):
        # No initial_frame_path: step 1's before-slot has no raw path → renders
        # nothing (not "frame missing"), step 1 after-slot shows the frame.
        f1 = tmp_path / "step_001.jpg"
        f1.write_bytes(b"FRAME1JPEG")
        t = {
            "conversation_id": "conv-x",
            "robot_name": "piper",
            "entries": [
                {"step": 1, "tool_name": "goto_xyzr", "input_params": {}, "success": True, "frame_path": str(f1)},
            ],
        }
        html = render_trace_html(t)
        assert "frame missing" not in html  # no raw before-path → no placeholder
        import base64 as _b64

        assert "data:image/jpeg;base64," + _b64.b64encode(b"FRAME1JPEG").decode() in html

    # --- Structural + interactive additions (toolbar, progress, collapse,
    # frame zoom lightbox, mini duration bar, hash-jump, no-JS fallback). ---

    def test_toolbar_has_filter_and_view_buttons(self):
        html = render_trace_html(_sample_trace())
        assert 'id="toolbar"' in html
        # Filter group: all (active by default) / fail / ok.
        assert 'data-filter="all"' in html and 'class="active"' in html
        assert 'data-filter="fail"' in html
        assert 'data-filter="ok"' in html
        # View group: collapse-all + expand-all.
        assert 'id="collapse-all"' in html
        assert 'id="expand-all"' in html

    def test_progress_bar_has_per_step_segments(self):
        t = _sample_trace()
        html = render_trace_html(t)
        # One .seg per entry; ok step → plain seg, fail step → seg.fail.
        assert html.count('class="seg "') + html.count('class="seg fail"') == len(t["entries"])
        assert 'class="seg fail"' in html  # step 2 fails
        # Each segment links to its step anchor and carries a tooltip.
        assert 'href="#step-1"' in html
        assert 'data-tip="[1] goto_xyzr' in html

    def test_step_has_id_data_status_and_chevron(self):
        html = render_trace_html(_sample_trace())
        assert 'id="step-1"' in html
        assert 'id="step-2"' in html
        assert 'data-status="ok"' in html
        assert 'data-status="fail"' in html
        assert 'class="chevron"' in html

    def test_failed_step_not_collapsed_by_default(self):
        # Failed steps stay expanded so problems surface without a click;
        # successful steps ship collapsed. Step 2 (failed) → no "collapsed".
        html = render_trace_html(_sample_trace())
        # The failed step's <li> carries id="step-2" and is NOT collapsed.
        li2 = html.split('id="step-2"', 1)[1].split(">", 1)[0]
        assert "collapsed" not in li2
        # The ok step's <li> IS collapsed (default for non-failed steps).
        assert 'class="step ok collapsed"' in html

    def test_head_has_duration_mini_bar(self):
        html = render_trace_html(_sample_trace())
        assert 'class="dur-bar"' in html
        # The bar fill is an <i> with a width:% style scaled to max duration.
        assert '<i style="width:' in html
        # Slowest step (step 2, 1.2s) fills 100%; step 1 (0.8s) is ~67%.
        assert "width:100%" in html

    def test_frame_zoom_wraps_image(self, tmp_path):
        # The <img> is wrapped in <button class="frame-zoom"> carrying the
        # step + label data the lightbox reads on click.
        f1 = tmp_path / "step_001.jpg"
        f1.write_bytes(b"FRAME1JPEG")
        t = {
            "conversation_id": "conv-x",
            "robot_name": "piper",
            "entries": [
                {"step": 1, "tool_name": "goto_xyzr", "input_params": {}, "success": True, "frame_path": str(f1)},
            ],
        }
        html = render_trace_html(t)
        assert '<button class="frame-zoom"' in html
        assert 'data-step="1"' in html
        assert 'data-label="after"' in html
        # The base64 img still lives inside the button.
        import base64 as _b64

        assert "data:image/jpeg;base64," + _b64.b64encode(b"FRAME1JPEG").decode() in html

    def test_lightbox_dialog_present(self):
        html = render_trace_html(_sample_trace())
        assert '<dialog class="lightbox"' in html
        assert 'id="lightbox"' in html
        assert 'id="lb-img"' in html
        assert 'id="lb-close"' in html

    def test_inline_script_and_noscript_present(self):
        html = render_trace_html(_sample_trace())
        assert "<script>" in html
        assert "frame-zoom" in html  # JS references the zoom buttons
        assert "<noscript>" in html  # no-JS fallback style block

    def test_empty_entries_has_no_progress_segments(self):
        t = _sample_trace()
        t["entries"] = []
        html = render_trace_html(t)
        # No steps → no progress bar rendered (toolbar still present).
        assert 'class="seg' not in html
        assert 'id="toolbar"' in html
