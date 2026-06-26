# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Render a recorded execution trace as a self-contained HTML page.

The replay CLI calls :func:`render_trace_html` to turn a ``traces/*.json``
(see :mod:`jiuwensymbiosis.agent.trace`) into one HTML file with every step's
saved JPEG frame inlined as base64. Each step becomes a card that fuses the
frame with a unified per-step *timeline* — the step's own failure (if any),
its rail events, and its log events all render in one chronological list with
a uniform row shape (badge + source + content), so a reader can follow the
causal chain ("this step failed → RecoveryRail recovered → warning logged")
instead of scanning three separate blocks. ``output_summary`` is pulled out
of the result area into its own labelled row so it doesn't drown among the
events.

The output is a single string — no external assets, no network, no optional
deps. Frame files are read best-effort; a missing frame renders a placeholder
rather than raising.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any, Optional

__all__ = ["render_trace_html"]


def _esc(value: Any) -> str:
    """JSON-encode then HTML-escape an arbitrary trace value for safe display."""
    try:
        s = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = repr(value)
    return html.escape(s)


def _esc_str(value: Any) -> str:
    """HTML-escape a plain string (no JSON wrapping)."""
    return html.escape(str(value))


def _resolve_frame(frame_path: Optional[str], trace_dir: Optional[Path]) -> Optional[Path]:
    """Resolve a stored ``frame_path`` to an existing file, or ``None``.

    Frame paths land in the JSON in one of three forms depending on how the
    recording run was configured:

    * **absolute** — when ``trace_dir`` was absolute (the default
      ``<workspace>/traces``); used as-is.
    * **JSON-dir relative** (``frames/{token}/step.jpg``) — the portable form,
      resolves against ``trace_dir`` (the directory the JSON lives in).
    * **workspace/cwd relative** (``traces/frames/{token}/step.jpg``) — what a
      relative ``trace_dir`` override (e.g. ``trace_dir: ./traces``) yields,
      since the path then carries its own ``traces/`` segment. The JSON sits in
      ``<base>/traces/`` so this resolves against ``trace_dir.parent`` (the
      workspace root), and lastly against the current working directory.

    Anchors are tried in that order; the first existing file wins. Returns
    ``None`` when nothing resolves so the caller renders a placeholder.
    """
    if not frame_path:
        return None
    p = Path(frame_path)
    if p.is_absolute():
        return p if p.is_file() else None
    candidates: list[Path] = []
    if trace_dir is not None:
        candidates.append(trace_dir / p)         # JSON-dir relative (portable form)
        candidates.append(trace_dir.parent / p)  # workspace relative (JSON in <ws>/traces)
    candidates.append(p)                          # cwd relative (last resort)
    for c in candidates:
        if c.is_file():
            return c
    return None


def _inline_frame(frame_path: Optional[str], trace_dir: Optional[Path]) -> str:
    """Return a ``data:image/jpeg;base64,...`` URI for a saved frame, or ``""``.

    Resolves ``frame_path`` via :func:`_resolve_frame`. Any unresolved path or
    read/decode failure → ``""`` so the caller renders a "frame missing"
    placeholder.
    """
    p = _resolve_frame(frame_path, trace_dir)
    if p is None:
        return ""
    try:
        data = p.read_bytes()
    except (OSError, ValueError):
        return ""
    if not data:
        return ""
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _step_timeline(entry: dict) -> list[dict]:
    """Merge a step's failure + rail + log events into a chronological list.

    Each item is ``{ts, badge, badge_cls, source, content}`` so the renderer
    can emit them in one uniform row shape. The step's own failure (when
    ``success`` is False and ``error`` is set) becomes the first item — a FAIL
    event — so the causal chain reads "step failed → rail recovered → logged".
    Rail/log events keep their own ``success``/``level`` for the badge colour.
    Items without a ``ts`` sort to the top (they describe the step itself).
    """
    items: list[dict] = []
    if not entry.get("success", True) and entry.get("error"):
        items.append(
            {
                "ts": 0.0,
                "badge": "FAIL",
                "badge_cls": "fail",
                "source": "step",
                "content": entry["error"],
            }
        )
    for ev in entry.get("rail_events", []) or []:
        ok = ev.get("success")
        items.append(
            {
                "ts": float(ev.get("ts") or 0.0),
                "badge": "ok" if ok else "FAIL",
                "badge_cls": "ok" if ok else "fail",
                "source": f'{ev.get("rail_name", "?")}/{ev.get("kind", "?")}',
                "content": ev.get("detail", {}) or {},
            }
        )
    for ev in entry.get("log_events", []) or []:
        lvl = ev.get("level", "INFO")
        cls = "fail" if str(lvl).upper() in ("ERROR", "CRITICAL") else "warn"
        items.append(
            {
                "ts": float(ev.get("ts") or 0.0),
                "badge": str(lvl),
                "badge_cls": cls,
                "source": ev.get("logger", "?"),
                "content": ev.get("msg", ""),
            }
        )
    # Stable sort by ts: keeps same-timestamp items in insertion order.
    items.sort(key=lambda it: it["ts"])
    return items


_HTML_HEAD = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Execution Trace</title>
<style>
  :root { --ok: #2e7d32; --fail: #c62828; --warn: #b58900; --muted: #6b7280;
          --bg: #f4f5f7; --card: #fff; --border: #e5e7eb; --code: #f3f4f6;
          --accent: #1d4ed8; }
  body { font: 14px/1.55 -apple-system, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
         margin: 0; background: var(--bg); color: #1f2937; }
  header { padding: 18px 24px; background: var(--card); border-bottom: 1px solid var(--border);
           display: flex; align-items: center; flex-wrap: wrap; gap: 6px 20px; }
  header .robot { font-size: 22px; font-weight: 700; color: var(--accent); }
  header .robot .icon { margin-right: 6px; }
  header .subtitle { color: var(--muted); font-size: 13px; }
  header .subtitle b { color: #374151; font-weight: 600; }
  header .query { flex-basis: 100%; color: #374151; font-size: 14px; margin-top: 2px; }
  header .summary { margin-left: auto; color: var(--muted); font-size: 13px; }
  main { max-width: 1120px; margin: 0 auto; padding: 16px; }
  ol.steps { list-style: none; margin: 0; padding: 0; }
  li.step { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
            margin-bottom: 14px; overflow: hidden; border-left: 4px solid var(--ok); }
  li.step.fail { border-left-color: var(--fail); }
  li.step .head { display: flex; align-items: baseline; gap: 8px; padding: 10px 14px;
                  border-bottom: 1px solid var(--border); flex-wrap: wrap; }
  li.step .num { font-variant-numeric: tabular-nums; color: var(--muted); min-width: 3ch; }
  li.step .name { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 600; }
  li.step .mark.fail { color: var(--fail); }
  li.step .mark.ok { color: var(--ok); }
  li.step .dur { color: var(--muted); font-size: 12px; margin-left: auto; }
  li.step .body { display: flex; gap: 16px; padding: 12px 14px; }
  li.step .frames { flex: 0 0 360px; max-width: 50%; display: flex; gap: 8px; }
  li.step .frame { flex: 1; min-width: 0; }
  li.step .frame-label { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  li.step .frame img { width: 100%; height: auto; border-radius: 4px;
                       border: 1px solid var(--border); display: block; }
  li.step .frame .missing { color: var(--muted); font-size: 12px;
                             border: 1px dashed var(--border); border-radius: 4px;
                             padding: 12px; text-align: center; }
  li.step .events { flex: 1; min-width: 0; }
  /* Highlighted error callout: a quick red box at the top of a failed step's
     events so "what went wrong" jumps out, in addition to the FAIL row in the
     timeline below. */
  li.step .error-callout { background: #fef2f2; border: 1px solid #fecaca;
                            border-radius: 6px; padding: 8px 10px; margin-bottom: 8px;
                            color: #991b1b; font-size: 13px; }
  li.step .error-callout .label { font-weight: 700; margin-right: 6px; }
  .timeline { margin: 0; padding: 0; list-style: none; }
  .timeline li { display: flex; align-items: flex-start; gap: 6px; margin: 4px 0;
                 font-size: 12px; line-height: 1.5; }
  .timeline .badge { flex: 0 0 auto; min-width: 4ch; text-align: center;
                     display: inline-block; padding: 0 5px; border-radius: 3px;
                     font-size: 11px; font-weight: 600; color: #fff; }
  .badge.ok { background: var(--ok); }
  .badge.fail { background: var(--fail); }
  .badge.warn { background: var(--warn); }
  .timeline .source { flex: 0 0 auto; color: var(--muted); font-family: ui-monospace,
                      SFMono-Regular, Menlo, monospace; font-size: 11px; }
  .timeline .content { flex: 1; min-width: 0; word-break: break-word; }
  .timeline code, .events code { background: var(--code); padding: 1px 4px; border-radius: 3px;
                         font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                         word-break: break-all; }
  .row.output { margin-top: 8px; border-top: 1px dashed var(--border); padding-top: 6px; }
  .row.output summary { cursor: pointer; color: var(--muted); font-size: 12px; }
  .row.output pre { margin: 4px 0 0; white-space: pre-wrap; word-break: break-word;
                    font-size: 12px; }
  .pose { margin: 4px 0; font-size: 12px; }
  .pose .label { color: var(--muted); display: inline-block; min-width: 5ch; }
  .trace-logs { margin-top: 16px; background: var(--card); border: 1px solid var(--border);
                border-radius: 8px; padding: 12px 14px; }
  .trace-logs h2 { font-size: 14px; margin: 0 0 6px; }
  footer { color: var(--muted); font-size: 12px; text-align: center; padding: 16px; }
</style>
</head>
<body>
"""


def render_trace_html(data: dict, *, trace_path: Optional[Path] = None) -> str:
    """Render a trace dict as a self-contained HTML page.

    Args:
        data: The parsed trace JSON (as produced by :class:`ExecutionTrace`).
        trace_path: Path to the trace JSON, used to resolve relative
            ``frame_path`` values against its directory. May be None, in which
            case only absolute frame paths can be inlined.

    Returns:
        A complete HTML document as a string. Frames are inlined as base64;
        missing frames render placeholders; no external assets are referenced.
    """
    trace_dir = trace_path.parent if trace_path is not None else None
    cid = data.get("conversation_id") or "?"
    robot = data.get("robot_name") or "?"
    query = data.get("query")
    entries = data.get("entries", []) or []
    trace_log = data.get("trace_log", []) or []

    n_ok = sum(1 for e in entries if e.get("success", True))
    n_fail = len(entries) - n_ok

    parts: list[str] = [_HTML_HEAD]
    parts.append("<header>\n")
    parts.append(f'<div class="robot"><span class="icon">🤖</span>{_esc_str(robot)}</div>\n')
    parts.append(f'<div class="subtitle">conversation <b>{_esc_str(cid)}</b></div>\n')
    if query:
        parts.append(f'<div class="query"><b>query:</b> {_esc_str(query)}</div>\n')
    parts.append(
        f'<div class="summary">{len(entries)} step(s)  ·  '
        f'<span class="mark ok">✅ {n_ok}</span>  ·  '
        f'<span class="mark fail">❌ {n_fail}</span></div>\n'
    )
    parts.append("</header>\n<main>\n")
    parts.append('<ol class="steps">\n')

    # The before-frame for step N: step 1 uses the trace's ``initial_frame_path``
    # (captured at invoke start); step N>1 reuses step N-1's after-frame
    # (``frame_path``) since the scene is unchanged between a step's end and the
    # next step's start. We walk entries carrying the previous after-frame.
    prev_frame_path = data.get("initial_frame_path")

    for e in entries:
        step = e.get("step", "?")
        name = e.get("tool_name", "?") or "?"
        params = e.get("input_params", {})
        success = e.get("success", True)
        dur = e.get("duration_s", 0.0) or 0.0
        cls = "ok" if success else "fail"
        mark = "✅" if success else "❌"
        parts.append(f'<li class="step {cls}">\n')
        parts.append(
            f'<div class="head"><span class="num">[{step}]</span>'
            f'<span class="mark {cls}">{mark}</span>'
            f'<span class="name">{_esc_str(name)}</span>'
            f"<code>({_esc(params)})</code>"
            f'<span class="dur">dur={dur:.2f}s</span></div>\n'
        )
        parts.append('<div class="body">\n')

        # Frames (left): before/after pair, side by side.
        before_uri = _inline_frame(prev_frame_path, trace_dir)
        after_uri = _inline_frame(e.get("frame_path"), trace_dir)
        parts.append('<div class="frames">')
        for label, uri, raw_path in (
            ("before", before_uri, prev_frame_path),
            ("after", after_uri, e.get("frame_path")),
        ):
            parts.append('<div class="frame">')
            parts.append(f'<div class="frame-label">{label}</div>')
            if uri:
                parts.append(f'<img alt="step {step} {label}" src="{uri}">')
            elif raw_path:
                parts.append('<div class="missing">frame missing</div>')
            parts.append("</div>")
        parts.append("</div>\n")

        prev_frame_path = e.get("frame_path") or prev_frame_path

        # Events (right): error callout + unified timeline + pose + output.
        parts.append('<div class="events">\n')
        # Highlighted error callout for failed steps — quick "what went wrong".
        if not success and e.get("error"):
            parts.append(
                f'<div class="error-callout"><span class="label">❌ ERROR</span>'
                f'{_esc_str(e["error"])}</div>\n'
            )
        obs = e.get("observation")
        if obs and obs.get("pose"):
            parts.append(
                f'<div class="pose"><span class="label">pose</span> '
                f'<code>{_esc(obs["pose"])}</code></div>\n'
            )
        # Unified chronological timeline: step failure + rail + log events.
        timeline = _step_timeline(e)
        if timeline:
            parts.append('<ul class="timeline">\n')
            for it in timeline:
                content = it["content"]
                # dict content (rail detail) → JSON code; string content → plain.
                if isinstance(content, (dict, list)):
                    chtml = f"<code>{_esc(content)}</code>"
                else:
                    chtml = _esc_str(content)
                parts.append(
                    f'<li><span class="badge {it["badge_cls"]}">{_esc_str(it["badge"])}</span>'
                    f'<span class="source">{_esc_str(it["source"])}</span>'
                    f'<span class="content">{chtml}</span></li>\n'
                )
            parts.append("</ul>\n")
        # Output summary pulled out of the event flow into its own labelled row,
        # so it doesn't masquerade as an event among rail/log rows.
        out_sum = e.get("output_summary")
        if out_sum:
            parts.append('<div class="row output"><details><summary>output</summary>')
            parts.append(f"<pre>{_esc_str(out_sum)}</pre></details></div>\n")
        parts.append("</div>\n")  # /events

        parts.append("</div>\n")  # /body
        parts.append("</li>\n")

    parts.append("</ol>\n")

    if trace_log:
        parts.append('<div class="trace-logs">\n<h2>trace-level logs</h2>\n<ul class="timeline">\n')
        for ev in trace_log:
            lvl = ev.get("level", "INFO")
            cls = "fail" if str(lvl).upper() in ("ERROR", "CRITICAL") else "warn"
            parts.append(
                f'<li><span class="badge {cls}">{_esc_str(lvl)}</span>'
                f'<span class="source">{_esc_str(ev.get("logger", "?"))}</span>'
                f'<span class="content">{_esc_str(ev.get("msg", ""))}</span></li>\n'
            )
        parts.append("</ul>\n</div>\n")

    parts.append("</main>\n")
    parts.append(f"<footer>{len(entries)} step(s) recorded.</footer>\n")
    parts.append("</body>\n</html>\n")
    return "".join(parts)
