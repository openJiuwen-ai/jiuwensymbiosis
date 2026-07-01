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
from typing import Any

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


def _resolve_frame(frame_path: str | None, trace_dir: Path | None) -> Path | None:
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
        candidates.append(trace_dir / p)
        candidates.append(trace_dir.parent / p)
    candidates.append(p)
    for c in candidates:
        if c.is_file():
            return c
    return None


def _inline_frame(frame_path: str | None, trace_dir: Path | None) -> str:
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
                "source": f"{ev.get('rail_name', '?')}/{ev.get('kind', '?')}",
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


# JiuwenSymbiosis brand assets for the trace page header.
#
# _LOGO_SRC is the logo <img> source — the official wordmark PNG, inlined as a
# base64 data: URI so the rendered page stays a single self-contained HTML file
# with zero external requests. The PNG ships at <repo>/docs/design/jiuwensymbiosis-
# logo.png, resolved relative to this module so it works regardless of cwd. If
# the file is missing the loader returns an empty string and the <img>'s onerror
# swaps in _LOGO_SVG, so the header never shows a broken-image icon.
_LOGO_PNG_LOCAL = Path(__file__).resolve().parents[2] / "docs" / "design" / "jiuwensymbiosis-logo.png"


def _load_logo_src() -> str:
    """Return the logo image src as a base64 data URI read from the local PNG,
    or ``""`` if the file is unavailable (the caller's onerror falls back to the
    inline SVG glyph). Rendering never raises on a missing logo."""
    try:
        data = _LOGO_PNG_LOCAL.read_bytes()
    except (OSError, ValueError):
        return ""
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


_LOGO_SRC = _load_logo_src()

# _LOGO_SVG is an inline fallback glyph (three interlinked nodes around a shared
# core — the "symbiosis" of multiple embodied agents). currentColour lets it pick
# up the active theme's --accent, so the fallback still re-colours per theme.
_LOGO_SVG = (
    '<svg class="logo" viewBox="0 0 32 32" width="20" height="20" aria-hidden="true" '
    'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<circle cx="16" cy="6.5" r="3.2"/>'
    '<circle cx="6.5" cy="22" r="3.2"/>'
    '<circle cx="25.5" cy="22" r="3.2"/>'
    '<line x1="16" y1="6.5" x2="6.5" y2="22"/>'
    '<line x1="16" y1="6.5" x2="25.5" y2="22"/>'
    '<line x1="6.5" y1="22" x2="25.5" y2="22"/>'
    '<circle cx="16" cy="16.5" r="2.4" fill="currentColor" stroke="none"/>'
    "</svg>"
)
# Module-load invariant: the SVG source must contain NO single quotes. A single
# quote would survive as a bare ' inside the JS string literal and close it
# early — the exact bug this constant was restructured to avoid. Use a real
# raise (not assert) so the check still fires under `python -O`.
if "'" in _LOGO_SVG:
    raise RuntimeError("_LOGO_SVG must use double-quoted attributes only")


_HTML_HEAD = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Execution Trace</title>
<style>
  /* Visual language: a dark, engineering-grade dashboard theme (default; see
     _THEMES for alternates). The :root block below carries every colour + font
     token; the rest of the stylesheet is token-driven (var(--...)), so swapping
     the root vars swaps the whole theme without touching DOM or rules. */
  {root_vars}
  html { color-scheme: dark; }
  body { font: 14px/1.6 var(--sans); margin: 0; background: var(--bg); color: var(--ink);
         font-variant-numeric: tabular-nums; -webkit-font-smoothing: antialiased; }
  header { padding: 18px 24px; background: var(--bg); border-bottom: 1px solid var(--border);
           display: flex; align-items: center; flex-wrap: wrap; gap: 8px 20px;
           border-top: 3px solid var(--accent); position: relative;
           padding-right: 260px; }
  /* Left cluster: trace subject. Robot name leads; subtitle trails on row 1.
     Row 2 is a .meta line: query + step tallies sharing one line, split by a
     middot. The brand is absolutely positioned top-right. */
  header .robot { font-size: 20px; font-weight: 600; color: var(--ink-strong);
                  letter-spacing: -0.01em; }
  header .subtitle { color: var(--muted); font-size: 13px; }
  header .subtitle b { color: var(--body); font-weight: 600; }
  header .meta { flex-basis: 100%; display: flex; flex-wrap: wrap; align-items: baseline;
                 gap: 10px; color: var(--body); font-size: 13px; margin-top: 2px; }
  header .meta .query { color: var(--body); }
  header .meta .query b { color: var(--ink-strong); font-weight: 600; }
  header .meta .sep { color: var(--border); opacity: .9; }
  header .meta .summary { color: var(--muted); }
  header .meta .summary .mark { font-variant-numeric: tabular-nums; }
  header .meta .summary .mark.ok { color: var(--ok); }
  header .meta .summary .mark.fail { color: var(--fail); }
  /* Brand pinned to the top-right corner. Absolute positioning + a vertical
     centre transform keep the logo + wordmark aligned to the header's midline
     regardless of how tall the left cluster grows, so the mark never reads as
     "偏下" against the multi-row metadata. */
  header .brand { position: absolute; top: 50%; right: 24px; transform: translateY(-50%);
                  display: flex; align-items: center; gap: 12px; }
  header .brandmark { width: 48px; height: 48px; border-radius: 9px; object-fit: contain;
                      display: block; }
  header .brandname { font-size: 18px; font-weight: 600; color: var(--ink-strong);
                      letter-spacing: -0.015em; line-height: 1; }
  .logo { display: block; color: var(--accent); }

  /* Toolbar: sticky filter + collapse controls (button-outline-on-dark chrome). */
  .toolbar { position: sticky; top: 0; z-index: 20; background: var(--bg);
             border-bottom: 1px solid var(--border); padding: 10px 24px;
             display: flex; align-items: center; flex-wrap: wrap; gap: 8px 12px; }
  .toolbar .group { display: flex; gap: 6px; align-items: center; }
  .toolbar .label { font-family: var(--mono); font-size: 11px; color: var(--accent);
                    text-transform: uppercase; letter-spacing: 0.06em; margin-right: 2px; opacity: .85; }
  .toolbar button { font: 600 13px/1.2 var(--sans); color: var(--body);
                    background: var(--card); border: 1px solid var(--border);
                    border-radius: 6px; padding: 5px 12px; cursor: pointer;
                    transition: border-color .12s, color .12s, background .12s; }
  .toolbar button:hover { color: var(--ink-strong); border-color: var(--muted); }
  .toolbar button.active { color: var(--accent); border-color: var(--accent);
                           background: color-mix(in srgb, var(--accent) 12%, transparent); }
  .toolbar .spacer { flex: 1; }

  /* Progress bar: one segment per step, ok=green fail=red, click jumps to step. */
  .progress { display: flex; gap: 2px; flex: 1 1 100%; min-width: 0;
              background: var(--card-deep); border: 1px solid var(--border-soft);
              border-radius: 6px; padding: 3px; }
  .progress .seg { flex: 1 1 0; min-width: 6px; height: 14px; border-radius: 3px;
                   background: var(--ok); cursor: pointer; opacity: .55;
                   transition: opacity .12s, transform .12s; position: relative; }
  .progress .seg.fail { background: var(--fail); }
  .progress .seg:hover { opacity: 1; transform: scaleY(1.15);
                          box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 60%, transparent); }
  .progress .seg::after { content: attr(data-tip); position: absolute; bottom: 18px;
                          left: 50%; transform: translateX(-50%); background: var(--card);
                          color: var(--ink); border: 1px solid var(--border); border-radius: 4px;
                          padding: 3px 7px; font: 11px var(--mono); white-space: nowrap;
                          opacity: 0; pointer-events: none; transition: opacity .12s; z-index: 30; }
  .progress .seg:hover::after { opacity: 1; }

  main { max-width: 1120px; margin: 0 auto; padding: 16px; }
  ol.steps { list-style: none; margin: 0; padding: 0; }
  li.step { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
            margin-bottom: 14px; overflow: hidden; border-left: 4px solid var(--ok); }
  li.step.fail { border-left-color: var(--fail); }
  li.step.hidden { display: none; }              /* filter: hidden by JS */
  li.step .head { display: flex; align-items: center; gap: 8px; padding: 10px 14px;
                  border-bottom: 1px solid var(--border-soft); flex-wrap: wrap;
                  cursor: pointer; user-select: none; }
  li.step.collapsed .head { border-bottom-color: transparent; }
  li.step .chevron { flex: 0 0 auto; color: var(--muted); font-size: 12px;
                     transition: transform .12s, color .12s; }
  li.step .head:hover .chevron { color: var(--accent); }
  li.step.collapsed .chevron { transform: rotate(-90deg); }
  li.step .num { font-family: var(--mono); font-variant-numeric: tabular-nums;
                 color: var(--muted); min-width: 3ch; font-size: 13px; }
  li.step .name { font-family: var(--mono); font-weight: 600; color: var(--ink-strong); font-size: 13px; }
  li.step .mark.fail { color: var(--fail); }
  li.step .mark.ok { color: var(--ok); }
  li.step .params { background: var(--card-deep); padding: 1px 6px; border-radius: 4px;
                    font-family: var(--mono); font-size: 12px; color: var(--body);
                    word-break: break-all; }
  li.step .dur { color: var(--muted); font-size: 12px; margin-left: auto;
                 font-family: var(--mono); font-variant-numeric: tabular-nums;
                 display: inline-flex; align-items: center; gap: 6px; }
  /* Mini duration bar: width scales to max step duration so slow steps stand out. */
  li.step .dur-bar { display: inline-block; width: 48px; height: 6px; border-radius: 3px;
                     background: var(--border); overflow: hidden; vertical-align: middle; }
  li.step .dur-bar > i { display: block; height: 100%; background: var(--ok); border-radius: 3px; }
  li.step.fail .dur-bar > i { background: var(--fail); }
  li.step .body { display: flex; gap: 16px; padding: 12px 14px; }
  li.step.collapsed .body { display: none; }
  li.step .frames { flex: 0 0 360px; max-width: 50%; display: flex; gap: 8px; }
  li.step .frame { flex: 1; min-width: 0; }
  li.step .frame-label { font-size: 11px; color: var(--muted); margin-bottom: 4px;
                         font-family: var(--mono); text-transform: uppercase; letter-spacing: 0.06em; }
  /* frame-zoom: wraps the <img> so a click opens the lightbox. */
  .frame-zoom { all: unset; display: block; cursor: zoom-in; }
  .frame-zoom img { width: 100%; height: auto; border-radius: 4px;
                    border: 1px solid var(--border); display: block;
                    transition: border-color .12s; }
  .frame-zoom:hover img { border-color: var(--accent); }
  li.step .frame .missing { color: var(--muted); font-size: 12px;
                             border: 1px dashed var(--border); border-radius: 4px;
                             padding: 12px; text-align: center; background: var(--card-deep); }
  li.step .events { flex: 1; min-width: 0; }
  /* Highlighted error callout: a quick red box at the top of a failed step's
     events so "what went wrong" jumps out, in addition to the FAIL row in the
     timeline below. */
  li.step .error-callout { background: color-mix(in srgb, var(--fail) 10%, transparent);
                            border: 1px solid color-mix(in srgb, var(--fail) 45%, transparent);
                            border-radius: 6px; padding: 8px 10px; margin-bottom: 8px;
                            color: color-mix(in srgb, var(--fail) 75%, var(--ink)); font-size: 13px; }
  li.step .error-callout .label { font-weight: 700; margin-right: 6px; color: var(--fail); }
  .timeline { margin: 0; padding: 0; list-style: none; }
  .timeline li { display: flex; align-items: flex-start; gap: 6px; margin: 4px 0;
                 font-size: 12px; line-height: 1.5; }
  .timeline .badge { flex: 0 0 auto; min-width: 4ch; text-align: center;
                     display: inline-block; padding: 1px 6px; border-radius: 4px;
                     font-size: 11px; font-weight: 600; font-family: var(--mono);
                     letter-spacing: 0.02em; }
  .badge.ok { background: var(--ok); color: var(--ok-ink); }
  .badge.fail { background: var(--fail); color: #fff; }
  .badge.warn { background: var(--warn); color: #1a1a1a; }
  .timeline .source { flex: 0 0 auto; color: var(--muted); font-family: var(--mono); font-size: 11px; }
  .timeline .content { flex: 1; min-width: 0; word-break: break-word; color: var(--body); }
  .timeline code, .events code { background: var(--card-deep); padding: 1px 5px; border-radius: 4px;
                         font-family: var(--mono); font-size: 12px; color: var(--ink);
                         word-break: break-all; }
  .row.output { margin-top: 8px; border-top: 1px dashed var(--border-soft); padding-top: 6px; }
  .row.output summary { cursor: pointer; color: var(--muted); font-size: 12px;
                        font-family: var(--mono); }
  .row.output pre { margin: 4px 0 0; white-space: pre-wrap; word-break: break-word;
                    font-size: 12px; font-family: var(--mono); color: var(--body);
                    background: var(--card-deep); padding: 8px; border-radius: 6px;
                    border: 1px solid var(--border-soft); }
  .pose { margin: 4px 0; font-size: 12px; color: var(--body); }
  .pose .label { color: var(--muted); display: inline-block; min-width: 5ch;
                 font-family: var(--mono); }
  .trace-logs { margin-top: 16px; background: var(--card); border: 1px solid var(--border);
                border-radius: 8px; padding: 12px 14px; }
  .trace-logs h2 { font-size: 13px; margin: 0 0 6px; color: var(--ink); font-weight: 600;
                   font-family: var(--mono); text-transform: uppercase; letter-spacing: 0.04em; }
  footer { color: var(--muted); font-size: 12px; text-align: center; padding: 16px;
           font-family: var(--mono); border-top: 1px solid var(--border-soft); }
  footer::before { content: ""; display: block; width: 40px; height: 2px;
                   margin: 0 auto 12px; background: var(--accent); border-radius: 2px; }

  /* Lightbox: native <dialog> for frame zoom, no deps. */
  dialog.lightbox { background: var(--card); border: 1px solid var(--border);
                    border-radius: 8px; padding: 12px; color: var(--ink);
                    max-width: min(92vw, 1100px); max-height: 92vh; }
  dialog.lightbox::backdrop { background: rgba(0, 0, 0, 0.82); }
  dialog.lightbox .lb-head { display: flex; align-items: center; gap: 8px;
                             margin-bottom: 8px; font-family: var(--mono); font-size: 12px; }
  dialog.lightbox .lb-head .lb-title { color: var(--ink-strong); font-weight: 600; }
  dialog.lightbox .lb-head .lb-label { color: var(--accent); }
  dialog.lightbox .lb-head .lb-close { margin-left: auto; font: 600 13px var(--sans);
                     background: var(--card-deep); color: var(--body); border: 1px solid var(--border);
                     border-radius: 6px; padding: 4px 10px; cursor: pointer; }
  dialog.lightbox .lb-close:hover { color: var(--ink-strong); border-color: var(--muted); }
  dialog.lightbox img { max-width: 100%; max-height: 78vh; border-radius: 6px;
                        border: 1px solid var(--border); display: block; }

  /* Flash: hash-jump target gets a brief accent-ring pulse. */
  @keyframes flash { 0% { box-shadow: 0 0 0 0 var(--accent), 0 0 0 0 var(--accent); }
                     35% { box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 55%, transparent),
                                          0 0 0 8px color-mix(in srgb, var(--accent) 20%, transparent); }
                     100% { box-shadow: 0 0 0 0 transparent, 0 0 0 0 transparent; } }
  li.step.flash { animation: flash 1.1s ease-out; }

  /* Responsive: narrow viewports stack frames under events, progress stays. */
  @media (max-width: 768px) {
    header, .toolbar { padding-left: 14px; padding-right: 14px; }
    header { padding-right: 14px; }
    header .brand { position: static; transform: none; padding-left: 0;
                    border-left: none; margin-left: 0;
                    border-top: 1px solid var(--border-soft); padding-top: 10px; }
    header .brandmark { width: 28px; height: 28px; }
    header .brandname { font-size: 15px; }
    main { padding: 12px; }
    li.step .body { flex-direction: column; }
    li.step .frames { flex: 0 0 auto; max-width: 100%; }
    li.step .params { display: none; }
    .progress .seg { min-width: 4px; height: 12px; }
  }
</style>
<noscript><style>
  /* No JS: every step renders expanded, the inert toolbar buttons are just
     visual. Without this, server-emitted .collapsed would hide bodies. */
  li.step.collapsed .body { display: flex; }
  li.step.collapsed .head { border-bottom-color: var(--border-soft); }
  li.step.collapsed .chevron { transform: none; }
  .toolbar { position: static; }
  .toolbar button { cursor: default; }
</style></noscript>
</head>
<body>
"""


# Theme root-variable blocks. Each maps a colour scheme onto the trace page's
# CSS variables. The stylesheet in _HTML_HEAD is fully token-driven, so a theme
# is just a :root block — no rule changes. --ok/--fail/--warn carry the status
# semantics (success / error / warning); --accent is the brand colour used on
# non-status chrome (header bar, active toolbar button, hover rings, etc.).
_THEMES: dict[str, str] = {
    "default": """  :root { --ok: #00d992; --ok-ink: #101010; --fail: #ff4d4f; --warn: #e0a800;
          --muted: #8b949e; --body: #bdbdbd; --ink: #f2f2f2; --ink-strong: #fff;
          --bg: #101010; --card: #1a1a1a; --card-deep: #0d0d0d;
          --border: #3d3a39; --border-soft: #2a2827; --accent: #00d992;
          --mono: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
          --sans: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans SC", sans-serif; }"""
}


# Inline vanilla JS for the trace page. No dependencies, runs at end of body.
# Handles: (1) filter steps by status, (2) collapse/expand step bodies +
# collapse-all/expand-all, (3) progress-segment → jump to step + flash,
# (4) frame-zoom → open a native <dialog> lightbox, (5) initial hash flash.
# Encapsulated in an IIFE so it never leaks globals; guarded so a missing
# element is a no-op rather than a thrown error.
_TRACE_JS = """<script>
(function(){
  var steps = Array.prototype.slice.call(document.querySelectorAll('li.step'));

  // (2) Collapse/expand: clicking a step head toggles its .collapsed class.
  steps.forEach(function(s){
    var head = s.querySelector('.head');
    if(!head) return;
    head.addEventListener('click', function(ev){
      // Don't toggle when the user clicks a link/badge/etc. inside the head.
      if(ev.target.closest('a, button, .params')) return;
      s.classList.toggle('collapsed');
    });
  });
  var collapseAll = document.getElementById('collapse-all');
  var expandAll = document.getElementById('expand-all');
  if(collapseAll) collapseAll.addEventListener('click', function(){
    steps.forEach(function(s){
      // Keep failed steps expanded even under collapse-all — problems stay visible.
      if(s.classList.contains('fail') && s.querySelector('.error-callout')) return;
      s.classList.add('collapsed');
    });
  });
  if(expandAll) expandAll.addEventListener('click', function(){
    steps.forEach(function(s){ s.classList.remove('collapsed'); });
  });

  // (1) Filter: all/fail/ok toggles which steps are visible.
  var filterBtns = Array.prototype.slice.call(
    document.querySelectorAll('.toolbar button[data-filter]'));
  filterBtns.forEach(function(b){
    b.addEventListener('click', function(){
      filterBtns.forEach(function(x){ x.classList.remove('active'); });
      b.classList.add('active');
      var f = b.getAttribute('data-filter');
      steps.forEach(function(s){
        var st = s.getAttribute('data-status');
        var show = (f === 'all') || (f === st);
        s.classList.toggle('hidden', !show);
      });
    });
  });

  // (5) + (3) Hash jump: scroll to #step-N and flash it. Runs on load and on
  // hashchange. The progress <a href="#step-N"> segments trigger this naturally.
  function flashStep(){
    var id = location.hash.slice(1);
    if(!id) return;
    var el = document.getElementById(id);
    if(!el || !el.classList.contains('step')) return;
    el.classList.remove('collapsed');
    el.classList.remove('flash');
    void el.offsetWidth;          // restart animation
    el.classList.add('flash');
  }
  window.addEventListener('hashchange', flashStep);
  if(document.readyState !== 'loading') flashStep();
  else document.addEventListener('DOMContentLoaded', flashStep);

  // (4) Lightbox: one reusable <dialog>, opened by any .frame-zoom button.
  var lb = document.getElementById('lightbox');
  var lbImg = document.getElementById('lb-img');
  var lbTitle = document.getElementById('lb-title');
  var lbLabel = document.getElementById('lb-label');
  var lbClose = document.getElementById('lb-close');
  if(lb && lbImg){
    Array.prototype.slice.call(document.querySelectorAll('.frame-zoom')).forEach(function(b){
      b.addEventListener('click', function(){
        var img = b.querySelector('img');
        if(!img) return;
        lbImg.src = img.src;
        lbImg.alt = img.alt;
        lbLabel.textContent = b.getAttribute('data-label') || '';
        lbTitle.textContent = 'step ' + (b.getAttribute('data-step') || '?');
        if(typeof lb.showModal === 'function') lb.showModal(); else lb.setAttribute('open','');
      });
    });
    function closeLb(){ if(typeof lb.close === 'function') lb.close(); else lb.removeAttribute('open'); }
    if(lbClose) lbClose.addEventListener('click', closeLb);
    lb.addEventListener('click', function(ev){
      // Click on backdrop (the dialog itself, not its children) closes.
      if(ev.target === lb) closeLb();
    });
    document.addEventListener('keydown', function(ev){
      if((ev.key === 'Escape') && lb.hasAttribute('open')){ ev.preventDefault(); closeLb(); }
    });
  }
})();
</script>
"""


def render_trace_html(data: dict, *, trace_path: Path | None = None, theme: str = "default") -> str:
    """Render a trace dict as a self-contained HTML page.

    Args:
        data: The parsed trace JSON (as produced by :class:`ExecutionTrace`).
        trace_path: Path to the trace JSON, used to resolve relative
            ``frame_path`` values against its directory. May be None, in which
            case only absolute frame paths can be inlined.
        theme: Visual theme name — a key in :data:`_THEMES`. The default
            is a dark dashboard scheme. The DOM and JS are identical across
            themes — only the CSS variable block differs.

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

    try:
        root_vars = _THEMES[theme]
    except KeyError:
        raise ValueError(f"unknown theme {theme!r}; expected one of {sorted(_THEMES)}") from None
    # .replace (not .format) so the stylesheet's own { } braces aren't touched.
    parts: list[str] = [_HTML_HEAD.replace("{root_vars}", root_vars)]
    parts.append("<header>\n")
    parts.append(f'<div class="robot">{_esc_str(robot)}</div>\n')
    parts.append(f'<div class="subtitle">conversation <b>{_esc_str(cid)}</b></div>\n')
    parts.append('<div class="meta">\n')
    if query:
        parts.append(f'<span class="query"><b>query:</b> {_esc_str(query)}</span>\n')
        parts.append('<span class="sep">|</span>\n')
    parts.append(
        f'<span class="summary">{len(entries)} step(s)  ·  '
        f'<span class="mark ok">✅ {n_ok}</span>  ·  '
        f'<span class="mark fail">❌ {n_fail}</span></span>\n'
    )
    parts.append("</div>\n")
    # onerror JS: escape only the SVG's `"` → &quot; (not <>, which the browser
    # would hand JS as literal entity text after a single attribute decode).
    # _LOGO_SVG is single-quote-free (module-load check), so the wrapping
    # '-literal can't close early — the original SyntaxError bug.
    onerror_js = f"this.outerHTML='{_LOGO_SVG.replace(chr(34), '&quot;')}'"
    parts.append(
        f'<div class="brand"><img class="brandmark" src="{_LOGO_SRC}" '
        f'alt="JiuwenSymbiosis" onerror="{onerror_js}">'
        f'<span class="brandname">JiuwenSymbiosis</span></div>\n'
    )
    parts.append("</header>\n")

    # Sticky toolbar: filter (all/fail/ok) + collapse-all/expand-all + a per-step
    # progress bar whose segments jump to the step on click. Hidden without JS via
    # <noscript> (progress segments still render but become inert anchors).
    has_entries = bool(entries)
    parts.append('<div class="toolbar" id="toolbar">\n')
    parts.append('<div class="group"><span class="label">filter</span>')
    parts.append('<button data-filter="all" class="active" type="button">all</button>')
    parts.append('<button data-filter="fail" type="button">fail</button>')
    parts.append('<button data-filter="ok" type="button">ok</button></div>')
    parts.append('<div class="group"><span class="label">view</span>')
    parts.append('<button id="collapse-all" type="button">collapse all</button>')
    parts.append('<button id="expand-all" type="button">expand all</button></div>')
    parts.append('<span class="spacer"></span>')
    if has_entries:
        parts.append('<div class="progress" id="progress" aria-label="step progress">')
        for e in entries:
            seg_cls = "fail" if not e.get("success", True) else ""
            step = e.get("step", "?")
            sname = e.get("tool_name", "?") or "?"
            sdur = e.get("duration_s", 0.0) or 0.0
            tip = f"[{step}] {sname} · {sdur:.2f}s"
            parts.append(f'<a class="seg {seg_cls}" href="#step-{_esc_str(step)}" data-tip="{_esc_str(tip)}"></a>')
        parts.append("</div>\n")
    parts.append("</div>\n")
    # No-JS fallback: the toolbar relies on JS; tell readers without JS it's inert.
    parts.append(
        '<noscript><p style="color:var(--muted);font:12px var(--mono);'
        'text-align:center;padding:8px">toolbar &amp; collapse require JavaScript'
        " — all steps shown expanded below.</p></noscript>\n"
    )

    parts.append("<main>\n")
    parts.append('<ol class="steps">\n')

    # Max duration across steps drives the per-step mini bar width, so the slowest
    # step fills its bar and others scale relative to it. Avoids divide-by-zero.
    max_dur = max((float(e.get("duration_s", 0.0) or 0.0) for e in entries), default=0.0)

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
        # Default collapsed unless this step failed — failed steps stay expanded
        # so problems surface without an extra click. step_ok classes mirror `cls`
        # for the JS filter (data-status) so CSS and JS share one source of truth.
        collapsed = "" if (not success and e.get("error")) else " collapsed"
        dur_pct = int(round((dur / max_dur) * 100)) if max_dur > 0 else 0
        parts.append(f'<li class="step {cls}{collapsed}" id="step-{_esc_str(step)}" data-status="{cls}">\n')
        parts.append(
            f'<div class="head"><span class="chevron">▾</span>'
            f'<span class="num">[{step}]</span>'
            f'<span class="mark {cls}">{mark}</span>'
            f'<span class="name">{_esc_str(name)}</span>'
            f'<span class="params">({_esc(params)})</span>'
            f'<span class="dur"><span class="dur-bar"><i style="width:{dur_pct}%"></i></span>'
            f"dur={dur:.2f}s</span></div>\n"
        )
        parts.append('<div class="body">\n')

        # Frames (left): before/after pair, side by side. Each rendered frame is
        # wrapped in a .frame-zoom button so a click opens the lightbox at full
        # size; a missing frame keeps its dashed placeholder (not clickable).
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
                parts.append(
                    f'<button class="frame-zoom" type="button" '
                    f'data-step="{_esc_str(step)}" data-label="{label}">'
                    f'<img alt="step {step} {label}" src="{uri}"></button>'
                )
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
                f'<div class="error-callout"><span class="label">❌ ERROR</span>{_esc_str(e["error"])}</div>\n'
            )
        obs = e.get("observation")
        if obs and obs.get("pose"):
            parts.append(f'<div class="pose"><span class="label">pose</span> <code>{_esc(obs["pose"])}</code></div>\n')
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

    # Lightbox host: one <dialog> reused for every frame-zoom; JS sets its img src
    # and caption from the clicked button's data attributes.
    parts.append(
        '<dialog class="lightbox" id="lightbox" aria-label="frame viewer">'
        '<div class="lb-head"><span class="lb-label" id="lb-label"></span>'
        '<span class="lb-title" id="lb-title"></span>'
        '<button class="lb-close" id="lb-close" type="button" autofocus>close</button></div>'
        '<img id="lb-img" alt="frame"></dialog>\n'
    )

    parts.append(f"<footer>{len(entries)} step(s) recorded.</footer>\n")

    parts.append(_TRACE_JS)
    parts.append("</body>\n</html>\n")
    return "".join(parts)
