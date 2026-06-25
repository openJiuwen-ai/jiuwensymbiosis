# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Lightweight CLI entry points registered as ``console_scripts`` in pyproject.toml."""

from __future__ import annotations

import argparse
import json
import logging
import runpy
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None

logger = logging.getLogger(__name__)

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _run_example(script: str) -> None:
    script_path = _EXAMPLES_DIR / script
    if not script_path.is_file():
        logger.error("Example script not found: %s", script_path)
        raise FileNotFoundError(script_path)
    runpy.run_path(str(script_path), run_name="__main__")


def piper_pick_demo() -> None:
    _run_example("piper_pick_demo.py")


def piper_watch_pick_place() -> None:
    _run_example("piper_watch_pick_place.py")


# ---------------------------------------------------------------------------
# Replay CLI — renders a recorded execution trace.
# (Issue #9) Usage: ``jiuwensymbiosis replay <trace.json>``
#
# By default a self-contained HTML page is written next to the trace JSON and
# its path is printed — every step's frame is inlined as base64 so the image
# and that step's params/error/rail events live in one card. The path is
# clickable in VSCode's terminal (opens in the built-in webview), which works
# the same for local and Remote-SSH/MobaXterm workflows without needing a
# browser on the trace host. ``--open`` additionally launches the OS default
# browser via ``xdg-open``/``open``/``startfile`` (only useful on a host that
# actually has a desktop — headless remote dev should rely on the clickable
# path). ``--text`` falls back to the original plain-text timeline (frames
# shown as paths only, no popup).
# ---------------------------------------------------------------------------
def _fmt_params(params: Any) -> str:
    try:
        s = json.dumps(params, ensure_ascii=False)
    except (TypeError, ValueError):
        s = repr(params)
    return s if len(s) <= 120 else s[:117] + "..."


def _load_trace(trace_path: str) -> Optional[tuple[dict, Path]]:
    """Load and validate a trace JSON. Returns ``(data, path)`` or None on error.

    Shared by the text and HTML replay paths so both surface the same
    "missing / invalid" diagnostics. Errors are printed to stderr.
    """
    path = Path(trace_path)
    if not path.is_file():
        print(f"trace not found: {path}", file=sys.stderr)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"invalid trace JSON: {exc}", file=sys.stderr)
        return None
    return data, path


def _open_in_viewer(path: str) -> bool:
    """Open ``path`` in the OS default application (image viewer or browser).

    Used by ``replay_html`` to open the generated HTML in the default browser.
    Uses only stdlib (no optional deps). Linux → ``xdg-open``, macOS → ``open``,
    Windows → ``os.startfile``. Failures are swallowed (replay must not crash
    on a missing opener); the caller prints the path either way as a fallback.
    """
    import os
    import platform
    import subprocess

    try:
        if platform.system() == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])  # nosec - user-supplied path on host
        else:
            subprocess.Popen(["xdg-open", path])  # nosec - user-supplied path on host
        return True
    except (OSError, FileNotFoundError, ValueError):
        return False


def replay(trace_path: str, *, out=sys.stdout) -> int:
    """Print a recorded execution trace as a text timeline.

    Args:
        trace_path: Path to a ``traces/*.json`` written by ``TraceRail``.
        out: Output stream (defaults to stdout).

    Returns:
        Process exit code (0 on success, 1 if the file is missing/invalid).
    """
    loaded = _load_trace(trace_path)
    if loaded is None:
        return 1
    data, path = loaded

    cid = data.get("conversation_id", "?")
    robot = data.get("robot_name", "?")
    query = data.get("query")
    entries = data.get("entries", [])
    trace_log = data.get("trace_log", [])

    print(f"=== Execution Trace: {path.name} ===", file=out)
    print(f"robot={robot}  conversation={cid}", file=out)
    if query:
        print(f"query: {query}", file=out)
    print("", file=out)

    if not entries:
        print("(no tool-call steps recorded)", file=out)
    for e in entries:
        step = e.get("step", "?")
        name = e.get("tool_name", "?")
        params = e.get("input_params", {})
        success = e.get("success", True)
        dur = e.get("duration_s", 0.0)
        mark = "✅" if success else "❌"
        print(f"[{step:>3}] {mark} {name}({_fmt_params(params)})", file=out)
        print(f"       dur={dur:.2f}s", file=out)
        if e.get("error"):
            print(f"       error: {e['error']}", file=out)
        obs = e.get("observation")
        if obs and obs.get("pose"):
            print(f"       pose: {obs['pose']}", file=out)
        if e.get("frame_path"):
            fp = e["frame_path"]
            extra = ""
            if _PILImage is not None:
                try:
                    with _PILImage.open(fp) as img:
                        extra = f" ({img.size[0]}x{img.size[1]})"
                except (OSError, ValueError):
                    pass
            print(f"       frame: {fp}{extra}", file=out)
        for ev in e.get("rail_events", []) or []:
            status = "ok" if ev.get("success") else "FAIL"
            print(
                f"       rail: [{status}] {ev.get('rail_name')}/{ev.get('kind')} "
                f"{ev.get('detail', {})}",
                file=out,
            )
        for ev in e.get("log_events", []) or []:
            print(
                f"       log:  [{ev.get('level')}] {ev.get('logger')}: {ev.get('msg')}",
                file=out,
            )

    if trace_log:
        print("\n--- trace-level logs ---", file=out)
        for ev in trace_log:
            print(
                f"  [{ev.get('level')}] {ev.get('logger')}: {ev.get('msg')}",
                file=out,
            )
    print(f"\n{len(entries)} step(s) recorded.", file=out)
    return 0


def replay_html(trace_path: str, *, open_browser: bool = False, out=sys.stdout) -> int:
    """Render a trace as a self-contained HTML page and print its path.

    The HTML is written next to the trace JSON as ``{json_stem}.html`` with
    every step's frame inlined as base64 — so the page is portable and shows
    each frame beside its params/error/rail events. The written path is
    printed; in VSCode's terminal it's clickable and opens in the built-in
    webview, which works identically for local and Remote-SSH/MobaXterm
    workflows (no browser needed on the trace host). If the trace directory
    isn't writable, the file falls back to the system temp directory.

    Args:
        trace_path: Path to a ``traces/*.json`` written by ``TraceRail``.
        open_browser: When False (default), don't launch anything — just write
            the file and print the path. When True, also call ``xdg-open`` /
            ``open`` / ``startfile`` on the generated HTML; only useful on a
            host that has a desktop. Set via the ``--open`` CLI flag.
        out: Output stream for the "wrote …" status line.

    Returns:
        Process exit code (0 on success, 1 if the trace is missing/invalid).
    """
    loaded = _load_trace(trace_path)
    if loaded is None:
        return 1
    data, path = loaded

    from jiuwensymbiosis.agent.trace_html import render_trace_html

    html_str = render_trace_html(data, trace_path=path)
    out_path = path.with_suffix(".html")
    try:
        out_path.write_text(html_str, encoding="utf-8")
    except OSError:
        import tempfile

        out_path = Path(tempfile.gettempdir()) / out_path.name
        try:
            out_path.write_text(html_str, encoding="utf-8")
        except OSError as exc:
            print(f"could not write HTML: {exc}", file=sys.stderr)
            return 1

    print(f"wrote {out_path}", file=out)
    if open_browser:
        print("opening browser…", file=out)
        if not _open_in_viewer(str(out_path)):
            print("(could not open browser; path above is clickable)", file=out)
    else:
        print("(open the path above in your browser; "
              "in VSCode it's clickable and opens in the built-in webview)", file=out)
    return 0


def replay_main() -> None:
    """Console-script entry: ``jiuwensymbiosis replay <trace_path>``."""
    parser = argparse.ArgumentParser(
        prog="jiuwensymbiosis-replay",
        description="Replay a recorded jiuwensymbiosis execution trace.",
    )
    parser.add_argument("trace_path", help="Path to a traces/*.json file")
    parser.add_argument(
        "--text",
        action="store_true",
        help="Print a text timeline instead of generating HTML.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML in the OS default browser. Only useful "
             "on a host with a desktop; for headless remote dev, rely on the "
             "clickable path printed instead.",
    )
    args = parser.parse_args()
    if args.text:
        raise SystemExit(replay(args.trace_path))
    raise SystemExit(replay_html(args.trace_path, open_browser=args.open))
