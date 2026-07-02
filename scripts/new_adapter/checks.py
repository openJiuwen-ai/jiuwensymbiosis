# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""checks — run the existing validator / smoke test against a generated adapter.

All checks shell out with ``PYTHONPATH=<repo_root>`` and ``cwd=<repo_root>`` so
they always exercise *this* repo (robust even when the editable install points
elsewhere) and see the freshly written adapter package.

``run_smoke`` connects the (mock) env first, then calls ``smoke_test_api`` — the
plain ``smoke_test_adapter`` CLI deliberately does NOT connect (real hardware
would block), so an offline mock adapter only goes green once connected.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_VALIDATE = REPO_ROOT / "scripts" / "validate_adapter.py"

_SMOKE_MARKER = "@@SMOKE@@"

# Connect the (mock) env, run smoke_test_api against the gated tools, report JSON.
_SMOKE_HARNESS = f"""
import importlib, json, sys
mod = sys.argv[1]
pkg = importlib.import_module(mod)
builder = None
for attr in dir(pkg):
    if attr.startswith("build_") and attr.endswith("_session"):
        builder = getattr(pkg, attr)
        break
if builder is None:
    print("{_SMOKE_MARKER}" + json.dumps({{"error": "no builder"}}))
    sys.exit(2)
session = builder.from_dict({{}})
env = session.env
env.connect()
try:
    from scripts.smoke_test_adapter import smoke_test_api
    results = smoke_test_api(session.api, env=env)
finally:
    try:
        env.disconnect()
    except Exception:
        pass
fails = [r for r in results if r.get("status") == "fail"]
print(
    "{_SMOKE_MARKER}"
    + json.dumps({{"fails": len(fails), "fail_names": [r["name"] for r in fails], "total": len(results)}})
)
sys.exit(1 if fails else 0)
"""


@dataclass
class Result:
    """Outcome of one check."""

    ok: bool
    title: str
    detail: str = ""


def format_with_black(paths: list[Path]) -> bool:
    """Best-effort: run black (line-length 100) on the generated files.

    Returns True on success; False if black is unavailable or errors (the files
    are still valid Python, just not auto-formatted).
    """
    if not paths:
        return True
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "black",
                "--line-length",
                "100",
                "--quiet",
                *(str(p) for p in paths),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(REPO_ROOT)}
    # Inherit the rest of the environment (PATH, conda, etc.).
    import os

    full_env = dict(os.environ)
    full_env.update(env)
    return subprocess.run(
        args,
        cwd=str(REPO_ROOT),
        env=full_env,
        capture_output=True,
        text=True,
    )


def run_validate(module: str) -> Result:
    """Static structural validation (validate_adapter). ok ⇔ 0 ERROR."""
    proc = _run([sys.executable, str(_VALIDATE), "--module", module])
    ok = proc.returncode == 0
    # Surface the validator's own summary/error lines, dropping noisy framework logs.
    detail = _tail_report(proc.stdout)
    return Result(ok=ok, title="validate (静态结构)", detail=detail)


def run_smoke(module: str) -> Result:
    """Runtime smoke: connect mock env, call every gated tool. ok ⇔ 0 FAIL."""
    proc = _run([sys.executable, "-c", _SMOKE_HARNESS, module])
    payload = _extract_marker(proc.stdout)
    if payload is None:
        return Result(
            ok=False,
            title="smoke (运行时)",
            detail=(proc.stdout[-400:] + proc.stderr[-400:]).strip() or "无输出",
        )
    fails = payload.get("fails", 0)
    total = payload.get("total", 0)
    if fails:
        detail = f"{fails}/{total} FAIL: {', '.join(payload.get('fail_names', []))}"
    else:
        detail = f"{total} 个工具全部通过"
    return Result(ok=(fails == 0), title="smoke (运行时)", detail=detail)


def _extract_marker(stdout: str):
    marker_len = len(_SMOKE_MARKER)
    for line in stdout.splitlines():
        if line.startswith(_SMOKE_MARKER):
            try:
                return json.loads(line[marker_len:])
            except json.JSONDecodeError:
                return None
    return None


def _keep_report_line(ln: str) -> bool:
    """Keep a validator result/error line; drop framework INFO log spam."""
    if "| INFO |" in ln:
        return False
    return (
        "结果:" in ln
        or "ERROR" in ln
        or "[E-" in ln
        or "[C-" in ln
        or "[A-" in ln
        or "[D-" in ln
    )


def _tail_report(stdout: str) -> str:
    """Keep the validator's result/error lines, drop framework INFO log spam."""
    keep = [ln for ln in stdout.splitlines() if _keep_report_line(ln)]
    return "\n".join(keep[-12:]).strip()


# ---------------------------------------------------------------------------
# Sentinel scan — which driver/vision methods are still generated mocks
# ---------------------------------------------------------------------------

_SENTINEL = "# >>> GENERATED-MOCK: replace with real hardware <<<"


def scan_pending(adapter_dir: Path) -> dict[str, list[str]]:
    """Return {relative file → [method names]} still carrying the mock sentinel."""
    pending: dict[str, list[str]] = {}
    for path in sorted(adapter_dir.glob("*.py")):
        methods = _methods_with_sentinel(path)
        if methods:
            pending[path.name] = methods
    return pending


def _methods_with_sentinel(path: Path) -> list[str]:
    methods: list[str] = []
    current = "<module>"
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and "(" in stripped:
            current = stripped[4:].split("(", 1)[0]
        elif _SENTINEL in line and current not in methods:
            methods.append(current)
    return methods
