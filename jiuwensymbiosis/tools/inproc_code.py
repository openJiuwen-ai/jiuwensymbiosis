# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""In-process Python execution as an openjiuwen Tool.

openjiuwen's built-in ``CodeTool`` (harness/tools/code.py) runs code in a
sandbox subprocess via ``SysOperation.code()``, which CANNOT see hot
in-memory objects on the agent side — and that is precisely what we need
for robotics: a live ``env`` handle to a connected robot, a detection client, an
already-warm RealSense pipeline.

This tool exec()s the code in the agent process with caller-injected
globals. It is intentionally NOT sandboxed; the agent code has the same
trust as the agent itself. The implementation is a simple in-process executor so prompts
that rely on ``env`` / ``APIS`` / ``RESULT`` semantics work unchanged.
"""

from __future__ import annotations

import contextlib
import io
import time
import traceback
from collections.abc import Callable
from typing import Any

from jiuwensymbiosis.agent.abstractions import LocalFunction, ToolCard


class InProcessCodeExec:
    """The exec engine without the openjiuwen tool wrapper. Useful for tests."""

    def __init__(self, globals_provider: Callable[[], dict[str, Any]]):
        """Store the globals provider callable."""
        self.gp = globals_provider

    def run(self, code: str) -> dict[str, Any]:
        """Execute Python code in-process and return the result dict."""
        g: dict[str, Any] = {**self.gp(), "__name__": "__main__", "RESULT": None}
        out, err = io.StringIO(), io.StringIO()
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(code, g, g)
            return {
                "ok": True,
                "stdout": out.getvalue(),
                "stderr": err.getvalue(),
                "result": g.get("RESULT"),
                "elapsed_s": round(time.time() - t0, 4),
            }
        except BaseException as exc:  # noqa: BLE001 - we want to surface anything
            tb = traceback.format_exc()
            return {
                "ok": False,
                "stdout": out.getvalue(),
                "stderr": err.getvalue() + "\n" + tb,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.time() - t0, 4),
            }


def make_inproc_code_tool(
    globals_provider: Callable[[], dict[str, Any]],
    *,
    name: str = "run_python",
    description: str | None = None,
) -> Any:
    """Build an openjiuwen ``LocalFunction`` Tool that exec()s Python in-process.

    The tool exposes a single string input ``code``. The agent should write
    multi-statement Python that calls ``api`` / ``env`` and assigns to
    ``RESULT`` if there is a meaningful return value.
    """
    exec_engine = InProcessCodeExec(globals_provider)

    description = description or (
        "Execute a block of Python in the agent process with live access to "
        "`api` (the robot control API) and `env` (the underlying hardware env). "
        "Assign final value to RESULT. stdout and stderr are captured. "
        "Use this for multi-step control flow that would be awkward as separate tool calls."
    )

    def _func(code: str) -> dict:
        """Run code via the in-process exec engine."""
        return exec_engine.run(code)

    card = ToolCard(
        name=name,
        description=description,
        input_params={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to exec()."},
            },
            "required": ["code"],
        },
    )
    return LocalFunction(card=card, func=_func)


# Convenience wrapper for tests / direct use without openjiuwen.
class InProcessCodeTool:
    """Thin facade: ``InProcessCodeTool(provider).run(code)`` -> result dict."""

    def __init__(self, globals_provider: Callable[[], dict[str, Any]]):
        """Wrap an InProcessCodeExec with the given provider."""
        self._engine = InProcessCodeExec(globals_provider)

    def run(self, code: str) -> dict[str, Any]:
        """Execute code and return the result dict."""
        return self._engine.run(code)

    def as_openjiuwen_tool(self, **kw) -> Any:
        """Build an openjiuwen LocalFunction from this engine."""
        return make_inproc_code_tool(self._engine.gp, **kw)
