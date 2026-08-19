# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Main-process side of the ROS 2 subprocess-worker pattern.

A ROS 2 body usually cannot talk to rclpy in-process: rclpy is built against the
system interpreter, while the agent runs under conda. The way out is the same for
every such body — put the ROS work in a small script, run it under the system
python, and speak **one line of JSON on stdout** back to the agent. This module is
the agent-side half of that protocol, so an adapter states *what* to run instead of
re-writing *how* to run it.

Two lifetimes, both here:

* **one-shot** (:func:`run_once`) — spawn, wait, read the last stdout line. Used for
  a blocking command that ends by itself.
* **resident** (:class:`ResidentWorker`) — keep a ``--serve`` worker alive and send
  it one request line per call, so rclpy/DDS discovery is paid once instead of per
  call. :func:`stop_and_collect` ends a self-bounding worker (one started to run
  until told to stop) and reaps its result.

The worker script itself is NOT covered here. Workers are loaded by file path under
another interpreter and cannot import this package (``jiuwensymbiosis/__init__.py``
eagerly imports openjiuwen), so their side of the protocol — one ``json.dumps`` plus
a flush — stays copied in each worker rather than shared through a second
file-path bootstrap.
"""

from __future__ import annotations

import json
import select
import subprocess
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)


def worker_path(module: str) -> Path:
    """Filesystem path of a worker module, for running it under another interpreter.

    Args:
        module: dotted module name, e.g. ``"jiuwensymbiosis.adapters.cruzr.ros2.wheel_worker"``.

    Raises:
        ModuleNotFoundError: if the module cannot be located — a typo would otherwise
            surface much later as a subprocess that exits non-zero.
    """
    spec = find_spec(module)
    if spec is None or not spec.origin:
        raise ModuleNotFoundError(f"cannot locate ROS 2 worker module {module!r}")
    return Path(spec.origin)


def read_line(proc: subprocess.Popen, timeout_s: float) -> str | None:
    """One stripped stdout line from a running worker, or ``None`` on timeout / EOF.

    ``select`` rather than a bare ``readline`` so a wedged worker cannot block the
    agent forever.
    """
    if proc.stdout is None:
        return None
    ready, _, _ = select.select([proc.stdout], [], [], timeout_s)
    if not ready:
        return None
    line = proc.stdout.readline()
    return line.strip() if line else None


def _last_json_line(text: str) -> dict | None:
    """Parse the last line of ``text`` as a JSON object, or ``None`` if it isn't one.

    Workers may log to stdout before their result, so only the final line is the
    payload.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped.splitlines()[-1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def run_once(
    cmd: list[str],
    *,
    timeout_s: float,
    label: str,
    reason_prefix: str = "",
    env: dict[str, str] | None = None,
) -> dict:
    """Run a worker to completion and return its result dict.

    Every failure converges to ``{"ok": False, "reason": ...}`` rather than raising,
    because the caller is a driver method whose contract is a structured result.

    Args:
        cmd: full argv, starting with the interpreter that can import rclpy.
        timeout_s: hard wall-clock cap on the subprocess.
        label: log prefix identifying the caller, e.g. ``"[CruzrNav] wheel"``.
        reason_prefix: prepended to the failure reasons, so one body's worker
            failures stay distinguishable from another's (``"wheel_"`` →
            ``wheel_worker_error`` / ``wheel_worker_failed`` / ``wheel_no_output`` /
            ``wheel_bad_output``).
        env: environment for the subprocess; ``None`` inherits the agent's.
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
    except Exception as exc:  # noqa: BLE001 - converge any spawn/timeout error to a structured result
        logger.warning("%s worker run failed: %s", label, exc)
        return {"ok": False, "reason": f"{reason_prefix}worker_error"}
    if proc.returncode != 0:
        logger.warning("%s worker rc=%d stderr=%s", label, proc.returncode, (proc.stderr or "").strip())
        return {"ok": False, "reason": f"{reason_prefix}worker_failed"}
    result = _last_json_line(proc.stdout or "")
    if result is None:
        if not (proc.stdout or "").strip():
            return {"ok": False, "reason": f"{reason_prefix}no_output"}
        return {"ok": False, "reason": f"{reason_prefix}bad_output"}
    return result


def stop_and_collect(
    proc: subprocess.Popen,
    *,
    label: str,
    kind: str,
    empty_result: dict,
    timeout_s: float = 15.0,
    kill_timeout_s: float = 5.0,
) -> dict:
    """Halt a self-bounding worker and reap its JSON result.

    Sends the ``stop`` sentinel for a clean actuator stop while the worker still
    runs, else just drains a finished one. Falls back to SIGTERM and then SIGKILL,
    so a wedged worker can never be left driving hardware.

    Args:
        label: log prefix identifying the caller.
        kind: names the failure reasons — ``{kind}_stop_failed`` / ``{kind}_bad_output``.
        empty_result: returned when the worker exits without printing anything (it
            did nothing, which is a success with a zero-valued measurement).
    """
    try:
        payload = "stop\n" if proc.poll() is None else None
        out, _err = proc.communicate(input=payload, timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 - timeout / broken pipe → force it down safely
        logger.warning("%s stop fallback (%s): terminating worker", label, exc)
        proc.terminate()
        try:
            out, _err = proc.communicate(timeout=kill_timeout_s)
        except Exception as kill_exc:  # noqa: BLE001 - last resort; the process must not survive
            logger.error("%s worker ignored SIGTERM (%s); killing", label, kill_exc)
            proc.kill()
            return {"ok": False, "reason": f"{kind}_stop_failed"}
    if not (out or "").strip():
        return dict(empty_result)
    result = _last_json_line(out or "")
    return result if result is not None else {"ok": False, "reason": f"{kind}_bad_output"}


class ResidentWorker:
    """A ``--serve`` worker kept alive across calls, driven one request line at a time.

    Restarting a worker per call means paying rclpy import + DDS discovery every
    time; keeping one warm turns that into a one-off. The cost is that a resident
    worker can die between calls, so :meth:`request` fails soft: it drops the dead
    handle and returns ``None``, letting the caller retry or fall back to a one-shot
    run rather than raising into a motion path.
    """

    def __init__(
        self,
        make_cmd: Callable[[], list[str]],
        *,
        label: str,
        env_fn: Callable[[list[str]], dict[str, str] | None] | None = None,
    ) -> None:
        """
        Args:
            make_cmd: builds the full ``--serve`` argv; called again on each restart.
            label: log prefix identifying the caller.
            env_fn: optional environment for the subprocess, derived from the argv.
        """
        self._make_cmd = make_cmd
        self._label = label
        self._env_fn = env_fn
        self._proc: subprocess.Popen | None = None

    @property
    def proc(self) -> subprocess.Popen | None:
        """The live handle, or ``None`` when no worker is currently running."""
        return self._proc if self._proc is not None and self._proc.poll() is None else None

    def _ensure(self) -> subprocess.Popen | None:
        """Return a live worker, starting one if absent or dead; ``None`` if it won't start."""
        if self.proc is not None:
            return self._proc
        cmd = self._make_cmd()
        try:
            # stderr → DEVNULL: a long-lived worker would otherwise block on a full pipe.
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env=self._env_fn(cmd) if self._env_fn is not None else None,
            )
        except Exception as exc:  # noqa: BLE001 - a worker that won't start is a soft failure
            logger.warning("%s resident worker start failed: %s", self._label, exc)
            self._proc = None
            return None
        return self._proc

    def request(self, line: str, timeout_s: float) -> str | None:
        """Send one request line and read back one reply line; ``None`` on any failure.

        A failure also drops the worker, so the next call starts a fresh one — a
        half-dead worker must not be reused for a motion command.
        """
        proc = self._ensure()
        if proc is None or proc.stdin is None:
            return None
        try:
            proc.stdin.write(line if line.endswith("\n") else line + "\n")
            proc.stdin.flush()
            reply = read_line(proc, timeout_s)
        except Exception as exc:  # noqa: BLE001 - broken pipe / dead worker → drop and report
            logger.warning("%s resident request failed: %s", self._label, exc)
            self.stop()
            return None
        if not reply:
            logger.warning("%s resident worker gave no reply", self._label)
            self.stop()
            return None
        return reply

    def request_json(self, line: str, timeout_s: float, *, bad_output_reason: str) -> dict | None:
        """:meth:`request` plus JSON parsing.

        ``None`` means "no usable worker" (the caller should fall back or retry); a
        dict with ``ok=False`` means the worker answered but unintelligibly — a live
        worker talking nonsense is not a reason to restart it.
        """
        reply = self.request(line, timeout_s)
        if reply is None:
            return None
        try:
            parsed = json.loads(reply)
        except json.JSONDecodeError:
            logger.warning("%s resident worker produced invalid JSON", self._label)
            return {"ok": False, "reason": bad_output_reason}
        return parsed if isinstance(parsed, dict) else {"ok": False, "reason": bad_output_reason}

    def stop(self) -> None:
        """Shut the worker down and forget it. Idempotent."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.communicate(input="stop\n", timeout=5.0)
        except Exception as exc:  # noqa: BLE001 - force it down; the process must not survive
            logger.warning("%s resident worker ignored stop (%s); terminating", self._label, exc)
            try:
                proc.terminate()
            except Exception as term_exc:  # noqa: BLE001 - already gone
                logger.debug("%s terminate after failed stop: %s", self._label, term_exc)
