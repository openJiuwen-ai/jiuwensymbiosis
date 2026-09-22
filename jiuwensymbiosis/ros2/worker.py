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
  until told to stop) and reaps its result. A resident request fails soft only
  when its worker can be stopped cleanly; an unconfirmed stop propagates so a
  caller does not issue a fallback command while the prior command may remain active.

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

from jiuwensymbiosis.agent.lifecycle import HardwareCleanupError
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

    Worker-reported action failures remain structured results. Once a process
    starts, timeout/abnormal exit/missing results leave actuator stop unconfirmed
    and raise HardwareCleanupError. The owning driver must retain that evidence.

    Args:
        cmd: full argv, starting with the interpreter that can import rclpy.
        timeout_s: hard wall-clock cap on the subprocess.
        label: log prefix identifying the caller, e.g. ``"[CruzrNav] wheel"``.
        reason_prefix: prepended to the spawn failure reason, e.g.
            ``"wheel_"`` → ``wheel_worker_error``. Unconfirmed stops raise.
        env: environment for the subprocess; ``None`` inherits the agent's.
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired as exc:
        # subprocess.run kills and reaps the child on timeout; SIGKILL cannot
        # execute the worker's final zero-velocity command.
        raise HardwareCleanupError(label, cleanup_errors=(exc,)) from exc
    except OSError as exc:
        logger.warning("%s worker run failed: %s", label, exc)
        return {"ok": False, "reason": f"{reason_prefix}worker_error"}
    if proc.returncode != 0:
        logger.warning("%s worker rc=%d stderr=%s", label, proc.returncode, (proc.stderr or "").strip())
        error = RuntimeError(f"worker exited with rc={proc.returncode}; actuator stop unconfirmed")
        raise HardwareCleanupError(label, cleanup_errors=(error,))
    result = _last_json_line(proc.stdout or "")
    if result is None:
        error = RuntimeError("worker returned no valid result; actuator stop unconfirmed")
        raise HardwareCleanupError(label, cleanup_errors=(error,))
    return result


def stop_and_collect(
    proc: subprocess.Popen,
    *,
    label: str,
    kind: str,
    timeout_s: float = 15.0,
    kill_timeout_s: float = 5.0,
) -> dict:
    """Halt a self-bounding worker and reap its JSON result.

    Sends the ``stop`` sentinel for a clean actuator stop while the worker still
    runs, else just drains a finished one. A clean exit with a JSON result
    confirms the worker's stop path ran. Forced termination only reaps the
    process; it cannot prove that the physical actuator stopped.

    Args:
        label: log prefix identifying the caller.
        kind: operation name included in cleanup errors (e.g. spin or drive).
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
            errors: list[BaseException] = [kill_exc]
            try:
                proc.kill()
                proc.communicate(timeout=kill_timeout_s)
            except Exception as reap_exc:
                errors.append(reap_exc)
            raise HardwareCleanupError(f"{label} {kind} stop", cleanup_errors=errors) from kill_exc
    result = _last_json_line(out or "")
    if proc.returncode != 0 or result is None:
        error = RuntimeError(f"{kind} worker rc={proc.returncode}, no confirmed actuator stop")
        raise HardwareCleanupError(label, cleanup_errors=(error,))
    return result


class ResidentWorker:
    """A ``--serve`` worker kept alive across calls, driven one request line at a time.

    Restarting a worker per call means paying rclpy import + DDS discovery every
    time; keeping one warm turns that into a one-off. The cost is that a resident
    worker can die between calls. A cleanly stopped failed request returns
    ``None`` and lets a caller retry or fall back to a one-shot run. If its
    shutdown cannot be confirmed, the cleanup error propagates into the caller's
    lifecycle so it cannot issue a second hardware command blindly.
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
        self._stop_uncertainty: BaseException | None = None

    @property
    def proc(self) -> subprocess.Popen | None:
        """The live handle, or ``None`` when no worker is currently running."""
        return self._proc if self._proc is not None and self._proc.poll() is None else None

    def _ensure(self) -> subprocess.Popen | None:
        """Return a reusable worker, starting one only after confirmed cleanup."""
        if self._stop_uncertainty is not None:
            raise RuntimeError(
                f"{self._label} resident worker shutdown remains unconfirmed"
            ) from self._stop_uncertainty
        if self.proc is not None:
            return self._proc
        if self._proc is not None:
            # Do not overwrite an unexpectedly dead handle and lose its missing
            # stop confirmation. stop() records that uncertainty and raises.
            self.stop()
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

        A failure also stops the worker before a caller can fall back. If the
        worker's stop cannot be confirmed, that cleanup error propagates and the
        worker remains marked uncertain; motion callers must not start a second
        command while the first worker's outcome is unknown.
        """
        proc = self._ensure()
        if proc is None:
            return None
        if proc.stdin is None:
            logger.warning("%s resident worker has no stdin pipe", self._label)
            self.stop()
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

        ``None`` means "no usable worker" after its stop was confirmed; a
        dict with ``ok=False`` means the worker answered but unintelligibly — a
        live worker talking nonsense is not a reason to restart it. An unconfirmed
        stop propagates so callers cannot fall back to another command path.
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
        """Stop and reap the worker, retaining evidence if shutdown is uncertain."""
        if self._stop_uncertainty is not None and self._proc is None:
            raise RuntimeError(
                f"{self._label} resident worker shutdown remains unconfirmed"
            ) from self._stop_uncertainty
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is not None:
            self._proc = None
            error = RuntimeError(f"{self._label} resident worker exited before a stop was confirmed")
            self._stop_uncertainty = error
            raise error

        try:
            proc.communicate(input="stop\n", timeout=5.0)
        except Exception as exc:  # noqa: BLE001 - force it down; the process must not survive
            logger.warning("%s resident worker ignored stop (%s); terminating", self._label, exc)
            escalation_errors: list[BaseException] = []
            try:
                proc.terminate()
            except Exception as term_exc:  # noqa: BLE001 - retain the failure as shutdown evidence
                escalation_errors.append(term_exc)
            try:
                proc.communicate(timeout=5.0)
            except Exception as wait_exc:  # noqa: BLE001 - last resort; process must not survive
                escalation_errors.append(wait_exc)
                try:
                    proc.kill()
                except Exception as kill_exc:  # noqa: BLE001 - retain the handle if kill failed
                    escalation_errors.append(kill_exc)
                try:
                    proc.communicate(timeout=5.0)
                except Exception as kill_wait_exc:  # noqa: BLE001
                    escalation_errors.append(kill_wait_exc)
            if proc.poll() is None:
                error = RuntimeError(f"{self._label} resident worker is still alive after terminate/kill")
                # Keep the handle available for stop retries, never new requests.
                self._stop_uncertainty = error
                raise error from exc
            # Forced process termination proves the process is gone, but not that
            # its outstanding robot command reached a safe stop. Keep this worker
            # permanently marked uncertain so resource owners fail closed.
            self._proc = None
            detail = "; ".join(f"{type(item).__name__}: {item}" for item in escalation_errors)
            error = RuntimeError(
                f"{self._label} resident worker required forced termination; stop is unconfirmed"
                + (f" ({detail})" if detail else "")
            )
            self._stop_uncertainty = error
            raise error from exc

        if proc.poll() is None:
            error = RuntimeError(f"{self._label} resident worker did not exit after its stop command")
            self._stop_uncertainty = error
            raise error
        if proc.returncode != 0:
            self._proc = None
            error = RuntimeError(f"{self._label} resident worker exited with status {proc.returncode} after stop")
            self._stop_uncertainty = error
            raise error
        self._proc = None
        self._stop_uncertainty = None
