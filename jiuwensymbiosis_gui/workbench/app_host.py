"""Process-owned runtimes and maintenance handles, beyond page lifetimes."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from jiuwensymbiosis.agent.lifecycle import HardwareCleanupError
from jiuwensymbiosis.runtime import CleanupReport, Runtime


class AppHost:
    def __init__(self):
        self._runtimes = {}
        self._tools = []
        self._lock = threading.RLock()
        self._closing = False
        self.current_engine = None

    def runtime(self, workspace):
        key = str(Path(workspace).expanduser().resolve())
        with self._lock:
            if key not in self._runtimes:
                if self._closing:
                    raise RuntimeError("application is closing")
                self._runtimes[key] = Runtime(key)
            return self._runtimes[key]

    def _prune_tools(self):
        # Called under the host lock; tools publish their own disposal boundary.
        self._tools[:] = [tool for tool in self._tools if not tool.can_dispose()]

    def start_tool(self, tool, start=None):
        """Own and start a tool atomically with respect to close and collection."""
        with self._lock:
            if self._closing:
                raise RuntimeError("application is closing")
            self._prune_tools()
            if tool not in self._tools:
                self._tools.append(tool)
            try:
                (start or tool.start)()
            finally:
                self._prune_tools()

    def is_busy(self):
        with self._lock:
            self._prune_tools()
            return (
                self._closing
                or bool(self._tools)
                or any(runtime.get_runtime_state()["busy"] for runtime in self._runtimes.values())
            )

    @property
    def is_closing(self):
        return self._closing

    def close(self, timeout=2.0):
        with self._lock:
            self._closing = True
            self._prune_tools()
            tools = list(self._tools)
            runtimes = list(self._runtimes.values())
        deadline = time.monotonic() + timeout
        errors = []
        for tool in tools:
            try:
                tool.stop()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        for runtime in runtimes:
            runtime.close(timeout=0)
        for tool in tools:
            try:
                tool.join(max(0, deadline - time.monotonic()))
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        results = [runtime.close(timeout=max(0, deadline - time.monotonic())) for runtime in runtimes]
        with self._lock:
            self._prune_tools()
            tools_disposed = not self._tools
        return {
            "closed": not errors and all(item["closed"] for item in results) and tools_disposed,
            "runtimes": results,
            "errors": errors,
        }

    def return_to(self, runtime, binding, joints, emit):
        """Existing return-to-start workflow, owned by this host as maintenance."""
        operation = ReturnToStart(runtime, binding, joints, emit)
        self.start_tool(operation)
        return operation


class ReturnToStart:
    def __init__(self, runtime, binding, joints, emit):
        self.runtime, self.binding, self.joints, self.emit = runtime, binding, list(joints), emit
        self._thread = None
        self._lease = None

    def start(self):
        self._lease = self.runtime.acquire_maintenance(self.binding, "return_to_start", stop=self.stop)
        self._thread = threading.Thread(target=self._run, name="jiuwen-return", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self.runtime.finish_maintenance(self._lease, CleanupReport())
            raise

    def _run(self):
        session = None
        outcome = {"ok": False}
        self.emit("pose_return_started", {"joints": self.joints})
        report = CleanupReport()
        errors = ()
        try:
            session = self.binding.build_session()
            with session:
                # Maintenance return: driver checks apply, with no SafetyRail
                # precheck. A previously observed endpoint does not prove the
                # return path is clear; the workbench asks the operator first.
                session.env.low_level.move_joint_blocking(self.joints)
            outcome = {"ok": True}
        except Exception as exc:
            outcome = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if isinstance(exc, HardwareCleanupError):
                errors = (str(exc),)
        finally:
            if session is not None:
                try:
                    session.disconnect()
                    report = session.cleanup_report()
                except Exception as exc:
                    errors += (f"{type(exc).__name__}: {exc}",)
            report = CleanupReport(report.pending_work, report.errors + errors, report.connected)
            released = self.runtime.finish_maintenance(self._lease, report)
            if not released:
                outcome = {"ok": False, "error": "硬件释放未确认，请检查占用记录。"}
            self.emit("pose_return_finished", outcome)

    @staticmethod
    def stop():
        # This existing driver call has no safe interrupt contract. Keep its
        # lease until it returns and disconnect is confirmed.
        return None

    def is_running(self):
        return bool(self._thread and self._thread.is_alive())

    def can_dispose(self):
        """Host ownership ends only after worker exit and confirmed release."""
        return not self.is_running() and (self._lease is None or self._lease.released)

    def join(self, timeout=None):
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout)
