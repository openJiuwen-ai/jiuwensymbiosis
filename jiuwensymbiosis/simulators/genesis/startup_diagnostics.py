"""Small, SDK-free startup measurements for the isolated Genesis worker."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import version
from typing import Any

_PACKAGES = ("genesis-world", "quadrants", "gs-nyx", "gs-nyx-plugin", "torch")
_CACHE_FIELDS = (
    "offline_cache",
    "offline_cache_file_path",
    "offline_cache_max_size_of_files",
    "offline_cache_cleaning_policy",
    "offline_cache_cleaning_factor",
)


def package_versions() -> dict[str, str | None]:
    """Read installed metadata once, without importing GPU or agent packages."""
    result: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            result[name] = version(name)
        except Exception:
            # Metadata can be absent or unreadable; it is diagnostic evidence,
            # never a prerequisite for importing and running the installed SDK.
            result[name] = None
    return result


def quadrants_cache_config(quadrants: Any) -> dict[str, Any]:
    """Read the active program config after gs.init, never infer it from env."""
    try:
        cfg = quadrants.lang.impl.current_cfg()
        values = {name: getattr(cfg, name) for name in _CACHE_FIELDS}
        # Refuse opaque SDK objects instead of stringifying arbitrary state.
        if any(type(value) not in (bool, int, float, str) for value in values.values()):
            raise TypeError("Unexpected Quadrants cache config value")
        return {"available": True, "source": "quadrants.lang.impl.current_cfg", **values}
    except Exception as exc:
        # Diagnostics must not prevent an otherwise supported SDK from starting.
        return {"available": False, "reason": type(exc).__name__}


class StartupDiagnostics:
    """Own one startup's measured phases and a single SDK configuration snapshot.

    Phase durations are wall time, without extra GPU synchronizations. Nested
    phases are inclusive: runtime_scene includes its named construction phases.
    They are not intended to be summed into total_ms. The worker finishes the
    operation after its first RGB frame, before publishing a ready status.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger,
        publish: Callable[[dict], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        operation: str = "startup",
    ) -> None:
        self._clock = clock
        self.operation = operation
        self.operation_id = uuid.uuid4().hex
        self._started = clock()
        self._ended: float | None = None
        self._logger = logger
        self._publish = publish
        self._stack: list[str] = []
        self._phases_ms: dict[str, float] = {}
        self._failed_phase: str | None = None
        self._first_frame_ms: float | None = None
        self._phase_started: list[float] = []
        self._progress: dict | None = None
        self.versions = package_versions()
        self.quadrants_cache: dict[str, Any] = {"available": False, "reason": "not_initialized"}

    @property
    def finished(self) -> bool:
        return self._ended is not None

    def collect_runtime(self, quadrants: Any) -> None:
        self.quadrants_cache = quadrants_cache_config(quadrants)
        self._logger.info(
            "Genesis startup runtime %s",
            json.dumps({"versions": self.versions, "quadrants_cache": self.quadrants_cache}, sort_keys=True),
        )
        self._notify()

    def snapshot(self) -> dict:
        ended = self._ended if self._ended is not None else self._clock()
        return {
            "operation": self.operation,
            "operation_id": self.operation_id,
            "state": "failed" if self._failed_phase else "complete" if self.finished else "running",
            "current_phase": self._stack[-1] if self._stack else None,
            "started_at": self._started,
            "phase_started_at": self._phase_started[-1] if self._phase_started else None,
            "progress": self._progress,
            "failed_phase": self._failed_phase,
            "phases_ms": self._phases_ms.copy(),
            "total_ms": round((ended - self._started) * 1000, 3),
            "first_frame_ms": self._first_frame_ms,
            "start_boundary": "run_preview_before_sdk_imports" if self.operation == "startup" else "request_accepted",
            "end_boundary": "first_rgb_frame_completed",
            "timing_scope": "wall_clock_inclusive_nested_phases",
            "versions": self.versions.copy(),
            "quadrants_cache": self.quadrants_cache.copy(),
        }

    def advance(self, completed: int, total: int) -> None:
        """Publish measured work within the current phase, never an estimated total."""
        if not self._stack or not 0 <= completed <= total or total <= 0:
            raise ValueError("Phase progress must name completed work within a positive total")
        self._progress = {"completed": completed, "total": total}
        self._notify()

    def _notify(self) -> None:
        if self._publish is not None:
            self._publish(self.snapshot())

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if self.finished or name in self._phases_ms or name in self._stack:
            raise ValueError(f"Startup phase must run once before completion: {name}")
        started = self._clock()
        self._stack.append(name)
        self._phase_started.append(started)
        self._progress = None
        self._notify()
        try:
            yield
        except BaseException:
            self._failed_phase = self._failed_phase or name
            raise
        finally:
            elapsed = round((self._clock() - started) * 1000, 3)
            self._phases_ms[name] = elapsed
            self._stack.pop()
            self._phase_started.pop()
            self._progress = None
            self._logger.info("Genesis startup phase %s: %.3f ms", name, elapsed)
            self._notify()

    def mark_first_frame(self) -> None:
        """Record the first main-view RGB completion, independently of ready.

        This is not WebRTC delivery or browser presentation. Do not republish a
        starting status: the worker's next regular status carries this milestone.
        """
        if self._first_frame_ms is None:
            self._first_frame_ms = round((self._clock() - self._started) * 1000, 3)
            self._logger.info("Genesis first RGB frame: %.3f ms since startup", self._first_frame_ms)

    def finish(self, *, error: bool = False) -> dict:
        if not self.finished:
            if error:
                self._failed_phase = self._failed_phase or "unscoped"
            self._ended = self._clock()
            self._logger.info("Genesis startup summary %s", json.dumps(self.snapshot(), sort_keys=True))
        return self.snapshot()
