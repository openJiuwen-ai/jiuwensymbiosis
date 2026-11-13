"""Wall-clock scheduling for the sole physics/render owner, without SDK imports."""

from __future__ import annotations

import time
from collections.abc import Callable


class PhysicsClock:
    """Catch up fixed physics steps in short, interruptible batches.

    A frame may take longer than a physics interval. Its debt survives across
    batches; the worker reads control and lease updates between those batches.
    This is serial scheduling, not concurrent access to Genesis or Nyx.
    """

    def __init__(
        self,
        dt: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        batch_budget_s: float = 0.008,
        max_batch_steps: int = 64,
    ) -> None:
        self.dt = dt
        self._monotonic = monotonic
        self._batch_budget_s = batch_budget_s
        self._max_batch_steps = max_batch_steps
        self._started = monotonic()
        self._steps = 0
        self.next_step = self._started + dt
        self._dropped_wall_time_s = 0.0
        self._overload_events = 0
        self._last_render = self._started

    def run_batch(self, step: Callable[[], None], *, stopping: Callable[[], bool] = lambda: False) -> int:
        """Return after the wall budget, hard step cap, or a stop request.

        One SDK step is atomic, so the wall budget can be exceeded by that step.
        Refresh wall time after every step, including time spent stepping itself.
        """
        started = self._monotonic()
        excess = started - self.next_step - 0.5
        if excess > 0:
            # Bound exceptional stalls, without pretending discarded wall time
            # was simulated. Ordinary render debt is never discarded.
            self._dropped_wall_time_s += excess
            self._overload_events += 1
            self.next_step += excess
        count = 0
        while count < self._max_batch_steps and not stopping():
            now = self._monotonic()
            if now < self.next_step or (count and now - started >= self._batch_budget_s):
                break
            step()
            count += 1
            self._steps += 1
            self.next_step = self._started + (self._steps + 1) * self.dt + self._dropped_wall_time_s
        return count

    def render_ready(self) -> bool:
        """Prioritize catch-up, without starving video near physics capacity.

        Stable debt need not reach the wall-time drop limit. When physics has
        not caught up, allow a fallback at least 0.5 s after the last completed
        render; the remaining time stays available for fixed physics steps.
        """
        now = self._monotonic()
        lag = now - self.next_step
        return lag <= self.dt or now - self._last_render >= 0.5

    def mark_render(self) -> None:
        """Account for both capture and video renders in the same fallback budget."""
        self._last_render = self._monotonic()

    def snapshot(self) -> dict[str, float | int | bool]:
        """Measured progress since this clock's start, including rendering time."""
        now = self._monotonic()
        elapsed = now - self._started
        simulated = self._steps * self.dt
        return {
            "dt_s": self.dt,
            "elapsed_wall_s": elapsed,
            "simulated_s": simulated,
            "actual_hz": self._steps / elapsed if elapsed else 0.0,
            "real_time_factor": simulated / elapsed if elapsed else 0.0,
            "lag_s": max(0.0, now - self.next_step),
            "dropped_wall_time_s": self._dropped_wall_time_s,
            "overload_events": self._overload_events,
            "degraded": self._overload_events > 0 and now - self.next_step >= 0.25,
        }
