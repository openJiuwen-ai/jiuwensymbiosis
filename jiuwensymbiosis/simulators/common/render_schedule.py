"""Bounded video work without coupling refresh to accumulated physics debt."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class RenderSchedule:
    """Pace fresh frames, spending at most a configured share on serial video.

    Physics still advances at its unchanged fixed timestep before each frame.
    A slow frame lowers the video rate; it never clears the physics clock's
    debt or invents simulated progress. Consumer backpressure remains external.
    """

    def __init__(self, fps: int, *, monotonic: Callable[[], float] = time.monotonic, budget: float = 0.2):
        self.fps = fps
        self.budget = budget
        self._now = monotonic
        self.next_frame = monotonic()
        self._cost = 0.0
        self._period = 1 / fps
        self._completed: deque[float] = deque(maxlen=600)
        self._missed = 0

    def ready(self) -> bool:
        return self._now() >= self.next_frame

    def pause(self) -> None:
        """Time with no requested video is not a missed delivery deadline."""
        self.next_frame = self._now()

    def finish(self, started: float, *, blocking_s: float | None = None) -> None:
        now = self._now()
        cost = now - started if blocking_s is None else blocking_s
        # React to a slow frame immediately; recover gradually after a spike.
        self._cost = max(cost, 0.8 * self._cost + 0.2 * cost)
        self._period = max(1 / self.fps, self._cost / self.budget)
        self._missed += max(0, int((started - self.next_frame) / self._period))
        self.next_frame = max(now, started + self._period)
        self._completed.append(now)

    def snapshot(self) -> dict:
        now = self._now()
        while len(self._completed) > 1 and self._completed[0] < now - 5:
            self._completed.popleft()
        elapsed = now - self._completed[0] if self._completed else 0.0
        return {
            "policy": "bounded_video_budget",
            "requested_fps": self.fps,
            "budget_fraction": self.budget,
            "effective_fps_limit": 1 / self._period,
            "blocking_cost_ms": self._cost * 1000,
            "recent_source_fps": (len(self._completed) - 1) / elapsed if elapsed > 0 else 0.0,
            "missed_frame_slots": self._missed,
        }
