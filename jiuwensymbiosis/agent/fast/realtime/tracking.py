# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Background perception tracker (decoupled from the control loop).

``BackgroundTracker`` runs a ``detect_fn`` in a daemon thread and keeps only the
**latest** target it produced. The servo ``ServoController`` (which ticks at
``control_hz``, e.g. 30 Hz) reads ``latest_target()`` without blocking, while
detection runs at whatever rate it can sustain (jiuwen's GroundingDINO+SAM2 is
seconds-scale; MediaPipe-class detectors are camera-rate). This two-rate split
is what lets a slow detector still drive a smooth high-rate servo: the loop
always slews toward the freshest known target instead of stalling on detection.

A target is a pose ``dict`` (``x/y/z`` mm + optional ``r``/``rz`` deg) or
``None`` (target not currently visible → the controller holds).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

Pose = dict[str, float]


class BackgroundTracker:
    """Run ``detect_fn`` in a thread; expose the most recent target pose.

    Args:
        detect_fn: ``() -> Optional[Pose]`` — one perception attempt. Should
            return the current target pose, or ``None`` if not detected.
        max_hz: cap on detection rate (detection often can't reach it anyway).
        staleness_s: a target older than this is reported as ``None`` (treated
            as "lost"), so the controller's lost-target grace can fire.
        name: thread name / log tag.
    """

    def __init__(
        self,
        detect_fn: Callable[[], Pose | None],
        *,
        max_hz: float = 10.0,
        staleness_s: float = 2.0,
        name: str = "track",
    ) -> None:
        self._detect_fn = detect_fn
        self._min_period = 1.0 / max_hz if max_hz > 0 else 0.0
        self._staleness_s = staleness_s
        self._name = name
        self._lock = threading.Lock()
        self._target: Pose | None = None
        self._stamp: float = 0.0
        self._detections = 0
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    # ----------------------------------------------------------------- lifecycle
    def start(self) -> BackgroundTracker:
        """Start the detection thread. Idempotent."""
        if self._thread is not None:
            return self
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, name=f"track-{self._name}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the detection thread. Idempotent."""
        self._stop_evt.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None

    def __enter__(self) -> BackgroundTracker:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------- access
    def latest_target(self) -> Pose | None:
        """Freshest target pose, or ``None`` if never detected / stale.

        Designed to be passed directly as ``ServoController(target_provider=...)``.
        """
        with self._lock:
            tgt, stamp = self._target, self._stamp
        if tgt is None:
            return None
        if self._staleness_s > 0 and (time.monotonic() - stamp) > self._staleness_s:
            return None
        return tgt

    @property
    def detections(self) -> int:
        """Total successful detections since start (diagnostic)."""
        return self._detections

    def wait_first(self, timeout_s: float = 5.0) -> bool:
        """Block until the first target lands (fresh), or ``timeout_s`` elapses."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.latest_target() is not None:
                return True
            time.sleep(0.02)
        return self.latest_target() is not None

    # -------------------------------------------------------------------- thread
    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            try:
                tgt = self._detect_fn()
            except Exception as exc:  # noqa: BLE001 - detection must never kill the thread
                logger.debug("[track-%s] detect error (ignored): %s", self._name, exc)
                tgt = None
            if tgt is not None:
                with self._lock:
                    self._target = dict(tgt)
                    self._stamp = t0
                    self._detections += 1
            dt = time.monotonic() - t0
            self._stop_evt.wait(max(self._min_period - dt, 0.005))
