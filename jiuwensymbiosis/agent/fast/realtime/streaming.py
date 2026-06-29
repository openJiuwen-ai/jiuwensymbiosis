# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Background continuous frame capture (decoupled perception rate).

``StreamingFrameSource`` runs a daemon thread that repeatedly calls a zero-arg
``grab_fn`` (e.g. ``RealSenseCamera.grab_frames`` or ``env.get_observation``-
derived) and keeps only the **latest** ``(rgb, depth)`` pair. Consumers read
``latest()`` without blocking — so a fast servo control loop never waits on a
camera read, and a slow perception step always works on the freshest frame.

This is the "持续感知" half of the real-time servo: frame acquisition runs at
the sensor's own rate, independent of the control-loop rate.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

Frame = tuple[Any, Any]  # (rgb, depth) — depth may be None


class StreamingFrameSource:
    """Daemon-thread frame grabber exposing the most recent frame.

    Args:
        grab_fn: Zero-arg callable returning ``(rgb, depth)`` or ``None``.
        max_hz: Upper bound on grab rate; the thread sleeps to honour it.
        name: Thread name / log tag.
    """

    def __init__(
        self,
        grab_fn: Callable[[], Frame | None],
        *,
        max_hz: float = 30.0,
        name: str = "frames",
    ) -> None:
        self._grab_fn = grab_fn
        self._min_period = 1.0 / max_hz if max_hz > 0 else 0.0
        self._name = name
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._stamp: float = 0.0
        self._frames_grabbed = 0
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    # ----------------------------------------------------------------- lifecycle
    def start(self) -> StreamingFrameSource:
        """Start the capture thread. Idempotent."""
        if self._thread is not None:
            return self
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, name=f"stream-{self._name}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the capture thread. Idempotent; safe if never started."""
        self._stop_evt.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None

    def __enter__(self) -> StreamingFrameSource:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------- access
    def latest(self) -> Frame | None:
        """Most recent ``(rgb, depth)`` pair, or ``None`` if nothing grabbed yet."""
        with self._lock:
            return self._latest

    def latest_stamped(self) -> tuple[Frame | None, float]:
        """Most recent frame plus its capture timestamp (``time.monotonic``)."""
        with self._lock:
            return self._latest, self._stamp

    @property
    def frames_grabbed(self) -> int:
        """Total frames successfully grabbed since start (diagnostic)."""
        return self._frames_grabbed

    def wait_first(self, timeout_s: float = 5.0) -> bool:
        """Block until the first frame lands, or ``timeout_s`` elapses."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.latest() is not None:
                return True
            time.sleep(0.01)
        return self.latest() is not None

    # -------------------------------------------------------------------- thread
    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            try:
                frame = self._grab_fn()
            except Exception as exc:  # noqa: BLE001 - capture must never die on a transient error
                logger.debug("[stream-%s] grab error (ignored): %s", self._name, exc)
                frame = None
            if frame is not None and frame[0] is not None:
                with self._lock:
                    self._latest = frame
                    self._stamp = t0
                    self._frames_grabbed += 1
            dt = time.monotonic() - t0
            sleep_s = self._min_period - dt
            # Always yield a little even when grab returned None, so a missing
            # camera doesn't spin a core.
            self._stop_evt.wait(max(sleep_s, 0.005))
