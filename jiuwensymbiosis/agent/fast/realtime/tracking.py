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
import math
import threading
import time
from collections.abc import Callable

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.errors import InferenceServiceError

logger = logging.getLogger(__name__)

Pose = dict[str, float]


class BackgroundTracker:
    """Run ``detect_fn`` in a thread; expose the most recent target pose.

    Args:
        detect_fn: ``() -> Optional[Pose]`` — one perception attempt. Should
            return the current target pose, or ``None`` if not detected.
        max_hz: cap on detection rate (detection often can't reach it anyway).
        staleness_s: **required (no default).** A positive value makes a target
            older than that read as ``None`` (lost); ``None`` explicitly opts
            into "never stale" (e.g. when freshness is driven via
            :meth:`target_is_live` instead). No default because an implicit
            "never stale" silently drives motion from an arbitrarily old frame.
        name: thread name / log tag.
    """

    def __init__(
        self,
        detect_fn: Callable[[], Pose | None],
        *,
        max_hz: float = 10.0,
        staleness_s: float | None,
        name: str = "track",
        cancel_token: CancelToken | None = None,
    ) -> None:
        self._detect_fn = detect_fn
        self._min_period = 1.0 / max_hz if max_hz > 0 else 0.0
        self._staleness_s = staleness_s
        self._name = name
        self._lock = threading.Lock()
        self._target: Pose | None = None
        self._stamp: float = 0.0
        self._completed_t: float = 0.0
        self._latency_s: float = 0.0
        self._detections = 0
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._cancel_token = cancel_token
        self._failure: Exception | None = None

    # ----------------------------------------------------------------- lifecycle
    def start(self) -> BackgroundTracker:
        """Start the detection thread. Idempotent."""
        if self._thread is not None:
            return self
        self._stop_evt.clear()
        self._failure = None
        finish = self._cancel_token.register_work(f"tracker:{self._name}") if self._cancel_token else lambda: None

        def tracked() -> None:
            try:
                self._loop()
            finally:
                finish()

        self._thread = threading.Thread(target=tracked, name=f"track-{self._name}", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._thread = None
            finish()
            raise
        return self

    def stop(self, *, timeout_s: float | None = None) -> None:
        """Join in-flight detection within its freshness budget; cancellation stays prompt.

        Without a freshness limit, use the HTTP client's default 30-second budget.
        An explicit timeout allows callers to bound cleanup more tightly.
        """
        budget = max(2.0, self._staleness_s or 30.0) if timeout_s is None else timeout_s
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("tracker stop timeout must be finite and nonnegative")
        with self._lock:
            self._stop_evt.set()
        t = self._thread
        if t is not None:
            deadline = time.monotonic() + budget
            while t.is_alive():
                # Retain both the handle and token work until detection really exits.
                # Cleanup must not replace cancellation with a recoverable action failure.
                if self._cancel_token is not None:
                    self._cancel_token.raise_if_set()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Tracker detection is still running; cleanup is incomplete")
                t.join(timeout=min(0.05, remaining))
        self._thread = None

    def raise_if_failed(self) -> None:
        """Raise a terminal detection failure, including one received during stop.

        Call after ``stop()`` before accepting a tracking result; stopping alone
        only releases resources and remains safe for repeated cleanup.
        """
        if self._failure is not None:
            raise self._failure

    def __enter__(self) -> BackgroundTracker:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------- access
    def latest_target(self) -> Pose | None:
        """Freshest target pose, or ``None`` if never detected / stale.

        Designed to be passed directly as ``ServoController(target_provider=...)``.
        ``staleness_s=None`` means a target is never reported stale; a
        configured positive value makes a target older than that None.
        """
        self.raise_if_failed()
        with self._lock:
            tgt, stamp = self._target, self._stamp
        if tgt is None:
            return None
        if self._staleness_s is not None and self._staleness_s > 0 and (time.monotonic() - stamp) > self._staleness_s:
            return None
        return tgt

    def latest_target_with_capture_time(self) -> tuple[Pose | None, float]:
        """Freshest ``(target, capture_time)`` pair, bypassing staleness filtering.

        ``capture_time`` is the ``monotonic()`` image-grab instant (start of
        ``_loop``'s detect call), letting a caller verify a frame was grabbed
        after some motion finished — not merely that its inference completed
        after (which also catches mid-motion frames with late inference).
        Staleness is intentionally not applied; wait-style callers reason about
        detection *generation* (``_detections``) themselves.
        """
        self.raise_if_failed()
        with self._lock:
            return (None if self._target is None else dict(self._target)), self._stamp

    def target_is_live(
        self,
        *,
        no_update_grace_s: float,
        max_image_age_s: float,
        latency_margin: float = 1.5,
    ) -> bool:
        """Whether the latest target is still safe for tracking motion.

        Two independent deadlines apply:

        * a detector-progress watchdog, measured from the last successful
          result. Its allowance adapts to the detector's measured latency so a
          normally slow detector is not mistaken for a stall;
        * an absolute image-age ceiling, measured from capture, so even an
          extremely slow/stuck detector can never make an old frame live
          indefinitely.

        Unlike ``latest_target()`` plus ``ServoController``'s ordinary grace,
        this health signal is intended to abort immediately when either
        deadline expires; callers must not add a second grace period.
        """
        self.raise_if_failed()
        with self._lock:
            has_target = self._target is not None
            capture_t = self._stamp
            completed_t = self._completed_t
            latency_s = self._latency_s
        if not has_target or capture_t <= 0.0 or completed_t <= 0.0:
            return False
        now = time.monotonic()
        update_allowance = max(float(no_update_grace_s), latency_s * float(latency_margin))
        return (now - completed_t) <= update_allowance and (now - capture_t) <= float(max_image_age_s)

    @property
    def detections(self) -> int:
        """Total successful detections since start (diagnostic)."""
        return self._detections

    def wait_first(self, timeout_s: float = 5.0, *, cancel_token: CancelToken | None = None) -> bool:
        """Block until the first target lands, or ``timeout_s`` elapses.

        Uses the internal target + detection counter directly (not the
        staleness-filtered ``latest_target``), so a slow detector whose
        per-result age exceeds ``staleness_s`` is still observed as "first
        frame arrived". ``cancel_token`` (GUI-only) is polled each iteration so a
        stop during this wait raises ``RunCancelled`` within ~20ms instead of
        blocking to ``timeout_s``.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.raise_if_failed()
            if cancel_token is not None:
                cancel_token.raise_if_set()
            with self._lock:
                if self._target is not None and self._detections > 0:
                    return True
            time.sleep(0.02)
        self.raise_if_failed()
        if cancel_token is not None:
            cancel_token.raise_if_set()
        with self._lock:
            return self._target is not None and self._detections > 0

    def wait_for_next(self, previous_detections: int, timeout_s: float = 5.0) -> tuple[Pose | None, float]:
        """Wait for a successful detection newer than ``previous_detections``.

        Returns ``(target, capture_time)`` for that new generation, or
        ``(None, 0.0)`` on timeout. A detector stall keeps ``_detections``
        frozen, so this returns ``(None, 0.0)`` rather than re-yielding a
        stale-but-under-staleness-limit frame.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.raise_if_failed()
            with self._lock:
                if self._detections > previous_detections and self._target is not None:
                    return dict(self._target), self._stamp
            time.sleep(0.02)
        self.raise_if_failed()
        return None, 0.0

    def wait_for_capture_after(
        self, capture_threshold_t: float, *, timeout_s: float = 5.0, cancel_token: CancelToken | None = None
    ) -> tuple[Pose, float] | None:
        """Block until a frame whose image capture time is ``>= capture_threshold_t``.

        Returns ``(target, capture_time)`` for the first such frame, or ``None``
        on timeout. Acceptance is keyed on the image-grab instant (``_stamp``),
        not a detection-generation counter, so a frame grabbed during a motion
        whose inference finishes after the motion is accepted iff the *grab*
        postdated ``capture_threshold_t`` — the post-motion-observation contract.
        ``cancel_token`` (GUI-only) is polled each iteration so a stop during this
        wait raises ``RunCancelled`` within ~20ms instead of blocking to ``timeout_s``.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.raise_if_failed()
            if cancel_token is not None:
                cancel_token.raise_if_set()
            with self._lock:
                target = self._target
                stamp = self._stamp
            if target is not None and stamp >= capture_threshold_t:
                return dict(target), stamp
            time.sleep(0.02)
        self.raise_if_failed()
        if cancel_token is not None:
            cancel_token.raise_if_set()
        return None

    # -------------------------------------------------------------------- thread
    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            try:
                if self._cancel_token is not None:
                    self._cancel_token.raise_if_set()
                tgt = self._detect_fn()
                if self._cancel_token is not None:
                    self._cancel_token.raise_if_set()
            except (RunCancelled, InferenceServiceError) as exc:
                with self._lock:
                    self._target = None
                    self._failure = exc
                return
            except Exception as exc:  # noqa: BLE001 - detection must never kill the thread
                logger.debug("[track-%s] detect error (ignored): %s", self._name, exc)
                tgt = None
            if tgt is not None and not self._stop_evt.is_set():
                completed_t = time.monotonic()
                with self._lock:
                    if self._stop_evt.is_set():
                        return
                    if self._cancel_token is not None and self._cancel_token.is_set():
                        self._target = None
                        self._failure = RunCancelled()
                        return
                    self._target = dict(tgt)
                    self._stamp = t0
                    self._completed_t = completed_t
                    self._latency_s = max(0.0, completed_t - t0)
                    self._detections += 1
            dt = time.monotonic() - t0
            self._stop_evt.wait(max(self._min_period - dt, 0.005))
