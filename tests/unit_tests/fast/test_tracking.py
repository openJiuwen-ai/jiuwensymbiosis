# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for the BackgroundTracker staleness/wait semantics.

These pin the F4 fix: ``staleness_s`` is a required keyword (no default — an
implicit "never stale" would silently drive motion from an arbitrarily old
frame); ``wait_first`` / ``wait_for_next`` / ``wait_for_capture_after`` observe
the internal target + detection counter directly rather than the
staleness-filtered ``latest_target`` — so a slow detector whose per-result age
exceeds a configured ``staleness_s`` is still seen as "a frame arrived".
"""

from __future__ import annotations

import time

from jiuwensymbiosis.agent.fast.realtime.servo import ServoConfig, ServoController
from jiuwensymbiosis.agent.fast.realtime.tracking import BackgroundTracker


def test_staleness_none_keeps_target_alive():
    # With ``staleness_s=None`` a target is never reported stale — a caller
    # that drives freshness via ``target_is_live`` opts into this explicitly.
    t = BackgroundTracker(lambda: {"x": 1.0, "y": 2.0, "z": 3.0}, max_hz=100.0, staleness_s=None)
    t.start()
    try:
        assert t.wait_first(timeout_s=1.0)
        # Even after a long wall-clock pause, the target remains "latest".
        time.sleep(0.05)
        assert t.latest_target() is not None
    finally:
        t.stop()


def test_wait_first_does_not_require_staleness_with_slow_detector():
    # A configured positive staleness must NOT defeat wait_first: the wait
    # methods reason about detection *generation*, not per-result age.
    import time as _time

    def slow_detect():
        _time.sleep(0.05)  # 50ms per detection
        return {"x": 1.0, "y": 0.0, "z": 0.0}

    t = BackgroundTracker(slow_detect, max_hz=100.0, staleness_s=0.01)
    t.start()
    try:
        # Even though each result is >0.01s old by the time wait_first checks,
        # the first frame is still observed.
        assert t.wait_first(timeout_s=2.0)
        assert t.detections >= 1
    finally:
        t.stop()


def test_wait_for_next_returns_target_with_capture_time():
    t = BackgroundTracker(lambda: {"x": 5.0, "y": 0.0, "z": 0.0}, max_hz=100.0, staleness_s=None)
    t.start()
    try:
        assert t.wait_first(timeout_s=1.0)
        baseline = t.detections
        target, capture_t = t.wait_for_next(baseline, timeout_s=2.0)
        assert target is not None
        assert target["x"] == 5.0
        assert capture_t > 0.0
    finally:
        t.stop()


def test_wait_for_next_returns_none_on_detector_stall():
    # A detector that returns None (target not visible) keeps _detections
    # frozen, so wait_for_next must return (None, 0.0) rather than re-yield
    # the same stale-but-under-staleness-limit frame.
    count = {"n": 0}

    def detect_once():
        count["n"] += 1
        return None if count["n"] > 1 else {"x": 1.0, "y": 0.0, "z": 0.0}

    t = BackgroundTracker(detect_once, max_hz=100.0, staleness_s=None)
    t.start()
    try:
        assert t.wait_first(timeout_s=1.0)
        baseline = t.detections
        target, _capture_t = t.wait_for_next(baseline, timeout_s=0.3)
        assert target is None
    finally:
        t.stop()


def test_latest_target_with_capture_time_bypasses_staleness():
    # Detect once, then stall so the target ages past staleness_s.
    count = {"n": 0}

    def detect_once():
        count["n"] += 1
        return {"x": 1.0, "y": 0.0, "z": 0.0} if count["n"] == 1 else None

    t = BackgroundTracker(detect_once, max_hz=100.0, staleness_s=0.05)
    t.start()
    try:
        assert t.wait_first(timeout_s=1.0)
        time.sleep(0.2)  # now the lone target is older than 0.05s
        assert t.latest_target() is None  # filtered out as stale
        # ... but the raw target + capture time is still available.
        target, capture_t = t.latest_target_with_capture_time()
        assert target is not None and target["x"] == 1.0
        assert capture_t > 0.0
    finally:
        t.stop()


def test_wait_for_capture_after_accepts_only_frames_grabbed_at_or_after_threshold():
    # Pin the post-descend race fix: wait_for_capture_after keys acceptance
    # solely on a frame's capture time, taking no baseline counter — so a frame
    # grabbed *after* the threshold is accepted even if it landed during the
    # gap between recording the threshold and reading a baseline (which the old
    # wait_for_next-based path could skip). Contract: the returned frame's
    # capture time is always >= the requested threshold.
    def detect_once():
        return {"x": 1.0, "y": 0.0, "z": 0.0}

    tracker = BackgroundTracker(detect_once, max_hz=200.0, staleness_s=None)
    tracker.start()
    try:
        assert tracker.wait_first(timeout_s=1.0)
        _, first_capture_t = tracker.latest_target_with_capture_time()
        # Threshold strictly after frame 1's grab: no already-landed frame can
        # satisfy it, so the call must block until a *new* frame is grabbed.
        threshold_t = first_capture_t + 0.05
        target, capture_t = tracker.wait_for_capture_after(threshold_t, timeout_s=2.0)
        assert target is not None
        assert capture_t >= threshold_t
    finally:
        tracker.stop()


def test_wait_for_capture_after_returns_none_when_no_frame_clears_threshold():
    # A detector that stalls after an old frame must time out rather than
    # accept the stale frame — no baseline counter is read, so the stall is
    # detected purely via capture time.
    calls = {"n": 0}

    def detect_once():
        calls["n"] += 1
        return {"x": 1.0, "y": 0.0, "z": 0.0} if calls["n"] == 1 else None

    tracker = BackgroundTracker(detect_once, max_hz=200.0, staleness_s=None)
    tracker.start()
    try:
        assert tracker.wait_first(timeout_s=1.0)
        _, first_capture_t = tracker.latest_target_with_capture_time()
        threshold_t = first_capture_t + 1.0  # no later frame will ever clear it
        result = tracker.wait_for_capture_after(threshold_t, timeout_s=0.2)
        assert result is None
    finally:
        tracker.stop()


def test_detector_stall_causes_real_servo_target_lost():
    calls = {"n": 0}

    def detect_once():
        calls["n"] += 1
        return {"x": 100.0, "y": 0.0, "z": 0.0} if calls["n"] == 1 else None

    tracker = BackgroundTracker(detect_once, max_hz=200.0, staleness_s=0.03)
    tracker.start()
    try:
        assert tracker.wait_first(timeout_s=1.0)
        result = ServoController(
            lambda: {"x": 0.0, "y": 0.0, "z": 0.0},
            lambda pose: None,
            tracker.latest_target,
            config=ServoConfig(
                control_hz=100.0,
                settle_ticks=2,
                timeout_s=1.0,
                lost_target_grace_s=0.03,
            ),
        ).run()
        assert result.reason == "target_lost"
        assert tracker.detections == 1
    finally:
        tracker.stop()


def test_detector_watchdog_does_not_stack_staleness_and_lost_grace():
    """A cached non-None target must not refresh the detector-stall deadline."""
    calls = {"n": 0}

    def detect_once():
        calls["n"] += 1
        return {"x": 100.0, "y": 0.0, "z": 0.0} if calls["n"] == 1 else None

    tracker = BackgroundTracker(detect_once, max_hz=200.0, staleness_s=8.0)
    tracker.start()
    try:
        assert tracker.wait_first(timeout_s=1.0)
        started = time.monotonic()
        result = ServoController(
            lambda: {"x": 0.0, "y": 0.0, "z": 0.0},
            lambda pose: None,
            tracker.latest_target,
            config=ServoConfig(
                control_hz=200.0,
                settle_ticks=2,
                timeout_s=1.0,
                lost_target_grace_s=0.04,
            ),
            target_is_live=lambda: tracker.target_is_live(
                no_update_grace_s=0.04,
                max_image_age_s=8.0,
            ),
        ).run()
        elapsed = time.monotonic() - started

        assert result.reason == "target_lost"
        assert elapsed < 0.2
        assert tracker.latest_target() is not None  # 8s cache did not defer the watchdog
    finally:
        tracker.stop()


def test_detector_watchdog_adapts_to_measured_latency():
    def slow_detect():
        time.sleep(0.05)
        return {"x": 1.0, "y": 0.0, "z": 0.0}

    tracker = BackgroundTracker(slow_detect, max_hz=100.0, staleness_s=8.0)
    tracker.start()
    try:
        assert tracker.wait_first(timeout_s=1.0)
        # 1.5 * measured latency is greater than the deliberately tiny grace,
        # so a healthy slow detector is not declared stalled immediately.
        assert tracker.target_is_live(no_update_grace_s=0.01, max_image_age_s=8.0)
    finally:
        tracker.stop()


def test_servo_controller_reads_moving_target_each_control_tick():
    targets = iter(
        [
            {"x": 30.0, "y": 0.0, "z": 0.0},
            {"x": 40.0, "y": 0.0, "z": 0.0},
            {"x": 50.0, "y": 0.0, "z": 0.0},
        ]
    )
    latest = {"x": 30.0, "y": 0.0, "z": 0.0}
    provider_calls = 0
    commands: list[dict[str, float]] = []

    def target_provider():
        nonlocal provider_calls, latest
        provider_calls += 1
        latest = next(targets, latest)
        return dict(latest)

    def servo_to(pose):
        commands.append(dict(pose))

    result = ServoController(
        lambda: {"x": 0.0, "y": 0.0, "z": 0.0},
        servo_to,
        target_provider,
        config=ServoConfig(control_hz=200.0, max_lin_step_mm=5.0, timeout_s=0.04, settle_ticks=2),
    ).run()

    assert result.reason == "timeout"
    assert provider_calls >= 3
    assert len(commands) >= 3
    assert result.target_pose == {"x": 50.0, "y": 0.0, "z": 0.0}
