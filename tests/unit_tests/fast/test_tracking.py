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

import math
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jiuwensymbiosis.agent.fast.realtime.mask_tracking import (
    MaskTargetFilter,
    MaskTrackingState,
)
from jiuwensymbiosis.agent.fast.realtime.servo import ServoConfig, ServoController, _slew
from jiuwensymbiosis.agent.fast.realtime.tracking import BackgroundTracker


def _mask_sample(
    mask,
    *,
    x=100.0,
    y=50.0,
    grasp_z=40.0,
    depth_m=0.5,
    score=0.9,
    depth_span_mm=5.0,
):
    return {
        "ok": True,
        "position": [x, y, grasp_z + 20.0],
        "grasp_z": grasp_z,
        "depth_m": depth_m,
        "score": score,
        "x": x,
        "y": y,
        "z": grasp_z + 20.0,
        "_tracking_mask": np.asarray(mask, dtype=bool),
        "_tracking_depth_span_mm": depth_span_mm,
        "_tracking_valid_depth_ratio": 1.0,
    }


def test_mask_filter_freezes_projection_noise_for_stationary_mask():
    mask = np.zeros((60, 80), dtype=bool)
    mask[20:40, 20:50] = True
    target_filter = MaskTargetFilter()

    initial = target_filter.update(_mask_sample(mask, x=100.0))
    stationary = target_filter.update(_mask_sample(mask, x=112.0))

    assert initial is not None
    assert stationary is not None
    assert stationary["x"] == 100.0
    assert stationary["_tracking_state"] == MaskTrackingState.VISIBLE_STABLE
    assert "_tracking_mask" not in stationary


def test_mask_filter_static_occlusion_keeps_last_trusted_xyz():
    mask = np.zeros((60, 80), dtype=bool)
    mask[20:40, 20:50] = True
    partial = np.zeros_like(mask)
    partial[22:38, 25:45] = True
    target_filter = MaskTargetFilter()

    target_filter.update(_mask_sample(mask, x=100.0))
    target_filter.update(_mask_sample(mask, x=101.0))  # establish stable history
    frozen = target_filter.update(_mask_sample(partial, x=180.0, grasp_z=10.0))

    assert frozen is not None
    assert frozen["x"] == 100.0
    assert frozen["grasp_z"] == 40.0
    assert frozen["_tracking_state"] == MaskTrackingState.OCCLUDED_STATIC


def test_mask_filter_contained_shrink_keeps_original_reference_and_target():
    """Any contained shrink is occlusion; bbox edge direction is irrelevant."""
    full = np.zeros((70, 90), dtype=bool)
    full[10:50, 20:60] = True
    clipped_once = np.zeros_like(full)
    clipped_once[12:48, 21:59] = True
    clipped_again = np.zeros_like(full)
    clipped_again[18:42, 25:55] = True
    target_filter = MaskTargetFilter()

    target_filter.update(_mask_sample(full, x=100.0, depth_m=0.50))
    first = target_filter.update(_mask_sample(clipped_once, x=140.0, depth_m=0.52))
    second = target_filter.update(_mask_sample(clipped_again, x=180.0, depth_m=0.54))

    assert first is not None and second is not None
    assert first["x"] == second["x"] == 100.0
    assert first["_tracking_state"] == MaskTrackingState.OCCLUDED_STATIC
    assert second["_tracking_state"] == MaskTrackingState.OCCLUDED_STATIC
    # The second frame is still compared with the original 40x40 mask, not the
    # first partial mask: the trusted reference never integrates occlusion.
    assert first["_tracking_metrics"]["area_ratio"] == pytest.approx(0.855)
    assert second["_tracking_metrics"]["area_ratio"] == pytest.approx(0.45)
    assert second["_tracking_metrics"]["containment"] == pytest.approx(1.0)


def test_mask_filter_occlusion_uses_mask_only_even_when_projection_is_unreliable():
    full = np.zeros((60, 80), dtype=bool)
    full[20:40, 20:60] = True
    partial = np.zeros_like(full)
    partial[22:38, 25:55] = True
    target_filter = MaskTargetFilter()
    target_filter.update(_mask_sample(full, x=100.0))

    frozen = target_filter.update(_mask_sample(partial, x=999.0, score=0.1, depth_span_mm=80.0))

    assert frozen is not None
    assert frozen["x"] == 100.0
    assert frozen["_tracking_state"] == MaskTrackingState.OCCLUDED_STATIC


def test_mask_filter_requires_two_consistent_frames_before_following_motion():
    anchor = np.zeros((50, 70), dtype=bool)
    anchor[20:30, 10:20] = True
    moved1 = np.zeros_like(anchor)
    moved1[20:30, 15:25] = True
    moved2 = np.zeros_like(anchor)
    moved2[20:30, 16:26] = True
    target_filter = MaskTargetFilter()

    target_filter.update(_mask_sample(anchor, x=100.0))
    assert target_filter.update(_mask_sample(moved1, x=105.0)) is None
    accepted = target_filter.update(_mask_sample(moved2, x=107.0))

    assert accepted is not None
    assert accepted["x"] == 107.0
    assert accepted["_tracking_state"] == MaskTrackingState.VISIBLE_MOVING


def test_mask_filter_uses_last_target_when_current_detection_is_unreliable():
    mask = np.zeros((60, 80), dtype=bool)
    mask[20:40, 20:60] = True
    tiny = np.zeros_like(mask)
    tiny[27:33, 35:45] = True
    target_filter = MaskTargetFilter()
    target_filter.update(_mask_sample(mask))
    target_filter.update(_mask_sample(mask))

    tiny_result = target_filter.update(_mask_sample(tiny, x=500.0))
    mixed_depth = target_filter.update(_mask_sample(mask, x=600.0, depth_span_mm=80.0))

    assert tiny_result is not None and tiny_result["x"] == 100.0
    assert tiny_result["_tracking_state"] == MaskTrackingState.OCCLUDED_STATIC
    assert mixed_depth is not None and mixed_depth["x"] == 100.0
    assert mixed_depth["_tracking_state"] == MaskTrackingState.BLIND_LAST_TARGET


def test_mask_filter_detector_miss_blind_grasps_last_trusted_target():
    mask = np.zeros((60, 80), dtype=bool)
    mask[20:40, 20:60] = True
    target_filter = MaskTargetFilter()
    target_filter.update(_mask_sample(mask, x=123.0))

    blind = target_filter.miss("not_detected")

    assert blind is not None
    assert blind["x"] == 123.0
    assert blind["_tracking_state"] == MaskTrackingState.BLIND_LAST_TARGET
    assert blind["_tracking_metrics"]["reason"] == "not_detected"


def test_mask_filter_detector_miss_without_reference_stays_lost():
    target_filter = MaskTargetFilter()

    assert target_filter.miss("not_detected") is None
    assert target_filter.state == MaskTrackingState.LOST


@pytest.mark.parametrize(
    "field_name",
    [
        "control_hz",
        "max_lin_step_mm",
        "max_ang_step_deg",
        "pos_tol_mm",
        "ang_tol_deg",
        "timeout_s",
        "absolute_timeout_s",
        "progress_pos_epsilon_mm",
        "progress_ang_epsilon_deg",
        "lost_target_grace_s",
        "settle_ticks",
    ],
)
@pytest.mark.parametrize("value", [True, False])
def test_servo_config_rejects_boolean_numeric_fields(field_name, value):
    with pytest.raises(ValueError, match=rf"ServoConfig\.{field_name}"):
        ServoConfig(**{field_name: value})


def test_slew_limits_xyz_as_one_translation_vector():
    """A diagonal target must not exceed the configured Cartesian step."""
    step = _slew(
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 6.0, "y": 6.0, "z": 6.0},
        max_lin=6.0,
        max_ang=5.0,
    )
    distance = sum(float(step[key]) ** 2 for key in ("x", "y", "z")) ** 0.5
    assert distance == pytest.approx(6.0)
    assert step["x"] == pytest.approx(step["y"])
    assert step["y"] == pytest.approx(step["z"])


def test_slew_limits_complete_orientation_on_so3_shortest_path():
    current = {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 10.0, "ry": 20.0, "rz": 30.0}
    target = {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 150.0, "ry": -40.0, "rz": -120.0}
    step = _slew(current, target, max_lin=5.0, max_ang=4.0)
    current_rotation = Rotation.from_euler("xyz", [current[key] for key in ("rx", "ry", "rz")], degrees=True)
    step_rotation = Rotation.from_euler("xyz", [step[key] for key in ("rx", "ry", "rz")], degrees=True)
    assert math.degrees((current_rotation.inv() * step_rotation).magnitude()) == pytest.approx(4.0)


def test_servo_result_preserves_typed_driver_error_code():
    class DriverError(ValueError):
        code = "ik_unreachable"

    def fail(_pose):
        raise DriverError("no solution")

    result = ServoController(
        lambda: {"x": 0.0, "y": 0.0, "z": 0.0},
        fail,
        lambda: {"x": 10.0, "y": 0.0, "z": 0.0},
        config=ServoConfig(control_hz=200.0),
    ).run()
    assert result.error_code == "ik_unreachable"
    assert result.as_dict()["error_code"] == "ik_unreachable"


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
    assert result.error_code == "servo_no_progress_timeout"
    assert provider_calls >= 3
    assert len(commands) >= 3
    assert result.target_pose == {"x": 50.0, "y": 0.0, "z": 0.0}


def test_servo_progress_refreshes_timeout_until_a_long_approach_reaches():
    """A moving live pose must not be cut off by the old total wall-clock limit."""
    current = {"x": 0.0, "y": 0.0, "z": 0.0}
    target = {"x": 10.0, "y": 0.0, "z": 0.0}

    def read_pose():
        return dict(current)

    def servo_to(_pose):
        current["x"] = min(target["x"], current["x"] + 1.0)

    result = ServoController(
        read_pose,
        servo_to,
        lambda: dict(target),
        config=ServoConfig(
            control_hz=200.0,
            max_lin_step_mm=1.0,
            settle_ticks=1,
            timeout_s=0.015,
            absolute_timeout_s=1.0,
        ),
    ).run()

    assert result.reason == "reached"
    assert result.elapsed_s > 0.015


def test_servo_absolute_timeout_remains_a_final_safety_ceiling():
    current = {"x": 0.0, "y": 0.0, "z": 0.0}

    def read_pose():
        return dict(current)

    def servo_to(_pose):
        current["x"] += 1.0

    result = ServoController(
        read_pose,
        servo_to,
        lambda: {"x": 1000.0, "y": 0.0, "z": 0.0},
        config=ServoConfig(
            control_hz=200.0,
            max_lin_step_mm=1.0,
            timeout_s=1.0,
            absolute_timeout_s=0.03,
        ),
    ).run()

    assert result.reason == "timeout"
    assert result.error_code == "servo_absolute_timeout"


def test_servo_controller_can_make_completion_position_only_without_dropping_command_orientation():
    current = {"x": 10.0, "y": 20.0, "z": 30.0, "rz": 0.0}
    target = {"x": 10.0, "y": 20.0, "z": 30.0, "rz": 90.0}
    commands: list[dict[str, float]] = []

    result = ServoController(
        lambda: dict(current),
        lambda pose: commands.append(dict(pose)),
        lambda: dict(target),
        config=ServoConfig(control_hz=200.0, settle_ticks=1),
        reached_angular_keys=(),
    ).run()

    assert result.reason == "reached"
    assert commands == []

    default_result = ServoController(
        lambda: dict(current),
        lambda pose: commands.append(dict(pose)),
        lambda: dict(target),
        config=ServoConfig(control_hz=200.0, settle_ticks=1),
        should_continue=lambda: not commands,
    ).run()
    assert default_result.reason == "stopped"
    assert commands[0]["rz"] != current["rz"]


def test_servo_controller_reports_live_pose_and_errors_on_reached_tick():
    ticks: list[dict] = []
    result = ServoController(
        lambda: {"x": 1.0, "y": 2.0, "z": 3.0, "rz": 45.0},
        lambda _pose: None,
        lambda: {"x": 1.0, "y": 2.0, "z": 3.0, "rz": 45.0},
        config=ServoConfig(control_hz=200.0, settle_ticks=1),
        on_tick=ticks.append,
    ).run()

    assert result.reason == "reached"
    assert ticks == [
        {
            "tick": 1,
            "pose": {"x": 1.0, "y": 2.0, "z": 3.0, "rz": 45.0},
            "target": {"x": 1.0, "y": 2.0, "z": 3.0, "rz": 45.0},
            "position_error_mm": 0.0,
            "angular_error_deg": 0.0,
            "in_tol": 1,
        }
    ]


def test_servo_controller_rejects_unknown_reached_angular_key():
    with pytest.raises(ValueError, match="unsupported keys"):
        ServoController(
            lambda: {"x": 0.0},
            lambda _pose: None,
            lambda: {"x": 0.0},
            reached_angular_keys=("yaw",),
        )


def test_servo_controller_can_chain_slew_from_previous_command():
    """Encoder lag must not pull an opt-in command trajectory back each tick."""
    target = {"x": 10.0, "y": 0.0, "z": 0.0}
    commands: list[dict[str, float]] = []

    result = ServoController(
        # Simulate an arm whose encoder has not caught up yet.
        lambda: {"x": 0.0, "y": 0.0, "z": 0.0},
        lambda pose: commands.append(dict(pose)),
        lambda: dict(target),
        config=ServoConfig(control_hz=200.0, max_lin_step_mm=2.0, timeout_s=1.0),
        should_continue=lambda: len(commands) < 3,
        slew_from_last_command=True,
    ).run()

    assert result.reason == "stopped"
    assert [command["x"] for command in commands] == pytest.approx([2.0, 4.0, 6.0])


def test_servo_controller_does_not_advance_chain_when_dispatch_is_skipped():
    """An explicit False acknowledgement must keep the previous slew origin."""
    attempts: list[dict[str, float]] = []

    def servo_to(pose):
        attempts.append(dict(pose))
        return len(attempts) > 1

    result = ServoController(
        lambda: {"x": 0.0, "y": 0.0, "z": 0.0},
        servo_to,
        lambda: {"x": 10.0, "y": 0.0, "z": 0.0},
        config=ServoConfig(control_hz=200.0, max_lin_step_mm=2.0, timeout_s=1.0),
        should_continue=lambda: len(attempts) < 3,
        slew_from_last_command=True,
    ).run()

    assert result.reason == "stopped"
    assert [attempt["x"] for attempt in attempts] == pytest.approx([2.0, 2.0, 4.0])


def test_servo_controller_keeps_live_pose_slew_as_default():
    """Adapters that do not opt in retain the existing live-pose behavior."""
    commands: list[dict[str, float]] = []

    ServoController(
        lambda: {"x": 0.0, "y": 0.0, "z": 0.0},
        lambda pose: commands.append(dict(pose)),
        lambda: {"x": 10.0, "y": 0.0, "z": 0.0},
        config=ServoConfig(control_hz=200.0, max_lin_step_mm=2.0, timeout_s=1.0),
        should_continue=lambda: len(commands) < 3,
    ).run()

    assert [command["x"] for command in commands] == pytest.approx([2.0, 2.0, 2.0])
