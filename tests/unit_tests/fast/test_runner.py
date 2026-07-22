# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for the C1 fast-path generic action-sequence runner.

Detection here uses the direct ``get_grasp_info_simple`` op (bound via ``bind``)
rather than the threaded ``track_detect``, so the core execution loop — param
resolution, detection binding, gripper-occlusion bookkeeping, failure retreat —
is tested deterministically. ``track_detect`` end-to-end (servo threads) is
covered by the mock smoke script.

A custom ``action_index`` is passed so no ``@robot_tool`` plumbing is needed; the
runner is task-agnostic, so the ops are just whatever the index provides.
"""

from __future__ import annotations

import types

import pytest

from jiuwensymbiosis.agent.fast import runner as runner_module
from jiuwensymbiosis.agent.fast.realtime.binding import ServoBinding
from jiuwensymbiosis.agent.fast.realtime.servo import ServoConfig, ServoResult
from jiuwensymbiosis.agent.fast.runner import SkillExecConfig, run_sequence
from jiuwensymbiosis.agent.fast.sequence import parse_sequence


class _FakeApi:
    """Records arm calls; returns canned detections."""

    def __init__(self, objects, fail_goto_at=None):
        self.calls = []
        self.objects = objects
        self.fail_goto_at = fail_goto_at  # raise on the Nth goto (1-based) if set
        self._n_goto = 0

    def home(self):
        self.calls.append(("home",))

    def goto_xyzr(self, x, y, z, r=None):
        self._n_goto += 1
        if self.fail_goto_at is not None and self._n_goto == self.fail_goto_at:
            raise RuntimeError("EXCEEDS_LIMIT")
        self.calls.append(("goto", round(x, 1), round(y, 1), round(z, 1)))

    def open_gripper(self):
        self.calls.append(("open",))
        return {"ok": True}

    def close_gripper(self):
        self.calls.append(("close",))
        return {"ok": True}

    def get_grasp_info_simple(self, object_name):
        return self.objects.get(object_name, {"ok": False, "reason": "not_found"})


def _session(api):
    return types.SimpleNamespace(api=api, env=None)


class _EyeToHandEnv:
    capabilities = frozenset(
        {"motion.servo", "vision.camera", "vision.depth", "vision.detection", "vision.eye_to_hand"}
    )
    z_min_safe = 10.0
    workspace_bounds = (-500.0, -500.0, 500.0, 500.0)

    def __init__(self, api):
        self.api = api

    def servo_to_flange(self, pose):
        self.api.pose = {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose["z"]),
            "rx": float(pose.get("rx", 180.0)),
            "ry": float(pose.get("ry", 0.0)),
            "rz": float(pose.get("rz", 0.0)),
        }


class _EyeToHandApi(_FakeApi):
    def __init__(self, objects):
        super().__init__(objects)
        self.pose = {"x": 0.0, "y": 0.0, "z": 100.0, "rx": 180.0, "ry": 0.0, "rz": 0.0}

    def get_pose(self):
        return dict(self.pose)

    def servo_to_tip(self, pose):
        self.env.servo_to_flange(pose)


def _index(api):
    return {
        "home": api.home,
        "goto_xyzr": api.goto_xyzr,
        "open_gripper": api.open_gripper,
        "close_gripper": api.close_gripper,
        "get_grasp_info_simple": api.get_grasp_info_simple,
    }


_GRASP_OBJ = {"box": {"ok": True, "position": [250.0, 90.0, 70.0], "grasp_z": 50.0, "place_z": 80.0, "score": 0.9}}


def _tracking_config(**kwargs):
    return SkillExecConfig(**kwargs)


def test_runner_executes_grasp_like_sequence_descends_to_grasp_z():
    api = _FakeApi(_GRASP_OBJ)
    raw = [
        {"op": "home"},
        {"op": "open_gripper"},
        {"op": "get_grasp_info_simple", "params": {"object_name": "box"}, "bind": "b"},
        {"op": "goto_xyzr", "params": {"x": "b.x", "y": "b.y", "z": "b.grasp_z"}},  # direct, no offset
        {"op": "close_gripper"},
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)), special_ops=frozenset())
    res = run_sequence(_session(api), steps, action_index=_index(api))

    assert res["ok"] is True and res["steps_done"] == 5
    assert ("goto", 250.0, 90.0, 50.0) in api.calls  # straight to grasp_z=50, no approach/lift
    assert api.calls.count(("close",)) == 1


def test_track_grasp_uses_absolute_grasp_target_for_both_phases():
    api = _EyeToHandApi({"banana": {"ok": True, "position": [200.0, 150.0, 70.0], "grasp_z": 50.0}})
    api.env = _EyeToHandEnv(api)
    raw = [
        {"op": "track_grasp", "params": {"object_name": "banana", "approach_mm": 40.0}, "bind": "banana"},
        {"op": "close_gripper"},
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)), special_ops={"track_grasp"})
    cfg = _tracking_config(
        detect_hz=100.0,
        first_target_timeout_s=1.0,
        servo=ServoConfig(control_hz=100.0, max_lin_step_mm=1000.0, settle_ticks=1, timeout_s=1.0),
    )
    res = run_sequence(types.SimpleNamespace(api=api, env=api.env), steps, config=cfg, action_index=_index(api))

    assert res["ok"] is True
    assert api.pose["x"] == 200.0 and api.pose["y"] == 150.0 and api.pose["z"] == 50.0
    assert ("close",) in api.calls


def test_track_grasp_real_controller_follows_moving_detection():
    class _MovingApi(_EyeToHandApi):
        def __init__(self):
            super().__init__({})
            self.detection_calls = 0

        def get_grasp_info_simple(self, object_name):
            self.detection_calls += 1
            x = min(10.0 * self.detection_calls, 30.0)
            return {"ok": True, "position": [x, 0.0, 50.0], "grasp_z": 50.0}

    class _RecordingEnv(_EyeToHandEnv):
        def __init__(self, api):
            super().__init__(api)
            self.servo_x: list[float] = []

        def servo_to_flange(self, pose):
            self.servo_x.append(float(pose["x"]))
            super().servo_to_flange(pose)

    api = _MovingApi()
    api.env = _RecordingEnv(api)
    steps = parse_sequence(_track_grasp_sequence(), allowed_ops=set(_index(api)), special_ops={"track_grasp"})
    cfg = _tracking_config(
        detect_hz=50.0,
        first_target_timeout_s=1.0,
        servo=ServoConfig(control_hz=200.0, max_lin_step_mm=5.0, settle_ticks=2, timeout_s=1.0),
    )

    result = run_sequence(
        types.SimpleNamespace(api=api, env=api.env),
        steps,
        config=cfg,
        action_index=_index(api),
    )

    assert result["ok"] is True
    assert api.detection_calls >= 3
    assert api.pose["x"] == 30.0
    assert len(set(api.env.servo_x)) >= 3
    assert ("close",) in api.calls


def test_track_grasp_requires_post_descend_detection_before_close():
    class _SingleUpdateApi(_EyeToHandApi):
        def __init__(self):
            super().__init__({"banana": {"ok": True, "position": [200.0, 150.0, 70.0], "grasp_z": 50.0}})
            self._detection_calls = 0

        def get_grasp_info_simple(self, object_name):
            self._detection_calls += 1
            # Early live frames arrive (approach/descend); no frame arrives
            # after descend, so the post-descend barrier must fail closed.
            if self._detection_calls <= 2:
                return self.objects[object_name]
            return {"ok": False, "reason": "detector_stalled"}

    api = _SingleUpdateApi()
    api.env = _EyeToHandEnv(api)
    raw = [
        {"op": "track_grasp", "params": {"object_name": "banana", "approach_mm": 40.0}, "bind": "banana"},
        {"op": "close_gripper"},
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)), special_ops={"track_grasp"})
    cfg = _tracking_config(
        detect_hz=100.0,
        first_target_timeout_s=0.1,
        servo=ServoConfig(control_hz=100.0, max_lin_step_mm=1000.0, settle_ticks=1, timeout_s=1.0),
    )
    result = run_sequence(types.SimpleNamespace(api=api, env=api.env), steps, config=cfg, action_index=_index(api))

    assert result["ok"] is False
    assert "post-descend" in result["steps"][-1]["reason"]
    assert ("close",) not in api.calls


def test_track_detect_servo_failure_aborts_before_close(monkeypatch):
    api = _EyeToHandApi({"banana": {"ok": True, "position": [200.0, 150.0, 70.0], "grasp_z": 50.0}})
    api.env = _EyeToHandEnv(api)

    class _FailedServo:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            return ServoResult(False, "timeout", 3, 0.1, api.pose, None)

    monkeypatch.setattr(runner_module, "ServoController", _FailedServo)
    raw = [
        {"op": "track_detect", "params": {"object_name": "banana"}, "bind": "banana"},
        {"op": "close_gripper"},
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)), special_ops={"track_detect"})
    result = run_sequence(
        types.SimpleNamespace(api=api, env=api.env),
        steps,
        config=_tracking_config(first_target_timeout_s=1.0),
        action_index=_index(api),
    )

    assert result["ok"] is False
    assert result["steps"][-1]["op"] == "track_detect"
    assert "timeout" in result["steps"][-1]["reason"]
    assert ("close",) not in api.calls


def test_track_detect_rejects_first_detection_that_is_already_stale(monkeypatch):
    api = _EyeToHandApi({"banana": {"ok": True, "position": [200.0, 150.0, 70.0], "grasp_z": 50.0}})
    api.env = _EyeToHandEnv(api)

    class _AlreadyStaleTracker:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return self

        def stop(self):
            return None

        def wait_first(self, timeout_s):
            return True

        def latest_target(self):
            return None

    monkeypatch.setattr(runner_module, "BackgroundTracker", _AlreadyStaleTracker)
    raw = [
        {"op": "track_detect", "params": {"object_name": "banana"}, "bind": "banana"},
        {"op": "close_gripper"},
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)), special_ops={"track_detect"})
    result = run_sequence(
        types.SimpleNamespace(api=api, env=api.env),
        steps,
        config=_tracking_config(first_target_timeout_s=1.0),
        action_index=_index(api),
    )

    assert result["ok"] is False
    assert "first detection was already stale" in result["steps"][-1]["reason"]
    assert ("close",) not in api.calls


def test_track_detect_stall_watchdog_beats_eight_second_cached_target(monkeypatch):
    api = _EyeToHandApi({"banana": {"ok": True, "position": [200.0, 150.0, 70.0], "grasp_z": 50.0}})
    api.env = _EyeToHandEnv(api)
    health_args: list[tuple[float, float]] = []

    class _StalledButCachedTracker:
        def __init__(self, *args, **kwargs):
            self._target = {"x": 200.0, "y": 150.0, "z": 70.0, "position": [200.0, 150.0, 70.0]}

        def start(self):
            return self

        def stop(self):
            return None

        def wait_first(self, timeout_s):
            return True

        def latest_target(self):
            return dict(self._target)

        def target_is_live(self, *, no_update_grace_s, max_image_age_s, latency_margin=1.5):
            health_args.append((no_update_grace_s, max_image_age_s))
            return False

    monkeypatch.setattr(runner_module, "BackgroundTracker", _StalledButCachedTracker)
    steps = parse_sequence(
        [
            {"op": "track_detect", "params": {"object_name": "banana"}, "bind": "banana"},
            {"op": "close_gripper"},
        ],
        allowed_ops=set(_index(api)),
        special_ops={"track_detect"},
    )
    cfg = _tracking_config(
        servo=ServoConfig(control_hz=100.0, timeout_s=1.0, lost_target_grace_s=0.03),
    )

    result = run_sequence(
        types.SimpleNamespace(api=api, env=api.env),
        steps,
        config=cfg,
        action_index=_index(api),
    )

    assert result["ok"] is False
    assert "target_lost" in result["steps"][-1]["reason"]
    assert health_args == [(0.03, runner_module._MAX_TRACKING_IMAGE_AGE_S)]
    assert ("close",) not in api.calls


class _ScriptedServo:
    """A ServoController mock that returns scripted results in order.

    Each ``run()`` call pops the next result. This lets a single test exercise
    approach-then-descend (and optional re-align) with controlled outcomes.
    """

    script: list[ServoResult] = []

    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        return self.script.pop(0)


def _track_grasp_sequence(object_name="banana", approach_mm=40.0):
    return [
        {"op": "track_grasp", "params": {"object_name": object_name, "approach_mm": approach_mm}, "bind": "banana"},
        {"op": "close_gripper"},
    ]


def _run_track_grasp(api, cfg, monkeypatch=None):
    steps = parse_sequence(_track_grasp_sequence(), allowed_ops=set(_index(api)), special_ops={"track_grasp"})
    return run_sequence(
        types.SimpleNamespace(api=api, env=api.env),
        steps,
        config=cfg,
        action_index=_index(api),
    )


def test_track_grasp_approach_failure_skips_descend_and_close(monkeypatch):
    api = _EyeToHandApi({"banana": {"ok": True, "position": [200.0, 150.0, 70.0], "grasp_z": 50.0}})
    api.env = _EyeToHandEnv(api)
    _ScriptedServo.script = [
        ServoResult(False, "timeout", 3, 0.1, api.pose, None),
        ServoResult(True, "reached", 1, 0.01, api.pose, None),  # must NOT run
    ]
    monkeypatch.setattr(runner_module, "ServoController", _ScriptedServo)
    result = _run_track_grasp(api, _tracking_config(first_target_timeout_s=1.0), monkeypatch)
    assert result["ok"] is False
    assert "approach failed" in result["steps"][-1]["reason"]
    assert ("close",) not in api.calls


def test_track_grasp_descend_failure_skips_close(monkeypatch):
    api = _EyeToHandApi({"banana": {"ok": True, "position": [200.0, 150.0, 70.0], "grasp_z": 50.0}})
    api.env = _EyeToHandEnv(api)
    _ScriptedServo.script = [
        ServoResult(True, "reached", 1, 0.01, api.pose, None),  # approach ok
        ServoResult(False, "target_lost", 3, 0.1, api.pose, None),  # descend fails
    ]
    monkeypatch.setattr(runner_module, "ServoController", _ScriptedServo)
    result = _run_track_grasp(api, _tracking_config(first_target_timeout_s=1.0), monkeypatch)
    assert result["ok"] is False
    assert "descend failed" in result["steps"][-1]["reason"]
    assert ("close",) not in api.calls


def test_track_grasp_target_lost_aborts_before_close(monkeypatch):
    api = _EyeToHandApi({"banana": {"ok": True, "position": [200.0, 150.0, 70.0], "grasp_z": 50.0}})
    api.env = _EyeToHandEnv(api)
    _ScriptedServo.script = [ServoResult(False, "target_lost", 2, 0.1, api.pose, None)]
    monkeypatch.setattr(runner_module, "ServoController", _ScriptedServo)
    result = _run_track_grasp(api, _tracking_config(first_target_timeout_s=1.0), monkeypatch)
    assert result["ok"] is False
    assert "approach failed" in result["steps"][-1]["reason"]
    assert ("close",) not in api.calls


def test_track_grasp_timeout_aborts_before_close(monkeypatch):
    api = _EyeToHandApi({"banana": {"ok": True, "position": [200.0, 150.0, 70.0], "grasp_z": 50.0}})
    api.env = _EyeToHandEnv(api)
    _ScriptedServo.script = [ServoResult(False, "timeout", 100, 5.0, api.pose, None)]
    monkeypatch.setattr(runner_module, "ServoController", _ScriptedServo)
    result = _run_track_grasp(api, _tracking_config(first_target_timeout_s=1.0), monkeypatch)
    assert result["ok"] is False
    assert "approach failed" in result["steps"][-1]["reason"]


def test_track_grasp_re_aligns_then_fails_closed_when_target_keeps_jumping(monkeypatch):
    # The post-descend detection jumps beyond the reach tolerance on every
    # generation. The runner must (a) attempt a re-align descend (proving the
    # post-descend check fired) and (b) fail-closed once the re-align budget
    # is exhausted, rather than close on a stale position. The servo mock does
    # NOT move the arm, so the tip-vs-target gap never closes.
    class _JumpingBanana(_EyeToHandApi):
        def __init__(self):
            super().__init__({})
            self._x = 200.0
            self.run_count = 0

        def get_grasp_info_simple(self, object_name):
            x = self._x
            self._x += 100.0  # +100mm per detection — always beyond tolerance
            return {"ok": True, "position": [x, 150.0, 70.0], "grasp_z": 50.0}

    api = _JumpingBanana()
    api.env = _EyeToHandEnv(api)
    runs: list[str] = []

    class _CountingNoOpServo:
        def __init__(self, read_pose, servo_to, target_provider, *, config=None, **kwargs):
            pass

        def run(self):
            runs.append("run")
            return ServoResult(True, "reached", 1, 0.01, api.pose, None)

    monkeypatch.setattr(runner_module, "ServoController", _CountingNoOpServo)
    cfg = _tracking_config(first_target_timeout_s=1.0, max_re_align_iters=1)
    result = _run_track_grasp(api, cfg, monkeypatch)
    assert result["ok"] is False
    # approach + descend + at least one re-align attempt (proves the
    # post-descend gap check fired before close).
    assert len(runs) >= 3
    assert ("close",) not in api.calls


class _DeterministicTracker:
    script: list[dict] = []

    def __init__(self, *args, **kwargs):
        targets = [dict(target) for target in self.script]
        self._latest = targets.pop(0)
        self._remaining = targets
        self._detections = 1

    @property
    def detections(self):
        return self._detections

    def start(self):
        return self

    def stop(self):
        return None

    def wait_first(self, timeout_s):
        return True

    def latest_target(self):
        return dict(self._latest)

    def wait_for_next(self, previous_detections, timeout_s):
        if not self._remaining:
            return None, 0.0
        self._latest = self._remaining.pop(0)
        self._detections += 1
        return dict(self._latest), runner_module.time.monotonic()

    def wait_for_capture_after(self, capture_threshold_t, *, timeout_s=5.0):
        # Pop the next scripted frame. Its capture time is "now" (monotonic),
        # which is >= any earlier descend_finished_t, so it is accepted —
        # mirroring how the real tracker stamps a freshly-grabbed frame.
        if not self._remaining:
            return None
        self._latest = self._remaining.pop(0)
        self._detections += 1
        return dict(self._latest), runner_module.time.monotonic()


def test_track_grasp_grasp_z_jump_re_aligns_then_fails_closed(monkeypatch):
    initial = {"x": 200.0, "y": 150.0, "z": 50.0, "position": [200.0, 150.0, 50.0], "grasp_z": 50.0}
    post = {**initial, "grasp_z": 70.0}
    final = {**initial, "grasp_z": 90.0}
    _DeterministicTracker.script = [initial, post, final]
    monkeypatch.setattr(runner_module, "BackgroundTracker", _DeterministicTracker)

    api = _EyeToHandApi({"banana": {"ok": True, "position": [200.0, 150.0, 50.0], "grasp_z": 50.0}})
    api.env = _EyeToHandEnv(api)
    api.pose.update(x=200.0, y=150.0, z=50.0)
    runs: list[str] = []

    class _NoOpServo:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            runs.append("run")
            return ServoResult(True, "reached", 1, 0.01, api.pose, None)

    monkeypatch.setattr(runner_module, "ServoController", _NoOpServo)
    result = _run_track_grasp(api, _tracking_config(max_re_align_iters=1), monkeypatch)

    assert result["ok"] is False
    assert len(runs) == 3
    assert ("close",) not in api.calls


def test_track_grasp_successful_re_align_allows_close(monkeypatch):
    initial = {"x": 200.0, "y": 150.0, "z": 50.0, "position": [200.0, 150.0, 50.0], "grasp_z": 50.0}
    moved = {**initial, "grasp_z": 70.0}
    _DeterministicTracker.script = [initial, moved, moved]
    monkeypatch.setattr(runner_module, "BackgroundTracker", _DeterministicTracker)

    api = _EyeToHandApi({"banana": {"ok": True, "position": [200.0, 150.0, 50.0], "grasp_z": 50.0}})
    api.env = _EyeToHandEnv(api)
    api.pose.update(x=200.0, y=150.0, z=50.0)

    class _MoveToTargetServo:
        def __init__(self, read_pose, servo_to, target_provider, *, config=None, **kwargs):
            self._read_pose = read_pose
            self._servo_to = servo_to
            self._target_provider = target_provider

        def run(self):
            target = self._target_provider()
            assert target is not None
            self._servo_to(target)
            return ServoResult(True, "reached", 1, 0.01, self._read_pose(), target)

    monkeypatch.setattr(runner_module, "ServoController", _MoveToTargetServo)
    result = _run_track_grasp(api, _tracking_config(max_re_align_iters=1), monkeypatch)

    assert result["ok"] is True
    assert api.pose["z"] == 70.0
    assert ("close",) in api.calls


def test_post_descend_barrier_skips_in_flight_detection():
    old = {"x": 1.0, "y": 2.0, "z": 3.0}
    fresh = {"x": 4.0, "y": 5.0, "z": 6.0}

    class _BarrierTracker:
        def __init__(self):
            self.detections = 1
            self._frames = [(old, 9.0), (fresh, 11.0)]

        def wait_for_capture_after(self, capture_threshold_t, *, timeout_s=5.0):
            # Pop scripted frames; accept only those whose capture time is
            # >= the threshold (mirroring the real tracker's stamp judgement).
            while self._frames:
                target, capture_t = self._frames.pop(0)
                self.detections += 1
                if capture_t >= capture_threshold_t:
                    return target, capture_t
            return None

    result = runner_module._wait_post_descend_target(_BarrierTracker(), 10.0, timeout_s=0.1)
    assert result == (fresh, 11.0)


def test_post_descend_barrier_accepts_frame_that_landed_in_baseline_gap():
    # Race regression: a frame grabbed *after* descend finished but whose
    # detection completed in the gap between recording ``descend_finished_t``
    # and the barrier reading a baseline counter. The old
    # ``wait_for_next(baseline)`` path would skip it (the baseline had already
    # advanced past it) and, on a subsequent detector stall, misread "no fresh
    # post-descend frame". The capture-time-keyed path accepts it directly.
    fresh = {"x": 4.0, "y": 5.0, "z": 6.0}

    class _AlreadyLandedTracker:
        # The fresh frame is already present (capture_t=11.0 >= threshold=10.0)
        # by the time the barrier runs — no new generation is needed.
        def __init__(self):
            self.detections = 2

        def wait_for_capture_after(self, capture_threshold_t, *, timeout_s=5.0):
            if 11.0 >= capture_threshold_t:
                return dict(fresh), 11.0
            return None

    result = runner_module._wait_post_descend_target(_AlreadyLandedTracker(), 10.0, timeout_s=0.1)
    assert result == (fresh, 11.0)


def test_safe_retreat_homes_when_release_raises():
    calls: list[str] = []

    class _FailingReleaseApi:
        def open_gripper(self):
            calls.append("open")
            raise RuntimeError("gripper jammed")

        def home(self):
            calls.append("home")

    runner_module._safe_retreat(types.SimpleNamespace(api=_FailingReleaseApi(), env=None))
    assert calls == ["open", "home"]


def test_servo_binding_applies_safety_rail_policy_before_dispatch():
    api = _EyeToHandApi({})
    api.env = _EyeToHandEnv(api)
    binding = ServoBinding(types.SimpleNamespace(api=api, env=api.env))

    with pytest.raises(ValueError, match="below z_floor"):
        binding.servo_to({"x": 0.0, "y": 0.0, "z": 5.0, "rz": 0.0})
    assert api.pose["z"] == 100.0


def test_servo_binding_dispatches_after_safety_passes():
    api = _EyeToHandApi({})
    api.env = _EyeToHandEnv(api)
    binding = ServoBinding(types.SimpleNamespace(api=api, env=api.env))

    binding.servo_to({"x": 10.0, "y": 20.0, "z": 80.0, "rz": 12.0})

    assert api.pose == {"x": 10.0, "y": 20.0, "z": 80.0, "rx": 180.0, "ry": 0.0, "rz": 12.0}


def test_runner_literal_offset_still_resolves():
    # No named constants exist, but a literal numeric offset in an expression
    # still evaluates — so a skill that DOES want a small clearance can write one.
    api = _FakeApi(_GRASP_OBJ)
    raw = [
        {"op": "get_grasp_info_simple", "params": {"object_name": "box"}, "bind": "b"},
        {"op": "goto_xyzr", "params": {"x": "b.x", "y": "b.y", "z": "b.grasp_z + 30"}},
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)), special_ops=frozenset())
    res = run_sequence(_session(api), steps, action_index=_index(api))
    assert res["ok"] is True
    assert ("goto", 250.0, 90.0, 80.0) in api.calls  # 50 + 30 literal


def test_runner_is_task_agnostic_position_only():
    # A detection with NO grasp_z/place_z — a generic "go to the object" task.
    api = _FakeApi({"thing": {"ok": True, "position": [100.0, 0.0, 30.0], "score": 0.8}})
    raw = [
        {"op": "get_grasp_info_simple", "params": {"object_name": "thing"}, "bind": "t"},
        {"op": "goto_xyzr", "params": {"x": "t.position[0]", "y": "t.position[1]", "z": "t.z"}},
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)), special_ops=frozenset())
    res = run_sequence(_session(api), steps, action_index=_index(api))
    assert res["ok"] is True
    assert ("goto", 100.0, 0.0, 30.0) in api.calls  # straight to detected z


def test_runner_stops_and_retreats_on_failure():
    api = _FakeApi(_GRASP_OBJ, fail_goto_at=1)  # first goto raises
    raw = [
        {"op": "get_grasp_info_simple", "params": {"object_name": "box"}, "bind": "b"},
        {"op": "goto_xyzr", "params": {"x": "b.x", "y": "b.y", "z": "b.grasp_z"}},
        {"op": "close_gripper"},  # must NOT run after the failure
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)), special_ops=frozenset())
    res = run_sequence(_session(api), steps, action_index=_index(api))

    assert res["ok"] is False
    assert res["steps"][-1]["op"] == "goto_xyzr" and not res["steps"][-1]["ok"]
    assert "EXCEEDS_LIMIT" in res["steps"][-1]["reason"]
    assert ("close",) not in api.calls  # stopped before close
    assert ("home",) in api.calls  # best-effort safe retreat ran


def test_runner_reports_unknown_op_on_robot():
    api = _FakeApi(_GRASP_OBJ)
    # 'wave' is allowed by schema (vocab) but not in the runtime action_index.
    steps = parse_sequence([{"op": "wave"}], allowed_ops={"wave"}, special_ops=frozenset())
    res = run_sequence(_session(api), steps, action_index=_index(api))
    assert res["ok"] is False and "not available" in res["steps"][-1]["reason"]


def test_runner_missing_detection_fails_cleanly():
    api = _FakeApi(_GRASP_OBJ)
    raw = [
        {"op": "get_grasp_info_simple", "params": {"object_name": "ghost"}, "bind": "g"},
        {"op": "goto_xyzr", "params": {"x": "g.x", "y": "g.y", "z": "g.z"}},
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)), special_ops=frozenset())
    res = run_sequence(_session(api), steps, action_index=_index(api))
    # detection returns ok=False → not bound → the goto referencing g.x fails clean
    assert res["ok"] is False


def test_runner_aborts_at_bind_step_when_detection_ran_but_returned_not_ok():
    # Mimics the REAL ability executor: the tool RAN (executor ok=True) but the
    # detection RESULT is ok=False (e.g. no valid depth at the target). Must abort
    # AT the detection step with the real cause — not skip the bind and let a later
    # goto reach the driver with an unresolved "<bind>.field" string.
    api = _FakeApi(_GRASP_OBJ)

    def executor(op, params):
        assert op == "get_grasp_info_simple", f"goto must not run after a failed detection, got {op!r}"
        return {"ok": True, "result": {"ok": False, "reason": "no_depth"}}

    raw = [
        {"op": "get_grasp_info_simple", "params": {"object_name": "white box"}, "bind": "w"},
        {"op": "goto_xyzr", "params": {"x": "w.position[0]", "y": "w.position[1]", "z": "w.place_z"}},
    ]
    steps = parse_sequence(raw, allowed_ops=set(_index(api)), special_ops=frozenset())
    res = run_sequence(_session(api), steps, executor=executor)

    assert res["ok"] is False
    failed = res["steps"][-1]
    assert failed["op"] == "get_grasp_info_simple" and not failed["ok"]
    assert "white box" in failed["reason"] and "no_depth" in failed["reason"]
    assert ("home",) in api.calls  # safe retreat ran
