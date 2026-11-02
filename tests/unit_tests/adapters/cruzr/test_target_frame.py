"""Target sensing keeps sensor identity and rejects motion during inference."""

import copy
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv
from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel
from jiuwensymbiosis.perception.frame import CameraFrame, FrameValidationError


def test_strict_driver_surfaces_are_explicit_optional_protocols():
    from jiuwensymbiosis.env.protocol import BaseFeedbackDriver, CalibratedCameraDriver

    driver = object.__new__(CruzrLowLevel)
    assert isinstance(driver, BaseFeedbackDriver)
    assert isinstance(driver, CalibratedCameraDriver)


def sample():
    return {
        "position_m": [0.0, 0.0, 0.0],
        "body_yaw_rad": 0.0,
        "linear_velocity_m_s": [0.0, 0.0, 0.0],
        "angular_velocity_rad_s": [0.0, 0.0, 0.0],
        "odom_frame": "odom",
        "base_frame": "base_link",
        "motion_generation": 0,
        "stamp_ns": 100_000_000_000,
        "measured_at": time.monotonic(),
    }


def fixture():
    state = sample()
    transform = np.eye(4)
    transform[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    frame = CameraFrame(
        np.zeros((96, 128, 3), np.uint8),
        np.full((96, 128), 2.0),
        np.array([[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]]),
        transform,
        frame_id="sensor-100",
        captured_monotonic_s=time.monotonic(),
        metadata={
            "color_stamp_ns": 100_000_000_000,
            "depth_stamp_ns": 100_000_000_000,
            "depth_sync_delta_ms": 0.0,
            "tf_used_latest": False,
            "base_before": copy.deepcopy(state),
            "base_sample": copy.deepcopy(state),
        },
    )
    env = CruzrEnv(CruzrConfig())
    env.low_level = SimpleNamespace(grab_calibrated_frame=lambda **kw: frame, read_base_state=lambda **kw: state)
    return env, frame, state


def test_stationary_frame_keeps_sensor_identity_and_can_be_localized():
    env, frame, _ = fixture()
    mask = np.ones(frame.depth_m.shape, bool)
    api = CruzrApi(env, detector_client=lambda *a, **kw: [{"mask": mask, "score": 0.9}])
    result = api.locate_target("box")
    assert result["ok"] and result["frame_id"] == "sensor-100"
    assert env.target_frame is frame
    assert result["captured_monotonic_s"] == frame.captured_monotonic_s


@pytest.mark.parametrize(
    "change,reason",
    [
        ("age", "stale_or_unsynchronized_rgbd"),
        ("future", "stale_or_unsynchronized_rgbd"),
        ("rgbd_mismatch", "stale_or_unsynchronized_rgbd"),
        ("latest_tf", "camera_tf_not_at_capture_time"),
        ("pose", "base_moved_during_sensing"),
        ("yaw", "base_moved_during_sensing"),
        ("moving", "base_moved_during_sensing"),
        ("moved_back", "base_moved_during_sensing"),
        ("odom_reset", "base_moved_during_sensing"),
    ],
)
def test_capture_guard_rejects_missing_or_changed_evidence(change, reason):
    env, frame, state = fixture()
    if change == "age":
        frame.captured_monotonic_s -= 6
    elif change == "future":
        frame.captured_monotonic_s += 1
    elif change == "rgbd_mismatch":
        frame.metadata["depth_sync_delta_ms"] = 121
    elif change == "latest_tf":
        frame.metadata["tf_used_latest"] = True
    elif change == "pose":
        state["position_m"][0] += 0.02
    elif change == "yaw":
        state["body_yaw_rad"] += 0.05
    elif change == "moving":
        state["linear_velocity_m_s"][0] = 0.1
    elif change == "moved_back":
        state["motion_generation"] += 2
    elif change == "odom_reset":
        state["odom_frame"] = "new_odom"
    with pytest.raises(FrameValidationError, match=reason):
        env.validate_target_frame(frame)


def test_inference_motion_rejected_before_location_cache_publication():
    env, frame, state = fixture()

    def infer(*args, **kwargs):
        state["motion_generation"] += 2  # moved away and returned during inference
        return [{"mask": np.ones(frame.depth_m.shape, bool), "score": 0.9}]

    api = CruzrApi(env, detector_client=infer)
    result = api.locate_target("box")
    assert not result["ok"] and result["reason"] == "base_moved_during_sensing"
    assert api.last_detection is None


def test_stale_odom_is_not_replaced_by_last_successful_motion():
    env, frame, _ = fixture()

    def unavailable(**kwargs):
        raise RuntimeError("odom stopped")

    env.low_level.read_base_state = unavailable
    with pytest.raises(FrameValidationError, match="stale_odometry"):
        env.validate_target_frame(frame)


def test_cruzr_target_turn_requires_a_body_profile_and_observed_depth():
    from jiuwensymbiosis.motion.target_approach import TargetApproachTuning

    env, frame, _ = fixture()
    env.target_frame = frame
    tuning = TargetApproachTuning()
    assert env.check_target_step(0.0, 0.2, frame.frame_id, tuning) == "body_geometry_unavailable"
    env.cfg.target_body_bounds_m = [[-0.4, -0.3, 0.0], [0.4, 0.3, 1.9]]
    assert env.check_target_step(0.0, 0.2, frame.frame_id, tuning) == "turn_clearance_unobserved"


@pytest.mark.parametrize("bounds", [[], [[0, 0]], [[0, 0, 0], [1, True, 2]], [[0, 0, 0], [1, 1, float("nan")]]])
def test_target_body_profile_rejects_invalid_configuration(bounds):
    with pytest.raises(ValueError, match="target_body_bounds_m"):
        CruzrConfig(target_body_bounds_m=bounds)


def message(stamp_ns, speed=0.0, x=0.0):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9), frame_id="odom"),
        child_frame_id="base_link",
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=0.0, z=0.0), orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=speed, y=0.0, z=0.0), angular=SimpleNamespace(x=0.0, y=0.0, z=0.0)
            )
        ),
    )


def test_odom_replay_delay_and_motion_generation(monkeypatch):
    driver = object.__new__(CruzrLowLevel)
    driver.cfg = CruzrConfig()
    driver._lock = threading.RLock()
    driver._base_state = None
    driver._base_motion_generation = 0
    driver._odom_clock_offset_s = None
    clock = [100_000_000_000]
    monkeypatch.setattr("jiuwensymbiosis.adapters.cruzr.lowlevel.time.monotonic", lambda: clock[0] / 1e9 + 1234.0)
    driver._node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=clock[0])))
    driver._on_base_odom(message(clock[0]))
    initial = copy.deepcopy(driver._base_state)
    driver._on_base_odom(message(clock[0], speed=1.0))
    assert driver._base_state == initial  # duplicate cannot refresh receipt or motion
    clock[0] += 500_000_000
    driver._on_base_odom(message(clock[0] - 300_000_000, speed=1.0))
    assert driver._base_state == initial  # delayed DDS delivery is not fresh odom
    driver._on_base_odom(message(clock[0], speed=0.1, x=0.02))
    clock[0] += 20_000_000
    driver._on_base_odom(message(clock[0]))
    assert driver._base_state["position_m"] == [0.0, 0.0, 0.0]
    assert driver._base_state["motion_generation"] == 1


def test_readback_waits_for_a_measurement_newer_than_the_capture(monkeypatch):
    from jiuwensymbiosis.adapters.cruzr import lowlevel

    driver = object.__new__(CruzrLowLevel)
    driver.cfg = CruzrConfig()
    driver._lock = threading.RLock()
    driver._closed = False
    driver._node = object()
    driver._odom_subscription = object()
    driver._base_state = sample()
    original = driver._base_state["stamp_ns"]
    waits = []

    def advance(dt):
        waits.append(dt)
        driver._base_state = {**sample(), "stamp_ns": original + 20_000_000}

    monkeypatch.setattr(lowlevel.time, "sleep", advance)
    result = driver.read_base_state(after_stamp_ns=original)
    assert waits and result["stamp_ns"] > original
