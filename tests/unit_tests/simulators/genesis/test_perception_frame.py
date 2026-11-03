"""RGB-D publication rejects reset, stale frames and intervening base movement."""

import copy
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.perception.frame import CameraFrame
from jiuwensymbiosis.simulators.genesis.agent import GenesisEnv


def fixture():
    base = {
        "position_m": [0.0, 0.0, 0.24],
        "body_yaw_rad": 0.0,
        "linear_velocity_m_s": [0.0, 0.0, 0.0],
        "angular_velocity_rad_s": [0.0, 0.0, 0.0],
    }
    state = {
        "state": "ready",
        "measured_at": time.monotonic(),
        "scene": {"instance_id": "a"},
        "config_revision": 1,
        "base_motion": {"id": "stopped-1"},
        "base_observation": copy.deepcopy(base),
    }
    driver = SimpleNamespace(
        actions_enabled=False,
        observation_mode="perception",
        cancelled=threading.Event(),
        preview=SimpleNamespace(status=lambda: state),
    )
    frame = CameraFrame(
        np.zeros((4, 4, 3)),
        frame_id="frame",
        captured_monotonic_s=time.monotonic(),
        metadata={"scene_instance_id": "a", "config_revision": 1, "motion_id": "stopped-1", "base_sample": base},
    )
    return GenesisEnv(driver), frame, state


def test_unchanged_stationary_frame_is_usable():
    env, frame, _ = fixture()
    env.validate_frame(frame)


@pytest.mark.parametrize("change", ["reset", "revision", "age", "state_age", "pose", "speed", "moved_back", "cancel"])
def test_stale_or_intervening_movement_cannot_publish_a_location(change):
    env, frame, state = fixture()
    if change == "reset":
        state["scene"]["instance_id"] = "b"
    elif change == "revision":
        state["config_revision"] += 1
    elif change == "age":
        frame.captured_monotonic_s -= 6
    elif change == "state_age":
        state["measured_at"] -= 3
    elif change == "pose":
        state["base_observation"]["position_m"][0] += 0.02
    elif change == "speed":
        state["base_observation"]["linear_velocity_m_s"][0] = 0.1
    elif change == "moved_back":
        state["base_motion"]["id"] = "stopped-3"
    else:
        env.driver.cancelled.set()
    with pytest.raises(RuntimeError):
        env.validate_frame(frame)


def test_pure_rotation_cannot_bypass_the_shared_target_guard():
    from jiuwensymbiosis.motion.target_approach import TargetApproachTuning

    env, frame, _ = fixture()
    frame.metadata["body_bounds_m"] = [[-0.4, -0.3, -0.24], [0.4, 0.3, 1.9]]
    env._last_frame = frame
    assert env.check_target_step(0.0, 0.2, frame.frame_id, TargetApproachTuning()) == "turn_clearance_unobserved"


@pytest.mark.parametrize("action", ["rotate", "relative_turn", "turn_and_translate", "forward", "backward", "lateral"])
def test_agent_motion_tools_cannot_bypass_unobserved_turn(action):
    from jiuwensymbiosis.simulators.genesis.agent import GenesisPerceptionMotionApi

    env, frame, _ = fixture()
    frame.metadata["body_bounds_m"] = [[-0.4, -0.3, -0.24], [0.4, 0.3, 1.9]]
    commands = []

    def navigate(*args):
        commands.append(args)
        return {"ok": True, "settled": True, "stop_confirmation": "measured"}

    env.driver.navigate_relative = navigate
    env.grab_calibrated_frame = lambda: frame
    api = GenesisPerceptionMotionApi(env, lambda *a, **kw: [])
    if action == "rotate":
        result = api.rotate_base(0.2)
    elif action in {"forward", "backward", "lateral"}:
        result = api.navigate_relative(-0.25 if action == "backward" else 0.25, 0.2 if action == "lateral" else 0)
    else:
        result = api.navigate_relative(0.25 if action == "turn_and_translate" else 0.0, 0.0, 0.2)
    assert result["reason"] == (
        "clearance_unavailable"
        if action == "forward"
        else "translation_clearance_unobserved"
        if action in {"backward", "lateral"}
        else "turn_clearance_unobserved"
    )
    assert not result["ok"] and not result["motion_executed"]
    assert result["settled"] and result["stop_confirmation"] == "measured"
    assert commands == [(0.0, 0.0, 0.0)]


@pytest.mark.parametrize("turn", [float("nan"), float("inf"), True, 4.0])
def test_invalid_turn_is_rejected_before_stop_or_capture(turn):
    env, _, _ = fixture()
    commands = []
    env.driver.navigate_relative = lambda *a: commands.append(a)
    with pytest.raises(ValueError):
        env.navigate_relative(0.0, 0.0, turn)
    assert not commands
