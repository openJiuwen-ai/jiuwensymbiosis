"""Actual feedback, not commanded displacement, determines navigation success."""

import math
import time
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jiuwensymbiosis.simulators.genesis.agent import make_session
from jiuwensymbiosis.simulators.genesis.navigation import NavigationMap


@pytest.fixture
def mobile():
    state = {
        "position_m": [0.0, 0.0, 0.2],
        "body_yaw_rad": math.pi / 2,
        "linear_velocity_m_s": [0, 0, 0],
        "angular_velocity_rad_s": [0, 0, 0],
    }
    calls = []
    preview = SimpleNamespace(acquire_motion=Mock(return_value="owner"), release_motion=Mock())

    def query(operation, params, **kwargs):
        result = {
            "source": "simulation_ground_truth",
            "scene_instance_id": "a",
            "reset_generation": 0,
            "captured_at": time.monotonic(),
            "robot": deepcopy(state),
        }
        if operation == "plan_navigation":
            result["plan"] = NavigationMap.plane([0, 0]).plan(
                state["position_m"], [params["x_m"], params["y_m"], params["yaw_rad"]]
            )
        return result

    def move(owner, dx, dy, yaw, **kwargs):
        calls.append((dx, dy, yaw))
        state["body_yaw_rad"] += yaw
        c, s = math.cos(state["body_yaw_rad"]), math.sin(state["body_yaw_rad"])
        state["position_m"][0] += c * dx - s * dy
        state["position_m"][1] += s * dx + c * dy
        return {"ok": True, "settled": True}

    preview.query_scene = Mock(side_effect=query)
    preview.navigate_relative = Mock(side_effect=move)
    session = make_session(preview, observation_mode="ground_truth")
    session.connect()
    yield session, preview, calls, state
    session.disconnect()


def test_world_goal_uses_measured_body_frame_and_exact_final_heading(mobile):
    session, preview, calls, state = mobile
    result = session.api.navigate_to_pose(1.2, 0.6, -math.pi / 2)
    assert result["ok"] and result["goal_reached"] and result["settled"]
    assert result["stop_confirmation"] == "measured"
    assert state["position_m"][:2] == pytest.approx([1.2, 0.6])
    assert state["body_yaw_rad"] == pytest.approx(-math.pi / 2)
    assert all(math.hypot(dx, dy) <= 1 for dx, dy, _ in calls)
    assert calls[0][0] > 0 and calls[0][1] < 0  # World +X is body-right at pi/2.


def test_successful_command_with_wrong_measured_position_is_not_arrival(mobile):
    session, preview, calls, state = mobile
    preview.navigate_relative.side_effect = lambda *a, **k: {"ok": True, "settled": True}
    result = session.api.navigate_to_pose(1, 0, 0)
    assert not result["ok"] and not result["goal_reached"]
    assert result["reason"] == "navigation_waypoint_error"
    assert preview.navigate_relative.call_count == 1
    assert session.env.low_level.results[-1] == result
    assert result["final_state"]["robot"]["position_m"] == state["position_m"]


def test_stopped_timeout_does_not_execute_rest_of_route(mobile):
    session, preview, calls, state = mobile
    preview.navigate_relative.side_effect = lambda *a, **k: {"ok": False, "settled": True, "reason": "motion_timeout"}
    result = session.api.navigate_to_pose(2, 1, 0)
    assert not result["goal_reached"] and result["settled"] and result["recoverable"]
    assert preview.navigate_relative.call_count == 1
    with pytest.raises(RuntimeError, match="sim_get_robot_state"):
        session.api.navigate_to_pose(2, 1, 0)


def test_cancel_between_waypoints_never_sends_another_command(mobile):
    session, preview, calls, state = mobile
    move = preview.navigate_relative.side_effect

    def cancel(*args, **kwargs):
        result = move(*args, **kwargs)
        session.env.low_level.cancelled.set()
        return result

    preview.navigate_relative.side_effect = cancel
    with pytest.raises(RuntimeError, match="取消"):
        session.api.navigate_to_pose(2, 0, 0)
    assert preview.navigate_relative.call_count == 1


@pytest.mark.parametrize("goal_x", [0, 2])
@pytest.mark.parametrize("fault", ["velocity", "unconfirmed"])
def test_new_velocity_after_reported_stop_cannot_certify_navigation_or_release(mobile, goal_x, fault):
    session, preview, calls, state = mobile
    move = preview.navigate_relative.side_effect

    def drifting(*args, **kwargs):
        result = move(*args, **kwargs)
        if fault == "velocity":
            state["linear_velocity_m_s"] = [0.1, 0, 0]
        else:
            result["settled"] = False
        return result

    preview.navigate_relative.side_effect = drifting
    preview.status = lambda: {"state": "ready"}
    result = session.api.navigate_to_pose(goal_x, 0, 0)
    assert not result["ok"] and not result["settled"] and result["stop_confirmation"] == "unknown"
    assert session.env.low_level.cancelled.is_set()
    assert preview.navigate_relative.call_count == 1
    with pytest.raises(RuntimeError, match="保留控制权"):
        session.env.low_level.close()
    assert session.env.low_level.owner == "owner"
    preview.release_motion.assert_not_called()
    preview.status = lambda: {"state": "stopped"}
