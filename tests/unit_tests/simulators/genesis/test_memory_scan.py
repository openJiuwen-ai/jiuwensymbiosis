"""Formal scan wiring uses the Session and checks admission before motion."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jiuwensymbiosis.simulators.common.agent import make_session
from jiuwensymbiosis.simulators.genesis.memory_scan import GenesisScanPort
from tests.unit_tests.spatial_memory.test_capture_session import moving_capture


def setup_port():
    raw = moving_capture()
    raw["base_sample"].update(linear_velocity_m_s=[0, 0, 0], position_m=[0, 0, 0], body_yaw_rad=0)
    state = {
        "state": "ready",
        "measured_at": time.monotonic(),
        "scene": {"instance_id": "epoch"},
        "base_observation": raw["base_sample"],
        "steps": 12,
        "sim_time": 1.0,
    }
    preview = SimpleNamespace(
        status=lambda: state,
        acquire_motion=Mock(return_value="owner"),
        release_motion=Mock(),
        capture_image=Mock(return_value=raw),
        motion_state=lambda _: state,
        navigate_relative=Mock(return_value={"ok": True, "settled": True, "stop_confirmation": "measured"}),
    )

    def query(operation, params, **kwargs):
        result = {"scene_instance_id": "epoch", "reset_generation": 0, "robot": raw["base_sample"]}
        if operation == "plan_navigation":
            result["plan"] = {"path_length_m": 0.0, "map_fingerprint": "static-map", "footprint_radius_m": 0.5}
        return result

    preview.query_scene = Mock(side_effect=query)
    session = make_session(preview, observation_mode="ground_truth")
    session.connect()
    return GenesisScanPort(session, map_frame="room"), preview


def test_connected_session_scan_admits_before_motion_and_releases_owner():
    port, preview = setup_port()
    try:
        obs = port.capture()
        result = port.execute({"path_xy_m": [[0, 0]], "heading_rad": 0.3, "remaining_distance_m": 1}, obs)
        assert result["ok"] and result["stopped"]
        assert preview.query_scene.call_args_list[-1].args[0] == "plan_navigation"
        assert preview.navigate_relative.call_count == 1
    finally:
        port.session.disconnect()
    preview.release_motion.assert_called_once_with("owner")


def test_refused_geometry_never_dispatches_motion():
    port, preview = setup_port()
    obs = port.capture()
    original = preview.query_scene.side_effect

    def refuse(operation, params, **kwargs):
        if operation == "plan_navigation":
            raise ValueError("navigation_goal_blocked")
        return original(operation, params, **kwargs)

    preview.query_scene.side_effect = refuse
    try:
        result = port.execute({"path_xy_m": [[0, 0]], "heading_rad": 0.3, "remaining_distance_m": 1}, obs)
        assert not result["motion_executed"] and result["stopped"]
        preview.navigate_relative.assert_not_called()
    finally:
        port.session.disconnect()


def test_cancellation_prevents_new_motion():
    port, preview = setup_port()
    obs = port.capture()
    port.driver.cancelled.set()
    try:
        with pytest.raises(RuntimeError, match="取消"):
            port.execute({"path_xy_m": [[0, 0]], "heading_rad": 0.3, "remaining_distance_m": 1}, obs)
        preview.navigate_relative.assert_not_called()
    finally:
        port.session.disconnect()


def test_unconfirmed_stop_revokes_motion_and_keeps_lease():
    port, preview = setup_port()
    obs = port.capture()
    preview.navigate_relative.return_value = {"ok": False, "settled": False, "stop_confirmation": "unknown"}
    result = port.execute({"path_xy_m": [[0, 0]], "heading_rad": 0.3, "remaining_distance_m": 1}, obs)
    assert not result["stopped"]
    assert port.driver.cancelled.is_set()
    with pytest.raises(RuntimeError, match="停车"):
        port.driver.close()
    preview.release_motion.assert_not_called()
    # Simulate the operator stopping this fake backend to permit cleanup.
    preview.status = lambda: {"state": "stopped"}
    port.session.disconnect()
