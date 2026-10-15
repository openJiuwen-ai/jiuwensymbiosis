"""Boundary tests for the HTTP/physics adapter; no Isaac or external service."""

from __future__ import annotations

import io
import json
from http.client import IncompleteRead
from unittest.mock import Mock
from urllib.error import URLError

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jiuwensymbiosis.env.protocol import NamedJointDriver, RobotDriver
from jiuwensymbiosis.errors import GraspNotConfirmedError
from jiuwensymbiosis.introspect import build_session
from jiuwensymbiosis.simulators.robosim import RobosimApi, RobosimConfig, RobosimEnv, build_robosim_session
from jiuwensymbiosis.simulators.robosim.lowlevel import RobosimError, RobosimLowLevel, base_pose
from jiuwensymbiosis.tools.robot_control_tool import RobotControlTool


def observation(joint: float = 0.0) -> dict:
    """Provide a measured pose in a translated and rotated base frame."""
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    base[:3, 3] = [1, 2, 0]
    return {
        "bridge_version": 1,
        "robot_type": "moz1",
        "world_from_base": base.tolist(),
        "joint_positions": {"wrist": joint},
        "joint_limits": {"wrist": [-1, 1]},
        "tcp_world": {"position": {"x": 1, "y": 2.3, "z": 0.4}, "rotation": {"roll": 0, "pitch": 0, "yaw": np.pi / 2}},
        "reported_state": {"holding": None, "grasps": []},
        "base_frame": "/base",
        "simulation_time_s": 1.0,
    }


def test_rotated_base_units():
    pose = base_pose(observation())
    assert [pose[k] for k in ("x", "y", "z", "rx", "ry", "rz")] == pytest.approx([300, 0, 400, 0, 0, 0])


def test_tcp_roundtrip_uses_world_frame(monkeypatch):
    driver = RobosimLowLevel(RobosimConfig())
    monkeypatch.setattr(driver, "observe", lambda: observation())
    request = Mock(return_value={})
    monkeypatch.setattr(driver, "request", request)
    driver.move_to_pose_blocking({"x": 300, "y": 0, "z": 400, "rx": 0, "ry": 0, "rz": 0})
    assert request.call_args.args[:2] == ("POST", "/skills/move_end_effector")
    payload = request.call_args.args[2]
    assert payload["resource_id"] == "left_arm"
    assert payload["target_pose"]["position"] == pytest.approx({"x": 1, "y": 2.3, "z": 0.4})
    assert payload["target_pose"]["rotation"]["yaw"] == pytest.approx(np.pi / 2)


def test_named_joint_readback_not_request_echo(monkeypatch):
    driver = RobosimLowLevel(RobosimConfig())
    states = iter([observation(), observation(0.03), observation(0.095)])
    monkeypatch.setattr(driver, "observe", lambda: next(states))
    request = Mock(return_value={"arm_joints": [0.1]})
    monkeypatch.setattr(driver, "request", request)
    driver.move_joints_blocking({"wrist": 0.1})
    assert request.call_count == 2


@pytest.mark.parametrize("targets", [{"unknown": 0}, {"wrist": float("nan")}, {"wrist": 2}, {"wrist": True}, {}])
def test_invalid_joint_never_dispatched(monkeypatch, targets):
    driver = RobosimLowLevel(RobosimConfig())
    monkeypatch.setattr(driver, "observe", observation)
    request = Mock()
    monkeypatch.setattr(driver, "request", request)
    with pytest.raises(ValueError):
        driver.move_joints_blocking(targets)
    request.assert_not_called()


def test_http_success_is_not_action_success():
    driver = RobosimLowLevel(RobosimConfig())
    driver.client._opener = Mock()
    driver.client._opener.open.return_value = io.BytesIO(
        json.dumps({"success": False, "error": "IK failed", "failure_reason": "ik_failed"}).encode()
    )
    with pytest.raises(RobosimError) as exc:
        driver.request("POST", "/move_end_effector", {})
    assert exc.value.code == "ik_failed"


@pytest.mark.parametrize("error", [URLError("timeout"), TimeoutError(), IncompleteRead(b"partial")])
def test_transport_error_never_retries(error):
    driver = RobosimLowLevel(RobosimConfig())
    driver.client._opener = Mock()
    driver.client._opener.open.side_effect = error
    with pytest.raises(RobosimError, match="outcome is unknown"):
        driver.request("POST", "/robot/joints", {"joint_values": {"wrist": 0.1}})
    assert driver.client._opener.open.call_count == 1
    with pytest.raises(RobosimError, match="Previous command"):
        driver.request("POST", "/robot/gripper/open", {})
    assert driver.client._opener.open.call_count == 1


def test_nonfinite_observation_is_not_success(monkeypatch):
    driver = RobosimLowLevel(RobosimConfig())
    state = observation(float("nan"))
    monkeypatch.setattr(driver, "request", lambda *args: state)
    with pytest.raises(RobosimError) as exc:
        driver.observe()
    assert exc.value.code == "robosim_observation_invalid"


def test_lifecycle_does_not_own_service():
    env = RobosimEnv(RobosimConfig())
    driver = Mock(spec=RobosimLowLevel)
    env.low_level = driver
    env.connect()
    env.connect()
    env.disconnect()
    env.disconnect()
    driver.connect.assert_called_once()
    driver.close.assert_called_once()
    driver.request.assert_not_called()


def test_partial_start_closes_client():
    env = RobosimEnv(RobosimConfig())
    driver = Mock(spec=RobosimLowLevel)
    driver.connect.side_effect = RobosimError("not ready")
    env.low_level = driver
    with pytest.raises(RobosimError):
        env.connect()
    driver.close.assert_called_once()
    with pytest.raises(RuntimeError, match="not connected"):
        _ = env.low_level


async def test_empty_close_does_not_commit_payload_effect():
    session = build_robosim_session(RobosimConfig())
    driver = Mock(spec=RobosimLowLevel)
    driver.set_gripper.return_value = {"holding": None}
    session.env.low_level = driver
    tool = RobotControlTool(session.api, env=session.env)
    result = await tool.invoke({"action": "close_gripper"})
    assert result.success is False
    assert "grasp_not_confirmed" in str(result.data)
    assert "payload.held" not in session.api.memory.self_state
    with pytest.raises(GraspNotConfirmedError):
        session.api.close_gripper()


def test_config_discovery_and_capability_gates():
    session = build_session("configs/robosim/moz1.yaml")
    assert isinstance(session.api, RobosimApi)
    assert session.api.capabilities & session.env.capabilities == frozenset(
        {"motion.joint", "motion.cartesian", "grasp.parallel"}
    )
    driver = RobosimLowLevel(RobosimConfig())
    assert isinstance(driver, RobotDriver)
    assert isinstance(driver, NamedJointDriver)


@pytest.mark.parametrize(
    "kwargs",
    [{"side": "both"}, {"service_url": "http://remote:8001"}, {"load_robot": "false"}, {"request_timeout_s": -1}],
)
def test_config_rejects_unsupported_values(kwargs):
    with pytest.raises(ValueError):
        RobosimConfig(**kwargs)
