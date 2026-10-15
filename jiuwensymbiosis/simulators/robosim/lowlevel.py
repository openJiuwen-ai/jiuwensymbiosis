"""Synchronous HTTP driver; physical execution remains owned by RoboSim."""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeVar

import numpy as np
from scipy.spatial.transform import Rotation

from jiuwensymbiosis.simulators.robosim.client import (
    RobosimClient,
    ensure_robot,
    identity_token,
    read_identity,
    read_robot,
)
from jiuwensymbiosis.simulators.robosim.config import RobosimConfig
from jiuwensymbiosis.simulators.robosim.operations import RobosimError as RobosimError

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _commanded(
    method: Callable[Concatenate[RobosimLowLevel, _P], _R],
) -> Callable[Concatenate[RobosimLowLevel, _P], _R]:
    """Hold command ownership until the driver's measured readback is complete."""

    @functools.wraps(method)
    def run(self: RobosimLowLevel, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self.client.commands.command(method.__name__):
            return method(self, *args, **kwargs)

    return run


def pose_matrix(pose: dict[str, Any]) -> np.ndarray:
    """Decode RoboSim world poses (meters, XYZ Euler radians)."""
    matrix = np.eye(4)
    matrix[:3, 3] = [pose["position"][k] for k in ("x", "y", "z")]
    matrix[:3, :3] = Rotation.from_euler("xyz", [pose["rotation"][k] for k in ("roll", "pitch", "yaw")]).as_matrix()
    return matrix


def base_pose(observation: dict[str, Any]) -> dict[str, float]:
    """Convert actual TCP world pose to the shared base-frame mm/degree contract."""
    matrix = np.linalg.inv(np.asarray(observation["world_from_base"])) @ pose_matrix(observation["tcp_world"])
    values = [*(matrix[:3, 3] * 1000), *Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)]
    return dict(zip(("x", "y", "z", "rx", "ry", "rz"), map(float, values), strict=True))


class RobosimLowLevel:
    """Named-joint and TCP driver for one arm on a separately owned RoboSim service."""

    def __init__(
        self,
        cfg: RobosimConfig,
        *,
        client: RobosimClient | None = None,
        expected_identity: tuple[str, str | None] | None = None,
    ) -> None:
        """Borrow a workspace client or own one standalone transport."""
        self.cfg = cfg
        self.client = client or RobosimClient(cfg)
        self._owns_client = client is None
        self._expected_identity = expected_identity
        self._closed = False
        self._gripper_state: bool | None = None
        self.home_pose: dict[str, float] | None = None

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        """Preserve the driver's request interface through the common transport."""
        if self._closed:
            raise RobosimError("Driver is closed", code="robosim_disconnected")
        if method != "GET":
            self._check_identity()
        return self.client.request(method, path, payload)

    def _check_identity(self, identity: dict | None = None) -> None:
        if self._expected_identity is not None:
            identity = read_identity(self.client.request) if identity is None else identity
            if identity.get("scene_phase") in {"loading", "failed"}:
                raise RobosimError("Scene is not ready; reset the scene first", code="robosim_scene_not_ready")
            if identity_token(identity) != self._expected_identity:
                raise RobosimError("Robot session is stale; start a new turn", code="robosim_robot_changed")

    def check_session(self, identity: dict | None = None) -> None:
        """Reject stale/closed sessions; a caller may share its freshly read snapshot."""
        if self._closed:
            raise RobosimError("Driver is closed", code="robosim_disconnected")
        self._check_identity(identity)

    def connect(self) -> None:
        """Attach to the configured body, optionally loading it into an empty service."""
        if not self.request("GET", "/sim/health").get("ready"):
            raise RobosimError("RoboSim is not ready", code="robosim_not_ready")
        identity = read_identity(self.client.request)
        self._check_identity(identity)
        self.client.commands.observe_service(identity["service_instance_id"])
        with self.client.commands.command("connect"):
            state = read_robot(self.request)
            if read_identity(self.client.request) != identity:
                raise RobosimError("Robot changed during connection", code="robosim_robot_changed")
            ensure_robot(self.request, self.cfg.robot_type, state, load=self.cfg.load_robot)
            current = read_identity(self.client.request)
            if current["service_instance_id"] != identity["service_instance_id"] or (
                state is not None and identity_token(current) != identity_token(identity)
            ):
                raise RobosimError("Robot changed during connection", code="robosim_robot_changed")
            if current["robot_instance_id"] is None:
                raise RobosimError(
                    "Robot loading was not confirmed", code="robosim_load_unconfirmed", outcome_unknown=True
                )
            self._expected_identity = identity_token(current)
        observation = self.observe()
        self.home_pose = base_pose(observation)

    def observe(self) -> dict[str, Any]:
        """Read physical joints/TCP, not the service's cached arm_joints vector."""
        self._check_identity()
        result = self.request("GET", f"/symbiosis/observation?side={self.cfg.side}")
        self._check_identity()
        if result.get("bridge_version") != 1 or result.get("robot_type") != self.cfg.robot_type:
            raise RobosimError("Observation bridge/body mismatch", code="robosim_protocol_error")
        transform = np.asarray(result["world_from_base"], dtype=float)
        joints = result["joint_positions"]
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RobosimError("Observation contains invalid physical state", code="robosim_observation_invalid")
        if not joints or not all(math.isfinite(v) for v in joints.values()):
            raise RobosimError("Observation contains invalid physical state", code="robosim_observation_invalid")
        if not np.isfinite(pose_matrix(result["tcp_world"])).all():
            raise RobosimError("Observation contains invalid physical state", code="robosim_observation_invalid")
        return result

    def get_joint_positions(self) -> dict[str, float]:
        """Return selected arm joint names and measured positions in radians."""
        return {name: float(value) for name, value in self.observe()["joint_positions"].items()}

    @_commanded
    def move_joints_blocking(self, targets: dict[str, float], *, timeout_s: float | None = None) -> dict[str, float]:
        """Hold a named setpoint until measured joints converge or the deadline expires."""
        timeout = self.cfg.joint_timeout_s if timeout_s is None else timeout_s
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout_s must be finite and positive")
        observation = self.observe()
        if not targets or set(targets) - observation["joint_positions"].keys():
            raise ValueError("targets must name joints on the selected arm")
        for name, value in targets.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Non-finite or non-numeric target: {name}")
            limits = observation["joint_limits"].get(name)
            if limits and not limits[0] <= value <= limits[1]:
                raise ValueError(f"Target outside authored joint limits: {name}")
        deadline = time.monotonic() + timeout
        while True:
            # The existing endpoint advances five physics ticks per command. Reissue
            # the same idempotent target to settle; never retry a failed request.
            self.request("POST", "/robot/joints", {"joint_values": targets})
            measured = self.get_joint_positions()
            if all(abs(measured[k] - v) <= self.cfg.joint_tolerance_rad for k, v in targets.items()):
                return measured
            if time.monotonic() >= deadline:
                raise RobosimError("Joint readback did not converge", code="robosim_joint_timeout")

    def get_tcp_pose(self) -> dict[str, float]:
        """Read selected TCP in base-frame mm and XYZ Euler degrees."""
        return base_pose(self.observe())

    @_commanded
    def move_tcp_blocking(self, pose: dict[str, float]) -> dict[str, float]:
        """Convert the shared TCP contract and use RoboSim's existing motion planner."""
        values = [pose[k] for k in ("x", "y", "z", "rx", "ry", "rz")]
        if not all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("Pose values must be finite numbers")
        local = np.eye(4)
        local[:3, 3] = np.asarray(values[:3]) / 1000
        local[:3, :3] = Rotation.from_euler("xyz", values[3:], degrees=True).as_matrix()
        world = np.asarray(self.observe()["world_from_base"]) @ local
        euler = Rotation.from_matrix(world[:3, :3]).as_euler("xyz")
        self.request(
            "POST",
            "/skills/move_end_effector",
            {
                "resource_id": f"{self.cfg.side}_arm",
                "target_pose": {
                    "position": dict(zip(("x", "y", "z"), world[:3, 3].tolist(), strict=True)),
                    "rotation": dict(zip(("roll", "pitch", "yaw"), euler.tolist(), strict=True)),
                },
            },
        )
        observation = self.observe()
        actual = pose_matrix(observation["tcp_world"])
        error_mm = float(np.linalg.norm(actual[:3, 3] - world[:3, 3]) * 1000)
        error_deg = float(np.degrees(Rotation.from_matrix(actual[:3, :3].T @ world[:3, :3]).magnitude()))
        if error_mm > self.cfg.position_tolerance_mm or error_deg > self.cfg.orientation_tolerance_deg:
            raise RobosimError(f"TCP readback error: {error_mm:.2f} mm, {error_deg:.2f} deg", code="robosim_pose_error")
        return base_pose(observation)

    @_commanded
    def set_gripper(self, on: bool) -> dict:
        """Actuate a gripper; closure alone does not establish payload ownership."""
        state = self.request("POST", "/robot/gripper/close" if on else "/robot/gripper/open", {"side": self.cfg.side})
        if (
            state.get("robot_type") != self.cfg.robot_type
            or not isinstance(state.get("gripper_open"), bool)
            or "holding" not in state
            or not (state["holding"] is None or isinstance(state["holding"], str))
            or "holding_arm_id" not in state
            or state["holding_arm_id"] not in (None, "left", "right")
        ):
            raise RobosimError("Invalid gripper state response", code="robosim_protocol_error", outcome_unknown=True)
        self._gripper_state = on
        return state

    @property
    def gripper_state(self) -> bool | None:
        """Last accepted command; this is not contact or payload evidence."""
        return self._gripper_state

    @_commanded
    def home(self) -> None:
        """Use the planned home operation without resetting physics or releasing payloads."""
        if self.observe()["reported_state"].get("holding"):
            raise RobosimError("Refusing home while carrying a payload", code="robosim_payload_held")
        state = self.request(
            "POST", "/skills/reset_to_home", {"arms": self.cfg.side, "release_gripper": False, "force": False}
        )
        if (
            not isinstance(state.get("arms_reset"), list)
            or self.cfg.side not in state["arms_reset"]
            or state.get("partial_success") is not False
            or state.get("gripper_released") is not False
            or state.get("failed_arms") != {}
        ):
            raise RobosimError(
                "Home completion was not confirmed", code="robosim_home_unconfirmed", outcome_unknown=True
            )

    def close(self) -> None:
        """Close this client; the independently launched simulation remains owned by its caller."""
        self._closed = True
        if self._owns_client:
            self.client.close()
