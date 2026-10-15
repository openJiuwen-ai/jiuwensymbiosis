"""RobotSession environment backed by an independently running RoboSim service."""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation
from jiuwensymbiosis.errors import GraspNotConfirmedError
from jiuwensymbiosis.simulators.robosim.client import RobosimClient
from jiuwensymbiosis.simulators.robosim.config import RobosimConfig
from jiuwensymbiosis.simulators.robosim.lowlevel import RobosimLowLevel, base_pose


class RobosimEnv(BaseRobotEnv):
    """Expose one MOZ1 arm; camera, mobile and dual-arm capabilities are not advertised."""

    name = "robosim_moz1"
    capabilities = frozenset({"motion.joint", "motion.cartesian", "grasp.parallel"})
    _joint_units = "rad"
    _default_orientation_policy = "preserve"

    def __init__(self, cfg: RobosimConfig) -> None:
        """Store configuration without touching the service."""
        self.cfg = cfg
        self._inner: RobosimLowLevel | None = None
        self._connected = False
        self._borrowed_client: RobosimClient | None = None
        self._expected_identity: tuple[str, str | None] | None = None

    @property
    def low_level(self) -> RobosimLowLevel:
        """Return the bound driver or fail explicitly before connection."""
        if self._inner is None:
            raise RuntimeError("RoboSim environment is not connected")
        return self._inner

    @low_level.setter
    def low_level(self, value: RobosimLowLevel) -> None:
        """Allow pre-connect injection for hardware-free adapter tests."""
        if self._inner is not None:
            raise AttributeError("Driver is already bound")
        self._inner = value

    def bind_transport(self, client: RobosimClient, identity: tuple[str, str | None]) -> None:
        """Keep the workspace's borrowed transport and identity across driver lifetimes."""
        if self._inner is not None or self._borrowed_client is not None:
            raise RuntimeError("Environment is already bound")
        self._borrowed_client = client
        self._expected_identity = identity

    def connect(self) -> None:
        """Own the client lifecycle, including cleanup after partial startup."""
        if self._connected:
            return
        driver = self._inner or RobosimLowLevel(
            self.cfg, client=self._borrowed_client, expected_identity=self._expected_identity
        )
        try:
            driver.connect()
        except Exception:
            driver.close()
            self._inner = None
            raise
        self._inner = driver
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect locally without unloading another process's robot."""
        try:
            if self._inner is not None:
                self._inner.close()
        finally:
            self._inner = None
            self._connected = False

    @property
    def joint_names(self) -> list[str] | None:
        """Read the selected execution resource's measured joint names."""
        return list(self.low_level.get_joint_positions()) if self._connected else None

    @joint_names.setter
    def joint_names(self, value: list[str] | None) -> None:
        """Keep joint ordering owned by the remote execution resource."""
        raise AttributeError("joint_names is read from RoboSim")

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]] | None:
        """Read authored USD limits, without inventing an envelope."""
        if not self._connected:
            return None
        return {name: tuple(bounds) for name, bounds in self.low_level.observe()["joint_limits"].items()} or None

    @joint_limits.setter
    def joint_limits(self, value: dict[str, tuple[float, float]] | None) -> None:
        """Prevent a client override from replacing authored USD limits."""
        raise AttributeError("joint_limits is read from RoboSim")

    @property
    def holding_payload(self) -> bool | None:
        """A known RoboSim grasp binding is positive evidence; absence is unknown."""
        if self._connected and self.low_level.observe()["reported_state"].get("holding"):
            return True
        return None

    def get_observation(self) -> RobotObservation:
        """Report measured joints/TCP with the source, frame and simulation timestamp."""
        observation = self.low_level.observe()
        joints = observation["joint_positions"]
        return RobotObservation(
            pose=base_pose(observation),
            joints=list(joints.values()),
            extra={
                "joint_positions": joints,
                "joint_units": "rad",
                "source": "robosim_physx",
                "base_frame": observation["base_frame"],
                "simulation_time_s": observation["simulation_time_s"],
                "reported_grasps": observation["reported_state"].get("grasps", []),
            },
        )

    def get_joint_positions(self) -> dict[str, float]:
        """Read measured named joints in radians through the environment boundary."""
        return self.low_level.get_joint_positions()

    def get_tcp_pose(self) -> dict[str, float]:
        """Read TCP in base-frame millimeters and XYZ Euler degrees."""
        return self.low_level.get_tcp_pose()

    def move_to_tcp(self, pose: dict[str, float]) -> dict[str, float]:
        """Move TCP and return the measured pose from the same guarded command."""
        return self.low_level.move_tcp_blocking(pose)

    def get_flange_pose(self) -> Any:
        """Reject FLANGE queries: the service only exposes TCP geometry."""
        raise NotImplementedError("RoboSim has no FLANGE transform; use get_tcp_pose")

    def move_to_flange(self, pose: Any) -> None:
        """Reject FLANGE commands rather than interpreting them as TCP targets."""
        raise NotImplementedError("RoboSim has no FLANGE transform; use move_to_tcp")

    def open_gripper(self) -> None:
        """Release the selected gripper through the common end-effector verb."""
        self.set_end_effector(False)

    def close_gripper(self) -> dict:
        """Return grasp evidence only when it belongs to the selected arm."""
        state = self.low_level.set_gripper(True)
        if not state or not state.get("holding") or state.get("holding_arm_id") != self.cfg.side:
            raise GraspNotConfirmedError("Gripper closed, but no payload binding confirms a grasp")
        return state

    def observe(self) -> dict[str, Any]:
        """Expose the measured backend snapshot for diagnostic consumers."""
        return self.low_level.observe()

    def home(self) -> None:
        """Delegate to RoboSim's planned posture restoration."""
        self.low_level.home()

    def release_effector(self) -> None:
        """Release the selected gripper for the framework's recovery path."""
        self.open_gripper()
