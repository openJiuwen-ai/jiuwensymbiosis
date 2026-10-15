"""A RoboSim connection usable before any robot session exists."""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.simulators.robosim.client import RobosimClient, ensure_robot, read_identity
from jiuwensymbiosis.simulators.robosim.config import RobosimConfig
from jiuwensymbiosis.simulators.robosim.operations import RobosimError


class RobosimWorkspace:
    """Own the transport and remote identity, never the backend's physical world."""

    def __init__(self, cfg: RobosimConfig, *, client: RobosimClient | None = None) -> None:
        self.cfg = cfg
        self.client = client or RobosimClient(cfg)
        self._owns_client = client is None
        self._identity: tuple[str, str | None] | None = None
        self.revision = 0

    def close(self) -> None:
        """Release owned transport; leave robot, backend and unknown writes intact."""
        if self._owns_client:
            self.client.close()

    def refresh(self) -> dict[str, Any]:
        """Read health, incarnation and robot state without loading or moving anything."""
        health = self.client.request("GET", "/sim/health")
        if health.get("ready") is not True:
            raise RobosimError("RoboSim is not ready", code="robosim_not_ready")
        identity = self.read_identity()
        service_id = identity["service_instance_id"]
        robot_id = identity.get("robot_instance_id")
        self.client.commands.observe_service(service_id)
        state = self.client.request("GET", "/robot/state") if robot_id is not None else None
        # Do not bind a state sampled across a robot replacement to the old identity.
        if self.read_identity() != identity:
            raise RobosimError("Robot changed during observation", code="robosim_robot_changed")
        current = (service_id, robot_id)
        if current != self._identity:
            self._identity = current
            self.revision += 1
        return {
            **identity,
            "revision": self.revision,
            "health": health,
            "robot_state": state,
            **self.client.commands.snapshot(),
        }

    def load_robot(self) -> dict:
        """Load configured MOZ1 into an empty world; never replace another body."""
        # Establish/reconcile incarnation before taking a lease, then recheck
        # under the lease so another local client cannot race this decision.
        self.refresh()
        with self.client.commands.command("load_robot"):
            snapshot = self.refresh()
            state = snapshot["robot_state"]
            ensure_robot(self.client.request, self.cfg.robot_type, state, load=True)
            if state is None:
                snapshot = self.refresh()
                if snapshot["robot_state"] is None or snapshot["robot_state"].get("robot_type") != self.cfg.robot_type:
                    raise RobosimError(
                        "Robot loading was not confirmed", code="robosim_load_unconfirmed", outcome_unknown=True
                    )
        return {**snapshot, **self.client.commands.snapshot()}

    def read_identity(self) -> dict:
        """Require the read-only identity bridge supplied by our launcher."""
        return read_identity(self.client.request)

    def list_robots(self) -> dict:
        """List backend specifications, without promising support for every body."""
        return self.client.request("GET", "/robots")

    def robot_spec(self) -> dict:
        """Query the configured robot's authoritative capabilities and resources."""
        return self.client.request("GET", f"/robots/{self.cfg.robot_type}")

    def scene_state(self) -> dict:
        """Label the registry snapshot distinctly from live physical measurements."""
        return {"source": "robosim_registry", "state": self.client.request("GET", "/scene/state")}

    def observe(self) -> dict:
        """Read the configured arm through the existing measured-state driver."""
        from jiuwensymbiosis.simulators.robosim.lowlevel import RobosimLowLevel

        snapshot = self.refresh()
        if snapshot["robot_state"] is None:
            raise RobosimError("Load a robot before observing its arm", code="robot_not_loaded")
        if snapshot["robot_state"].get("robot_type") != self.cfg.robot_type:
            raise RobosimError("Active robot differs from config", code="robosim_robot_mismatch")
        return RobosimLowLevel(
            self.cfg,
            client=self.client,
            expected_identity=(snapshot["service_instance_id"], snapshot["robot_instance_id"]),
        ).observe()
