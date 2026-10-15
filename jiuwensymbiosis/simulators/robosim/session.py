"""Factory for the separately hosted RoboSim session."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.simulators.robosim.api import RobosimApi
from jiuwensymbiosis.simulators.robosim.client import identity_token
from jiuwensymbiosis.simulators.robosim.config import RobosimConfig
from jiuwensymbiosis.simulators.robosim.env import RobosimEnv
from jiuwensymbiosis.simulators.robosim.lowlevel import RobosimLowLevel
from jiuwensymbiosis.simulators.robosim.operations import RobosimError
from jiuwensymbiosis.simulators.robosim.tools import build_workspace_tools

if TYPE_CHECKING:
    from jiuwensymbiosis.simulators.robosim.workspace import RobosimWorkspace

__all__ = ["build_robosim_session", "bind_robosim_session"]

build_robosim_session = make_builder(RobosimConfig, RobosimEnv, RobosimApi)


def bind_robosim_session(workspace: RobosimWorkspace, snapshot: dict) -> tuple[RobotSession, list[Any]]:
    """Bind native tools and a disconnected session to one fresh workspace snapshot.

    The workspace owns the transport; disconnecting this session only retires its
    driver. Every later motion rechecks identity, including before connect/load.
    """
    state = snapshot.get("robot_state")
    if not state or state.get("robot_type") != workspace.cfg.robot_type or not all(identity_token(snapshot)):
        raise RobosimError("No supported robot session to bind", code="robot_not_loaded")
    if snapshot.get("scene_phase") in {"loading", "failed"}:
        raise RobosimError("Scene is not ready", code="robosim_scene_not_ready")
    session = build_robosim_session(workspace.cfg, include_sidecars=False)
    session.env.low_level = RobosimLowLevel(
        workspace.cfg, client=workspace.client, expected_identity=identity_token(snapshot)
    )
    tools = build_workspace_tools(workspace, allow_load=False)
    return session, tools
