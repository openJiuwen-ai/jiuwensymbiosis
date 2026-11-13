# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""XxxApi — this body's implementation of the shared action vocabulary.

Porting a robot is **not** inventing actions: pick the ones your hardware can do from
the shared vocabulary (``jiuwensymbiosis-actions --vocabulary``) and bind each with
``@implements(SPEC)``. The contract — name, capability gate, parameters, result shape,
pre-conditions and effects — comes from that one spec, so a plan or a SKILL.md written
for another robot means the same thing here. Your job is only *how*.

Two kinds of method below:

* **generic** — the implementation is the Env verb, so forward to ``api.defaults``.
  One line each; they are here (rather than inherited) so this file is the complete
  list of what the body offers.
* **body-specific** — your geometry / SDK / calibration. Write it out.

Bring-up, calibration and debug views are NOT actions: leave them undecorated and drive
them from a script under ``scripts/``. A plan written against a one-robot tool would not
survive being moved, and the agent's tool list is the shared vocabulary only.

Key patterns:
  - Motion / end-effector go through the Env verbs (``self.env.home()`` /
    ``move_to_flange()`` / ``move_joint()`` / ``set_end_effector()``).
  - Body constants come from Env properties (``self.env.home_pose`` /
    ``self.env.tool_offset_mm``).
  - Vision calibration reads ``self.env.low_level`` (the ``RobotDriver`` protocol) —
    a controlled penetration for sensor data that does not belong on the Env body
    abstraction.
  - Every action returns an ``{"ok": True/False, ...}``-shaped dict.
"""

from __future__ import annotations

from typing import Literal

from jiuwensymbiosis.api import defaults
from jiuwensymbiosis.api.actions import (
    GET_HOME_POSE,
    GET_POSE,
    GOTO_XYZR,
    MOVE_DIRECTION,
    # CLOSE_GRIPPER,          # [选填] parallel gripper
    # OPEN_GRIPPER,
    # ACTIVATE_SUCTION,       # [选填] suction end-effector
    # DEACTIVATE_SUCTION,
    # GET_IMAGE,              # [选填] camera
    # GET_GRASP_INFO_SIMPLE,  # [选填] detection → ready-to-use grasp pose
    # PIXEL_TO_BASE_XYZ,
    # MOVE_JOINT,             # [选填] joint-space motion
    implements,
)
from jiuwensymbiosis.api.base import BaseRobotApi


class XxxApi(BaseRobotApi):
    """Robot API for Xxx — TODO: replace with your robot description."""

    # If your Api.__init__ needs extra parameters beyond env (e.g. detector URL,
    # calibration constants), declare them here. The session builder passes them
    # via ``api_kwargs_from_cfg``.

    # ================================================================ Motion
    # ``home`` comes from BaseRobotApi (delegates to env.home()). Override it only
    # if returning safely takes more than that — e.g. straightening a torso first.

    @implements(MOVE_DIRECTION)
    def move_direction(self, direction: str, distance_mm: float) -> dict:
        """Relative nudge with a bounds check — generic, nothing to write."""
        return defaults.move_direction(self, direction, distance_mm)

    @implements(GET_HOME_POSE)
    def get_home_pose(self) -> dict:
        return defaults.get_home_pose(self)

    @implements(GET_POSE)
    def get_pose(self) -> dict:
        """TIP pose. The generic default assumes tip == flange; this body has a tool
        offset, so it subtracts it. Delete this and call ``defaults.get_pose(self)``
        if your tip IS the flange.
        """
        p = self.env.get_flange_pose()
        return {
            "x": p.x,
            "y": p.y,
            "z": p.z - self.env.tool_offset_mm,
            "rx": p.rx,
            "ry": p.ry,
            "rz": p.rz,
        }

    @implements(GOTO_XYZR)
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None,
                  orientation_policy: Literal["top_down"] = "top_down") -> None:
        """Move the tip to an absolute pose; ``orientation_policy`` picks the tilt.

        The generic default (``defaults.goto_xyzr``) does ``top_down`` and ``preserve``, so
        widen the Literal to match whichever your body really honours — it is what tells a
        planner this body's options, and a value you accept but ignore is a silent lie.
        Keep this override only if your body needs the tip↔flange conversion or a tilted
        tool; otherwise forward to the default.
        """
        raise NotImplementedError("TODO: convert the TIP target to a flange command and dispatch it")

    # ================================================================ Joint  [选填]
    # @implements(MOVE_JOINT)
    # def move_joint(self, targets: dict[str, float]) -> Any:
    #     return defaults.move_joint(self, targets)

    # ================================================================ End effector  [选填]
    # Two-state bodies forward to the defaults; width_mm / force_n are accepted for
    # contract parity and ignored — the contract already calls both a HINT, so there is
    # nothing to add. A body with real width or force control writes its own instead.
    #
    # @implements(OPEN_GRIPPER)
    # def open_gripper(self, width_mm: float = 80.0) -> dict:
    #     return defaults.open_gripper(self, width_mm)
    #
    # @implements(CLOSE_GRIPPER)
    # def close_gripper(self, force_n: float | None = None) -> dict:
    #     return defaults.close_gripper(self, force_n)
    #
    # @implements(ACTIVATE_SUCTION)
    # def activate_suction(self) -> dict:
    #     return defaults.activate_suction(self)
    #
    # @implements(DEACTIVATE_SUCTION)
    # def deactivate_suction(self) -> dict:
    #     return defaults.deactivate_suction(self)

    # ================================================================ Vision  [选填]
    # ``get_image`` is generic; the rest need YOUR detector client and hand-eye
    # calibration, which is why they have no default.
    #
    # @implements(GET_IMAGE)
    # def get_image(self):
    #     return defaults.get_image(self)
    #
    # @implements(PIXEL_TO_BASE_XYZ)
    # def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
    #     """Back-project a pixel to base XYZ using this body's calibration."""
    #     raise NotImplementedError
    #
    # @implements(GET_GRASP_INFO_SIMPLE)
    # def get_grasp_info_simple(self, object_name: str):
    #     """Detect + project + apply gripper geometry. See
    #     ``perception/vision.py:default_get_grasp_info_simple`` — it factors out the
    #     whole eye-in-hand pipeline, leaving you a ``seg_fn`` and a pose→TF callback.
    #     """
    #     raise NotImplementedError
