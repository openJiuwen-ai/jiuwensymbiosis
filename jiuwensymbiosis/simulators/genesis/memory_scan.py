"""Genesis execution port for bounded Rule scans through a RobotSession.

The policy sees RGB-D only. Static simulator geometry admits (or refuses) a
short full-body motion, never supplies exploration candidates or completeness.
"""

from __future__ import annotations

import math

import numpy as np

from jiuwensymbiosis.rails.safety import SafetyRail
from jiuwensymbiosis.simulators.genesis.spatial_capture import observation_from_capture
from jiuwensymbiosis.spatial_memory.exploration import wrap
from jiuwensymbiosis.spatial_memory.models import Observation


class GenesisScanPort:
    """Borrow a connected ground-truth Genesis session; never own its resources."""

    def __init__(self, session, *, map_frame: str, depth_stride: int = 4):
        if "motion.base" not in session.env.capabilities & session.api.capabilities:
            raise ValueError("Scan requires motion.base")
        if session.env.driver.observation_mode != "ground_truth":
            raise ValueError("Genesis scan admission requires explicit simulator ground-truth execution")
        if not map_frame.strip() or not 4 <= depth_stride <= 32:
            raise ValueError("Require a map frame and depth stride in [4, 32]")
        self.session = session
        self.driver = session.env.driver
        self.preview = self.driver.preview
        self.map_frame = map_frame
        self.depth_stride = depth_stride
        self.safety = SafetyRail(session)
        self.last_base: dict | None = None
        self.binding: str | None = None

    def capture(self) -> Observation:
        self.driver.require_motion()
        raw = self.preview.capture_image(
            "head_cam",
            include_depth=True,
            depth_stride=self.depth_stride,
            capture_mode="spatial_memory",
            cancelled=self.driver.cancelled,
        )
        observation = observation_from_capture(raw, map_frame=self.map_frame, allow_motion=True)
        # Frozen sampling supports motion; this scan specifically requires settled views.
        for key, limit in (("linear_velocity_m_s", 0.008), ("angular_velocity_rad_s", 0.02)):
            if np.linalg.norm(raw["base_sample"][key]) >= limit:
                raise RuntimeError("Scan capture was not stationary")
        if self.binding is not None and self.binding != observation.scene_epoch:
            raise RuntimeError("Scene changed during scan")
        self.binding = observation.scene_epoch
        self.last_base = raw["base_sample"]
        return observation

    def execute(self, choice: dict, observation: Observation) -> dict:
        self.driver.require_motion()
        if self.last_base is None or observation.scene_epoch != self.binding:
            raise ValueError("Capture before choosing a move")
        current = self.driver.query_truth("robot_state", {})
        robot = current["robot"]
        if current["scene_instance_id"] != observation.scene_epoch.rsplit(":", 1)[0]:
            raise RuntimeError("Scene changed before motion")
        if (
            math.dist(robot["position_m"], self.last_base["position_m"]) > 0.02
            or abs(wrap(robot["body_yaw_rad"] - self.last_base["body_yaw_rad"])) > 0.02
        ):
            raise RuntimeError("Scan start changed after observation")
        yaw = robot["body_yaw_rad"]
        delta = np.zeros(2)
        if len(choice["path_xy_m"]) > 1:
            delta = np.asarray(choice["path_xy_m"][1]) - observation.map_from_camera[:2, 3]
            length = float(np.linalg.norm(delta))
            if length > 0:
                delta *= min(0.2, choice["remaining_distance_m"], length) / length
        angle = (
            0.0
            if np.linalg.norm(delta) > 0.02
            else float(np.clip(wrap(choice["heading_rad"] - yaw), -math.pi / 6, math.pi / 6))
        )
        goal = np.asarray(robot["position_m"][:2]) + delta
        try:
            admission = self.driver.query_truth(
                "plan_navigation", {"x_m": float(goal[0]), "y_m": float(goal[1]), "yaw_rad": wrap(yaw + angle)}
            )
        except (ValueError, RuntimeError) as exc:
            return {"ok": False, "motion_executed": False, "stopped": self.confirm_stopped(), "reason": str(exc)}
        plan = admission["plan"]
        # A map-approved detour would change the observation-only policy. Reject it.
        if plan["path_length_m"] > float(np.linalg.norm(delta)) + 1e-6:
            return {
                "ok": False,
                "motion_executed": False,
                "stopped": self.confirm_stopped(),
                "reason": "admission_requires_detour",
            }
        c, s = math.cos(yaw), math.sin(yaw)
        command = {
            "dx_m": float(c * delta[0] + s * delta[1]),
            "dy_m": float(-s * delta[0] + c * delta[1]),
            "dyaw_rad": angle,
        }
        self.safety.validate_motion("navigate_relative", command)
        result = self.session.api.navigate_relative(**command)
        stopped = (
            result.get("settled") is True and result.get("stop_confirmation") == "measured" and self.confirm_stopped()
        )
        if not stopped:
            self.driver.fail_stop_confirmation()
        return {
            "ok": result.get("ok") is True,
            "motion_executed": True,
            "stopped": stopped,
            "command": command,
            "feedback": result,
            "admission": {k: plan[k] for k in ("map_fingerprint", "footprint_radius_m")},
            "admission_source": "simulator_static_geometry_full_body",
        }

    def confirm_stopped(self) -> bool:
        measured = self.driver.observe()
        return all(
            np.isfinite(measured[key]).all() and np.linalg.norm(measured[key]) < limit
            for key, limit in (("linear_velocity_m_s", 0.008), ("angular_velocity_rad_s", 0.02))
        )
