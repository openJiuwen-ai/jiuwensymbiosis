"""Scene-owned relative motion, measured in the MOZ1 forward/left body frame.

No renderer, agent or process imports. The caller ticks this before physics and
owns cancellation; only the four wheel motors are actuated.
"""

from __future__ import annotations

import math

# This controller executes body-frame translation along both horizontal axes.
MOZ1_SUPPORTS_LATERAL = True


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def validate_move(dx: float, dy: float, dyaw: float) -> None:
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (dx, dy, dyaw)):
        raise ValueError("Relative base motion requires finite numbers")
    if math.hypot(dx, dy) > 1.0 or abs(dyaw) > math.pi:
        raise ValueError("Base step exceeds experimental limits: 1 metre / pi radians")


def measure(robot) -> dict:
    """Read real root state; yaw describes the body front, not URDF root +X."""
    import torch

    entity = robot.entity
    # Queue all authored-root readings before one device-to-host synchronization.
    # No step runs between these getters: every field belongs to the same state.
    tensor = torch.cat((entity.get_pos()[0], entity.get_quat()[0], entity.get_vel()[0], entity.get_ang()[0]))
    values = tensor.cpu().tolist()
    pos, quat, vel, ang = values[:3], values[3:7], values[7:10], values[10:13]
    w, x, y, z = quat
    # Rotate body front (root -Y) into world coordinates, including tilt.
    yaw = math.atan2(-(1 - 2 * (x * x + z * z)), -2 * (x * y - w * z))
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError("Non-finite physical base observation")
    return {
        "position_m": pos,
        "quaternion_wxyz": quat,
        "body_yaw_rad": yaw,
        "linear_velocity_m_s": vel,
        "angular_velocity_rad_s": ang,
        "frame": "world",
        "command_frame": "body_forward_left_up",
    }


class RelativeMotion:
    """Turn, translate, brake, then require measured stability for 0.6 seconds."""

    def __init__(self, robot):
        self.robot = robot
        self.result: dict = {}
        self.active = False
        self.elapsed = self.stable = 0.0
        self.failure: str | None = None
        self._control_elapsed = 0.0
        self.target: list[float] = []
        self.anchor: list[float] = []
        self.phase = "idle"
        self.brake_elapsed = 0.0

    def accept_request(self, request: dict | None, *, scene_id: str, revision: int, now: float, step: int) -> None:
        """Deduplicate a single-owner request, enforce scene binding and dead-man lease."""
        if request is None:
            if self.active:
                self.cancel("cancelled_or_lease_expired")
            return
        bound = request["scene_instance_id"] == scene_id and request["config_revision"] == revision
        live = bound and not request["cancelled"] and request["lease_until"] > now
        same = request["id"] == self.result.get("id")
        if self.active:
            if not live or not same:
                self.cancel("cancelled_or_lease_expired")
        elif bound and not same:
            if live:
                self.start(request["id"], request["dx"], request["dy"], request["dyaw"], step=step)
            else:
                # Never initialize motion for an already invalidated request.
                # No physical observation was taken, so stopping is unconfirmed.
                self.result = {
                    "id": request["id"],
                    "status": "completed",
                    "ok": False,
                    "reason": "cancelled_or_lease_expired",
                    "settled": False,
                    "end_step": step,
                    "duration_sim_s": 0.0,
                }

    def start(self, request_id: str, dx: float, dy: float, dyaw: float, *, step: int) -> None:
        validate_move(dx, dy, dyaw)
        if self.active:
            raise RuntimeError("A base action is already running")
        before = measure(self.robot)
        yaw = wrap(before["body_yaw_rad"] + dyaw)
        c, s = math.cos(yaw), math.sin(yaw)
        self.target = [before["position_m"][0] + c * dx - s * dy, before["position_m"][1] + s * dx + c * dy, yaw]
        self.anchor = before["position_m"][:2]
        self.elapsed = self.stable = 0.0
        self._control_elapsed = 0.0
        self.active = True
        self.phase = "rotate" if abs(dyaw) > 0.012 else "translate"
        self.failure = None
        self.result = {
            "id": request_id,
            "status": "running",
            "ok": False,
            "requested": {"dx_m": dx, "dy_m": dy, "dyaw_rad": dyaw},
            "before": before,
            "start_step": step,
            "target_world": self.target,
        }

    def cancel(self, reason: str = "cancelled") -> None:
        if self.active and self.failure is None:
            self.failure = reason
            self.phase = "brake"
            self.stable = 0.0
            self.brake_elapsed = 0.0
            self._control_elapsed = 0.0
        self.robot.stop()

    def tick(self, dt: float, step: int) -> None:
        if not self.active:
            return
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("Invalid physics timestep")
        self.elapsed += dt
        self._control_elapsed += dt
        if self.elapsed > 40 and self.failure is None:
            self.cancel("motion_timeout")
        # Low-speed navigation updates at 60 Hz, independently of video FPS.
        # Physics still integrates every tick; braking/stability keep full-rate
        # observations so reducing command traffic cannot shorten settling.
        if self.phase in {"rotate", "translate"} and self._control_elapsed < 1 / 60 - 1e-9:
            return
        dt = self._control_elapsed
        self._control_elapsed = 0.0
        state = measure(self.robot)
        x, y = state["position_m"][:2]
        yaw = state["body_yaw_rad"]
        yaw_error = wrap(self.target[2] - yaw)
        distance = math.hypot(self.target[0] - x, self.target[1] - y)
        linear_speed = math.sqrt(sum(v * v for v in state["linear_velocity_m_s"]))
        angular_speed = math.sqrt(sum(v * v for v in state["angular_velocity_rad_s"]))
        if self.phase == "rotate" and abs(yaw_error) < 0.012:
            self.phase = "translate"
        at_goal = distance <= 0.008 and abs(yaw_error) <= 0.012
        if self.phase == "translate" and at_goal:
            self.phase = "settle"
        if self.phase == "settle" and (distance > 0.015 or abs(yaw_error) > 0.025):
            self.phase = "translate"
            self.stable = 0.0
        if self.phase in {"settle", "brake"}:
            self.robot.stop()
            self.stable = self.stable + dt if linear_speed < 0.008 and angular_speed < 0.02 else 0.0
            if self.phase == "brake":
                self.brake_elapsed += dt
            settled = self.stable >= 0.6
            if settled or (self.phase == "brake" and self.brake_elapsed >= 5):
                self.active = False
                self.result.update(
                    status="completed",
                    ok=settled and self.failure is None,
                    reason=self.failure or "reached",
                    settled=settled,
                    after=state,
                    end_step=step,
                    duration_sim_s=self.elapsed,
                    position_error_m=distance,
                    yaw_error_rad=yaw_error,
                )
                return
        else:
            tx, ty = self.anchor if self.phase == "rotate" else self.target[:2]
            c, s = math.cos(yaw), math.sin(yaw)
            ex, ey = tx - x, ty - y
            vx, vy = 1.8 * (c * ex + s * ey), 1.8 * (-s * ex + c * ey)
            magnitude = math.hypot(vx, vy)
            if magnitude > 0.15:
                vx, vy = vx * 0.15 / magnitude, vy * 0.15 / magnitude
            wz = max(-0.3, min(0.3, 1.8 * yaw_error))
            self.robot.command(vx, vy, wz, min(2 * dt, 0.2))
        self.result.update(
            phase=self.phase, measured=state, measured_step=step, position_error_m=distance, yaw_error_rad=yaw_error
        )
