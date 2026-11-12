"""Lease-bound Jumper policy actions; physics and native FSM stay worker-owned."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def validate_policy_action(kind: str, *, side: str = "left", closed: bool = False) -> None:
    if not isinstance(kind, str) or kind not in {"jump", "set_gripper"}:
        raise ValueError("Unsupported Jumper action")
    if not isinstance(side, str) or side not in {"left", "right"} or not isinstance(closed, bool):
        raise ValueError("Jumper gripper requires left/right side and boolean closed")


POLICY_TIMEOUT_S = 20
# Allow five more seconds to finish landing/braking and measure a stable stop.
STOP_TIMEOUT_S = 25


class JumperMotion:
    """Serialize navigation, one jump and one carried claw on the existing lease.

    A closed claw remains held between calls. Changing sides, navigating or
    jumping requires opening it first; no command silently releases a payload.
    Airborne cancellation waits for the official recording to finish and land.
    """

    def __init__(self, robot: Any, base: Any) -> None:
        self.robot, self.base = robot, base
        self.task: dict | None = None
        self._result: dict = {}
        self._navigation = False
        self.claw_side: str | None = None
        self.claw_closed = False

    @property
    def result(self) -> dict:
        return cast(dict, self.base.result) if self._navigation else self._result

    @property
    def active(self) -> bool:
        return self.task is not None or bool(self.base.active)

    def accept_request(self, request: dict | None, *, scene_id: str, revision: int, now: float, step: int) -> None:
        bound = (
            request is not None and request["scene_instance_id"] == scene_id and request["config_revision"] == revision
        )
        live = bound and request is not None and not request["cancelled"] and request["lease_until"] > now
        same = request is not None and request["id"] == self.result.get("id")
        if self.active:
            if not live or not same:
                self.cancel("cancelled_or_lease_expired")
        elif bound and not same and request is not None:
            if live:
                self.start(request, step)
            else:
                self._navigation = False
                self._result = {
                    "id": request["id"],
                    "status": "completed",
                    "ok": False,
                    "reason": "cancelled_or_lease_expired",
                    "settled": False,
                    "end_step": step,
                }

    def start(self, request: dict, step: int) -> None:
        kind = request.get("kind", "navigate")
        if kind == "navigate":
            from base_motion import validate_move

            validate_move(request["dx"], request["dy"], request["dyaw"])
        else:
            validate_policy_action(kind, side=request.get("side", "left"), closed=request.get("closed", False))
        self._navigation = False
        self.task = {
            **request,
            "kind": kind,
            "phase": "prepare",
            "elapsed": 0.0,
            "stable": 0.0,
            "failure": None,
            "saw_mode": False,
            "previous_claw": (self.claw_side, self.claw_closed),
        }
        self._result = {
            "id": request["id"],
            "kind": kind,
            "status": "running",
            "ok": False,
            "start_step": step,
            "before": self.robot.observe_base(),
            "flight_s": 0.0,
            "max_height_m": float(self.robot.data.qpos[2]),
        }
        mode = self.robot.policy.fsm.mode
        different = kind != "set_gripper" or request.get("side", "left") != self.claw_side
        if different and self.claw_closed:
            self.task.update(phase="rejected", failure="gripper_requires_open")
        elif mode not in {"locomotion", "claw_left", "claw_right"}:
            self.task.update(phase="rejected", failure="policy_state_invalid")
        elif mode.startswith("claw_") and self.claw_side != mode.removeprefix("claw_"):
            self.task.update(phase="rejected", failure="gripper_state_unknown")
        elif not self._still():
            self.task.update(phase="rejected", failure="unsafe_start")
        elif mode.startswith("claw_") and different:
            try:
                self.robot.policy.claw(self.claw_side, False)
                self.robot.policy.tap_mode(mode)
                self.task["phase"] = "stow"
            except (KeyError, ValueError, RuntimeError):
                self._reject_transition()
        self.robot.stop()

    def _reject_transition(self) -> None:
        """Reject a controller mismatch without letting it escape the worker tick."""
        if self.task is None:
            raise RuntimeError("Cannot reject a policy transition without an active task")
        self.task.update(phase="rejected", failure="policy_transition_failed")
        self.robot.policy.key_events.clear()
        self.robot.stop()

    def cancel(self, reason: str = "cancelled") -> None:
        self.robot.stop()
        if self.base.active:
            self.base.cancel(reason)
        elif self.task and self.task["failure"] is None:
            self.task["failure"] = reason
            # A started jump/closure finishes its bounded motion. A not-yet
            # started action must never be triggered by a cancellation.
            if self.task["phase"] in {"prepare", "stow"}:
                self.task["phase"] = "rejected"
            elif (
                not self.task["saw_mode"]
                and self.robot.policy.fsm.mode == "locomotion"
                and self.robot.policy.key_events
            ):
                self.robot.policy.key_events.clear()
                self.claw_side, self.claw_closed = self.task["previous_claw"]
                self.task["phase"] = "rejected"

    def _still(self) -> bool:
        state = self.robot.observe_base()
        return bool(
            self.robot.policy.fsm.running_policy
            and self.robot.data.qpos[2] > 0.07
            and self._support_count() >= 3
            and np.linalg.norm(state["linear_velocity_m_s"]) < 0.02
            and np.linalg.norm(state["angular_velocity_rad_s"]) < 0.2
        )

    def _support_count(self) -> int:
        r = self.robot
        return int(
            sum((r.model.geom_bodyid[c.geom1] == 0) != (r.model.geom_bodyid[c.geom2] == 0) for c in r.data.contact)
        )

    def sample_physics(self) -> None:
        if self.task is None or self.task["kind"] != "jump":
            return
        height = float(self.robot.data.qpos[2])
        self._result["max_height_m"] = max(self._result["max_height_m"], height)
        if self.robot.policy.fsm.mode == "jump" and self.robot.policy.fsm.running_policy:
            self.task["saw_mode"] = True
        if (
            self.task["saw_mode"]
            and self._support_count() == 0
            and height > self._result["before"]["position_m"][2] + 0.015
        ):
            self._result["flight_s"] += self.robot.policy.dt

    def tick(self, dt: float, step: int) -> None:
        if self.base.active:
            self.base.tick(dt, step)
            return
        if self.task is None:
            return
        task, r = self.task, self.robot
        task["elapsed"] += dt
        if task["elapsed"] > POLICY_TIMEOUT_S and task["failure"] is None:
            task["failure"] = "policy_timeout"
        if r.policy.fsm.mode == "safe":
            task["failure"] = "policy_safe"
        if task["phase"] == "stow":
            task["stable"] = (
                task["stable"] + dt
                if r.policy.fsm.mode == "locomotion" and r.policy.fsm.running_policy and self._still()
                else 0.0
            )
            if task["stable"] >= 0.3:
                self.claw_side = None
                task.update(phase="prepare", stable=0.0)
            elif task["failure"]:
                task["phase"] = "rejected"
            else:
                self._result.update(phase="stow", duration_sim_s=task["elapsed"])
                return
        if task["phase"] == "prepare":
            if task["failure"]:
                task["phase"] = "rejected"
            else:
                try:
                    if task["kind"] == "navigate":
                        r.policy.use_operators(False)
                        self.base.start(task["id"], task["dx"], task["dy"], task["dyaw"], step=step)
                        self._navigation = True
                        self.task = None
                        return
                    elif task["kind"] == "jump":
                        r.policy.use_operators(False)
                        r.policy.tap_mode("jump")
                        task["phase"] = "execute"
                    else:
                        side = task.get("side", "left")
                        r.policy.use_operators(True)
                        if r.policy.fsm.mode != "claw_" + side:
                            r.policy.tap_mode("claw_" + side)
                        r.policy.claw(side, task["closed"])
                        self.claw_side, self.claw_closed = side, task["closed"]
                        task["phase"] = "execute"
                except (KeyError, ValueError, RuntimeError):
                    self._reject_transition()
        reached = task["phase"] in {"rejected", "handoff"}
        if task["phase"] == "execute":
            if task["kind"] == "jump":
                reached = task["saw_mode"] and r.policy.fsm.mode == "locomotion" and r.policy.fsm.running_policy
                if reached and self._result["flight_s"] < 0.02:
                    task["failure"] = task["failure"] or "flight_unconfirmed"
            else:
                side = task.get("side", "left")
                if r.policy.fsm.mode == "claw_" + side:
                    task["saw_mode"] = True
                name = ("LF" if side == "left" else "RF") + "_J4_joint"
                index = r.policy.wire.index(name)
                sign = 1 if side == "left" else -1
                angle = float(r.data.qpos[r.qids[index]])
                velocity = float(r.data.qvel[r.vids[index]])
                # Compare the actual finger against the official hook target;
                # closure is actuator completion, never evidence of a payload.
                reached = (
                    task["saw_mode"]
                    and r.policy.fsm.running_policy
                    and abs(angle - r.targets[index]) < 0.07
                    and abs(velocity) < 0.03
                    and (sign * angle > 0.03 if task["closed"] else sign * angle < -0.5)
                )
                self._result.update(
                    side=side,
                    closed=task["closed"],
                    joint_name=name,
                    joint_angle_rad=angle,
                    joint_velocity_rad_s=velocity,
                    joint_target_rad=float(r.targets[index]),
                    grasp_verified=False,
                )
        if task["failure"] in {"policy_timeout", "policy_safe"}:
            reached = True
        task["stable"] = task["stable"] + dt if reached and self._still() else 0.0
        settled = task["stable"] >= 1.0
        if settled or task["elapsed"] >= STOP_TIMEOUT_S:
            # Rebuilding the controller may change finger targets. Never do it
            # automatically while a claw is closed, even after an FSM mode exit.
            handoff = r.policy.fsm.mode == "locomotion" and r.policy.operators and not self.claw_closed
            if settled and handoff:
                try:
                    r.policy.use_operators(False)
                except (KeyError, ValueError, RuntimeError):
                    self._reject_transition()
                else:
                    task.update(phase="handoff", stable=0.0)
                    self._result["phase"] = "input_handoff"
                    return
            self._result.update(
                status="completed",
                ok=settled and task["failure"] is None,
                reason=task["failure"] or "reached",
                settled=settled,
                after=r.observe_base(),
                end_step=step,
                duration_sim_s=task["elapsed"],
                support_contacts=int(self._support_count()),
            )
            self.task = None
        else:
            self._result.update(phase=task["phase"], duration_sim_s=task["elapsed"])

    def gripper_snapshot(self) -> dict:
        return {"side": self.claw_side, "closed_command": self.claw_closed, "grasp_verified": False}
