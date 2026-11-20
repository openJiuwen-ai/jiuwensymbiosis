"""Lease-bound Jumper policy actions; physics and native FSM stay worker-owned."""

from __future__ import annotations

from typing import Any

import numpy as np

from .policy_actions import JUMPER_ACTIONS, RECORDINGS, validate_action
from .policy_motion import PolicyMotion

JUMPS = {"jump", "ref_free_jump"}
PLAYBACK = JUMPS.union(RECORDINGS)
# The standalone actor has no recording clock. Its official training episode ends after 2.5 s.
REF_FREE_EPISODE_S = 2.5
CLAW_ACTIONS = {"set_gripper", "claw_aperture", "claw_pose", "claw_step"}


POLICY_TIMEOUT_S = 20
# Allow five more seconds to finish landing/braking and measure a stable stop.
STOP_TIMEOUT_S = 25


class JumperMotion(PolicyMotion):
    """Serialize navigation, one jump and one carried claw on the existing lease.

    A closed claw remains held between calls. Changing sides, navigating or
    jumping requires opening it first; no command silently releases a payload.
    Airborne cancellation waits for the official recording to finish and land.
    """

    def __init__(self, robot: Any, base: Any) -> None:
        super().__init__(robot, base)
        self.claw_side: str | None = None
        self.claw_closed = False

    def start(self, request: dict, step: int) -> None:
        request_id = request.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Jumper motion requires a request ID")
        kind = request.get("kind", "navigate")
        if kind == "navigate":
            from ..common.base_motion import validate_move

            validate_move(request["dx"], request["dy"], request["dyaw"])
        else:
            if kind not in JUMPER_ACTIONS:
                raise ValueError("Unsupported Jumper action")
            values = request.get("params", {})
            if kind == "set_gripper" and "params" not in request:
                values = {k: request[k] for k in ("side", "closed") if k in request}
            params = validate_action(kind, values)
            request = {**request, "params": params, **params}
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
            "id": request_id,
            "kind": kind,
            "status": "running",
            "ok": False,
            "start_step": step,
            "before": self.robot.observe_base(),
            "flight_s": 0.0,
            "max_height_m": float(self.robot.data.qpos[2]),
        }
        mode = self.robot.policy.fsm.mode
        different = kind not in CLAW_ACTIONS or request.get("side", "left") != self.claw_side
        if kind != "navigate" and kind not in self.robot.actions:
            self.task.update(phase="rejected", failure="policy_unavailable")
        elif self.claw_closed and (different or kind == "claw_step"):
            self.task.update(phase="rejected", failure="gripper_requires_open")
        elif mode not in {"locomotion", "claw_left", "claw_right"}:
            self.task.update(phase="rejected", failure="policy_state_invalid")
        elif mode.startswith("claw_") and self.claw_side != mode.removeprefix("claw_"):
            self.task.update(phase="rejected", failure="gripper_state_unknown")
        elif not self._still():
            self.task.update(phase="rejected", failure="unsafe_start")
        elif mode.startswith("claw_") and different:
            try:
                self.robot.policy.arm_pose(self.claw_side, "rest")
                self.robot.policy.claw(self.claw_side, False)
                self.robot.policy.tap_mode(mode)
                self.task["phase"] = "stow"
            except (KeyError, ValueError, RuntimeError):
                self._reject_transition()
        if kind in RECORDINGS and kind in self.robot.actions:
            self.task["recording_s"] = self.robot.policy.durations[kind]
        self.robot.stop()

    def _reject_transition(self) -> None:
        """Reject a controller mismatch without letting it escape the worker tick."""
        if self.task is None:
            raise RuntimeError("Cannot reject a policy transition without an active task")
        self.task.update(phase="rejected", failure="policy_transition_failed")
        self.robot.policy.cancel_keys()
        self.robot.stop()

    def cancel(self, reason: str = "cancelled") -> None:
        self.robot.stop()
        if self.base.active:
            self.base.cancel(reason)
        elif self.task and self.task["failure"] is None:
            cancelled_phase = self.task["phase"]
            self.task["failure"] = reason
            # A started jump/closure finishes its bounded motion. A not-yet
            # started action must never be triggered by a cancellation.
            if self.task["kind"] in RECORDINGS and self.robot.policy.fsm.mode == self.task["kind"]:
                self.robot.policy.cancel_keys()
                self.robot.policy.tap_mode(self.task["kind"])
                self.task["saw_mode"] = True
            elif self.task["kind"] in {"claw_step", "claw_pose"} and self.task["phase"] == "execute":
                self.robot.policy.cancel_keys()
                if self.robot.policy.fsm.mode == "locomotion":
                    self.claw_side, self.claw_closed = self.task["previous_claw"]
                self.task["phase"] = "rejected"
            elif self.task["kind"] == "claw_aperture" and not self.task.get("sent"):
                self.robot.policy.cancel_keys()
                if self.robot.policy.fsm.mode == "locomotion":
                    self.claw_side, self.claw_closed = self.task["previous_claw"]
                self.task["phase"] = "rejected"
            elif self.task["phase"] in {"prepare", "stow"}:
                self.task["phase"] = "rejected"
            elif (
                not self.task["saw_mode"]
                and self.robot.policy.fsm.mode == "locomotion"
                and self.robot.policy.key_events
            ):
                self.robot.policy.cancel_keys()
                self.claw_side, self.claw_closed = self.task["previous_claw"]
                self.task["phase"] = "rejected"
            # Diagnostics must never interrupt cancellation or leave queued actions.
            cancellation = {"reason": reason, "phase": cancelled_phase}
            self._result["cancellation"] = cancellation
            try:
                cancellation.update(
                    sim_time_s=float(self.robot.data.time),
                    policy_mode=self.robot.policy.fsm.mode,
                    support_contacts=self._support_count(),
                    base=self.robot.observe_base(),
                    flight_s=self._result.get("flight_s", 0.0),
                )
            except Exception as exc:
                cancellation["diagnostic_error"] = f"{type(exc).__name__}: {exc}"

    def _still(self) -> bool:
        state = self.robot.observe_base()
        posture = self.robot.policy.fsm.mode == "locomotion" and self.robot.policy.posture
        # A held tilt balances with small servo oscillations on the compliant feet.
        linear_still, angular_still = (0.09, 0.8) if posture else (0.02, 0.2)
        return bool(
            self.robot.policy.fsm.running_policy
            and self.robot.data.qpos[2] > 0.07
            and self._support_count() >= 3
            and np.linalg.norm(state["linear_velocity_m_s"]) < linear_still
            and np.linalg.norm(state["angular_velocity_rad_s"]) < angular_still
        )

    def _support_count(self) -> int:
        r = self.robot
        return int(
            sum((r.model.geom_bodyid[c.geom1] == 0) != (r.model.geom_bodyid[c.geom2] == 0) for c in r.data.contact)
        )

    def sample_physics(self) -> None:
        if self.task is None or self.task["kind"] not in PLAYBACK:
            return
        height = float(self.robot.data.qpos[2])
        self._result["max_height_m"] = max(self._result["max_height_m"], height)
        if self.robot.policy.fsm.mode == self.task["kind"] and self.robot.policy.fsm.running_policy:
            self.task["saw_mode"] = True
        if self.task["kind"] not in JUMPS or not self.task["saw_mode"]:
            return
        takeoff_height = self._result["before"]["position_m"][2] + 0.015
        if height > takeoff_height and self._support_count() == 0:
            self._result["flight_s"] += self.robot.policy.dt

    def _dispatch(self, task: dict, step: int) -> None:
        r = self.robot
        if task["kind"] == "navigate":
            r.policy.use_operators(False)
            self.base.start(task["id"], task["dx"], task["dy"], task["dyaw"], step=step)
            self._navigation = True
            self.task = None
            return
        if task["kind"] in PLAYBACK:
            r.policy.use_operators(False)
            r.policy.posture = {}
            r.policy.tap_mode(task["kind"])
            task["phase"] = "execute"
        elif task["kind"] in {"jumper_posture", "stow_claw"}:
            r.policy.use_operators(False)
            if task["kind"] == "jumper_posture":
                r.policy.posture = dict(task["params"])
            task["phase"] = "hold"
        else:
            side = task.get("side", "left")
            r.policy.use_operators(True)
            if r.policy.fsm.mode != "claw_" + side:
                r.policy.tap_mode("claw_" + side)
            self.claw_side = side
            if task["kind"] == "set_gripper":
                r.policy.claw(side, task["closed"])
                self.claw_closed = task["closed"]
            elif task["kind"] == "claw_pose":
                r.policy.arm_pose(side, task["pose"])
            elif task["kind"] == "claw_step":
                task["step_key"] = r.policy.step_key(side, task["direction"])
                task["held_s"] = 0.0
            task["phase"] = "execute"

    def _playback_finished(self, task: dict, dt: float) -> bool:
        r = self.robot
        if task["kind"] == "ref_free_jump" and task.get("saw_mode") and not task.get("leaving"):
            task["active_s"] = task.get("active_s", 0.0) + dt
            supported = self._support_count() >= 3
            task["landed_s"] = task.get("landed_s", 0.0) + dt if supported else 0.0
            flown = self._result["flight_s"] >= 0.02
            if task["landed_s"] >= 0.1 and (flown or task["active_s"] >= REF_FREE_EPISODE_S):
                if not flown:
                    task["failure"] = task["failure"] or "flight_unconfirmed"
                r.policy.tap_mode("ref_free_jump")
                task["leaving"] = True
        reached = task["saw_mode"] and r.policy.fsm.mode == "locomotion" and r.policy.fsm.running_policy
        if reached and task["kind"] in JUMPS and self._result["flight_s"] < 0.02:
            task["failure"] = task["failure"] or "flight_unconfirmed"
        return bool(reached)

    def _gripper_reached(self, task: dict) -> bool:
        r = self.robot
        side = task.get("side", "left")
        in_claw = r.policy.fsm.mode == "claw_" + side
        if in_claw:
            task["saw_mode"] = True
        proportional = task["kind"] == "claw_aperture"
        if proportional and not task.get("sent"):
            controller_ready = in_claw and r.policy.fsm.running_policy and not r.policy.key_events
            if controller_ready and not task["failure"]:
                r.policy.claw_aperture(side, task["opening"])
                self.claw_closed = task["opening"] < 1
                task["sent"] = True
            return False
        name = ("LF" if side == "left" else "RF") + "_J4_joint"
        index = r.policy.wire.index(name)
        sign = 1 if side == "left" else -1
        angle = float(r.data.qpos[r.qids[index]])
        velocity = float(r.data.qvel[r.vids[index]])
        opening = task["opening"] if proportional else float(not task["closed"])
        if 0 < opening < 1:
            # Jumper's five_foot hook interpolates the 60-degree travel from its 0.10 rad shut stop.
            at_aperture = abs(sign * angle - (0.10 - np.deg2rad(60) * opening)) < 0.07
        else:
            at_aperture = sign * angle > 0.03 if opening == 0 else sign * angle < -0.5
        # Compare the actual finger against the official hook target;
        # closure is actuator completion, never evidence of a payload.
        reached = (
            task["saw_mode"]
            and r.policy.fsm.running_policy
            and abs(angle - r.targets[index]) < 0.07
            and abs(velocity) < 0.03
            and at_aperture
        )
        self._result.update(
            side=side,
            closed=opening == 0,
            opening=opening,
            joint_name=name,
            joint_angle_rad=angle,
            joint_velocity_rad_s=velocity,
            joint_target_rad=float(r.targets[index]),
            grasp_verified=False,
        )
        return bool(reached)

    def tick(self, dt: float, step: int) -> None:
        if self.base.active:
            self.base.tick(dt, step)
            return
        if self.task is None:
            return
        task, r = self.task, self.robot
        task["elapsed"] += dt
        if task["elapsed"] > task.get("recording_s", 0) + POLICY_TIMEOUT_S and task["failure"] is None:
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
                    self._dispatch(task, step)
                    if self._navigation:
                        return
                except (KeyError, ValueError, RuntimeError):
                    self._reject_transition()
        reached = task["phase"] in {"rejected", "handoff", "hold"}
        if task["phase"] == "execute":
            if task["kind"] in PLAYBACK:
                reached = self._playback_finished(task, dt)
            elif task["kind"] in {"claw_pose", "claw_step"}:
                in_claw = r.policy.fsm.mode == "claw_" + task["side"] and r.policy.fsm.running_policy
                if in_claw and not r.policy.key_events:
                    if task["kind"] == "claw_pose":
                        reached = True
                    else:
                        if not task["held_s"] and not task["failure"]:
                            r.policy.key_events.append((task["step_key"], True))
                        task["held_s"] += dt
                        if task["held_s"] >= task["duration_s"] or task["failure"]:
                            r.policy.key_events.append((task["step_key"], False))
                            task["phase"] = "hold"
            else:
                reached = self._gripper_reached(task)
        if task["failure"] in {"policy_timeout", "policy_safe"}:
            reached = True
        if reached and task["kind"] == "jumper_posture" and not task["failure"]:
            state = r.observe_base()
            position = np.asarray(state["position_m"])
            quat = np.asarray(state["quaternion_wxyz"])
            anchor, orientation = task.setdefault("balance_pose", (position, quat))
            # Measure drift across the whole hold interval as well as instantaneous speed.
            # Small alternating foot loads can spike the latter even at a fixed tilt.
            if np.linalg.norm(position - anchor) > 0.008 or abs(np.dot(quat, orientation)) < np.cos(0.05 / 2):
                task["balance_pose"] = (position, quat)
                reached = False
        task["stable"] = task["stable"] + dt if reached and self._still() else 0.0
        settled = task["stable"] >= 1.0
        if settled or task["elapsed"] >= task.get("recording_s", 0) + STOP_TIMEOUT_S:
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
                reason=task["failure"] or ("reached" if settled else "policy_timeout"),
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
        squeeze = next(iter(self.robot.policy.trigger_axes.values()), float(self.claw_closed))
        return {
            "side": self.claw_side,
            "closed_command": self.claw_closed if squeeze in (0, 1) else None,
            "opening_command": 1.0 - squeeze,
            "grasp_verified": False,
        }
