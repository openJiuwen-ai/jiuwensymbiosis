"""Lease-bound Pollen policy selection and measured completion in native MuJoCo."""

from __future__ import annotations

import math

import numpy as np

from .policy_actions import ARENA_GAITS, validate_action
from .policy_motion import PolicyMotion

# At this phase the published roller crouch holds both wheels level under the lowered body.
ROLLER_SEATED_PHASE = 0.28
SPRINT_DISTANCE_M = 2.0
# The standing policy needs room to brake after the sprint policy releases control.
SPRINT_BRAKE_M = 1.90
SPRINT_DRIVE_LIMIT_S = 12.0


class MicroduckMotion(PolicyMotion):
    def start(self, request: dict, step: int) -> None:
        r, p = self.robot, self.robot.policy
        kind = request.get("kind", "navigate")
        before = r.observe_base()
        self._navigation = False
        self._result = {
            "id": request["id"],
            "kind": kind,
            "status": "running",
            "ok": False,
            "before": before,
            "start_step": step,
        }
        self.task = {"kind": kind, "elapsed": 0.0, "stable": 0.0, "failure": None, "wait": 0.25}
        r.stop()
        if p.sit_mode and kind not in {"sit", "stand", "head_pose", "mouth"}:
            self.task["failure"] = "stand_required"
            return
        if kind != "recover_stance" and not p.sit_mode and not r.upright():
            self.task["failure"] = "recovery_required"
            return
        if kind == "navigate" and not r.rollers:
            self.base.start(request["id"], request["dx"], request["dy"], request["dyaw"], step=step)
            self._navigation, self.task = True, None
            return
        if kind not in r.actions:
            self.task["failure"] = "policy_unavailable"
            return
        values = validate_action(kind, request.get("params", {}))
        if kind == "mouth":
            r.set_mouth(values["opening"])
        elif kind in {"head_pose", "body_pose"}:
            if r.gait in ARENA_GAITS and not p.sit_mode:
                r.select_gait("velstand_gait")
            if kind == "head_pose":
                r.set_policy_pose(head=list(values.values()))
            else:
                r.set_policy_pose(body=list(values.values()))
        elif kind in {"sit", "stand"}:
            if kind == "sit" and not p.sit_mode:
                # Start the sitstand episode in its neutral head/body command frame.
                # Once seated, the head can be steered again.
                r.set_policy_pose(head=0, body=0)
            if r.rollers:
                if kind == "sit" and not p.sit_mode:
                    p.action_scale = r.specs["roller_crouch"]["action_scale"]
                    p.trigger_ground_pick()
                    self.task["seat_entry"] = True
                elif kind == "stand":
                    r.hold_roller_seat = p.sit_mode = False
            else:
                if p.sit_mode != (kind == "sit"):
                    p.action_scale = 1.0
                    p.toggle_sit()
                timing = r.specs["alpha_sitstand"]
                self.task["wait"] = timing["ramp_s"] if kind == "sit" else timing["unwind_s"]
        elif kind == "sprint_2m":
            r.select_gait(kind)
            self.task["sprinting"] = True
        elif kind in {"classic_gait", "velstand_gait", "roller_gait", "recover_stance"} | ARENA_GAITS.keys():
            r.select_gait("velstand_gait" if kind == "recover_stance" else kind)
        elif kind == "got_a_talent":
            p.head_offset[:] = p.body_cmd[:] = 0
            p.ort_session = r.sessions[kind]
            p.current_policy = kind
            r.show_started = r.data.time
            self.task["wait"] = r.timeline["duration_s"]
        elif kind in {"ground_pick", "roller_crouch"}:
            p.trigger_ground_pick()
            p.command[3:] = 0
            if kind == "roller_crouch":
                p.action_scale = r.specs[kind]["action_scale"]
        elif kind == "roller_skate":
            self.task.update(skate=values, wait=values["duration_s"])
        else:
            # The released one-shots start from a quiet, neutral stance.
            r.set_policy_pose(head=0, body=0)
            self.task["prepare"] = True
            if kind == "roulade":
                self.task["rolls"] = values["repetitions"]
                self.task["remaining"] = values["repetitions"] - 1
        self._result["policy"] = p.current_policy

    def cancel(self, reason: str = "cancelled") -> None:
        self.robot.stop()
        if self.base.active:
            self.base.cancel(reason)
        elif self.task and self.task["failure"] is None:
            # Let the bounded reference finish a roll, kick, sit or ground contact before handoff.
            self.task["failure"] = reason
            if self.task["kind"] == "got_a_talent" and self.robot.show_started is not None:
                # The closing frame of the published show is a neutral stand.
                last = self.robot.timeline["keyframes"][-1]["t"]
                self.robot.show_started = self.robot.data.time - last
                self.task["wait"] = self.task["elapsed"] + self.robot.timeline["duration_s"] - last

    def _sprint_distance(self, pose: dict) -> float:
        before = self._result["before"]
        x, y = pose["position_m"][:2]
        start_x, start_y = before["position_m"][:2]
        yaw = before["body_yaw_rad"]
        return float((x - start_x) * math.cos(yaw) + (y - start_y) * math.sin(yaw))

    def tick(self, dt: float, step: int) -> None:
        if self.base.active:
            if not self.robot.upright():
                self.base.cancel("recovery_required")
            self.base.tick(dt, step)
            return
        if self.task is None:
            return
        task, r = self.task, self.robot
        p = r.policy
        task["elapsed"] += dt
        kind = task["kind"]
        if task.get("sprinting"):
            if not r.upright():
                task["failure"] = task["failure"] or "recovery_required"
            distance = self._sprint_distance(r.observe_base())
            if not task["failure"] and distance < SPRINT_BRAKE_M and task["elapsed"] < SPRINT_DRIVE_LIMIT_S:
                r.command(r.max_linear_rate, 0, 0, min(1, 2 * dt))
                self._result["policy"] = "walking"
                return
            r.stop()
            task.update(sprinting=False, wait=task["elapsed"])
        if task.get("seat_entry") and not task["failure"]:
            if p.ground_pick_mode and p.ground_pick_phase >= ROLLER_SEATED_PHASE and r.data.qpos[2] < 0.085:
                r.hold_roller_seat = p.sit_mode = True
                task["seat_entry"] = False
            elif not p.ground_pick_mode:
                task["failure"] = "seat_unconfirmed"
        if task.pop("prepare", False) and not task["failure"]:
            pose = r.observe_base()
            quiet = (
                np.linalg.norm(pose["linear_velocity_m_s"]) < 0.035
                and np.linalg.norm(pose["angular_velocity_rad_s"]) < 0.28
            )
            task["stable"] = task["stable"] + dt if quiet and r.upright() else 0.0
            if task["stable"] >= 0.4:
                if kind.startswith("kick_"):
                    task["ball_start"] = r.data.qpos[p.ball_qpos_adr : p.ball_qpos_adr + 2].copy()
                    r.watch_kick(kind.removeprefix("kick_"))
                p.trigger_behavior(kind)
                task["stable"] = 0.0
                self._result["policy"] = p.current_policy
            elif task["elapsed"] < task.get("prepare_started", 0) + 10:
                task["prepare"] = True
                return
            else:
                task["failure"] = "unsafe_start"
        if kind == "roller_skate" and "skate" in task:
            skate = task["skate"]
            if not r.upright():
                task["failure"] = task["failure"] or "recovery_required"
            if task["elapsed"] < task["wait"] and not task["failure"]:
                r.command(skate["forward"], 0, skate["heading"], min(1, dt * 2))
            else:
                r.stop()
        if kind == "roulade":
            up = r.data.body(r.base_body).xmat[8]
            if up < -0.2 and not task.get("inverted"):
                task["inverted"] = self._result["inverted"] = True
                self._result["rolls_observed"] = self._result.get("rolls_observed", 0) + 1
            if up > 0.7:
                task["inverted"] = False
        if r.show_started is not None:
            return
        if (p.ground_pick_mode and not r.hold_roller_seat) or p.behavior_mode:
            if task["elapsed"] < 10 + task.get("rolls", 0):
                return
            task["failure"] = "policy_timeout"
            r.select_gait(r.gait)
        if task["elapsed"] < task["wait"] and not task["failure"]:
            return
        if kind == "stand" and not p.sit_mode and p.current_policy == "sit":
            # The published rise window ends before the gait takes over to balance.
            r.select_gait(r.gait)
            task["stable"] = 0
        after = r.observe_base()
        stable = (
            np.linalg.norm(after["linear_velocity_m_s"]) < 0.035
            and np.linalg.norm(after["angular_velocity_rad_s"]) < 0.28
        )
        task["stable"] = task["stable"] + dt if stable else 0.0
        settled = task["stable"] >= 0.4
        if p.sit_mode:
            pose_reached = r.data.body(r.base_body).xmat[8] > 0.7 and after["position_m"][2] < 0.09
        else:
            pose_reached = r.upright()
        if kind == "mouth":
            angle = float(r.data.qpos[r.mouth_qid])
            pose_reached = abs(angle - r.mouth_target) < 0.05 and abs(r.data.qvel[r.mouth_vid]) < 0.05
            self._result.update(joint_angle_rad=angle, joint_target_rad=r.mouth_target, grasp_verified=False)
        ready = settled and (pose_reached or task["failure"])
        if not ready and task["elapsed"] < task.get("prepare_started", 0) + task["wait"] + 10:
            return
        if kind == "roulade" and ready and task.get("remaining") and not task["failure"]:
            # Re-arm the released one-second actor from its measured standing entry pose.
            task.update(remaining=task["remaining"] - 1, prepare=True, prepare_started=task["elapsed"], stable=0.0)
            return
        if kind == "roulade" and self._result.get("rolls_observed", 0) < task.get("rolls", 1):
            task["failure"] = task["failure"] or "roll_unconfirmed"
        if kind == "sprint_2m" and settled:
            distance = self._sprint_distance(after)
            self._result["forward_displacement_m"] = distance
            if ready and distance < SPRINT_DISTANCE_M:
                task["failure"] = task["failure"] or "sprint_short"
            if "sprinting" in task:
                # Braking already uses the velstand actor at zero twist. Restore
                # its walking slot before another command can reach the sprint actor.
                r.select_gait("velstand_gait")
        if kind == "roller_skate" and task.get("skate", {}).get("forward", 0) > 0:
            distance = math.dist(self._result["before"]["position_m"][:2], after["position_m"][:2])
            self._result["distance_m"] = distance
            if distance < 0.01:
                task["failure"] = task["failure"] or "skate_unconfirmed"
        if "ball_start" in task:
            self._result["ball_displacement_m"] = math.dist(
                task["ball_start"], r.data.qpos[p.ball_qpos_adr : p.ball_qpos_adr + 2]
            )
            self._result["foot_ball_contact"] = r.kick_contact
            r.kicking_bodies.clear()
            if ready and (not r.kick_contact or self._result["ball_displacement_m"] < 0.01):
                task["failure"] = task["failure"] or "kick_unconfirmed"
        self._result.update(
            status="completed",
            ok=bool(ready and not task["failure"]),
            reason=task["failure"] or ("reached" if ready else "policy_timeout"),
            settled=settled,
            after=after,
            end_step=step,
            duration_sim_s=task["elapsed"],
        )
        self.task = None
