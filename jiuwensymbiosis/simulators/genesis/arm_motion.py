"""Scene-owned arm planning and execution; background work only uses snapshots."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import numpy as np


def sibling(name):
    return import_module(f"{__package__}.{name}" if __package__ else name)


class ArmMotion:
    def __init__(self, runtime, profile: dict, directory: Path):
        if find_spec("curobo") is None or find_spec("fcl") is None:
            raise RuntimeError("Arm planning requires cuRobo V2 and python-fcl in the simulation environment")
        self.runtime, self.entity = runtime, runtime.robot.entity
        self.directory = Path(directory)
        self.planner = sibling("arm_planning").CuroboPlanner(runtime.robot.model_manifest, profile, directory)
        self.profile = self.planner.profile
        self.names = [n for n, j in self.planner.joints.items() if j.get("type") != "fixed"]
        self.dofs = [self.entity.get_joint(n).dofs_idx_local[0] for n in self.names]
        self.process = None
        self.follower = sibling("arm_trajectory").TrajectoryFollower(self.read, self.write, self.binding)
        self.result: dict = {}
        self.phase, self.anchor, self.reason = "idle", None, None
        self.stable = self.elapsed = 0.0
        configurations = self.profile.get("effectors", {})
        motors = {n for config in configurations.values() for n in config["joints"]}
        if any(motors.intersection(group["joints"]) for group in self.profile["groups"].values()):
            raise ValueError("arm groups and effector motors must be disjoint")
        self.effectors = sibling("end_effectors").EndEffectors(self.entity, self.planner.root, configurations)
        self.gripper_request = None
        self.execution_path = self.plan_directory = None
        self.goal_poses = self.captured = self.static = self.payloads = None
        self.started = 0.0

    @property
    def active(self):
        return self.phase in {"planning", "executing", "gripping", "braking", "confirming"}

    def snapshot(self):
        # mmap status is bounded. Detailed arrays stay in retained artifacts.
        keys = (
            "id status ok goal_reached settled stop_confirmation plan_success "
            "reason failure_stage backend_success max_tracking_error grasp_confirmed "
            "holding_payload contact_confirmed placement_confirmed"
        ).split()
        result = {key: self.result[key] for key in keys if key in self.result}
        if result.get("reason"):
            result["reason"] = result["reason"][:240]
        return result

    def read(self):
        q = self.entity.get_dofs_position(self.dofs)[0].detach().cpu().numpy()
        v = self.entity.get_dofs_velocity(self.dofs)[0].detach().cpu().numpy()
        if not np.isfinite([q, v]).all():
            raise ValueError("nonfinite robot state")
        return dict(zip(self.names, q.tolist(), strict=True)), dict(zip(self.names, v.tolist(), strict=True))

    def write(self, targets):
        names = list(targets)
        self.entity.control_dofs_position([targets[n] for n in names], [self.dofs[self.names.index(n)] for n in names])

    def base_pose(self):
        link = self.entity.get_link(self.profile["base_link"])
        return sibling("arm_collision").link_pose(link)

    def binding(self):
        if self.anchor is None:
            return ""
        pose = self.base_pose()
        error = sibling("arm_planning").pose_error(pose, self.anchor)
        changed = error["position_m"] > 0.003 or error["rotation_rad"] > 0.005
        q, _ = self.read()
        changed |= any(abs(q[n] - v) > 0.01 for n, v in self.static.items())
        if getattr(self, "captured", None) is not None:
            changed |= not sibling("arm_collision").world_unchanged(self.entity, self.captured)
            changed |= any(e.state == "unknown" for e in self.effectors.evidence.values())
            changed |= not self.effectors.unchanged(self.payloads)
        return "changed" if changed else f"{self.runtime.info['instance_id']}:{self.planner.model_id}"

    def tool_poses(self):
        inverse = np.linalg.inv(self.base_pose())
        names = {name for group in self.profile["groups"].values() for name in group["tools"]}
        result = {}
        for name in sorted(names):
            link = self.entity.get_link(name)
            pose = inverse @ sibling("arm_collision").link_pose(link)
            result[name] = [
                *pose[:3, 3].tolist(),
                *sibling("arm_planning").Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True).tolist(),
            ]
        return result

    def check_tool_goal(self):
        if not self.goal_poses:
            return
        actual = self.tool_poses()
        errors = {}
        for name, goal in self.goal_poses.items():
            planning = sibling("arm_planning")
            error = planning.pose_error(
                planning.transform(goal[:3], goal[3:]), planning.transform(actual[name][:3], actual[name][3:])
            )
            errors[name] = error
            if error["position_m"] > 0.005 or error["rotation_rad"] > 0.05:
                self.reason = "tool_goal_not_reached"
        self.result["tool_goal_errors"] = errors

    def stopped(self):
        _, v = self.read()
        root = sibling("base_motion").measure(self.runtime.robot)
        return (
            max(abs(value) for name, value in v.items() if name not in self.planner.uncontrolled) < 0.02
            and np.linalg.norm(root["linear_velocity_m_s"]) < 0.008
            and np.linalg.norm(root["angular_velocity_rad_s"]) < 0.02
        )

    def accept_request(self, request, *, scene_id, revision, now, step):
        same = request is not None and request.get("id") == self.result.get("id")
        live = (
            request is not None
            and request.get("scene_instance_id") == scene_id
            and request.get("config_revision") == revision
            and not request.get("cancelled")
            and request.get("lease_until", 0) > now
        )
        if self.active:
            if not live or not same:
                self.cancel("cancelled_or_lease_expired")
        elif request is not None and not same:
            self.result = {
                "id": request["id"],
                "status": "running",
                "ok": False,
                "goal_reached": False,
                "settled": False,
                "stop_confirmation": "unknown",
                "plan_success": False,
            }
            self.reason = None
            self.execution_path = self.directory / f"execution-{uuid.uuid4().hex}.json"
            self.elapsed = self.stable = 0.0
            try:
                if not live:
                    raise ValueError("cancelled_or_lease_expired")
                self.start(request)
            except Exception as exc:
                self.cancel(f"{type(exc).__name__}: {exc}")

    def start(self, request):
        if self.follower.terminal or (self.process is not None and self.process.poll() is None):
            raise RuntimeError("planner busy or stop unconfirmed")
        if self.runtime.motion.active or not self.stopped():
            raise ValueError("base and joints must be measured stopped before planning")
        self.runtime.robot.stop()
        state, _ = self.read()
        self.goal_poses = None
        self.captured = None
        self.gripper_request = None
        if request.get("effector") is not None:
            if request.get("targets") is not None or request.get("poses") is not None:
                raise ValueError("effector and arm goals are exclusive")
            self.effectors.start(request["effector"], opened=request.get("opened"))
            self.gripper_request = (request["effector"], request["opened"])
            self.phase = "gripping"
            return
        group = request.get("group")
        if group is None:
            wanted = set(request.get("targets") or {})
            candidates = [
                (len(g["joints"]), name)
                for name, g in self.profile["groups"].items()
                if wanted and wanted <= set(g["joints"])
            ]
            if not candidates:
                raise ValueError("no configured group contains targets")
            group = min(candidates)[1]
        settings = self.profile["groups"][group]
        self.goal_poses = request.get("poses")
        if self.goal_poses is not None and set(self.goal_poses) != set(settings["tools"]):
            raise ValueError("all group tool poses required")
        self.anchor = self.base_pose()
        self.static = {
            n: v for n, v in state.items() if n not in settings["joints"] and n not in self.planner.uncontrolled
        }
        binding = self.binding()
        if binding == "changed":
            raise ValueError("state changed during snapshot")
        fk = sibling("arm_planning").forward_kinematics(self.planner.root, state)
        errors = {}
        for name in settings["tools"]:
            link = self.entity.get_link(name)
            measured = np.linalg.inv(self.anchor) @ sibling("arm_collision").link_pose(link)
            expected = np.linalg.inv(fk[self.profile["base_link"]]) @ fk[name]
            error = sibling("arm_planning").pose_error(expected, measured)
            errors[name] = error
            if error["position_m"] > 0.002 or error["rotation_rad"] > 0.005:
                raise ValueError("Genesis tool FK differs from planning model")
        self.payloads = self.effectors.attachments()
        captured = sibling("arm_collision").snapshot(self.entity, self.profile, self.payloads)
        self.captured = captured
        self.result["fk_errors"] = errors
        self.phase = "planning"
        self.started = time.monotonic()
        self.plan_directory = self.directory / uuid.uuid4().hex
        self.plan_directory.mkdir()
        data = {
            "manifest": self.planner.model,
            "profile": self.profile,
            "group": group,
            "state": state,
            "captured": captured,
            "binding": binding,
            "request": request,
        }
        path = self.plan_directory / "request.json"
        path.write_text(json.dumps(data, default=lambda value: value.tolist(), allow_nan=False))
        with (self.plan_directory / "worker.log").open("w") as log:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("arm_plan_worker.py")), str(path)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
            )

    def _stop_planner(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2.0)
            self.process = None

    def cancel(self, reason="cancelled"):
        if self.phase == "braking":
            return
        self.reason = reason
        self._stop_planner()
        self.effectors.cancel()
        self.follower.cancel(reason)
        self.runtime.robot.stop()
        self.phase = "braking"
        self.stable = self.elapsed = 0.0

    def tick(self, dt):
        try:
            self._tick(dt)
        except Exception as exc:
            self.follower.terminal = True
            self.result.update(
                status="completed",
                ok=False,
                goal_reached=False,
                settled=False,
                stop_confirmation="unknown",
                reason=f"arm_tick_failed: {exc}",
            )
            self.cancel(self.result["reason"])
            self.phase = "completed"

    def _tick(self, dt):
        try:
            self.effectors.tick(dt)
        except Exception as exc:
            self.effectors.invalidate()
            if self.active:
                self.cancel(f"payload_readback_failed: {exc}")
        if not self.active:
            return
        self.runtime.robot.stop()
        self.elapsed += dt
        if self.phase == "gripping":
            if self.effectors.command is None:
                self.result.update(self.effectors.result)
                self.reason = self.effectors.result.get("reason")
                self.phase = "confirming"
                self.stable = self.elapsed = 0.0
        elif self.phase == "planning":
            if self.binding() == "changed":
                self.cancel("planning_world_or_payload_changed")
            elif time.monotonic() - self.started > 90:
                self.cancel("planning_timeout_result_discarded")
            elif self.process.poll() is not None:
                try:
                    path = self.plan_directory / "result.json"
                    result = json.loads(path.read_text())
                    self.result.update(
                        {key: result[key] for key in ("backend_success", "failure_stage", "timing") if key in result}
                    )
                    if not result["ok"]:
                        raise ValueError(result["reason"])
                    data = result["trajectory"]
                    data["names"] = tuple(data["names"])
                    trajectory = sibling("arm_trajectory").Trajectory(**data)
                    validation = result["collision_validation"]
                    self.follower.start(trajectory)
                    self.result.update(plan_success=True, trajectory_path=str(path), collision_validation=validation)
                    self.phase = "executing"
                except Exception as exc:
                    self.cancel(f"{type(exc).__name__}: {exc}")
        elif self.phase == "executing":
            self.follower.tick(dt)
            if not self.follower.active:
                self.result.update(self.follower.result)
                self.result.update(status="running", ok=False, settled=False, stop_confirmation="unknown")
                self.reason = self.follower.result.get("reason")
                self.phase = "confirming"
                self.stable = self.elapsed = 0.0
        else:
            self.follower.tick(dt)
            if self.phase == "confirming" and self.captured is not None and self.binding() == "changed":
                self.reason = "world_or_payload_changed_during_settle"
            self.stable = self.stable + dt if self.stopped() else 0.0
            if self.stable >= 0.25 or self.elapsed > 5.0:
                settled = self.stable >= 0.25 and not self.follower.terminal
                if settled and self.reason is None:
                    self.check_tool_goal()
                    if self.gripper_request is not None:
                        name, opened = self.gripper_request
                        evidence = self.effectors.evidence[name]
                        self.result.update(evidence.report())
                        confirmed = (
                            (
                                evidence.holding is False
                                and (evidence.object_id is None or evidence.report()["placement_confirmed"])
                            )
                            if opened
                            else evidence.state in {"contact", "held"} and evidence.contact_time >= evidence.contact_s
                        )
                        if not confirmed:
                            self.reason = "grasp_or_release_confirmation_lost"
                self.result.update(
                    status="completed",
                    ok=settled and self.reason is None,
                    goal_reached=settled and self.reason is None,
                    reason=self.reason,
                    settled=settled,
                    stop_confirmation="measured" if settled else "unknown",
                )
                states = list(self.effectors.evidence.values())
                if states and self.gripper_request is None:
                    held = (
                        True
                        if any(e.holding is True for e in states)
                        else False
                        if all(e.holding is False for e in states)
                        else None
                    )
                    self.result.update(holding_payload=held, grasp_confirmed=held is True)
                self.phase = "completed"
                try:
                    self.execution_path.write_text(json.dumps(self.result, indent=2, allow_nan=False))
                except OSError:
                    self.result.update(ok=False, reason="evidence_unwritable")
                if not settled:
                    self.follower.terminal = True

    def close(self):
        self.cancel("scene_closed")
        self._stop_planner()
        self.planner.close()
