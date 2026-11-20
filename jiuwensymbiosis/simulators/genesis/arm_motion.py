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
    pick_lift = None

    def __init__(self, runtime, profile: dict, directory: Path):
        if find_spec("curobo") is None:
            raise RuntimeError("Arm planning requires cuRobo V2 in the simulation environment")
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
        self.effector_followers = {
            n
            for n, joint in self.planner.joints.items()
            if (mimic := joint.find("mimic")) is not None and mimic.get("joint") in motors
        }
        if any(motors.intersection(group["joints"]) for group in self.profile["groups"].values()):
            raise ValueError("arm groups and effector motors must be disjoint")
        self.effectors = sibling("end_effectors").EndEffectors(self.entity, self.planner.root, configurations)
        self.pick_lift = None
        self.gripper_request = None
        self.execution_path = self.plan_directory = None
        self.goal_poses = self.captured = self.static = self.payloads = None
        self.started = 0.0
        self.command_started = self.execution_started = None

    def warmup(self):
        """Prepare configured grasp groups during load; never execute a trajectory."""
        started = time.monotonic()
        groups = sorted({config["group"] for config in self.profile.get("grasp_pose_backends", {}).values()})
        report = {"groups": {}, "robot_motion": False}
        try:
            if not groups:
                return report
            state, _ = self.read()
            captured = sibling("arm_collision").snapshot(self.entity, self.profile)
            fk = sibling("arm_planning").forward_kinematics(self.planner.root, state)
            for group in groups:
                directory = self.directory / f"warmup-{group}-{uuid.uuid4().hex}"
                directory.mkdir()
                path = directory / "request.json"
                begin = time.monotonic()
                self._submit_plan(
                    path,
                    {
                        "manifest": self.planner.model,
                        "profile": self.profile,
                        "group": group,
                        "state": state,
                        "captured": captured,
                        "binding": "",
                        "request": {"warmup": True},
                    },
                )
                result_path = directory / "result.json"
                while not result_path.exists():
                    if self.process.poll() is not None:
                        raise RuntimeError("cuRobo planner exited during loading")
                    if time.monotonic() - begin > 90:
                        raise TimeoutError("cuRobo planner loading timeout")
                    time.sleep(0.01)
                result = json.loads(result_path.read_text())
                report["groups"][group] = {"motion": result, "motion_request_wall_s": time.monotonic() - begin}
                if not result["ok"]:
                    raise RuntimeError(result["reason"])
                tool = self.profile["groups"][group]["tools"][0]
                target = np.linalg.inv(fk[self.profile["base_link"]]) @ fk[tool]
                begin = time.monotonic()
                result = sibling("arm_ik").solve(self, group, tool, np.asarray([target]), warmup=True)
                report["groups"][group].update(ik=result, ik_request_wall_s=time.monotonic() - begin)
            report["ok"] = True
            return report
        except Exception as exc:
            report.update(ok=False, reason=f"{type(exc).__name__}: {exc}")
            self._stop_planner()
            sibling("arm_ik").stop_worker(self)
            raise
        finally:
            report["total_s"] = time.monotonic() - started
            (self.directory / "warmup.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    @property
    def active(self):
        return self.phase in {"planning", "executing", "closing_for_lift", "gripping", "braking", "confirming"}

    def snapshot(self):
        # mmap status is bounded. Detailed arrays stay in retained artifacts.
        keys = (
            "id",
            "status",
            "ok",
            "goal_reached",
            "settled",
            "stop_confirmation",
            "plan_success",
            "reason",
            "failure_stage",
            "backend_success",
            "execution_mode",
            "max_tracking_error",
            "grasp_confirmed",
            "holding_payload",
            "contact_confirmed",
            "placement_confirmed",
            "lift_m",
            "held_s",
        )
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
        # Mimic linkage motion can amplify small motor tracking deviations.
        tolerances = {n: 0.02 if n in self.effector_followers else 0.01 for n in self.static}
        drift = {n: q[n] - v for n, v in self.static.items() if abs(q[n] - v) > tolerances[n]}
        peaks = self.result.setdefault("effector_follower_max_drift_rad", {})
        for n in self.effector_followers.intersection(self.static):
            peaks[n] = max(peaks.get(n, 0.0), abs(q[n] - self.static[n]))
        changed |= bool(drift)
        if changed:
            self.result["binding_change"] = {
                "base_error": error,
                "stationary_joint_drift": drift,
                "joint_tolerances_rad": {n: tolerances[n] for n in drift},
            }
        if getattr(self, "captured", None) is not None:
            changed |= not sibling("arm_collision").world_unchanged(self.entity, self.captured)
            changed |= any(e.state == "unknown" for e in self.effectors.evidence.values())
            if not self.pick_lift or not self.pick_lift.started:
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
        from ..common.base_motion import measure

        _, v = self.read()
        root = measure(self.runtime.robot)
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
            self.command_started = time.monotonic()
            self.execution_started = None
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
        if self.follower.terminal or self.active:
            raise RuntimeError("planner busy or stop unconfirmed")
        if self.runtime.motion.active or not self.stopped():
            raise ValueError("base and joints must be measured stopped before planning")
        self.runtime.robot.stop()
        state, _ = self.read()
        self.goal_poses = None
        self.captured = None
        self.pick_lift = None
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
        trial = request.get("pick_lift")
        if trial is not None:
            if self.goal_poses or request.get("targets") or "grasp_pose_backends" not in self.profile:
                raise ValueError("trial lift requires a configured grasp backend and a vertical goal")
            effector = trial["effector"]
            tool = self.profile["effectors"][effector]["tool"]
            if settings["tools"] != [tool]:
                raise ValueError("invalid lift group")
            self.pick_lift = sibling("pick_lift").PickLift(self, effector, trial["center_m"])
            target = self.tool_poses()[tool]
            target[:3] = (np.asarray(target[:3]) + self.base_pose()[:3, :3].T @ [0.0, 0.0, 0.08]).tolist()
            self.goal_poses = {tool: target}
            if "joint_goal" in trial:
                q = trial["joint_goal"]
                if set(q) != set(settings["joints"]) or not np.isfinite(list(q.values())).all():
                    raise ValueError("invalid trial lift joint goal")
                fk = sibling("arm_planning").forward_kinematics(self.planner.root, state | q)
                predicted = np.linalg.inv(fk[self.profile["base_link"]]) @ fk[tool]
                goal = sibling("arm_planning").transform(target[:3], target[3:])
                delta = np.linalg.inv(goal) @ predicted
                if (
                    np.linalg.norm(delta[:3, 3]) > 0.005
                    or sibling("arm_planning").Rotation.from_matrix(delta[:3, :3]).magnitude() > 0.02
                ):
                    raise ValueError("trial lift joint goal does not match the vertical lift")
                request = {**request, "targets": q}
            else:
                request = {**request, "poses": self.goal_poses}

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
        self.payloads = self.pick_lift.payload if self.pick_lift else self.effectors.attachments()
        if self.pick_lift:
            motors = self.effectors.configurations[self.pick_lift.name]["joints"]
            for n, joint in self.planner.joints.items():
                mimic = joint.find("mimic")
                if n in motors or (mimic is not None and mimic.get("joint") in motors):
                    self.static.pop(n, None)
        snapshot_started = time.monotonic()
        captured = sibling("arm_collision").snapshot(self.entity, self.profile, self.payloads)
        self.result.setdefault("timing", {})["scene_snapshot_s"] = time.monotonic() - snapshot_started
        self.captured = captured
        self.result["fk_errors"] = errors
        self.result["execution_mode"] = "curobo_planned"
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
        self._submit_plan(path, data)

    def _submit_plan(self, path, data):
        """Share the scene-owned worker between load-time warmup and action requests."""
        path.write_text(json.dumps(data, default=lambda value: value.tolist(), allow_nan=False))
        if self.process is None or self.process.poll() is not None:
            self._stop_planner()
            with (self.directory / "planner-worker.log").open("a") as log:
                self.process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).with_name("arm_plan_worker.py"))],
                    stdin=subprocess.PIPE,
                    env=sibling("arm_ik").worker_environment(),
                    stdout=log,
                    stderr=log,
                    text=True,
                )
        self.process.stdin.write(str(path.resolve()) + "\n")
        self.process.stdin.flush()

    def _stop_planner(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2.0)
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process = None

    def cancel(self, reason="cancelled"):
        if self.phase == "braking":
            return
        self.reason = reason
        if self.pick_lift:
            self.pick_lift.uncertain()
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
            if (
                self.pick_lift
                and self.pick_lift.started
                and self.phase in {"closing_for_lift", "executing", "confirming"}
            ):
                self.effectors.drive_closing(dt)
                if self.phase == "confirming":
                    self.pick_lift.measure(dt)
            else:
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
            elif (self.plan_directory / "result.json").exists() or self.process.poll() is not None:
                try:
                    path = self.plan_directory / "result.json"
                    result = json.loads(path.read_text())
                    result.setdefault("timing", {})["request_wall_s"] = time.monotonic() - self.started
                    self.result.setdefault("timing", {}).update(result["timing"])
                    self.result.update(
                        {key: result[key] for key in ("backend_success", "failure_stage") if key in result}
                    )
                    if not result["ok"]:
                        raise ValueError(result["reason"])
                    data = result["trajectory"]
                    data["names"] = tuple(data["names"])
                    trajectory = sibling("arm_trajectory").Trajectory(**data)
                    validation = result["collision_validation"]
                    self.follower.start(trajectory)
                    self.execution_started = time.monotonic()
                    self.result.update(plan_success=True, trajectory_path=str(path), collision_validation=validation)
                    if self.pick_lift:
                        self.pick_lift.close()
                        self.phase = "closing_for_lift"
                    else:
                        self.phase = "executing"
                except Exception as exc:
                    self.cancel(f"{type(exc).__name__}: {exc}")
        elif self.phase == "closing_for_lift":
            if self.binding() == "changed":
                self.cancel("scene_or_base_changed")
            elif self.pick_lift.name not in self.effectors.closing:
                self.phase = "executing"
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
            hold_window = 2.0 if self.pick_lift and self.pick_lift.started and self.phase == "confirming" else 0.0
            if (self.stable >= 0.25 and self.elapsed >= hold_window) or self.elapsed > 5.0:
                settled = self.stable >= 0.25 and not self.follower.terminal
                if self.pick_lift and self.pick_lift.started:
                    self.result.update(lift_m=self.pick_lift.rise, held_s=self.pick_lift.stable_s)
                if not settled and self.reason is None:
                    self.reason = "stop_unconfirmed"
                if settled and self.reason is None:
                    self.check_tool_goal()
                    if self.pick_lift and self.pick_lift.started:
                        if self.pick_lift.finish():
                            self.result.update(self.effectors.evidence[self.pick_lift.name].report())
                        else:
                            self.reason = "target_hold_unconfirmed"
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
                if self.pick_lift and (self.reason or not settled):
                    self.pick_lift.uncertain()
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
                now = time.monotonic()
                timing = self.result.setdefault("timing", {})
                if self.command_started is not None:
                    timing["command_wall_s"] = now - self.command_started
                if self.execution_started is not None:
                    timing["execution_and_settle_s"] = now - self.execution_started
                try:
                    self.execution_path.write_text(json.dumps(self.result, indent=2, allow_nan=False))
                except OSError:
                    self.result.update(ok=False, reason="evidence_unwritable")
                if not settled:
                    self.follower.terminal = True

    def close(self):
        sibling("arm_ik").stop_worker(self)
        self.cancel("scene_closed")
        self._stop_planner()
        self.planner.close()
