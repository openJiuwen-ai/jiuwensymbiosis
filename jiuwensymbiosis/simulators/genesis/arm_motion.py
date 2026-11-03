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
        self.phase = "idle"
        self.anchor = None
        self.reason = None
        self.stable = self.elapsed = 0.0

    @property
    def active(self):
        return self.phase in {"planning", "executing", "braking", "confirming"}

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
            "max_tracking_error",
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
        return sibling("arm_planning").transform(link.get_pos()[0].cpu().numpy(), link.get_quat()[0].cpu().numpy())

    def binding(self):
        if self.anchor is None:
            return ""
        pose = self.base_pose()
        changed = (
            np.linalg.norm(pose[:3, 3] - self.anchor[:3, 3]) > 0.003
            or sibling("arm_planning").Rotation.from_matrix(pose[:3, :3] @ self.anchor[:3, :3].T).magnitude() > 0.005
        )
        q, _ = self.read()
        changed |= any(abs(q[n] - v) > 0.01 for n, v in self.static.items())
        return "changed" if changed else f"{self.runtime.info['instance_id']}:{self.planner.model_id}"

    def tool_poses(self):
        inverse = np.linalg.inv(self.base_pose())
        names = {name for group in self.profile["groups"].values() for name in group["tools"]}
        result = {}
        for name in sorted(names):
            link = self.entity.get_link(name)
            pose = inverse @ sibling("arm_planning").transform(
                link.get_pos()[0].cpu().numpy(), link.get_quat()[0].cpu().numpy()
            )
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
            position = float(np.linalg.norm(np.array(actual[name][:3]) - goal[:3]))
            rotation = float(
                (
                    sibling("arm_planning").Rotation.from_quat(actual[name][3:], scalar_first=True).inv()
                    * sibling("arm_planning").Rotation.from_quat(goal[3:], scalar_first=True)
                ).magnitude()
            )
            errors[name] = {"position_m": position, "rotation_rad": rotation}
            if position > 0.005 or rotation > 0.05:
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
            measured = np.linalg.inv(self.anchor) @ sibling("arm_planning").transform(
                link.get_pos()[0].cpu().numpy(), link.get_quat()[0].cpu().numpy()
            )
            expected = np.linalg.inv(fk[self.profile["base_link"]]) @ fk[name]
            position = float(np.linalg.norm(expected[:3, 3] - measured[:3, 3]))
            angle = float(
                sibling("arm_planning").Rotation.from_matrix(expected[:3, :3] @ measured[:3, :3].T).magnitude()
            )
            errors[name] = {"position_m": position, "rotation_rad": angle}
            if position > 0.002 or angle > 0.005:
                raise ValueError("Genesis tool FK differs from planning model")
        captured = sibling("arm_collision").snapshot(self.entity, self.profile)
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
        self.follower.cancel(reason)
        self.runtime.robot.stop()
        self.phase = "braking"
        self.stable = self.elapsed = 0.0

    def tick(self, dt):
        if not self.active:
            return
        self.runtime.robot.stop()
        self.elapsed += dt
        if self.phase == "planning":
            if time.monotonic() - self.started > 90:
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
            self.stable = self.stable + dt if self.stopped() else 0.0
            if self.stable >= 0.25 or self.elapsed > 5.0:
                settled = self.stable >= 0.25 and not self.follower.terminal
                if settled and self.reason is None:
                    self.check_tool_goal()
                self.result.update(
                    status="completed",
                    ok=settled and self.reason is None,
                    goal_reached=settled and self.reason is None,
                    reason=self.reason,
                    settled=settled,
                    stop_confirmation="measured" if settled else "unknown",
                )
                self.phase = "completed"
                self.execution_path.write_text(json.dumps(self.result, indent=2, allow_nan=False))
                if not settled:
                    self.follower.terminal = True

    def close(self):
        self.cancel("scene_closed")
        self._stop_planner()
        self.planner.close()
