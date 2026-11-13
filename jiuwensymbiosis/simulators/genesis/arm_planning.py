"""Body-independent cuRobo planning, bound to frozen non-group joints and world state."""

from __future__ import annotations

import copy
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def sibling(name):
    return import_module(f"{__package__}.{name}" if __package__ else name)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def transform(position, quaternion):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
    result[:3, 3] = position
    return result


def pose_error(expected, actual):
    return {
        "position_m": float(np.linalg.norm(expected[:3, 3] - actual[:3, 3])),
        "rotation_rad": float(Rotation.from_matrix(expected[:3, :3] @ actual[:3, :3].T).magnitude()),
    }


def required_element(parent: ET.Element, tag: str) -> ET.Element:
    value = parent.find(tag)
    if value is None:
        raise ValueError(f"missing required model element: {tag}")
    return value


def joint_transform(joint, value):
    origin = joint.find("origin")
    attrs = {} if origin is None else origin.attrib
    result = transform(
        np.fromstring(attrs.get("xyz", "0 0 0"), sep=" "),
        Rotation.from_euler("xyz", np.fromstring(attrs.get("rpy", "0 0 0"), sep=" ")).as_quat(scalar_first=True),
    )
    if joint.get("type") == "fixed":
        return result
    axis = joint.find("axis")
    axis = np.fromstring("1 0 0" if axis is None else axis.get("xyz", "1 0 0"), sep=" ")
    axis /= np.linalg.norm(axis)
    if joint.get("type") in {"revolute", "continuous"}:
        result[:3, :3] @= Rotation.from_rotvec(axis * value).as_matrix()
    elif joint.get("type") == "prismatic":
        result[:3, 3] += result[:3, :3] @ (axis * value)
    return result


def forward_kinematics(root, positions):
    joints = list(root.findall("joint"))
    children = {required_element(j, "child").get("link") for j in joints}
    roots = {link.get("name") for link in root.findall("link")} - children
    if len(roots) != 1:
        raise ValueError("expected one robot root")
    frames = {roots.pop(): np.eye(4)}
    pending = joints.copy()
    q = dict(positions)
    while pending:
        progress = False
        for j in pending[:]:
            parent = required_element(j, "parent").get("link")
            mimic = j.find("mimic")
            if parent not in frames or (mimic is not None and mimic.get("joint") not in q):
                continue
            value = (
                q.get(j.get("name"), 0.0)
                if mimic is None
                else q[mimic.get("joint")] * float(mimic.get("multiplier", "1")) + float(mimic.get("offset", "0"))
            )
            q[j.get("name")] = value
            frames[required_element(j, "child").get("link")] = frames[parent] @ joint_transform(j, value)
            pending.remove(j)
            progress = True
        if not progress:
            raise ValueError("unresolved model or mimic joints")
    return frames


class CuroboPlanner:
    def __init__(self, manifest: dict, profile: dict, directory: Path):
        sibling("robot_model").verify_model_files(manifest)
        self.model, self.profile = copy.deepcopy(manifest), copy.deepcopy(profile)
        self.root = ET.parse(manifest["urdf"]).getroot()
        self.joints = {j.get("name"): j for j in self.root.findall("joint")}
        links = {link.get("name") for link in self.root.findall("link")}
        if profile.get("schema_version") != 1 or profile["base_link"] not in links:
            raise ValueError("invalid planning profile")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        for group in profile["groups"].values():
            names = group["joints"]
            invalid_joints = not names or len(set(names)) != len(names) or set(names) - self.joints.keys()
            tools = group["tools"]
            invalid_tools = not tools or len(set(tools)) != len(tools) or set(tools) - links
            if invalid_joints or invalid_tools:
                raise ValueError("invalid planning group names")
            for name in names:
                if self.joints[name].find("mimic") is not None or self.joints[name].get("type") == "fixed":
                    raise ValueError("planning group requires independent movable joints")
                limits = profile["limits"][name]
                if any(
                    not np.isfinite(limits[k]) or limits[k] <= 0 for k in ("velocity", "acceleration", "jerk", "effort")
                ):
                    raise ValueError("invalid planning limits")
        ignores = profile.get("self_collision_ignore", {})
        if set(ignores) - links or any(set(values) - links for values in ignores.values()):
            raise ValueError("unknown self-collision ignore links")
        provenance = profile.get("collision_provenance")
        if provenance:
            source = Path(provenance["path"])
            if hashlib.sha256(source.read_bytes()).hexdigest() != provenance["sha256"]:
                raise ValueError("collision profile source changed")
        spheres = profile["collision_spheres"]
        if not spheres or set(spheres) - links:
            raise ValueError("unknown collision sphere links")
        for entries in spheres.values():
            for s in entries:
                if len(s["center"]) != 3 or not np.isfinite([*s["center"], s["radius"]]).all() or s["radius"] <= 0:
                    raise ValueError("invalid collision sphere")
        # Only verified centred support spheres can ignore orientation; never arms/arbitrary joints.
        ignored = profile.get("uncontrolled_support_joints", [])
        children_with_joints = {required_element(j, "parent").get("link") for j in self.joints.values()}
        all_group_names = {n for g in profile["groups"].values() for n in g["joints"]}
        self.uncontrolled = set(ignored)
        for name in ignored:
            joint = self.joints.get(name)
            if joint is None or name in all_group_names or joint.get("type") != "continuous":
                raise ValueError("invalid uncontrolled support joint")
            child = required_element(joint, "child").get("link")
            link = next(item for item in self.root.findall("link") if item.get("name") == child)
            collisions = link.findall("collision")
            if child not in profile.get("support_links", []) or child in children_with_joints or not collisions:
                raise ValueError("uncontrolled support must be a supported leaf sphere")
            for collision in collisions:
                origin = collision.find("origin")
                if collision.find("geometry/sphere") is None or (
                    origin is not None and np.linalg.norm(np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")) > 1e-9
                ):
                    raise ValueError("uncontrolled support collision must be a centred sphere")
        self.backend: Any = None
        self.key = None
        self.model_id = identity({"model": manifest["model_sha256"], "profile": profile})

    def _prepare(self, group_name, state, world):
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        group = self.profile["groups"][group_name]
        names = group["joints"]
        static = {n: state[n] for n, j in self.joints.items() if j.get("type") != "fixed" and n not in names}
        key = identity({"group": group_name, "static": static, "world": world})
        if key == self.key:
            return group
        self.close()
        root = copy.deepcopy(self.root)
        for j in root.findall("joint"):
            name = j.get("name")
            if name not in names and j.get("type") != "fixed":
                pose = joint_transform(j, static[name])
                origin = j.find("origin")
                if origin is None:
                    origin = ET.SubElement(j, "origin")
                origin.set("xyz", " ".join(map(str, pose[:3, 3])))
                origin.set("rpy", " ".join(map(str, Rotation.from_matrix(pose[:3, :3]).as_euler("xyz"))))
                j.set("type", "fixed")
                for child in list(j):
                    if child.tag in {"axis", "limit", "mimic", "dynamics"}:
                        j.remove(child)
            elif name in names:
                limit = required_element(j, "limit")
                limit.set("velocity", str(self.profile["limits"][name]["velocity"]))
                limit.set("effort", str(self.profile["limits"][name]["effort"]))
        path = self.directory / f"planning-{key}.urdf"
        ET.ElementTree(root).write(path)
        kin = {
            "format_version": 2.0,
            "urdf_path": str(path.resolve()),
            "base_link": self.profile["base_link"],
            "tool_frames": group["tools"],
            "collision_link_names": list(self.profile["collision_spheres"]),
            "collision_spheres": self.profile["collision_spheres"],
            "collision_sphere_buffer": 0.0,
            "self_collision_ignore": self.profile.get("self_collision_ignore", {}),
            "self_collision_buffer": dict.fromkeys(self.profile["collision_spheres"], 0.0),
            "cspace": {
                "joint_names": names,
                "default_joint_position": [state[n] for n in names],
                "cspace_distance_weight": [1.0] * len(names),
                "null_space_weight": [1.0] * len(names),
                "max_acceleration": min(self.profile["limits"][n]["acceleration"] for n in names),
                "max_jerk": min(self.profile["limits"][n]["jerk"] for n in names),
            },
        }
        self.backend = MotionPlanner(
            MotionPlannerCfg.create(
                robot={"robot_cfg": {"kinematics": kin}},
                scene_model=world,
                self_collision_check=True,
                collision_cache={"cuboid": max(8, len(world.get("cuboid", {})))},
                position_tolerance=0.002,
                orientation_tolerance=0.03,
                interpolation_dt=0.025,
            )
        )
        if self.backend.joint_names != names:
            self.close()
            raise ValueError("planner joint mapping mismatch")
        self.backend.warmup(enable_graph=True)
        self.key = key
        return group

    @staticmethod
    def _state(q, names):
        import torch
        from curobo.types import JointState

        return JointState.from_position(torch.tensor([q], device="cuda", dtype=torch.float32), joint_names=names)

    def plan(
        self, group: str, state: dict, world: dict, binding: str, *, targets=None, poses=None, speed=1.0, timeout_s=30.0
    ):
        """Targets use radians/metres; tool poses use base-link metres and wxyz."""
        if isinstance(speed, bool) or not np.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout must be finite and positive")
        if group not in self.profile["groups"] or (targets is None) == (poses is None):
            raise ValueError("one known group and one goal type required")
        movable = {n for n, j in self.joints.items() if j.get("type") != "fixed"}
        if movable - set(state) or not np.isfinite(list(state.values())).all():
            raise ValueError("complete finite measured state required")
        if targets is not None:
            expected = set(self.profile["groups"][group]["joints"])
            if not isinstance(targets, dict) or not targets or set(targets) - expected:
                raise ValueError("unknown/empty joint targets")
            for name in expected:
                value = targets.get(name, state[name])
                limit = required_element(self.joints[name], "limit")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("target outside joint limits")
                if not np.isfinite(value) or not float(limit.get("lower", "-inf")) <= value <= float(
                    limit.get("upper", "inf")
                ):
                    raise ValueError("target outside joint limits")
        sibling("robot_model").verify_model_files(self.model)
        settings = self._prepare(group, state, world)
        names = settings["joints"]
        start = self._state([state[n] for n in names], names)
        fk = self.backend.compute_kinematics(start)
        reference = forward_kinematics(self.root, state)
        for tool in settings["tools"]:
            expected = np.linalg.inv(reference[self.profile["base_link"]]) @ reference[tool]
            actual = fk.tool_poses.get_link_pose(tool)
            position = actual.position.detach().cpu().numpy().reshape(3)
            quaternion = actual.quaternion.detach().cpu().numpy().reshape(4)
            error = pose_error(expected, transform(position, quaternion))
            if error["position_m"] > 1e-4 or error["rotation_rad"] > 1e-3:
                raise ValueError("planner FK differs from independent model")
        begin = time.monotonic()
        if targets is not None:
            q = [targets.get(n, state[n]) for n in names]
            result = self.backend.plan_cspace(self._state(q, names), start, max_attempts=3)
        else:
            import torch
            from curobo.types import GoalToolPose

            if set(poses) != set(settings["tools"]):
                raise ValueError("provide each group's tool target")
            values = np.asarray([poses[t] for t in settings["tools"]], dtype=float)
            if (
                values.shape != (len(settings["tools"]), 7)
                or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values[:, 3:], axis=1), 1, atol=1e-5)
            ):
                raise ValueError("invalid tool poses")
            goal = GoalToolPose(
                tool_frames=self.backend.tool_frames,
                position=torch.tensor(values[:, :3], device="cuda", dtype=torch.float32).reshape(1, 1, -1, 1, 3),
                quaternion=torch.tensor(values[:, 3:], device="cuda", dtype=torch.float32).reshape(1, 1, -1, 1, 4),
            )
            result = self.backend.plan_pose(goal, start, max_attempts=3)
        if time.monotonic() - begin > timeout_s:
            raise TimeoutError("planning deadline exceeded; result discarded")
        if result is None or not bool(result.success.all().item()):
            raise RuntimeError("planning_failed")
        q = result.get_interpolated_plan().position.detach().cpu().numpy().reshape(-1, len(names))
        if targets is not None:
            held = [i for i, n in enumerate(names) if n not in targets]
            if held and np.max(np.abs(q[:, held] - np.array([state[names[i]] for i in held]))) > 1e-5:
                raise ValueError("planner moved joints omitted from targets")
        # C4 path, zero endpoint velocity/acceleration; retime using analytic extrema including boundaries.
        trajectory_type = sibling("arm_trajectory").Trajectory
        limits = [
            np.array([self.profile["limits"][n][key] for n in names]) for key in ("velocity", "acceleration", "jerk")
        ]
        unit = trajectory_type(tuple(names), q, 1.0, binding, dict(state), *[np.full(len(names), 1e12)] * 3)
        dt = 0.025 / speed
        for order, (maximum, bound) in enumerate(zip(unit.derivative_maxima(), limits, strict=True), 1):
            dt = max(dt, float(np.max(maximum / bound)) ** (1 / order))
        trajectory = trajectory_type(
            tuple(names),
            q,
            dt * (1 + 1e-6),
            binding,
            dict(state),
            *limits,
            stationary_names=tuple(n for n in state if n not in self.uncontrolled),
        )
        for name, values in zip(names, trajectory.position_bounds(), strict=True):
            limit = required_element(self.joints[name], "limit")
            if (
                values.min() < float(limit.get("lower", "-inf")) - 1e-6
                or values.max() > float(limit.get("upper", "inf")) + 1e-6
            ):
                raise ValueError("smoothed trajectory exceeds joint bounds")
        return trajectory

    def close(self):
        if self.backend is not None:
            self.backend.destroy()
            self.backend = None
        self.key = None
