"""MOZ1 wheel-driven base with an articulated upper-body position hold.

Wheel velocity and upper-body PD targets address disjoint motors. Root pose is
never integrated or teleported by the controller. The URDF's diagonal omni-wheel
layout is read from its axes instead of assuming parallel-axis mecanum geometry.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

WHEEL_NAMES = tuple(f"wheel{i:02}_joint" for i in range(1, 5))


def _vector(element: ET.Element | None, key: str, default: str) -> np.ndarray:
    values = np.fromstring(element.get(key, default) if element is not None else default, sep=" ")
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid URDF {key}")
    return values


def _joint_link(joint: ET.Element, tag: str) -> str:
    element = joint.find(tag)
    link = None if element is None else element.get("link")
    if not link:
        raise ValueError(f"Joint {joint.get('name')} requires a {tag} link")
    return link


@dataclass(frozen=True)
class OmniWheelLayout:
    """Wheel rows map URDF base_link (vx m/s, vy m/s, CCW rad/s) to rad/s.

    These geometric axes do not imply the robot's semantic forward direction.
    """

    rows: tuple[tuple[float, float, float], ...]
    radius_m: float
    roller_names: tuple[str, ...]

    @classmethod
    def from_urdf(cls, path: Path) -> OmniWheelLayout:
        root = ET.parse(path).getroot()
        joints = {j.attrib["name"]: j for j in root.findall("joint")}
        links = {link.attrib["name"]: link for link in root.findall("link")}
        wheel_rows: list[tuple[float, float, float]] = []
        radii: list[float] = []
        rollers: list[str] = []
        for name in WHEEL_NAMES:
            joint = joints.get(name)
            if joint is None or joint.get("type") != "continuous":
                raise ValueError("MOZ1 wheel physics requires wheel01_joint through wheel04_joint")
            if _joint_link(joint, "parent") != "base_link":
                raise ValueError("Wheel axes must be expressed relative to base_link")
            origin = joint.find("origin")
            position = _vector(origin, "xyz", "0 0 0")
            axis = Rotation.from_euler("xyz", _vector(origin, "rpy", "0 0 0")).apply(
                _vector(joint.find("axis"), "xyz", "1 0 0")
            )
            if abs(axis[2]) > 1e-5 or not np.isclose(np.linalg.norm(axis), 1):
                raise ValueError("Wheel axes must be horizontal unit vectors")
            direction = np.cross(axis, (0, 0, 1))
            wheel_link = _joint_link(joint, "child")
            passive = [j for j in joints.values() if _joint_link(j, "parent") == wheel_link]
            if len(passive) != 10 or any(j.get("type") != "continuous" for j in passive):
                raise ValueError("Expected ten independently rotating rollers per MOZ1 wheel")
            wheel_radii = []
            for roller in passive:
                child_name = _joint_link(roller, "child")
                if child_name not in links:
                    raise ValueError(f"Missing roller link: {child_name}")
                link = links[child_name]
                spheres = link.findall("collision/geometry/sphere")
                if len(spheres) != 1:
                    raise ValueError("Expected the exported MOZ1 spherical roller collision proxy")
                offset = _vector(roller.find("origin"), "xyz", "0 0 0")
                radius = float(spheres[0].get("radius", "nan")) + float(np.linalg.norm(offset[1:]))
                if not math.isfinite(radius) or radius <= 0:
                    raise ValueError("Invalid roller radius")
                wheel_radii.append(radius)
                rollers.append(roller.attrib["name"])
            if np.ptp(wheel_radii) > 1e-5:
                raise ValueError("Inconsistent roller support radii")
            radius = float(np.mean(wheel_radii))
            radii.append(radius)
            wheel_rows.append(
                (
                    float(direction[0]) / radius,
                    float(direction[1]) / radius,
                    float(-direction[0] * position[1] + direction[1] * position[0]) / radius,
                )
            )
        if np.ptp(radii) > 1e-5 or np.linalg.matrix_rank(wheel_rows) != 3:
            raise ValueError("Unsupported omni-wheel geometry")
        return cls(tuple(wheel_rows), float(np.mean(radii)), tuple(rollers))

    def wheel_rates(self, vx: float, vy: float, wz: float) -> np.ndarray:
        """Validate conservative experiment limits, not hardware operating limits."""
        validate_twist(vx, vy, wz)
        return np.asarray(self.rows) @ (vx, vy, wz)


def validate_twist(vx: float, vy: float, wz: float) -> None:
    for value in (vx, vy, wz):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Base velocity must contain finite numbers")
    if math.hypot(vx, vy) > 0.3 or abs(wz) > 0.6:
        raise ValueError("Experimental base limits: translation <= 0.3 m/s, yaw <= 0.6 rad/s")


def mobile_home_targets(source: Path, layout: OmniWheelLayout) -> dict[str, float]:
    """Keep articulated joints and mimic constraints; return the upper-body home.

    The source is only read. Zero exported motor limits
    are overridden explicitly by the simulation-only posture controller.
    """
    tree = ET.parse(source)
    root = tree.getroot()
    active = set(WHEEL_NAMES) | set(layout.roller_names)
    positions: dict[str, float] = {}
    pending = list(root.findall("joint"))
    while pending:
        progressed = False
        for joint in pending[:]:
            mimic = joint.find("mimic")
            if mimic is not None and mimic.get("joint") not in positions:
                continue
            value = (
                0.0
                if mimic is None
                else (
                    float(mimic.get("multiplier", "1")) * positions[mimic.attrib["joint"]]
                    + float(mimic.get("offset", "0"))
                )
            )
            positions[joint.attrib["name"]] = value
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("Unresolved mimic references")
    return {
        j.attrib["name"]: positions[j.attrib["name"]]
        for j in root.findall("joint")
        if j.get("type") != "fixed" and j.attrib["name"] not in active
    }


class UpperBodyHold:
    """Finite-force simulation PD hold, with passive URDF mimic followers.

    Gains and force caps are simulation tuning, not MOZ1 hardware specifications.
    The target is fixed at construction, never recaptured from a sagging posture.
    Genesis retains these motor commands while the wheel controller runs.
    """

    def __init__(self, entity: Any, urdf: Path, targets: dict[str, float]) -> None:
        self.entity = entity
        self.names = list(targets)
        self.target = np.asarray(list(targets.values()))
        self.dofs: list[int] = []
        self.motor_dofs: list[int] = []
        joints = {j.attrib["name"]: j for j in ET.parse(urdf).findall("joint")}
        self.driven = [i for i, name in enumerate(self.names) if joints[name].find("mimic") is None]
        self.kp, self.kv, self.effort = [], [], []
        for i in self.driven:
            name = self.names[i]
            if name.startswith("tn__LegWaist"):
                gains = (30000.0, 1500.0, 500.0)
            elif name.startswith(("tn__LeftArm", "tn__RightArm")):
                gains = (8000.0, 200.0, 120.0)
            elif name in {"left_hand_narrow1_joint", "right_hand_narrow1_joint"}:
                gains = (120.0, 5.0, 16.5)
            else:
                raise ValueError(f"No MOZ1 simulation posture gains for {name}")
            if joints[name].get("type") != "revolute":
                raise ValueError(f"Expected revolute MOZ1 posture joint: {name}")
            self.kp.append(gains[0])
            self.kv.append(gains[1])
            self.effort.append(gains[2])
        for name, value in targets.items():
            limit = joints[name].find("limit")
            if not math.isfinite(value) or (
                limit is not None and not float(limit.get("lower", "-inf")) <= value <= float(limit.get("upper", "inf"))
            ):
                raise ValueError(f"MOZ1 initial posture outside joint limits: {name}")

    def build(self) -> None:
        self.dofs = [self.entity.get_joint(name).dofs_idx_local[0] for name in self.names]
        self.motor_dofs = [self.dofs[i] for i in self.driven]
        # Initialization only, before the first physical step. No pose writes in
        # motion/stop/reset: subsequent holding is entirely through motor forces.
        self.entity.set_dofs_position(self.target, self.dofs)
        self.entity.set_dofs_kp(np.zeros(len(self.dofs)), self.dofs)
        self.entity.set_dofs_kv(np.zeros(len(self.dofs)), self.dofs)
        self.entity.set_dofs_force_range(np.zeros(len(self.dofs)), np.zeros(len(self.dofs)), self.dofs)
        self.entity.set_dofs_kp(self.kp, self.motor_dofs)
        self.entity.set_dofs_kv(self.kv, self.motor_dofs)
        self.entity.set_dofs_force_range(-np.asarray(self.effort), self.effort, self.motor_dofs)
        self.hold()

    def hold(self) -> None:
        """Restore the original motor target after a physical checkpoint reset."""
        self.entity.control_dofs_position(self.target[self.driven], self.motor_dofs)

    def snapshot(self) -> dict:
        position = self.entity.get_dofs_position(self.dofs)[0].cpu().numpy()
        velocity = self.entity.get_dofs_velocity(self.dofs)[0].cpu().numpy()
        error = position - self.target
        return {
            "mode": "position_hold",
            "parameter_source": "simulation_tuning_not_hardware_limits",
            "joint_names": self.names,
            "target_rad": self.target.tolist(),
            "position_rad": position.tolist(),
            "error_rad": error.tolist(),
            "velocity_rad_s": velocity.tolist(),
            "max_abs_error_rad": float(np.max(np.abs(error), initial=0.0)),
            "max_abs_velocity_rad_s": float(np.max(np.abs(velocity), initial=0.0)),
            "motor_joint_names": [self.names[i] for i in self.driven],
            "kp_nm_rad": self.kp,
            "kv_nm_s_rad": self.kv,
            "force_limit_nm": self.effort,
            "mimic_followers": len(self.names) - len(self.driven),
        }


class Moz1Base:
    """Scene-owned wheel controller. Caller owns Genesis init/build/step/destroy."""

    def __init__(
        self,
        scene: Any,
        source: Path,
        directory: Path,
        *,
        name: str = "MOZ1",
        render_urdf: Path | None = None,
        spawn: tuple[float, float, float] = (0.0, 0.0, 0.0),
        collision_policy: dict | None = None,
        model_directory: Path,
    ) -> None:
        import genesis as gs

        self.layout = OmniWheelLayout.from_urdf(source)
        targets = mobile_home_targets(source, self.layout)
        model_module = import_module(f"{__package__}.robot_model" if __package__ else "robot_model")
        self.model_manifest = model_module.prepare_robot_model(source, model_directory, policy=collision_policy)
        self.path = Path(self.model_manifest["urdf"])
        self.source, self.name, self.render_urdf = source, name, render_urdf
        self.entity = scene.add_entity(
            morph=gs.morphs.URDF(
                file=str(self.path),
                **model_module.admission_morph_options(fixed=False, decimate=True),
                file_meshes_are_zup=True,
                pos=(spawn[0], spawn[1], spawn[2] + self.layout.radius_m + 0.005),
            )
        )
        self.twist = (0.0, 0.0, 0.0)
        self.remaining_s = 0.0
        self.wheels: list[int] | range = []
        self._wheel_hold: Any = None
        self.model_audit = None
        self._velocity_target: tuple[float, float, float] | None = None
        self.upper_body = UpperBodyHold(self.entity, self.path, targets)

    def build(self) -> None:
        audit_module = import_module(f"{__package__}.collision_audit" if __package__ else "collision_audit")
        self.model_audit = audit_module.audit_loaded_model(
            self.entity, self.model_manifest, self.path.parent / "loaded-model.json"
        )
        self.wheels = [self.entity.get_joint(name).dofs_idx_local[0] for name in WHEEL_NAMES]
        stride = self.wheels[1] - self.wheels[0]
        if stride > 0 and all(b - a == stride for a, b in zip(self.wheels, self.wheels[1:], strict=False)):
            # Preserve the actual joint indices as a strided slice. Genesis can
            # then address motors without building a CUDA index tensor per tick.
            self.wheels = range(self.wheels[0], self.wheels[-1] + 1, stride)
        self.entity.set_dofs_kp([2000.0] * 4, self.wheels)
        self.entity.set_dofs_kv([50.0] * 4, self.wheels)
        self.entity.set_dofs_force_range([-80.0] * 4, [80.0] * 4, self.wheels)
        self.upper_body.build()
        self.stop()

    @property
    def bounds(self) -> list:
        return list(self.entity.get_AABB()[0].cpu().tolist())

    def command(self, vx: float, vy: float, wz: float, duration_s: float = 0.25) -> None:
        """Command body forward, body left (m/s), and CCW yaw (rad/s)."""
        validate_twist(vx, vy, wz)
        if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
            raise ValueError("Base command duration must be a number")
        if not math.isfinite(duration_s) or not 0 < duration_s <= 2:
            raise ValueError("Base command duration must be in (0, 2] seconds")
        if vx == vy == wz == 0:
            self.stop()
            return
        self._wheel_hold = None
        self.twist = (vx, vy, wz)
        self.remaining_s = duration_s

    def stop(self) -> None:
        """Brake and hold the wheel angles captured on entry, without re-anchoring."""
        self.twist = (0.0, 0.0, 0.0)
        self.remaining_s = 0.0
        if self._wheel_hold is None:
            hold = self.entity.get_dofs_position(self.wheels)[0].clone()
            # Genesis retains motor targets between steps. Re-uploading the
            # same target on every idle tick creates needless host/device work.
            self.entity.control_dofs_position(hold, self.wheels)
            self._wheel_hold = hold
            self._velocity_target = None

    def before_step(self, dt: float) -> None:
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("Physics timestep must be in (0, 0.1] seconds")
        if self.remaining_s <= 1e-9:
            self.stop()
        else:
            # MOZ1 faces base_link -Y; its left shoulder is on base_link +X.
            # Matches RoboSim MOZ1 relative_forward_offset_deg=-90.
            # The wheel matrix uses URDF geometry, independent of USD placement.
            forward, left, ccw = self.twist
            if self.twist != getattr(self, "_velocity_target", None):
                self.entity.control_dofs_velocity(self.layout.wheel_rates(left, -forward, ccw), self.wheels)
                self._velocity_target = self.twist
            self.remaining_s = max(0.0, self.remaining_s - dt)

    def reset_controller(self) -> None:
        """Restore wheel parking and upper-body targets after a checkpoint."""
        self._wheel_hold = self._velocity_target = None
        self.stop()
        self.upper_body.hold()

    def collision_settings(self) -> dict[str, bool]:
        """Read actual solver flags, including Genesis' default pair exclusions."""
        solver = self.entity.solver
        # Pinned Genesis 1.4.4 exposes no public accessor for these solver flags.
        # Read the live flags, not a copy of the requested scene options.
        # pylint: disable=protected-access
        return {
            "enabled": bool(solver._enable_collision),  # noqa: SLF001 -- pinned SDK readback
            "self_collision": bool(solver._enable_self_collision),  # noqa: SLF001 -- pinned SDK readback
            "adjacent_collision": bool(solver._enable_adjacent_collision),  # noqa: SLF001 -- pinned SDK readback
            "neutral_collision": bool(solver._enable_neutral_collision),  # noqa: SLF001 -- pinned SDK readback
        }
        # pylint: enable=protected-access

    def snapshot(self) -> dict:
        collisions = self.collision_settings()
        return {
            "name": self.name,
            "source_format": "URDF",
            "urdf": str(self.source),
            "fixed_base": False,
            "control_mode": "wheel_velocity_physics",
            "collision": collisions["enabled"],
            "self_collision": collisions["self_collision"],
            "collision_settings": collisions,
            "upper_body": "position_hold",
            "upper_body_state": self.upper_body.snapshot(),
            "planning_connected": False,
            "wheel_radius_m": self.layout.radius_m,
            "base_position_m": self.entity.get_pos()[0].cpu().tolist(),
            "base_quaternion_wxyz": self.entity.get_quat()[0].cpu().tolist(),
            "wheel_names": list(WHEEL_NAMES),
            "wheel_velocity_rad_s": self.entity.get_dofs_velocity(self.wheels)[0].cpu().tolist(),
            "parking_active": self._wheel_hold is not None,
            "parking_wheel_target_rad": self._wheel_hold.cpu().tolist() if self._wheel_hold is not None else None,
            "command_body_twist": list(self.twist),
            "command_frame": "body_forward_left_up",
            "body_forward_axis_in_base_link": [0.0, -1.0, 0.0],
            "command_remaining_s": self.remaining_s,
        }
