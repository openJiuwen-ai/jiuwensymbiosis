"""MOZ1 wheel-driven base experiment, with a passive upper-body posture.

Only wheel motors receive commands. Root pose is never integrated or teleported
by the controller. The URDF's diagonal omni-wheel layout is deliberately read
from its axes, rather than assuming the usual parallel-axis mecanum matrix.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
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


def prepare_mobile_urdf(source: Path, target: Path, layout: OmniWheelLayout) -> None:
    """Freeze non-wheel joints at the inspection posture in a disposable copy.

    Exported upper-body effort limits are zero. Locking those joints makes this
    explicitly a base experiment without inventing arm/waist motor limits.
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
            if joint.attrib["name"] not in active and joint.get("type") != "fixed":
                origin = joint.find("origin")
                if origin is None:
                    origin = ET.SubElement(joint, "origin")
                rotation = Rotation.from_euler("xyz", _vector(origin, "rpy", "0 0 0"))
                axis = _vector(joint.find("axis"), "xyz", "1 0 0")
                if joint.get("type") == "prismatic":
                    xyz = _vector(origin, "xyz", "0 0 0") + rotation.apply(axis * value)
                    origin.set("xyz", " ".join(map(str, xyz)))
                else:
                    rpy = (rotation * Rotation.from_rotvec(axis * value)).as_euler("xyz")
                    origin.set("rpy", " ".join(map(str, rpy)))
                joint.set("type", "fixed")
                for child in list(joint):
                    if child.tag in {"axis", "limit", "dynamics", "mimic"}:
                        joint.remove(child)
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("Unresolved mimic references")
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str((source.parent / mesh.attrib["filename"]).resolve()))
    ET.indent(tree, space="  ")
    tree.write(target, encoding="utf-8", xml_declaration=True)


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
    ) -> None:
        import genesis as gs

        self.layout = OmniWheelLayout.from_urdf(source)
        self.path = directory / "moz1-mobile.urdf"
        prepare_mobile_urdf(source, self.path, self.layout)
        self.source, self.name, self.render_urdf = source, name, render_urdf
        self.entity = scene.add_entity(
            morph=gs.morphs.URDF(
                file=str(self.path),
                fixed=False,
                merge_fixed_links=False,
                collision=True,
                convexify=True,
                decimate=True,
                file_meshes_are_zup=True,
                pos=(spawn[0], spawn[1], spawn[2] + self.layout.radius_m + 0.005),
            )
        )
        self.twist = (0.0, 0.0, 0.0)
        self.remaining_s = 0.0
        self.wheels: list[int] | range = []
        self._wheel_hold: Any = None
        self._velocity_target: tuple[float, float, float] | None = None

    def build(self) -> None:
        self.wheels = [self.entity.get_joint(name).dofs_idx_local[0] for name in WHEEL_NAMES]
        stride = self.wheels[1] - self.wheels[0]
        if stride > 0 and all(b - a == stride for a, b in zip(self.wheels, self.wheels[1:], strict=False)):
            # Preserve the actual joint indices as a strided slice. Genesis can
            # then address motors without building a CUDA index tensor per tick.
            self.wheels = range(self.wheels[0], self.wheels[-1] + 1, stride)
        self.entity.set_dofs_kp([2000.0] * 4, self.wheels)
        self.entity.set_dofs_kv([50.0] * 4, self.wheels)
        self.entity.set_dofs_force_range([-80.0] * 4, [80.0] * 4, self.wheels)
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

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "source_format": "URDF",
            "urdf": str(self.source),
            "fixed_base": False,
            "control_mode": "wheel_velocity_physics",
            "collision": True,
            "self_collision": False,
            "upper_body": "locked_inspection_posture",
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
