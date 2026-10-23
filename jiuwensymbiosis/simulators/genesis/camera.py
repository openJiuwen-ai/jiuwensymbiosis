"""Bounded Z-up camera controls shared by the UI bridge and renderer worker."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class PreviewSettings:
    """Preview quality and rate, independent from the physics timestep."""

    width: int = 1280
    height: int = 720
    fps: int = 60
    spp: int = 64
    dt: float = 1 / 120
    scene_id: str = "demo"
    robot_mode: str = "inspection"
    robot_id: str = ""
    render_mode: str = "reference"
    camera_name: str = "observer"

    def __post_init__(self) -> None:
        if (self.width, self.height) not in {(960, 540), (1280, 720), (1920, 1080)}:
            raise ValueError("预览支持 540p、720p、1080p。")
        if type(self.fps) is not int or not 1 <= self.fps <= 120:
            raise ValueError("帧率必须在 1–120 之间。")
        if type(self.spp) is not int or not 1 <= self.spp <= 64:
            raise ValueError("spp 必须在 1–64 之间。")
        if self.render_mode not in {"fast", "reference"}:
            raise ValueError("渲染模式必须为 fast 或 reference。")
        if not isinstance(self.camera_name, str) or not self.camera_name.isidentifier() or len(self.camera_name) > 64:
            raise ValueError("相机名称无效。")
        if not math.isfinite(self.dt) or not 0 < self.dt <= 0.1:
            raise ValueError("物理步长必须在 (0, 0.1] 之间。")
        if not isinstance(self.scene_id, str) or not self.scene_id or len(self.scene_id) > 1024:
            raise ValueError("场景 ID 无效。")
        if not isinstance(self.robot_id, str) or len(self.robot_id) > 1024:
            raise ValueError("机器人资产 ID 无效。")
        if self.robot_id and (self.scene_id == "demo" or self.robot_mode != "moz1_wheels"):
            raise ValueError("室内机器人需要 USD 场景和 MOZ1 底盘物理模式。")
        if self.robot_mode not in {"inspection", "moz1_wheels"}:
            raise ValueError("机器人模式无效。")


@dataclass(frozen=True)
class OrbitCamera:
    """Z-up pose supporting orbit and free flight, in world metres."""

    yaw: float = -0.95
    pitch: float = 0.48
    distance: float = 3.0
    target: tuple[float, float, float] = (0.0, 0.0, 0.45)
    revision: int = 0
    scene_scale: float = 1.0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def command(
        self,
        action: str,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        delta: float = 0.0,
        forward: float = 0.0,
        strafe: float = 0.0,
        lift: float = 0.0,
    ) -> OrbitCamera:
        """Pointer deltas are viewport fractions; flight distances are scene-scale fractions.

        A fly command applies look and translation together, so simultaneous mouse
        and keyboard input cannot overwrite one another in the control mailbox.
        """
        for value in (dx, dy, delta, forward, strafe, lift):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("相机增量必须是数值。")
            if not math.isfinite(value) or abs(value) > 1:
                raise ValueError("相机增量必须是 [-1, 1] 内的有限数值。")
        next_revision = self.revision + 1
        if action == "reset":
            return OrbitCamera(scene_scale=self.scene_scale, center=self.center, revision=next_revision)
        if action == "orbit":
            return replace(
                self,
                yaw=(self.yaw - dx * math.pi) % (2 * math.pi),
                pitch=max(-1.48, min(1.48, self.pitch + dy * math.pi)),
                revision=next_revision,
            )
        if action == "fly":
            # Turning the head preserves the eye position, unlike orbiting.
            eye = self.snapshot()["pos"]
            turned = replace(
                self,
                yaw=(self.yaw - dx * math.pi) % (2 * math.pi),
                pitch=max(-1.48, min(1.48, self.pitch + dy * math.pi)),
                revision=next_revision,
            )
            offset = _offset(turned.yaw, turned.pitch)
            target = tuple(eye[i] - self.distance * offset[i] for i in range(3))
            turned = replace(turned, target=target)
            right = (-math.sin(turned.yaw), math.cos(turned.yaw), 0.0)
            return _translate(
                turned,
                tuple(
                    self.scene_scale * (-forward * offset[i] + strafe * right[i] + (lift if i == 2 else 0))
                    for i in range(3)
                ),
            )
        if action == "zoom":
            return replace(
                self,
                distance=max(0.02, min(max(8.0 * self.scene_scale, self.distance), self.distance * math.exp(delta))),
                revision=next_revision,
            )
        if action == "pan":
            right = (-math.sin(self.yaw), math.cos(self.yaw), 0.0)
            up = (
                -math.cos(self.yaw) * math.sin(self.pitch),
                -math.sin(self.yaw) * math.sin(self.pitch),
                math.cos(self.pitch),
            )
            return _translate(
                replace(self, revision=next_revision),
                tuple(self.distance * (-dx * right[i] + dy * up[i]) for i in range(3)),
            )
        raise ValueError("不支持的相机操作。")

    def focus(self, bounds, *, fov_y: float, aspect: float) -> OrbitCamera:
        """Frame all corners from the current viewing side, fitting both axes."""
        from itertools import product

        if len(bounds) != 2 or any(len(row) != 3 for row in bounds):
            raise ValueError("聚焦对象的包围盒形状无效。")
        if any(not math.isfinite(v) for row in bounds for v in row):
            raise ValueError("聚焦对象的包围盒必须为有限数值。")
        if any(bounds[0][i] > bounds[1][i] for i in range(3)):
            raise ValueError("聚焦对象的包围盒上下界无效。")
        if not 0 < fov_y < 179 or not math.isfinite(aspect) or aspect <= 0:
            raise ValueError("聚焦对象的视场角或宽高比无效。")
        center = tuple((bounds[0][i] + bounds[1][i]) / 2 for i in range(3))
        offset = [self.snapshot()["pos"][i] - center[i] for i in range(3)]
        length = math.sqrt(sum(v * v for v in offset))
        yaw, pitch = self.yaw, self.pitch
        if length > 0.001:
            yaw = math.atan2(offset[1], offset[0])
            pitch = max(-1.48, min(1.48, math.asin(max(-1, min(1, offset[2] / length)))))
        outward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
        right = (-math.sin(yaw), math.cos(yaw), 0.0)
        up = (
            -math.cos(yaw) * math.sin(pitch),
            -math.sin(yaw) * math.sin(pitch),
            math.cos(pitch),
        )
        tan_y = math.tan(math.radians(fov_y) / 2)
        distance = 0.05
        for corner in product(*zip(bounds[0], bounds[1], strict=True)):
            relative = [corner[i] - center[i] for i in range(3)]
            depth = sum(relative[i] * outward[i] for i in range(3))
            x = abs(sum(relative[i] * right[i] for i in range(3)))
            y = abs(sum(relative[i] * up[i] for i in range(3)))
            distance = max(distance, depth + 1.15 * max(x / (tan_y * aspect), y / tan_y) + 0.02)
        return replace(self, target=center, yaw=yaw, pitch=pitch, distance=distance, revision=self.revision + 1)

    def snapshot(self) -> dict:
        offset = _offset(self.yaw, self.pitch)
        return {
            **asdict(self),
            "pos": [self.target[i] + self.distance * offset[i] for i in range(3)],
            "lookat": list(self.target),
            "up": [0.0, 0.0, 1.0],
        }


def _translate(camera: OrbitCamera, movement: tuple[float, ...]) -> OrbitCamera:
    """Bound the eye, not the look-at point, without snapping when changing modes."""
    if not any(movement):
        return camera
    eye = camera.snapshot()["pos"]
    bound = 20 * camera.scene_scale
    target = tuple(
        camera.target[i]
        + max(
            min(0.0, camera.center[i] - bound - eye[i]),
            min(max(0.0, camera.center[i] + bound - eye[i]), movement[i]),
        )
        for i in range(3)
    )
    return replace(camera, target=target)


def _offset(yaw: float, pitch: float) -> tuple[float, float, float]:
    return (
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    )
