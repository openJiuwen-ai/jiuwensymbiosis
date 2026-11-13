"""Shared native-world lifecycle; each body owns its controller and actuator physics."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .mujoco_rendering import MujocoRenderer


class MujocoPolicyRuntime(ABC):
    """One model/data owner with a replaceable renderer and a fixed physical clock."""

    model: Any
    data: Any
    motion: Any
    cameras: dict
    pose: dict
    renderer: Any
    indoor: Any
    selection: dict | None
    base_body: str
    physics_dt: float

    def __init__(self, settings, directory):
        self.settings, self.directory = settings, Path(directory)
        self.renderer = self.indoor = self.selection = None
        self.bodies = []
        self.steps = self.cycle = self.physics_steps = 0
        self._accumulator = 0.0
        self._command = (0.0, 0.0, 0.0)
        self._command_until = 0.0

    @abstractmethod
    def _physics_step(self):
        raise NotImplementedError("A body must supply its native policy/actuator step")

    def step(self):
        if self.motion is not None:
            self.motion.tick(self.settings.dt, self.steps)
        self._accumulator += self.settings.dt
        while self._accumulator >= self.physics_dt - 1e-10:
            self._physics_step()
            self._accumulator -= self.physics_dt
        self.steps += 1

    def stop(self):
        self._command = (0.0, 0.0, 0.0)
        self._command_until = 0.0

    def observe_base(self):
        import mujoco as mj

        body = self.data.body(self.base_body)
        velocity = np.zeros(6)
        mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, body.id, velocity, 0)
        if not np.isfinite(np.concatenate((body.xpos, body.xquat, velocity))).all():
            raise RuntimeError("Non-finite native base observation")
        return {
            "position_m": body.xpos.tolist(),
            "quaternion_wxyz": body.xquat.tolist(),
            "body_yaw_rad": math.atan2(body.xmat[3], body.xmat[0]),
            "linear_velocity_m_s": velocity[3:].tolist(),
            "angular_velocity_rad_s": velocity[:3].tolist(),
            "frame": "world",
            "command_frame": "body_forward_left_up",
        }

    def frame_size(self, name):
        if name == "observer":
            return self.settings.width, self.settings.height
        if name not in self.cameras:
            raise ValueError(f"Native world has no camera: {name}")
        return self.cameras[name].resolution

    def build_renderer(self, settings):
        self.renderer, self.settings = MujocoRenderer(self), settings

    def close_renderer(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def update_camera(self, pose):
        self.pose = pose

    @staticmethod
    def selection_snapshot():
        return None

    @staticmethod
    def pick(u, v):
        raise ValueError("MuJoCo 场景暂不支持物体点选。")

    @staticmethod
    def query_truth(operation, params):
        raise ValueError("MuJoCo 场景尚不支持真值查询。")

    def close(self):
        self.close_renderer()


class MujocoCamera:
    name = "head_cam"
    resolution = (640, 480)
    mjcf_name: str
    link_name: str

    def __init__(self, runtime):
        self.runtime = runtime

    def metadata(self):
        width, height = self.resolution
        fovy = float(self.runtime.model.camera(self.mjcf_name).fovy[0])
        focal = height / (2 * math.tan(math.radians(fovy) / 2))
        return {
            "name": self.name,
            "link": self.link_name,
            "width": width,
            "height": height,
            "K": [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
            "D": [0] * 5,
            "distortion_model": "pinhole",
            "optical_frame": "x_right_y_down_z_forward",
            "fov_y_deg": fovy,
            "source": f"MJCF {self.mjcf_name}; model FOV, not hardware calibration",
        }
