"""Shared native MuJoCo camera renderer; physics stays owned by each runtime."""

from __future__ import annotations

import math


class MujocoRenderer:
    fov_y = 45

    def __init__(self, runtime):
        self.runtime = runtime
        self.renderers = {}
        self.render_counts = dict.fromkeys(("observer", *runtime.cameras), 0)
        self.last_camera = {}

    def read_rgb(self, name="observer"):
        import mujoco as mj

        width, height = self.runtime.frame_size(name)
        key = (width, height)
        renderer = self.renderers.pop(key, None)
        if renderer is None:
            if len(self.renderers) >= 2:
                self.renderers.pop(next(iter(self.renderers))).close()
            renderer = mj.Renderer(self.runtime.model, height=height, width=width)
        self.renderers[key] = renderer
        option = mj.MjvOption()
        for group in self.runtime.hidden_geom_groups:
            option.geomgroup[group] = 0  # Keep collision geometry physical but invisible.
        camera = mj.MjvCamera()
        if name == "observer":
            pose = self.runtime.pose
            camera.type = mj.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = pose["target"]
            camera.distance = pose["distance"]
            camera.azimuth = math.degrees(pose["yaw"])
            camera.elevation = -math.degrees(pose["pitch"])
            self.last_camera = pose.copy()
        else:
            camera.type = mj.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = self.runtime.model.camera(self.runtime.cameras[name].mjcf_name).id
            c = self.runtime.data.camera(self.runtime.cameras[name].mjcf_name)
            rotation = c.xmat.reshape(3, 3)
            self.last_camera = {
                **self.runtime.cameras[name].metadata(),
                "pos": c.xpos.tolist(),
                "lookat": (c.xpos - rotation[:, 2]).tolist(),
                "up": rotation[:, 1].tolist(),
            }
        renderer.update_scene(self.runtime.data, camera, scene_option=option)
        rgb = renderer.render().copy()
        self.render_counts[name] += 1
        return rgb

    @staticmethod
    def read_rgbd(name="head_cam", **kwargs):
        raise ValueError("MuJoCo 场景尚不支持 RGB-D 深度采样。")

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
