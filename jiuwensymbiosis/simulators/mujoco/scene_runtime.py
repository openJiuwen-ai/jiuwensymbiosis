"""Native MJCF inspection and a self-contained primitive physics demo."""

from __future__ import annotations

import uuid
from dataclasses import replace

from ..common.camera import OrbitCamera
from ..common.scene_catalog import resolve_scene
from .mujoco_runtime import MujocoCamera, MujocoPolicyRuntime

DEMO_XML = """<mujoco model="Symbiosis demo">
  <option timestep="0.002"/>
  <visual><global offwidth="1920" offheight="1080"/></visual>
  <worldbody>
    <light pos="0 -1 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 .1" rgba=".3 .34 .4 1"/>
    <camera name="scene_cam" pos="2 -2 1.5" xyaxes=".707 .707 0 -.35 .35 .87"/>
    <body pos="0 0 1"><freejoint/><geom type="box" size=".12 .12 .12" rgba=".8 .1 .1 1"/></body>
    <body pos=".4 0 1.4"><freejoint/><geom type="sphere" size=".12" rgba=".1 .7 .2 1"/></body>
    <body pos="-.4 0 .8"><freejoint/><geom type="capsule" size=".08 .15" rgba=".1 .3 .8 1"/></body>
  </worldbody>
</mujoco>"""


class RuntimeScene(MujocoPolicyRuntime):
    """Inspect an indexed MJCF without advertising robot actuation capabilities."""

    hidden_geom_groups = ()

    def __init__(self, settings, directory, revision=0, *, startup=None):
        import mujoco as mj

        super().__init__(settings, directory)
        self.robot = self.motion = None
        self.model = (
            mj.MjModel.from_xml_string(DEMO_XML)
            if settings.scene_id == "demo"
            else mj.MjModel.from_xml_path(str(resolve_scene(settings.scene_id, "mujoco").preview_path))
        )
        self.model.vis.global_.offwidth = 1920
        self.model.vis.global_.offheight = 1080
        self.data = mj.MjData(self.model)
        self.physics_dt = float(self.model.opt.timestep)
        mj.mj_forward(self.model, self.data)
        self._initial_state = self.data.qpos.copy()
        self.info = {
            "id": settings.scene_id,
            "instance_id": uuid.uuid4().hex,
            "mode": "mjcf_inspection",
            "engine": "MuJoCo",
            "collision": True,
            "reset_interval_steps": 0,
            "control_description": "MJCF 物理检视（无机器人控制）",
        }
        pose = OrbitCamera()
        self.pose = self.initial_pose = replace(pose, revision=revision).snapshot()
        self.cameras = {}
        for camera in range(self.model.ncam):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_CAMERA, camera)
            if name and name != "observer" and name.isidentifier():
                self.cameras[name] = ModelCamera(self, name)
        try:
            self.frame_size(settings.camera_name)
            self.build_renderer(settings)
        except BaseException:
            self.close()
            raise

    def _physics_step(self):
        import mujoco as mj

        mj.mj_step(self.model, self.data)
        self.physics_steps += 1

    def reset(self, *, phase, stopping):
        import mujoco as mj

        if stopping():
            raise RuntimeError("复位已取消。")
        previous = self.info["instance_id"]
        with phase("reset_restore"):
            mj.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self._initial_state
            mj.mj_forward(self.model, self.data)
            self.steps = self.physics_steps = 0
            self._accumulator = 0.0
            self.cycle += 1
            self.info.update(instance_id=uuid.uuid4().hex, reset_generation=self.cycle)
        return {
            "previous_instance_id": previous,
            "scene_instance_id": self.info["instance_id"],
            "reset_generation": self.cycle,
            "stop_confirmation": "no_controller",
        }


class ModelCamera(MujocoCamera):
    def __init__(self, runtime, name):
        super().__init__(runtime)
        self.name = self.mjcf_name = name
        self.link_name = runtime.model.body(int(runtime.model.camera(name).bodyid[0])).name
