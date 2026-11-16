"""Worker-owned native MuJoCo world for the official Jumper hexapod."""

from __future__ import annotations

import math
import uuid
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np

from ..common.base_motion import MotionTuning, RelativeMotion
from ..common.camera import OrbitCamera
from ..common.scene_catalog import resolve_scene
from .jumper_motion import JumperMotion
from .jumper_policy import JumperPolicy
from .mujoco_runtime import MujocoCamera, MujocoPolicyRuntime


class JumperRuntime(MujocoPolicyRuntime):
    base_body = "base_link"
    hidden_geom_groups = (1,)

    def __init__(self, settings, directory, revision=0, *, startup=None):
        import mujoco as mj
        import yaml

        super().__init__(settings, directory)
        self.asset = resolve_scene(settings.scene_id)
        if self.asset.asset.preview_mode != "jumper" or settings.robot_id:
            raise ValueError("Jumper 模式需要索引中的官方 Jumper 模型和策略包。")
        try:
            self.policy = JumperPolicy(self.asset.paths["policy_bundle"])
            self.physics_dt = self.policy.dt
            root = ET.parse(self.asset.paths["source"]).getroot()
            root.find("compiler").set("meshdir", str(self.asset.paths["source"].parent))
            root.find("option").set("timestep", str(self.policy.dt))
            world = root.find("worldbody")
            # Match the upstream HYBRID_COLLISION, preserving original meshes.
            legs = ("LF", "RF", "LM", "RM", "LR", "RR")
            for geom in world.iter("geom"):
                name = geom.get("name", "")
                if not name.endswith("_meshcol"):
                    continue
                leg = name[:2]
                bit = 1 << (legs.index(leg) + 1) if leg in legs else 0
                foot = "palm_pad_b" in name or "foot_tip" in name
                for key, value in {
                    "contype": str(1 | bit),
                    "conaffinity": str(126 & ~bit if bit else 0),
                    "condim": "3" if foot else "1",
                    "priority": "1" if foot else "0",
                    "friction": "1.2 .01 .01" if foot else "0 0 0",
                    "solref": ".02 1",
                    "solimp": ".8 .9 .002 .5 2",
                }.items():
                    geom.set(key, value)
            actuator = ET.SubElement(root, "actuator")
            for name in self.policy.wire:
                ET.SubElement(actuator, "motor", name=name, joint=name)
            ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", diffuse=".8 .8 .8", castshadow="true")
            assets = root.find("asset")
            ET.SubElement(
                assets,
                "texture",
                name="floor_grid",
                type="2d",
                builtin="checker",
                width="256",
                height="256",
                rgb1=".28 .32 .36",
                rgb2=".4 .44 .48",
            )
            ET.SubElement(
                assets, "material", name="floor_grid", texture="floor_grid", texrepeat="12 12", texuniform="true"
            )
            ET.SubElement(
                world, "geom", name="floor", type="plane", size="3 3 .1", material="floor_grid", friction="1 .01 .01"
            )
            # Small, visible navigation landmarks; no artificial motion or target truth for the Agent.
            for name, pos, color in (
                ("red_box", ".8 .35 .1", ".8 .06 .06 1"),
                ("green_box", ".8 -.35 .1", ".06 .7 .1 1"),
                ("blue_box", "-.7 0 .1", ".08 .2 .8 1"),
            ):
                ET.SubElement(world, "geom", name=name, type="box", pos=pos, size=".1 .1 .1", rgba=color)
            self.model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
            self.model.vis.global_.offwidth = 1920
            self.model.vis.global_.offheight = 1080
            self.model.vis.map.znear = 0.01
            self.data = mj.MjData(self.model)
            self.qids = np.asarray([self.model.joint(n).qposadr[0] for n in self.policy.wire])
            self.vids = np.asarray([self.model.joint(n).dofadr[0] for n in self.policy.wire])
            self.data.qpos[:3] = self.policy.contract["base_init_pos"]
            self.data.qpos[3:7] = (1, 0, 0, 0)
            self.data.qpos[self.qids] = [self.policy.contract["default_joint_pos"][n] for n in self.policy.wire]
            mj.mj_forward(self.model, self.data)
            spec = yaml.safe_load(self.asset.paths["motor_config"].read_text())["motors"]["joint_servo"]["torque_speed"]
            if spec["model"] != "exp_decay_with_cutoff":
                raise ValueError("Unsupported Jumper servo model")
            self.plateau, self.continuous = (
                float(spec["plateau_torque_nm"]),
                float(spec["thermal_continuous_torque_nm"]),
            )
            self.corner, self.decay, self.cutoff = (
                float(spec[k + "_speed_rpm"]) * math.pi / 30 for k in ("corner", "decay", "cutoff")
            )
            self.thermal_tau = float(spec["peak_duration_s"]) / math.log(
                1 / (1 - (self.continuous / self.plateau) ** 2)
            )
            self.heat = np.zeros(len(self.policy.wire))
            self.targets = self.data.qpos[self.qids].copy()
            self.kp = self.kd = np.zeros(len(self.policy.wire))
            self.robot = self  # Same scene-owned robot interface as the Genesis runtime.
            self.bounds = [[-0.35, -0.35, 0], [0.35, 0.35, 0.22]]
            base_motion = RelativeMotion(
                self,
                MotionTuning(
                    position_tolerance=0.02,
                    yaw_tolerance=0.06,
                    position_resume=0.03,
                    yaw_resume=0.08,
                    linear_still=0.02,
                    angular_still=0.2,
                    settle_s=1.0,
                    min_linear=0.075,
                    min_angular=0.3,
                    max_angular=0.6,
                ),
            )
            self.motion = JumperMotion(self, base_motion)
            self.info = {
                "id": settings.scene_id,
                "robot_id": self.asset.asset.id,
                "instance_id": uuid.uuid4().hex,
                "mode": "robot_mobile",
                "collision": True,
                "engine": "MuJoCo",
                "renderer": "MuJoCo OpenGL",
                "control_description": "Jumper 官方步态策略 · 六足接触物理",
                "reset_interval_steps": None,
                "bounds_m": self.bounds,
                "robot_bounds_m": self.bounds,
                "policy": self.policy.summary,
            }
            # Settle under policy before exposing a controllable scene.
            for _ in range(round(3 / self.policy.dt)):
                self._physics_step()
            if self.data.qpos[2] < 0.07 or self.policy.fsm.mode != "locomotion":
                raise ValueError("Jumper failed to stand under the official policy")
            pose = OrbitCamera(yaw=-2.4, pitch=0.45, distance=1.4, target=(0, 0, 0.1), scene_scale=0.5)
            self.pose = self.initial_pose = replace(pose, revision=revision).snapshot()
            self.cameras = {"head_cam": JumperCamera(self)}
            self.frame_size(settings.camera_name)
            self.build_renderer(settings)
        except BaseException:
            self.close()
            raise

    def command(self, vx, vy, wz, duration):
        from ..genesis.omni_base import validate_twist

        validate_twist(vx, vy, wz)
        if any((vx, vy, wz)) and (self.policy.fsm.mode != "locomotion" or self.policy.operators):
            raise ValueError("请先张开夹爪，再由 Agent 移动以收回机械臂；当前策略不接受手动移动。")
        if not math.isfinite(duration) or not 0 < duration <= 1:
            raise ValueError("Invalid Jumper command duration")
        self._command = (vx, vy, wz)
        self._command_until = self.data.time + duration

    def _physics_step(self):
        import mujoco as mj

        if self.physics_steps % round(1 / (self.policy.hz * self.policy.dt)) == 0:
            command = self._command if self.data.time < self._command_until else (0.0, 0.0, 0.0)
            self.targets, self.kp, self.kd = self.policy.tick(
                self.data.qpos[self.qids],
                self.data.qvel[self.vids],
                self.data.actuator_force,
                self.data.qpos[3:7],
                self.data.sensor("imu_ang_vel").data,
                command,
                round(self.physics_steps * self.policy.dt * 1e6),
            )
        speed = np.abs(self.data.qvel[self.vids])
        limit = np.where(
            speed >= self.cutoff, 0.0, self.plateau * np.exp(-np.maximum(speed - self.corner, 0) / self.decay)
        )
        limit = np.minimum(limit, np.where(self.heat >= 1, self.continuous, self.plateau))
        torque = np.clip(
            self.kp * (self.targets - self.data.qpos[self.qids]) - self.kd * self.data.qvel[self.vids], -limit, limit
        )
        self.data.ctrl[:] = torque
        self.heat += ((torque / self.continuous) ** 2 - self.heat) * (self.policy.dt / self.thermal_tau)
        mj.mj_step(self.model, self.data)
        self.physics_steps += 1
        self.motion.sample_physics()

    def snapshot(self):
        return {
            "name": "Jumper",
            "source_format": "MJCF",
            "engine": "MuJoCo",
            "links": self.model.nbody - 1,
            "dofs": len(self.policy.wire),
            "joint_names": self.policy.wire,
            "joint_positions": self.data.qpos[self.qids].tolist(),
            "fixed_base": False,
            "control_mode": "policy_locomotion",
            "supports_lateral": True,
            "collision": True,
            "policy_mode": self.policy.fsm.mode,
            "running_policy": self.policy.fsm.running_policy,
            "policy_actions": ["jump", "set_gripper"],
            "gripper": self.motion.gripper_snapshot(),
            "command": list(self._command),
            "motion_tolerances": {
                "position_m": self.motion.base.tuning.position_resume,
                "yaw_rad": self.motion.base.tuning.yaw_resume,
                "settle_s": self.motion.base.tuning.settle_s,
            },
        }


class JumperCamera(MujocoCamera):
    mjcf_name = "onboard"
    link_name = "camera_link"
