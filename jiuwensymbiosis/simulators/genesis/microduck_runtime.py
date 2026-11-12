"""Native Microduck world using Pollen's unmodified CPU policy/actuator reference."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import uuid
from contextlib import redirect_stdout
from dataclasses import replace

import numpy as np
from base_motion import MotionTuning, RelativeMotion
from camera import OrbitCamera
from mujoco_rendering import MujocoRenderer
from scene_catalog import resolve_scene


def load_model(reference, scene):
    """Pollen's CPU MjSpec setup with BAM's corrected DOF-friction implementation.

    Mirrors microduck_rl/scripts/infer_policy.py (Apache-2.0, Pollen Robotics).
    BAM 1.0.3 expresses supply sag as resistance instead of the older torque gain.
    Use its physical supply model; no copied policy or actuator equations.
    """
    import mujoco as mj
    from bam.mujoco import MujocoController

    bam = reference.load_bam_model(reference.BAM_KP_FW, 7.4, None)
    limit = bam.actuator.vin * bam.kt.value / bam.R.value
    spec = mj.MjSpec.from_file(str(scene))
    names = []
    for actuator in spec.actuators:
        target = actuator.target
        if target.startswith("passive_"):
            continue
        actuator.set_to_motor()
        actuator.forcelimited = True
        actuator.forcerange = (-limit, limit)
        actuator.ctrllimited = False
        actuator.gear = [1, 0, 0, 0, 0, 0]
        names.append(actuator.name)
        joint = next(joint for joint in spec.joints if joint.name == target)
        joint.damping = np.zeros((3, 1))
        joint.frictionloss = 0
        joint.solref_friction = reference.BAM_STIFF_SOLREF_FRICTION
        joint.solimp_friction = reference.BAM_STIFF_SOLIMP_FRICTION
    model = spec.compile()
    model.opt.timestep = 0.005
    data = mj.MjData(model)
    actuators = MujocoController(bam, names, model, data, vin_drop_resistance=0.1, vin_min=reference.BAM_VIN_MIN)
    return model, data, actuators


class MicroduckRuntime:
    hidden_geom_groups = (3,)

    def __init__(self, settings, directory, revision=0, *, startup=None):
        import mujoco as mj

        self.settings = settings
        self.renderer = None
        self.indoor = None
        self.bodies = []
        self.selection = None
        self.steps = self.cycle = self.physics_steps = 0
        self._accumulator = 0.0
        self._command = (0.0, 0.0, 0.0)
        self._policy_command = (0.0, 0.0, 0.0)
        self._command_until = 0.0
        self.asset = resolve_scene(settings.scene_id)
        if self.asset.asset.preview_mode != "microduck" or settings.robot_id:
            raise ValueError("Microduck 模式需要索引中的官方模型和行走策略。")
        try:
            paths = self.asset.paths
            manifest = json.loads(paths["policy_manifest"].read_text())
            if (manifest.get("obs_len"), manifest.get("action_len"), manifest["robot"]["control_hz"]) != (61, 14, 50):
                raise ValueError("Unsupported Microduck policy contract")
            spec = importlib.util.spec_from_file_location("_microduck_reference", paths["policy_reference"])
            if spec is None or spec.loader is None:
                raise ImportError("Cannot load official Microduck CPU reference")
            reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference)
            self.model, self.data, self.actuators = load_model(reference, paths["runtime_scene"])
            self.policy = reference.PolicyInference(
                self.model,
                self.data,
                walking_onnx_path=str(paths["walking_policy"]),
                bam_ctrl=self.actuators,
                new_cmd_obs=True,
                use_projected_gravity=True,
            )
            self.wire = [self.model.joint(self.model.actuator_trnid[i, 0]).name for i in range(self.model.nu)]
            session = self.policy.walking_session
            metadata = session.get_modelmeta().custom_metadata_map
            if (
                session.get_inputs()[0].shape != [1, 61]
                or session.get_outputs()[0].shape != [1, 14]
                or metadata.get("joint_names", "").split(",") != self.wire
                or float(metadata.get("action_scale", "nan")) != 1.0
                or metadata.get("observation_names")
                != "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command"
            ):
                raise ValueError("Microduck ONNX dimensions/joint order/action scale mismatch")
            self.qids = np.asarray(self.policy.joint_qpos_indices)
            self.data.qpos[:7] = (0, 0, 0.125, 1, 0, 0, 0)
            self.data.qpos[self.qids] = self.policy.default_pose
            self.actuators.reset()
            self.policy.set_position_targets(self.policy.default_pose)
            mj.mj_forward(self.model, self.data)
            self.model.vis.global_.offwidth = 1920
            self.model.vis.global_.offheight = 1080
            self.model.vis.map.znear = 0.01
            # The CAD export's camera looks back into its lens, with Y up.
            # Render-only correction: look out along head -Z, with head +X up;
            # move 5 mm beyond the lens surface. Physics and source stay intact.
            camera = self.model.camera("head_camera").id
            self.model.cam_quat[camera] = (math.sqrt(0.5), 0, 0, -math.sqrt(0.5))
            self.model.cam_pos[camera, 2] -= 0.005
            self.robot = self
            self.bounds = [[-0.3, -0.3, 0], [0.3, 0.3, 0.3]]
            self.motion = RelativeMotion(
                self,
                MotionTuning(
                    position_tolerance=0.025,
                    position_resume=0.04,
                    yaw_tolerance=0.06,
                    yaw_resume=0.1,
                    linear_still=0.025,
                    angular_still=0.25,
                    settle_s=1.0,
                    min_linear=0.15,
                    min_angular=0.6,
                    max_angular=0.6,
                ),
            )
            self.info = {
                "id": settings.scene_id,
                "robot_id": self.asset.asset.id,
                "instance_id": uuid.uuid4().hex,
                "mode": "robot_mobile",
                "collision": True,
                "engine": "MuJoCo",
                "renderer": "MuJoCo OpenGL",
                "control_description": "Microduck 官方 velstand 策略 · BAM XL330 执行器",
                "reset_interval_steps": None,
                "bounds_m": self.bounds,
                "robot_bounds_m": self.bounds,
                "policy": {
                    "backend": "ONNX Runtime CPU / official PolicyInference + BAM",
                    "sha256": hashlib.sha256(paths["walking_policy"].read_bytes()).hexdigest(),
                    "control_hz": 50,
                    "physics_dt": 0.005,
                },
            }
            for _ in range(600):
                self._physics_step()
            if self.data.qpos[2] < 0.08 or self.data.body("trunk_base").xmat[8] < 0.9:
                raise ValueError("Microduck failed to stand under the official policy")
            pose = OrbitCamera(yaw=2.4, pitch=0.3, distance=0.8, target=(0, 0, 0.13), scene_scale=0.3)
            self.pose = self.initial_pose = replace(pose, revision=revision).snapshot()
            self.cameras = {"head_cam": MicroduckCamera(self)}
            self.frame_size(settings.camera_name)
            self.build_renderer(settings)
        except BaseException:
            self.close()
            raise

    def observe_base(self):
        import mujoco as mj

        body = self.data.body("trunk_base")
        vel = np.zeros(6)
        mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, body.id, vel, 0)
        if not np.isfinite(np.concatenate((body.xpos, body.xquat, vel))).all():
            raise RuntimeError("Non-finite Microduck observation")
        return {
            "position_m": body.xpos.tolist(),
            "quaternion_wxyz": body.xquat.tolist(),
            "body_yaw_rad": math.atan2(body.xmat[3], body.xmat[0]),
            "linear_velocity_m_s": vel[3:].tolist(),
            "angular_velocity_rad_s": vel[:3].tolist(),
            "frame": "world",
            "command_frame": "body_forward_left_up",
        }

    def command(self, vx, vy, wz, duration):
        from omni_base import validate_twist

        validate_twist(vx, vy, wz)
        if not math.isfinite(duration) or not 0 < duration <= 1:
            raise ValueError("Invalid Microduck command duration")
        self._command = (vx, vy, wz)
        self._command_until = self.data.time + duration

    def stop(self):
        self._command = (0.0, 0.0, 0.0)
        self._command_until = 0.0

    def _physics_step(self):
        import mujoco as mj

        if self.physics_steps % 4 == 0:
            command = self._command if self.data.time < self._command_until else (0.0, 0.0, 0.0)
            # The released gait has a substantial start deadband in CPU MuJoCo.
            # Use a bounded gait command; RelativeMotion still closes the loop on
            # measured distance/yaw, never on an assumed commanded velocity.
            vx, vy, wz = command
            norm = math.hypot(vx, vy)
            self._policy_command = (
                0.4 * vx / norm if norm else 0.0,
                0.4 * vy / norm if norm else 0.0,
                math.copysign(1.5, wz) if wz else 0.0,
            )
            if not np.allclose(self.policy.vel_cmd, self._policy_command, atol=1e-6):
                # Avoid the reference CLI's 50-Hz terminal prints in worker.log.
                with redirect_stdout(io.StringIO()):
                    self.policy.set_vel_cmd(*self._policy_command)
            action = self.policy.infer()
            if not np.isfinite(action).all():
                raise RuntimeError("Microduck policy produced non-finite actions")
            self.policy.apply_action(action)
        self.actuators.update()
        mj.mj_step(self.model, self.data)
        self.physics_steps += 1
        if not np.isfinite(self.data.qpos).all() or self.data.body("trunk_base").xmat[8] < 0.7:
            self.stop()
            raise RuntimeError("Microduck lost balance; restart the scene")

    def step(self):
        self.motion.tick(self.settings.dt, self.steps)
        self._accumulator += self.settings.dt
        while self._accumulator >= 0.005 - 1e-10:
            self._physics_step()
            self._accumulator -= 0.005
        self.steps += 1

    def snapshot(self):
        return {
            "name": "Microduck",
            "source_format": "MJCF",
            "engine": "MuJoCo",
            "links": self.model.nbody - 1,
            "dofs": len(self.wire),
            "joint_names": self.wire,
            "joint_positions": self.data.qpos[self.qids].tolist(),
            "fixed_base": False,
            "control_mode": "policy_locomotion",
            "supports_lateral": True,
            "collision": True,
            "command": list(self._command),
            "policy_command": list(self._policy_command),
        }

    def frame_size(self, name):
        if name == "observer":
            return self.settings.width, self.settings.height
        if name not in self.cameras:
            raise ValueError(f"Microduck has no camera: {name}")
        return self.cameras[name].resolution

    def build_renderer(self, settings):
        self.renderer = MujocoRenderer(self)
        self.settings = settings

    def close_renderer(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def update_camera(self, pose):
        self.pose = pose

    def selection_snapshot(self):
        return None

    def pick(self, u, v):
        raise ValueError("MuJoCo 场景暂不支持物体点选。")

    def close(self):
        self.close_renderer()


class MicroduckCamera:
    name = "head_cam"
    mjcf_name = "head_camera"
    resolution = (640, 480)

    def __init__(self, runtime):
        self.runtime = runtime

    def metadata(self):
        fovy = float(self.runtime.model.camera(self.mjcf_name).fovy[0])
        focal = 480 / (2 * math.tan(math.radians(fovy) / 2))
        return {
            "name": self.name,
            "link": "head_camera",
            "width": 640,
            "height": 480,
            "K": [[focal, 0, 320], [0, focal, 240], [0, 0, 1]],
            "D": [0] * 5,
            "distortion_model": "pinhole",
            "optical_frame": "x_right_y_down_z_forward",
            "fov_y_deg": fovy,
            "source": "official MJCF head_camera; model FOV, not hardware calibration",
        }
