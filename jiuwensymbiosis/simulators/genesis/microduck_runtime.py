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
from mujoco_runtime import MujocoCamera, MujocoPolicyRuntime
from scene_catalog import resolve_scene

REFERENCE_SHA256 = "03bfd0642a880cd379e057536ee819bd1377d4af7b5206a57907d10afd1d4270"
WALKING_SHA256 = "ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a"


def load_reference(path, policy):
    """Execute only the pinned upstream reference, after verifying both inputs."""
    for source, expected in ((path, REFERENCE_SHA256), (policy, WALKING_SHA256)):
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Microduck reference/policy digest mismatch: {source.name}")
    spec = importlib.util.spec_from_file_location("_microduck_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load official Microduck CPU reference")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return reference


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


class MicroduckRuntime(MujocoPolicyRuntime):
    base_body = "trunk_base"
    hidden_geom_groups = (3,)
    physics_dt = 0.005
    # Explicit native-policy envelope; the generic/manual path retains its 0.6 rad/s cap.
    max_linear_rate = 0.4
    max_yaw_rate = 1.5

    def __init__(self, settings, directory, revision=0, *, startup=None):
        import mujoco as mj

        super().__init__(settings, directory)
        self._policy_command = (0.0, 0.0, 0.0)
        self.asset = resolve_scene(settings.scene_id)
        if self.asset.asset.preview_mode != "microduck" or settings.robot_id:
            raise ValueError("Microduck 模式需要索引中的官方模型和行走策略。")
        try:
            paths = self.asset.paths
            manifest = json.loads(paths["policy_manifest"].read_text())
            if (manifest.get("obs_len"), manifest.get("action_len"), manifest["robot"]["control_hz"]) != (61, 14, 50):
                raise ValueError("Unsupported Microduck policy contract")
            reference = load_reference(paths["policy_reference"], paths["walking_policy"])
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
            contract_matches = (
                session.get_inputs()[0].shape == [1, 61],
                session.get_outputs()[0].shape == [1, 14],
                metadata.get("joint_names", "").split(",") == self.wire,
                float(metadata.get("action_scale", "nan")) == 1.0,
                metadata.get("observation_names")
                == "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command",
            )
            if not all(contract_matches):
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
                    min_linear=self.max_linear_rate,
                    max_linear=self.max_linear_rate,
                    min_angular=self.max_yaw_rate,
                    max_angular=self.max_yaw_rate,
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

    def command(self, vx, vy, wz, duration):
        from omni_base import validate_twist

        validate_twist(vx, vy, wz, max_linear=self.max_linear_rate, max_angular=self.max_yaw_rate)
        if not math.isfinite(duration) or not 0 < duration <= 1:
            raise ValueError("Invalid Microduck command duration")
        self._command = (vx, vy, wz)
        self._command_until = self.data.time + duration

    def _physics_step(self):
        import mujoco as mj

        if self.physics_steps % 4 == 0:
            command = self._command if self.data.time < self._command_until else (0.0, 0.0, 0.0)
            # Commands keep their validated SI magnitude; never boost small requests.
            self._policy_command = command
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
            "command_limits": {"linear_m_s": self.max_linear_rate, "angular_rad_s": self.max_yaw_rate},
        }


class MicroduckCamera(MujocoCamera):
    mjcf_name = "head_camera"
    link_name = "head_camera"
