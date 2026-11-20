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
from types import SimpleNamespace

import numpy as np

from ..assets.policy_files import MICRODUCK_ARENA, MICRODUCK_FILES
from ..common.base_motion import MotionTuning, RelativeMotion
from ..common.camera import OrbitCamera
from ..common.scene_catalog import resolve_scene
from .microduck_motion import ROLLER_SEATED_PHASE, MicroduckMotion
from .microduck_mouth import CLOSED, OPEN, add_mouth
from .mujoco_runtime import MujocoCamera, MujocoPolicyRuntime
from .policy_actions import ARENA_GAITS, action_controls

REFERENCE_SHA256 = "03bfd0642a880cd379e057536ee819bd1377d4af7b5206a57907d10afd1d4270"
WALKING_SHA256 = MICRODUCK_FILES["velstand.onnx"]


def load_reference(path, policy, expected_policy=None):
    """Execute only the pinned upstream reference, after verifying both inputs."""
    for source, expected in ((path, REFERENCE_SHA256), (policy, expected_policy or WALKING_SHA256)):
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Microduck reference/policy digest mismatch: {source.name}")
    spec = importlib.util.spec_from_file_location("_microduck_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load official Microduck CPU reference")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return reference


def load_model(reference, scene, ball=None):
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
    if ball is not None:
        spec.attach(mj.MjSpec.from_file(str(ball)), prefix="kick_", frame=spec.worldbody.add_frame())
        # Pollen's kick reference looks up this free joint to place its training ball.
        spec.joint("kick_ball_free").name = "ball_free"
    add_mouth(spec)
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
        self.hold_roller_seat = False
        self.kicking_bodies: set[int] = set()
        self._policy_command = (0.0, 0.0, 0.0)
        self.asset = resolve_scene(settings.scene_id)
        if self.asset.asset.preview_mode != "microduck" or settings.robot_id:
            raise ValueError("Microduck 模式需要索引中的官方模型和行走策略。")
        try:
            paths = self.asset.paths
            self.rollers = bool(self.asset.asset.options.get("rollers"))
            primary = "roller" if self.rollers else "velstand"
            primary_path = paths["roller" if self.rollers else "walking_policy"]
            manifest_bytes = paths["policy_manifest"].read_bytes()
            if hashlib.sha256(manifest_bytes).hexdigest() != MICRODUCK_FILES["manifest.json"]:
                raise ValueError("Microduck manifest digest mismatch")
            manifest = json.loads(manifest_bytes)
            self.specs = {entry["file"].removesuffix(".onnx"): entry for entry in manifest["policies"]}
            reference = load_reference(paths["policy_reference"], primary_path, MICRODUCK_FILES[primary + ".onnx"])
            self.model, self.data, self.actuators = load_model(
                reference, paths["runtime_scene"], paths.get("roller_ball")
            )
            # Pollen's policy controls the first fourteen servos. The independent
            # mouth servo is last on the simulator wire and shares the BAM supply.
            mouth = self.model.actuator("mouth").id
            if mouth != self.model.nu - 1 or mouth != 14:
                raise ValueError("Microduck requires 14 policy actuators and an independent mouth")

            class ScenePolicyInference(reference.PolicyInference):
                def _place_ball(self, _behavior):
                    """Keep the live ball under MuJoCo control throughout a kick."""

            self.policy = ScenePolicyInference(
                self.model,
                self.data,
                walking_onnx_path=str(primary_path),
                bam_ctrl=SimpleNamespace(q_target=self.actuators.q_target[:mouth]),
                new_cmd_obs=True,
                use_projected_gravity=True,
                action_scale=self.specs[primary].get("action_scale", 1.0),
            )
            self.policy.n_joints = mouth
            self.policy.joint_qpos_indices = self.policy.joint_qpos_indices[:mouth]
            self.policy.joint_qvel_indices = self.policy.joint_qvel_indices[:mouth]
            self.policy.last_action = self.policy.last_action[:mouth]
            self.wire = [self.model.joint(self.model.actuator_trnid[i, 0]).name for i in range(mouth)]
            self.mouth_qid = self.model.joint("mouth").qposadr[0]
            self.mouth_vid = self.model.joint("mouth").dofadr[0]
            self.set_mouth(0.0)
            self._load_policies(paths, primary)
            self.qids = np.asarray(self.policy.joint_qpos_indices)
            self.data.qpos[:7] = (0, 0, 0.1385 if self.rollers else 0.125, 1, 0, 0, 0)
            if self.rollers:
                self.max_linear_rate, self.max_yaw_rate = 0.6, 1.0
                for joint in range(self.model.njnt):
                    if self.model.joint(joint).name.startswith("passive_"):
                        self.model.dof_frictionloss[self.model.jnt_dofadr[joint]] = 0.003
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
            base_motion = RelativeMotion(
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
            self.motion = MicroduckMotion(self, base_motion)
            self.info = {
                "id": settings.scene_id,
                "robot_id": self.asset.asset.id,
                "instance_id": uuid.uuid4().hex,
                "mode": "robot_mobile",
                "collision": True,
                "engine": "MuJoCo",
                "renderer": "MuJoCo OpenGL",
                "control_description": f"Microduck · {primary} · BAM XL330",
                "reset_interval_steps": None,
                "bounds_m": self.bounds,
                "robot_bounds_m": self.bounds,
                "policy": {
                    "backend": "ONNX Runtime CPU / official PolicyInference + BAM",
                    "sha256": MICRODUCK_FILES[primary + ".onnx"],
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

    def _load_policies(self, paths, primary):
        import onnxruntime as ort

        p = self.policy
        self.sessions = {primary: p.walking_session}
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        for name in self.specs.keys() & paths.keys() - {primary}:
            content = paths[name].read_bytes()
            if hashlib.sha256(content).hexdigest() != MICRODUCK_FILES[name + ".onnx"]:
                raise ValueError(f"Microduck policy digest mismatch: {name}")
            self.sessions[name] = ort.InferenceSession(
                content, sess_options=options, providers=["CPUExecutionProvider"]
            )
        self.timeline: dict = {}
        self.show_started: float | None = None
        for name, release in MICRODUCK_ARENA.items():
            if f"{name}/policy.onnx" not in paths:
                continue
            files = {}
            for filename, digest in release["files"].items():
                data = paths[f"{name}/{filename}"].read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError(f"Microduck Arena policy digest mismatch: {name}/{filename}")
                files[filename] = data
            manifest_bytes = files.get("manifest.json")
            policy_bytes = files.get("policy.onnx")
            if manifest_bytes is None or policy_bytes is None:
                raise ValueError(f"Microduck Arena policy or manifest missing: {name}")
            manifest = json.loads(manifest_bytes)
            if ("rollers" in manifest["robot"]["accessories"]) != self.rollers:
                raise ValueError("Microduck Arena policy and feet mismatch")
            self.sessions[name] = ort.InferenceSession(
                policy_bytes, sess_options=options, providers=["CPUExecutionProvider"]
            )
            if "timeline.json" in files:
                self.timeline = json.loads(files["timeline.json"])
        for session in self.sessions.values():
            metadata = session.get_modelmeta().custom_metadata_map
            if not all(
                (
                    session.get_inputs()[0].shape == [1, 61],
                    session.get_outputs()[0].shape == [1, 14],
                    metadata.get("joint_names", "").split(",") == self.wire,
                    float(metadata.get("action_scale", "nan")) == 1,
                    metadata.get("observation_names")
                    == "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command",
                )
            ):
                raise ValueError("Microduck ONNX dimensions/joint order/action scale mismatch")
        self.gait = "roller_gait" if self.rollers else "velstand_gait"
        self.gait_scale = p.action_scale
        self.actions = ["roller_skate"] if self.rollers else ["head_pose", "body_pose", "recover_stance"]
        self.actions.append("mouth")
        if {"alpha_walking", "alpha_stand"} <= self.sessions.keys():
            self.actions.append("classic_gait")
        if not self.rollers and "alpha_sitstand" in self.sessions:
            p.sit_session = self.sessions["alpha_sitstand"]
            p.is_sitstand = True
            self.actions += ["sit", "stand"]
        phase = "roller_crouch" if self.rollers else "alpha_ground_pick"
        if phase in self.sessions:
            p.ground_pick_session = self.sessions[phase]
            p.ground_pick_period = self.specs[phase]["command"]["period_s"]
            self.actions.append("roller_crouch" if self.rollers else "ground_pick")
            if self.rollers:
                self.actions += ["sit", "stand"]
        for name in ("kick_left", "kick_right", "roulade"):
            weight = "ball_" + name if name.startswith("kick_") else name
            if weight in self.sessions:
                p.behavior_sessions[name] = self.sessions[weight]
                p.behavior_durations[name] = self.specs[weight]["duration_s"]
                self.actions.append(name)
        self.actions.extend(name for name in ARENA_GAITS if name in self.sessions)
        if "classic_gait" in self.actions or self.sessions.keys() & ARENA_GAITS.keys():
            self.actions.append(self.gait)
        if self.timeline:
            self.actions.append("got_a_talent")
        self.controls = action_controls(self.actions)

    def select_gait(self, gait):
        p = self.policy
        self.gait = gait
        if gait in ARENA_GAITS:
            # These releases declare unused head/body slots as zero.
            p.head_offset[:] = 0
            p.body_cmd[:] = 0
        gait_file = {"velstand_gait": "velstand", "classic_gait": "alpha_walking", "roller_gait": "roller"}.get(
            gait, gait
        )
        p.walking_session = self.sessions[gait_file]
        if gait in ARENA_GAITS:
            # These are walk-slot policies. Use the factory stance to brake and hold at zero twist.
            p.standing_session = self.sessions["roller" if self.rollers else "velstand"]
        else:
            p.standing_session = self.sessions.get("alpha_stand") if gait == "classic_gait" else None
        self.gait_scale = self.specs.get(gait_file, {}).get("action_scale", 1.0)
        p.action_scale = self.gait_scale
        p.current_policy, p.ort_session = "walking", p.walking_session
        p.behavior_mode, p.ground_pick_mode, p.sit_mode = None, False, False
        self.set_gait_velocity((0, 0, 0))

    def set_gait_velocity(self, command):
        p = self.policy
        with redirect_stdout(io.StringIO()):
            p.set_vel_cmd(*command)
        if p.current_policy == "walking":
            p.action_scale = self.gait_scale
        elif p.current_policy == "standing" and self.gait in ARENA_GAITS:
            default = "roller" if self.rollers else "velstand"
            p.action_scale = self.specs[default].get("action_scale", 1.0)

    def update_show(self):
        if self.show_started is None:
            return
        elapsed = self.data.time - self.show_started
        if elapsed >= self.timeline["duration_s"]:
            self.show_started = None
            self.select_gait(self.gait)
            return
        # Arena timeline v1 holds the last whole frame at or before this control tick.
        frame = next(f for f in reversed(self.timeline["keyframes"]) if f["t"] <= elapsed + 1e-9)
        self.policy.command[:] = (
            *frame.get("twist", (0,) * 3),
            *frame.get("head", (0,) * 4),
            *frame.get("body", (0,) * 6),
        )

    def upright(self):
        return self.data.body(self.base_body).xmat[8] > 0.7 and self.data.qpos[2] > 0.08

    def set_mouth(self, opening):
        self.mouth_target = CLOSED + opening * (OPEN - CLOSED)
        self.actuators.set_q_target("mouth", self.mouth_target)

    def set_policy_pose(self, *, head=None, body=None):
        p = self.policy
        if head is not None:
            p.head_offset[:] = head
        if body is not None:
            p.body_cmd[:] = body
        # The pinned 13-slot layout keeps twist or sit/ground-pick phase in
        # the first three slots. Kick and roll actors require all-zero commands.
        if p.behavior_mode is None:
            p.command[3:7] = p.head_offset
            p.command[7:13] = p.body_cmd

    def watch_kick(self, side):
        import mujoco as mj

        # Existing contact must separate before a new contact can count as a strike.
        mj.mj_forward(self.model, self.data)
        if self.rollers:
            bodies = ("tire", "tire_2") if side == "left" else ("tire_3", "tire_4")
        else:
            bodies = ("ankle_" + side,)
        self.kicking_bodies = {self.model.body(name).id for name in bodies}
        self.ball_geom = self.model.geom("kick_ball_geom" if self.rollers else "ball_geom").id
        self.kick_contact = False
        self.kick_clear = not self.foot_touches_ball()

    def foot_touches_ball(self):
        for contact in self.data.contact:
            a, b = contact.geom1, contact.geom2
            if a == self.ball_geom:
                a, b = b, a
            if b == self.ball_geom and self.model.geom_bodyid[a] in self.kicking_bodies and contact.dist <= 0:
                return True
        return False

    def command(self, vx, vy, wz, duration):
        from ..genesis.omni_base import validate_twist

        validate_twist(vx, vy, wz, max_linear=self.max_linear_rate, max_angular=self.max_yaw_rate)
        if not math.isfinite(duration) or not 0 < duration <= 1:
            raise ValueError("Invalid Microduck command duration")
        if self.rollers and (vy or vx < -0.5):
            raise ValueError("轮滑策略仅支持前进、制动和航向指令。")
        if any((vx, vy, wz)) and (self.policy.sit_mode or not self.upright()):
            raise ValueError("请先让 Microduck 站稳。")
        self._command = (vx, vy, wz)
        self._command_until = self.data.time + duration

    def _physics_step(self):
        import mujoco as mj

        if self.physics_steps % 4 == 0:
            command = self._command if self.data.time < self._command_until else (0.0, 0.0, 0.0)
            # Commands keep their validated SI magnitude; never boost small requests.
            # Velstand's zero twist is its native fall-recovery input.
            if not self.upright() and not self.rollers:
                self.stop()
                command = (0.0, 0.0, 0.0)
            self._policy_command = command
            idle_handoff = self.gait in ARENA_GAITS and self.policy.current_policy == "walking" and not any(command)
            if idle_handoff or not np.allclose(self.policy.vel_cmd, self._policy_command, atol=1e-6):
                self.set_gait_velocity(command)
            ground_pick = self.policy.ground_pick_mode
            if self.hold_roller_seat:
                self.policy.ground_pick_phase = ROLLER_SEATED_PHASE
            self.policy.update_ground_pick_phase(4 * self.physics_dt)
            if ground_pick and not self.policy.ground_pick_mode:
                self.policy.action_scale = self.gait_scale
            self.policy.update_behavior(4 * self.physics_dt)
            self.update_show()
            action = self.policy.infer()
            if not np.isfinite(action).all():
                raise RuntimeError("Microduck policy produced non-finite actions")
            self.policy.apply_action(action)
        self.actuators.update()
        mj.mj_step(self.model, self.data)
        if self.kicking_bodies and not self.kick_contact:
            if not self.foot_touches_ball():
                self.kick_clear = True
            elif self.kick_clear:
                self.kick_contact = True
        self.physics_steps += 1
        if not np.isfinite(self.data.qpos).all():
            self.stop()
            raise RuntimeError("Non-finite Microduck state")

    def snapshot(self):
        return {
            "name": "Microduck",
            "source_format": "MJCF",
            "engine": "MuJoCo",
            "links": self.model.nbody - 1,
            "dofs": self.model.nu,
            "policy_dofs": len(self.wire),
            "joint_names": [*self.wire, "mouth"],
            "joint_positions": [*self.data.qpos[self.qids].tolist(), float(self.data.qpos[self.mouth_qid])],
            "fixed_base": False,
            "control_mode": "policy_locomotion",
            "supports_lateral": not self.rollers,
            "relative_navigation": not self.rollers,
            "gait": self.gait,
            "policy_mode": self.policy.current_policy,
            "policy_actions": self.actions,
            "policy_controls": self.controls,
            "mouth": {"angle_rad": float(self.data.qpos[self.mouth_qid]), "target_rad": self.mouth_target},
            "collision": True,
            "command": list(self._command),
            "policy_command": list(self._policy_command),
            "command_limits": {"linear_m_s": self.max_linear_rate, "angular_rad_s": self.max_yaw_rate},
        }


class MicroduckCamera(MujocoCamera):
    mjcf_name = "head_camera"
    link_name = "head_camera"
