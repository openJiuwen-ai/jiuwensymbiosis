# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SO-101 adapter config.

SO-101 is a 5-DoF underactuated arm (Feetech STS3215 servos) driven by the
LeRobot 0.6.x ``SOFollower``. This config captures the per-robot knobs the
adapter needs (serial port, explicit safe home joints, soft limits, IK weights,
gripper two-state percentages) and validates them at load time.

See ``.claude/plans/so101-adapter.md`` for the verified baseline.
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from jiuwensymbiosis.adapters.so101.lowlevel import ARM_JOINT_ORDER

# Canonical 5-arm-joint order; the gripper is a separate end effector.
_ARM_JOINT_SET = frozenset(ARM_JOINT_ORDER)


@dataclass
class DetectorServerConfig:
    """Connection and spawn settings for the GroundingDINO/SAM2 sidecar.

    When ``spawn`` is enabled, ``url`` is canonical and ``host``/``port`` are
    derived from it during validation.
    """

    url: str = "http://127.0.0.1:8114"
    spawn: bool = False
    host: str = "127.0.0.1"
    port: int = 8114
    device: str = "cuda"
    startup_timeout_s: float = 300.0
    gdino_model_id: str = "IDEA-Research/grounding-dino-base"
    sam2_model_id: str = "facebook/sam2.1-hiera-large"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    use_sam2: bool = True

    @classmethod
    def from_dict(cls, raw: Any) -> DetectorServerConfig:
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raise ValueError(f"So101Config: detector must be a mapping, got {type(raw).__name__}.")
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"So101Config: unknown detector fields: {sorted(unknown)}.")
        return cls(**raw)

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url:
            raise ValueError("So101Config: detector.url must be a non-empty string.")
        if not isinstance(self.spawn, bool):
            raise ValueError("So101Config: detector.spawn must be bool.")
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("So101Config: detector.host must be a non-empty string.")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise ValueError(f"So101Config: detector.port must be an integer in [1, 65535], got {self.port!r}.")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("So101Config: detector.device must be a non-empty string.")
        for name in ("startup_timeout_s", "box_threshold", "text_threshold"):
            value = getattr(self, name)
            if not _is_finite(value):
                raise ValueError(f"So101Config: detector.{name} must be finite, got {value!r}.")
        if self.startup_timeout_s <= 0:
            raise ValueError("So101Config: detector.startup_timeout_s must be > 0.")
        if not 0.0 <= self.box_threshold <= 1.0:
            raise ValueError("So101Config: detector.box_threshold must be in [0, 1].")
        if not 0.0 <= self.text_threshold <= 1.0:
            raise ValueError("So101Config: detector.text_threshold must be in [0, 1].")
        if not isinstance(self.use_sam2, bool):
            raise ValueError("So101Config: detector.use_sam2 must be bool.")
        if self.spawn:
            try:
                parsed = urlparse(self.url)
                url_port = parsed.port
            except ValueError as exc:
                raise ValueError(f"So101Config: invalid detector.url {self.url!r}: {exc}.") from exc
            if parsed.scheme != "http" or not parsed.hostname:
                raise ValueError("So101Config: detector.url must be an absolute http URL when detector.spawn=True.")
            # ``url`` is the canonical endpoint used by the API. Derive the
            # sidecar bind address from it so the subprocess cannot start on a
            # different port than the client calls.
            self.host = parsed.hostname
            self.port = url_port or 80


def _is_finite(value: float) -> bool:
    """True only when ``value`` is a real number (not NaN / Inf)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _server_value(server: dict[str, Any], field_name: str, default: Any) -> Any:
    """Return an api-server value, treating explicit YAML null as absent."""
    value = server.get(field_name, default)
    return default if value is None else value


def _server_number(server: dict[str, Any], field_name: str, default: int | float, converter: Any) -> Any:
    """Convert one numeric detector field with a configuration-specific error."""
    value = _server_value(server, field_name, default)
    type_name = "an integer" if converter is int else "a number"
    if isinstance(value, bool):
        raise ValueError(f"So101Config: api_servers detector.{field_name} must be {type_name}, got {value!r}.")
    try:
        return converter(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"So101Config: api_servers detector.{field_name} must be {type_name}, got {value!r}.") from exc


def _server_bool(server: dict[str, Any], field_name: str, default: bool) -> bool:
    """Read one boolean detector field without truthiness coercion."""
    value = _server_value(server, field_name, default)
    if not isinstance(value, bool):
        raise ValueError(f"So101Config: api_servers detector.{field_name} must be bool, got {value!r}.")
    return value


def _extract_detector_from_api_servers(api_servers: list[Any]) -> DetectorServerConfig:
    """Pick the GroundingDINO+SAM2 server entry from a top-level ``api_servers:`` list.

    Mirrors piper's loader so SO-101 YAMLs use the same ``api_servers:`` shape
    (``_target_`` containing ``grounding_dino``/``gdino`` identifies the detector).
    ``GDINO_MODEL_ID`` / ``SAM2_MODEL_ID`` env vars override the YAML model ids,
    letting the GUI prime local offline model dirs without editing YAML.
    """
    defaults = DetectorServerConfig()
    for s in api_servers or []:
        if not isinstance(s, dict):
            continue
        target = str(s.get("_target_", "")).lower()
        if "grounding_dino" not in target and "gdino" not in target:
            continue
        host = _server_value(s, "host", "127.0.0.1")
        port = _server_number(s, "port", defaults.port, int)
        return DetectorServerConfig(
            url=f"http://{host}:{port}",
            spawn=True,
            host=host,
            port=port,
            device=_server_value(s, "device", defaults.device),
            startup_timeout_s=_server_number(s, "startup_timeout_s", defaults.startup_timeout_s, float),
            gdino_model_id=os.environ.get("GDINO_MODEL_ID")
            or _server_value(s, "gdino_model_id", defaults.gdino_model_id),
            sam2_model_id=os.environ.get("SAM2_MODEL_ID") or _server_value(s, "sam2_model_id", defaults.sam2_model_id),
            box_threshold=_server_number(s, "box_threshold", defaults.box_threshold, float),
            text_threshold=_server_number(s, "text_threshold", defaults.text_threshold, float),
            use_sam2=_server_bool(s, "use_sam2", defaults.use_sam2),
        )
    return DetectorServerConfig(
        gdino_model_id=os.environ.get("GDINO_MODEL_ID") or defaults.gdino_model_id,
        sam2_model_id=os.environ.get("SAM2_MODEL_ID") or defaults.sam2_model_id,
    )  # no detector entry -> fail-closed defaults (spawn=False)


@dataclass
class So101Config:
    """SO-101 adapter configuration.

    Required safety fields (no defaults) must come first; the rest are
    overridable per deployment.
    """

    # --- required safety config ---
    port: str
    home_joints_deg: list[float]
    joint_limits: dict[str, tuple[float, float]]

    # Fail-closed gate: home_joints_deg / joint_limits ship as unverified
    # placeholders. connect() refuses to open hardware until the operator
    # explicitly confirms (in YAML) that these were validated on the real
    # robot — so a fresh template config cannot silently drive RecoveryRail/home
    # to an unverified pose. Default False; the operator sets `safety_validated:
    # true` only after manually confirming a safe home and tightened limits.
    safety_validated: bool = False

    # --- connection & LeRobot calibration ---
    robot_id: str = "so101"
    calibration_dir: str | None = None
    disable_torque_on_disconnect: bool = True
    # Native motor units: 5 arm joints in degree, gripper 0..100 percentage.
    # Float only: LeRobot's ensure_safe_goal_position requires dict keys to match
    # the action keys EXACTLY, which is incompatible with our split arm/gripper
    # actions. We always pass a float, applied per-action by the driver.
    max_relative_target: float | None = 5.0

    # --- kinematics ---
    urdf_path: str | None = None  # None -> packaged so101_new_calib.urdf
    ik_target_frame: str = "gripper_frame_link"  # passed to RobotKinematics ctor; must exist in URDF
    ik_orientation_weight: float = 0.01
    ik_position_tolerance_mm: float = 3.0
    ik_orientation_tolerance_deg: float | None = None  # None -> record only, never hard-reject

    # Cartesian path planning: the planner splits the SE(3) path start->target
    # into N = ceil(max(translation_mm, rotation_deg) / cartesian_interp_step_mm)
    # steps and solves IK for each, seeded by the previous step's solution (a
    # continuous seed chain matching lerobot's ``InverseKinematicsEEToJoints``
    # with ``initial_guess_current_joints=False``). This keeps placo inside its
    # convergence basin so IK does not jump branches: verified ~0.005 mm residual
    # on real IK for z +/-50 mm, vs 3-30 mm with the prior SE(3) bisection (which
    # re-solved from a stale seed and oscillated between two branches under the
    # jump/residual tolerance squeeze). Smaller = smoother/slower; must be > 0.
    cartesian_interp_step_mm: float = 1.0
    # Optional ease-in-out (sin^2) weighting on the interpolation parameter so the
    # first/last steps move least (better IK convergence at path ends). Default
    # False (linear) for predictability; the seed chain already gives sub-mm
    # residuals, so this is an opt-in smoothness tweak.
    cartesian_ease_in_out: bool = False

    # --- cartesian pose convergence loop (joint-space integral trim) ---
    # NOTE: ``settle_overcompensate`` (below, default on) closes the PD
    # steady-state error INSIDE the settle loop of every move (home, move_joint,
    # and each Cartesian waypoint), so the Cartesian convergence trim is largely
    # redundant. Default 0 (disabled); set >0 only for an extra Cartesian-domain
    # end-point trim when settle over-compensation alone is insufficient.
    pose_convergence_max_iters: int = 0
    # Convergence tolerance (mm, Cartesian translation): position_error_mm(actual,
    # target) <= tol means "arrived" and stops iterating. Must be > 0. Default
    # 1.0: below the measured ~5mm shortfall so compensation triggers, with room
    # to stop early when "close enough".
    pose_convergence_tolerance_mm: float = 1.0

    # --- motion & settle ---
    trajectory_hz: float = 30.0
    max_joint_step_deg: float = 2.0
    # Real-time Cartesian servo joint slew cap.  Kept separate from the
    # blocking-path interpolation cap because servo commands are generated one
    # tick at a time from a live encoder seed.
    servo_max_joint_step_deg: float = 1.0
    # Real-time servo velocity enforcement (hardware-boundary safety). The
    # caller's tick rate is untrusted (a busy-loop or a misconfigured
    # ``control_hz`` can call ``servo_to_pose`` far faster than the arm can
    # track), so the driver caps the *actual* joint velocity itself rather than
    # trusting per-call step clipping alone.
    #   - servo_min_send_period_s: minimum elapsed time between two dispatched
    #     servo actions; a call within this window is skipped (non-blocking
    #     semantics: no accumulation, no catch-up burst).
    #   - servo_max_joint_vel_dps: hard deg/s cap. The per-call step is
    #     re-clipped against ``vel_cap = servo_max_joint_vel_dps * dt`` where
    #     ``dt`` is the real inter-send interval, so speed is independent of
    #     the caller's tick rate.
    servo_min_send_period_s: float = 0.02
    servo_max_joint_vel_dps: float = 30.0
    # Settle "arrived" tolerance (deg, joint space, max norm). The settle loop
    # returns once max|actual - target| <= this for ``settle_samples`` consecutive
    # reads. Default 1.5: paired with ``settle_overcompensate=True`` the servo now
    # reaches the target (over-compensation closes the STS3215 PD steady-state
    # error), so the tolerance can be tight (~0.5mm end-effector). Still > encoder
    # read noise to avoid false non-convergence. With ``settle_overcompensate=False``
    # keep this >= ~3.5 to cover the ~2.46 deg elbow steady-state error (else the
    # settle loop times out -- re-sending the bare target cannot close PD error).
    joint_tolerance_deg: float = 1.5
    settle_samples: int = 3
    move_timeout_s: float = 30.0
    # Settle-loop tuning (true-robot safety). The arm settle loop re-sends the
    # final joint target after the interpolation sweep so a LeRobot-clipped goal
    # can still be driven to completion. Re-sending at ``trajectory_hz`` (30 Hz)
    # overdrives STS3215 servos on gravity-loaded joints (e.g. elbow_flex): the
    # servo cannot track, drifts under gravity, and the loop pushes the joint
    # toward a mechanical limit. ``settle_resend_period_s`` caps the re-send
    # rate; ``settle_drift_abort_samples`` aborts if the max joint error grows
    # for that many consecutive re-sends (servo under load moving the wrong way).
    # 0 for either restores legacy behavior (re-send at trajectory_hz / no abort).
    settle_resend_period_s: float = 0.2
    settle_drift_abort_samples: int = 5
    # Settle real-time over-compensation (software I term for STS3215 PD). The
    # STS3215 firmware position-loop I term is inert (PID experiment: I=2/5/50
    # zero movement), so a gravity-loaded joint (elbow_flex) settles at
    # ``target - e`` instead of ``target`` (~2.46 deg elbow error -> ~9mm TCP x
    # drift at home). With this ON, the settle loop re-sends ``target + e``
    # (``e = target - actual``, read fresh from the encoder each re-send) instead
    # of the bare ``target``: the servo parks AT ``target`` in one over-command,
    # so home / move_joint reach the configured joint angles (verified elbow
    # 0.000 deg vs 2.462 deg baseline). Falls back to the bare target if the
    # over-command would break a soft limit (fail-closed). False = legacy (re-send
    # bare target; then keep ``joint_tolerance_deg`` >= ~3.5 to avoid timeout).
    settle_overcompensate: bool = True

    # --- safety bounds ---
    z_min_safe_mm: float = 30.0
    workspace_bounds: tuple[float, float, float, float] | None = None

    # --- gripper (two-state percentage) ---
    gripper_open_pos: float = 100.0
    gripper_close_pos: float = 0.0
    # Polling settle loop for the gripper: under the default max_relative_target
    # a single send_action cannot move 0->100 or 100->0 in one step, so set_gripper
    # re-sends the target until the observed gripper position converges within
    # gripper_tolerance, bounded by gripper_timeout_s. gripper_settle_s is an
    # optional post-converge dwell (kept for back-compat with the old contract).
    gripper_tolerance: float = 2.0
    gripper_timeout_s: float = 5.0
    gripper_settle_s: float = 0.0

    # --- vision (milestone B): eye-to-hand desktop RealSense ---
    # Camera is a desktop-fixed D405 (eye-to-hand), NOT wrist-mounted. The
    # hand-eye calibration therefore solves ``T_base_cam`` (camera-in-base, a
    # CONSTANT) rather than piper's ``T_flange_cam`` (which varies with the
    # flange). Projection: p_base = T_base_cam @ p_cam (no flange read per step).
    # camera_serial None -> no camera (milestone A behaviour preserved).
    camera_serial: str | None = None
    camera_resolution: tuple[int, int] = (640, 480)
    camera_fps: int = 30
    # Hand-eye calibration JSON (schema-2, ``T_base_cam`` field). None -> no
    # calibration; vision tools raise at call time (fail-closed, like piper).
    calib_path: str | None = None

    # Open-vocabulary detection server (GroundingDINO + SAM2). The session
    # builder spawns it as a sidecar when detector.spawn=True (piper-style).
    detector: DetectorServerConfig = field(default_factory=DetectorServerConfig)

    # --- grasp geometry (eye-to-hand projection → grasp/place z) ---
    # Constant base-frame Z correction added to every detection (piper: 0).
    z_correction_mm: float = 0.0
    # Offset from detected TOP to the grasp point (negative = below top).
    grasp_z_offset_mm: float = -25.0
    chip_thickness_mm: float = 75.0

    task_prompt: str | None = None
    name: str = "so101"

    # ----------------------------------------------------------------- loaders
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> So101Config:
        """Build from a flat dict OR an ``env.cfg.low_level`` nested shape.

        Accepts the legacy nested YAML layout (``env.cfg.low_level.*``) and the
        ``env.cfg.prompt`` override, mirroring how other adapters load config.
        """
        if isinstance(data.get("env"), dict):
            ll = data.get("env", {}).get("cfg", {}).get("low_level", {})
            prompt = data.get("env", {}).get("cfg", {}).get("prompt")
        else:
            ll = None
            prompt = None

        kw: dict[str, Any] = (
            {k: v for k, v in ll.items() if not k.startswith("_")} if isinstance(ll, dict) and ll else dict(data)
        )

        if "workspace_bounds" in kw and kw["workspace_bounds"] is not None:
            wb = kw["workspace_bounds"]
            if isinstance(wb, (list, tuple)) and len(wb) == 4:
                kw["workspace_bounds"] = tuple(float(v) for v in wb)
            else:
                raise ValueError(f"workspace_bounds must be a 4-element list/tuple, got {wb!r}")

        if "home_joints_deg" in kw and kw["home_joints_deg"] is not None:
            hj = kw["home_joints_deg"]
            if not isinstance(hj, (list, tuple)):
                raise ValueError(f"home_joints_deg must be a list, got {type(hj).__name__}")
            kw["home_joints_deg"] = [float(v) for v in hj]

        if "joint_limits" in kw and kw["joint_limits"] is not None:
            kw["joint_limits"] = _normalise_joint_limits(kw["joint_limits"])

        # ``api_servers`` is the unified detector config shape (same as piper):
        # a top-level (or ``env.cfg``) list whose ``_target_`` identifies the
        # GroundingDINO+SAM2 server entry. Extracted into ``cfg.detector``.
        env_cfg = data.get("env", {}).get("cfg", {}) if isinstance(data.get("env"), dict) else {}
        api_servers = data.get("api_servers") or env_cfg.get("api_servers") or []
        kw["detector"] = _extract_detector_from_api_servers(api_servers)

        if "camera_resolution" in kw and kw["camera_resolution"] is not None:
            cr = kw["camera_resolution"]
            if not isinstance(cr, (list, tuple)) or len(cr) != 2:
                raise ValueError(f"So101Config: camera_resolution must be a 2-element list, got {cr!r}.")
            kw["camera_resolution"] = (int(cr[0]), int(cr[1]))

        if "max_relative_target" in kw and kw["max_relative_target"] is not None:
            mrt = kw["max_relative_target"]
            if isinstance(mrt, dict):
                raise ValueError(
                    "So101Config: max_relative_target must be a float, not a dict. "
                    "LeRobot's ensure_safe_goal_position requires dict keys to match "
                    "the action keys exactly, which is incompatible with our split "
                    "arm/gripper actions. Set a single float (degrees per step)."
                )
            if not isinstance(mrt, (int, float)) or not _is_finite(float(mrt)):
                raise ValueError(f"So101Config: max_relative_target must be a finite number, got {mrt!r}.")
            kw["max_relative_target"] = float(mrt)

        if "settle_resend_period_s" in kw and kw["settle_resend_period_s"] is not None:
            srp = kw["settle_resend_period_s"]
            if not isinstance(srp, (int, float)) or not _is_finite(float(srp)) or float(srp) < 0.0:
                raise ValueError(
                    f"So101Config: settle_resend_period_s must be a non-negative finite number, got {srp!r}."
                )
            kw["settle_resend_period_s"] = float(srp)

        if "settle_drift_abort_samples" in kw and kw["settle_drift_abort_samples"] is not None:
            sda = kw["settle_drift_abort_samples"]
            if isinstance(sda, bool) or not isinstance(sda, int) or sda < 0:
                raise ValueError(f"So101Config: settle_drift_abort_samples must be a non-negative int, got {sda!r}.")
            kw["settle_drift_abort_samples"] = int(sda)

        if prompt is not None:
            kw["task_prompt"] = prompt

        valid = {f.name for f in dataclasses.fields(cls)}
        clean = {k: v for k, v in kw.items() if k in valid}
        return cls(**clean)

    @classmethod
    def from_yaml(cls, path: str | Path) -> So101Config:
        """Load config from a YAML file, resolving relative paths.

        ``urdf_path`` and ``calibration_dir`` are resolved unconditionally against
        the YAML file's directory (after ``~`` expansion). Existence is NOT
        required: LeRobot may legitimately create ``calibration_dir`` during
        ``lerobot-calibrate``, so resolving only-when-exists would leave a relative
        path to be re-resolved against the process cwd (or create the calibration
        directory in the wrong place). Absolute paths and ``~``-prefixed paths are
        passed through unchanged.
        """
        path = Path(path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls.from_dict(data)
        yaml_dir = path.parent
        if cfg.urdf_path:
            urdf = Path(cfg.urdf_path).expanduser()
            cfg.urdf_path = str(urdf if urdf.is_absolute() else (yaml_dir / urdf).resolve())
        if cfg.calibration_dir:
            calib = Path(cfg.calibration_dir).expanduser()
            cfg.calibration_dir = str(calib if calib.is_absolute() else (yaml_dir / calib).resolve())
        if cfg.calib_path:
            calib_p = Path(cfg.calib_path).expanduser()
            cfg.calib_path = str(calib_p if calib_p.is_absolute() else (yaml_dir / calib_p).resolve())
        return cfg

    def __post_init__(self) -> None:
        """Validate required fields, value finiteness, ordering and ranges."""
        # --- required fields ---
        if not self.port or not isinstance(self.port, str):
            raise ValueError("So101Config: 'port' is required (str), e.g. '/dev/ttyUSB0'.")
        if not isinstance(self.safety_validated, bool):
            raise ValueError(f"So101Config: safety_validated must be bool, got {type(self.safety_validated).__name__}.")

        if self.home_joints_deg is None:
            raise ValueError("So101Config: 'home_joints_deg' is required (5 floats, deg).")
        if not isinstance(self.home_joints_deg, (list, tuple)):
            raise ValueError(f"So101Config: home_joints_deg must be a list, got {type(self.home_joints_deg).__name__}.")
        if len(self.home_joints_deg) != len(ARM_JOINT_ORDER):
            raise ValueError(
                f"So101Config: home_joints_deg must have {len(ARM_JOINT_ORDER)} arm joints "
                f"({list(ARM_JOINT_ORDER)}), got {len(self.home_joints_deg)}."
            )
        for i, v in enumerate(self.home_joints_deg):
            if not _is_finite(v):
                raise ValueError(f"So101Config: home_joints_deg[{i}] is not finite: {v!r}.")
        # Freeze into a plain list of floats (drop tuple/numpy surprises).
        self.home_joints_deg = [float(v) for v in self.home_joints_deg]

        # --- joint_limits: exactly the 5 arm joints, by ARM_JOINT_ORDER ---
        if self.joint_limits is None:
            raise ValueError(
                f"So101Config: 'joint_limits' is required (dict over {list(ARM_JOINT_ORDER)}, (lo, hi) deg pairs)."
            )
        # __post_init__ also runs for direct construction (bypassing from_dict).
        self.joint_limits = _normalise_joint_limits(self.joint_limits)

        # --- workspace_bounds ---
        if self.workspace_bounds is not None:
            if len(self.workspace_bounds) != 4:
                raise ValueError(
                    f"So101Config: workspace_bounds must have 4 elements, got {len(self.workspace_bounds)}."
                )
            if not all(_is_finite(v) for v in self.workspace_bounds):
                raise ValueError(f"So101Config: workspace_bounds has non-finite value: {self.workspace_bounds!r}.")
            xmin, ymin, xmax, ymax = self.workspace_bounds
            if xmin > xmax or ymin > ymax:
                raise ValueError(
                    "So101Config: workspace_bounds must be ordered "
                    f"(xmin<=xmax, ymin<=ymax); got {self.workspace_bounds!r}."
                )
            self.workspace_bounds = (
                float(self.workspace_bounds[0]),
                float(self.workspace_bounds[1]),
                float(self.workspace_bounds[2]),
                float(self.workspace_bounds[3]),
            )

        # --- max_relative_target: float-only, validated + normalised on construction ---
        # The from_dict loader also normalises ints -> float and rejects dicts,
        # but direct So101Config(...) construction bypasses the loader, so we
        # must enforce the same contract here.
        #
        # LeRobot's ensure_safe_goal_position() does ``isinstance(mrt, float)``;
        # an int is NOT a float subclass, so an int value makes SOFollower raise
        # TypeError on the FIRST motion (after a successful connect). We must
        # normalise int -> float. A dict is forbidden because its keys must
        # exactly match the action keys — incompatible with our split arm/gripper
        # actions.
        mrt = self.max_relative_target
        if mrt is not None:
            if isinstance(mrt, dict):
                raise ValueError(
                    "So101Config: max_relative_target must be a float, not a dict. "
                    "LeRobot's ensure_safe_goal_position requires dict keys to match "
                    "the action keys exactly, which is incompatible with our split "
                    "arm/gripper actions. Set a single float (degrees per step)."
                )
            if isinstance(mrt, bool) or not isinstance(mrt, (int, float)):
                raise ValueError(f"So101Config: max_relative_target must be a number, got {type(mrt).__name__}.")
            if not math.isfinite(float(mrt)):
                raise ValueError(f"So101Config: max_relative_target must be finite, got {mrt!r}.")
            if float(mrt) <= 0:
                raise ValueError(f"So101Config: max_relative_target must be > 0, got {mrt}.")
            self.max_relative_target = float(mrt)

        # --- safety/IK finiteness ---
        for name, val in (
            ("z_min_safe_mm", self.z_min_safe_mm),
            ("trajectory_hz", self.trajectory_hz),
            ("max_joint_step_deg", self.max_joint_step_deg),
            ("servo_max_joint_step_deg", self.servo_max_joint_step_deg),
            ("servo_min_send_period_s", self.servo_min_send_period_s),
            ("servo_max_joint_vel_dps", self.servo_max_joint_vel_dps),
            ("joint_tolerance_deg", self.joint_tolerance_deg),
            ("settle_samples", self.settle_samples),
            ("move_timeout_s", self.move_timeout_s),
            ("ik_orientation_weight", self.ik_orientation_weight),
            ("ik_position_tolerance_mm", self.ik_position_tolerance_mm),
            ("cartesian_interp_step_mm", self.cartesian_interp_step_mm),
            ("pose_convergence_tolerance_mm", self.pose_convergence_tolerance_mm),
        ):
            if not _is_finite(val):
                raise ValueError(f"So101Config: {name} must be finite, got {val!r}.")
        if isinstance(self.settle_samples, float) or not isinstance(self.settle_samples, int):
            raise ValueError(f"So101Config: settle_samples must be int, got {self.settle_samples!r}.")
        if self.settle_samples < 1:
            raise ValueError(f"So101Config: settle_samples must be >= 1, got {self.settle_samples}.")
        if self.trajectory_hz <= 0:
            raise ValueError(f"So101Config: trajectory_hz must be > 0, got {self.trajectory_hz}.")
        if self.max_joint_step_deg <= 0:
            raise ValueError(f"So101Config: max_joint_step_deg must be > 0, got {self.max_joint_step_deg}.")
        if self.servo_max_joint_step_deg <= 0:
            raise ValueError(f"So101Config: servo_max_joint_step_deg must be > 0, got {self.servo_max_joint_step_deg}.")
        if self.servo_min_send_period_s <= 0:
            raise ValueError(f"So101Config: servo_min_send_period_s must be > 0, got {self.servo_min_send_period_s}.")
        if self.servo_max_joint_vel_dps <= 0:
            raise ValueError(f"So101Config: servo_max_joint_vel_dps must be > 0, got {self.servo_max_joint_vel_dps}.")
        if self.joint_tolerance_deg <= 0:
            raise ValueError(f"So101Config: joint_tolerance_deg must be > 0, got {self.joint_tolerance_deg}.")
        if self.move_timeout_s <= 0:
            raise ValueError(f"So101Config: move_timeout_s must be > 0, got {self.move_timeout_s}.")
        if self.ik_position_tolerance_mm <= 0:
            raise ValueError(f"So101Config: ik_position_tolerance_mm must be > 0, got {self.ik_position_tolerance_mm}.")
        if self.cartesian_interp_step_mm <= 0:
            raise ValueError(f"So101Config: cartesian_interp_step_mm must be > 0, got {self.cartesian_interp_step_mm}.")
        if not isinstance(self.cartesian_ease_in_out, bool):
            raise ValueError(f"So101Config: cartesian_ease_in_out must be bool, got {self.cartesian_ease_in_out!r}.")
        if not isinstance(self.settle_overcompensate, bool):
            raise ValueError(f"So101Config: settle_overcompensate must be bool, got {self.settle_overcompensate!r}.")
        if self.settle_overcompensate and self.max_relative_target is None:
            raise ValueError(
                "So101Config: settle_overcompensate=True requires a finite "
                "max_relative_target; disable settle_overcompensate when LeRobot "
                "relative-target clipping is intentionally disabled."
            )
        if isinstance(self.pose_convergence_max_iters, bool) or not isinstance(self.pose_convergence_max_iters, int):
            raise ValueError(
                "So101Config: pose_convergence_max_iters must be a non-negative "
                f"int, got {self.pose_convergence_max_iters!r}."
            )
        if self.pose_convergence_max_iters < 0:
            raise ValueError(
                f"So101Config: pose_convergence_max_iters must be >= 0, got {self.pose_convergence_max_iters}."
            )
        if self.pose_convergence_tolerance_mm <= 0:
            raise ValueError(
                f"So101Config: pose_convergence_tolerance_mm must be > 0, got {self.pose_convergence_tolerance_mm}."
            )
        if self.ik_orientation_weight < 0:
            raise ValueError(f"So101Config: ik_orientation_weight must be >= 0, got {self.ik_orientation_weight}.")
        if self.ik_orientation_tolerance_deg is not None and (
            not _is_finite(self.ik_orientation_tolerance_deg) or self.ik_orientation_tolerance_deg < 0
        ):
            raise ValueError(
                "So101Config: ik_orientation_tolerance_deg must be >= 0 "
                f"or None, got {self.ik_orientation_tolerance_deg!r}."
            )

        # --- gripper range [0, 100] + settle loop tunables ---
        for name, val in (
            ("gripper_open_pos", self.gripper_open_pos),
            ("gripper_close_pos", self.gripper_close_pos),
            ("gripper_tolerance", self.gripper_tolerance),
            ("gripper_timeout_s", self.gripper_timeout_s),
            ("gripper_settle_s", self.gripper_settle_s),
        ):
            if not _is_finite(val):
                raise ValueError(f"So101Config: {name} must be finite, got {val!r}.")
        if not (0.0 <= self.gripper_open_pos <= 100.0):
            raise ValueError(f"So101Config: gripper_open_pos must be in [0, 100], got {self.gripper_open_pos}.")
        if not (0.0 <= self.gripper_close_pos <= 100.0):
            raise ValueError(f"So101Config: gripper_close_pos must be in [0, 100], got {self.gripper_close_pos}.")
        if self.gripper_tolerance <= 0:
            raise ValueError(f"So101Config: gripper_tolerance must be > 0, got {self.gripper_tolerance}.")
        if self.gripper_timeout_s <= 0:
            raise ValueError(f"So101Config: gripper_timeout_s must be > 0, got {self.gripper_timeout_s}.")
        if self.gripper_settle_s < 0:
            raise ValueError(f"So101Config: gripper_settle_s must be >= 0, got {self.gripper_settle_s}.")

        # --- vision (milestone B): eye-to-hand camera + grasp geometry ---
        if self.camera_serial is not None and not isinstance(self.camera_serial, str):
            raise ValueError(
                f"So101Config: camera_serial must be str or None, got {type(self.camera_serial).__name__}."
            )
        if len(self.camera_resolution) != 2 or not all(isinstance(v, int) and v > 0 for v in self.camera_resolution):
            raise ValueError(
                f"So101Config: camera_resolution must be two positive ints, got {self.camera_resolution!r}."
            )
        if not isinstance(self.camera_fps, int) or self.camera_fps <= 0:
            raise ValueError(f"So101Config: camera_fps must be a positive int, got {self.camera_fps!r}.")
        for name, val in (
            ("z_correction_mm", self.z_correction_mm),
            ("grasp_z_offset_mm", self.grasp_z_offset_mm),
            ("chip_thickness_mm", self.chip_thickness_mm),
        ):
            if not _is_finite(val):
                raise ValueError(f"So101Config: {name} must be finite, got {val!r}.")

        # --- detector sidecar -------------------------------------------------
        if not isinstance(self.detector, DetectorServerConfig):
            self.detector = DetectorServerConfig.from_dict(self.detector)


def _normalise_joint_limits(raw: Any) -> dict[str, tuple[float, float]]:
    """Rebuild ``joint_limits`` over exactly ``ARM_JOINT_ORDER``.

    Rejects missing keys, extra keys, and unordered (lo, hi) pairs. Always
    returns a fresh dict keyed in ``ARM_JOINT_ORDER`` so SafetyRail's
    ``len(q) == len(names)`` check and the ``q[i]`` index labels stay stable.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"joint_limits must be a dict, got {type(raw).__name__}.")

    raw_keys = set(raw.keys())
    extra = raw_keys - _ARM_JOINT_SET
    missing = _ARM_JOINT_SET - raw_keys
    if extra or missing:
        raise ValueError(
            f"joint_limits keys must be exactly {list(ARM_JOINT_ORDER)}; "
            f"missing={sorted(missing)}, unexpected={sorted(extra)}."
        )

    normalised: dict[str, tuple[float, float]] = {}
    for name in ARM_JOINT_ORDER:
        val = raw[name]
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(f"joint_limits['{name}'] must be a (lo, hi) pair, got {val!r}.")
        lo, hi = val
        if not _is_finite(lo) or not _is_finite(hi):
            raise ValueError(f"joint_limits['{name}'] has non-finite bound: {val!r}.")
        lo_f, hi_f = float(lo), float(hi)
        if lo_f > hi_f:
            raise ValueError(f"joint_limits['{name}'] must be ordered (lo<=hi), got ({lo_f}, {hi_f}).")
        normalised[name] = (lo_f, hi_f)
    return normalised
