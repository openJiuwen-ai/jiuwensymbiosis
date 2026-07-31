# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""SO-101 low-level driver wrapping LeRobot 0.6.x ``SOFollower``.

Design contract (see ``.claude/plans/so101-adapter.md`` §A3):

- **Module import is cheap**: the constants below and the class definition do
  NOT import LeRobot. LeRobot is imported lazily inside :meth:`So101Driver.connect`
  so that ``import jiuwensymbiosis.adapters.so101`` works without the ``so101``
  extra installed (e.g. on a dev box without the hardware SDK).
- **One cleanup path**: :meth:`disconnect` and :meth:`close` share the same
  idempotent teardown — no two state machines.
- **Joints in degrees, gripper in 0..100 %**: native ``SOFollower`` units.
- **No ``ee.*`` on the command path**: we call ``RobotKinematics`` FK/IK and
  send ``{"shoulder_pan.pos": ...}`` motor targets; ``ee.x``/``ee.wx`` are a
  kinematic-processor intermediate format we never touch.
- **Reachable-or-reject**: non-finite values, out-of-soft-limit targets, IK
  residual over tolerance, and unreachable poses all raise ``ValueError``
  *before* the first ``send_action``; we never silently clamp.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from jiuwensymbiosis.adapters._common.lerobot_backend import LerobotKinematicsBackend
from jiuwensymbiosis.adapters.so101.geometry import (
    So101Pose,
    matrix_m_to_pose_mm_deg,
    orientation_error_deg,
    pose_mm_deg_to_matrix_m,
    position_error_mm,
)
from jiuwensymbiosis.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-only typing helpers
    from jiuwensymbiosis.adapters.so101.config import So101Config

__all__ = [
    "ARM_JOINT_ORDER",
    "MOTOR_ORDER",
    "So101CartesianServoError",
    "So101HardwareSendMismatch",
    "So101Driver",
    "So101PreDispatchError",
    "So101PoseConvergenceError",
]

_logger = get_logger(__name__)

# Cap on the number of SE(3) interpolation steps in Cartesian path planning. The
# planner splits start->target into N = ceil(max(translation_mm, rotation_deg) /
# cartesian_interp_step_mm) steps (one IK per step, seeded by the previous step's
# solution). A count cap — not a wall-clock timeout — keeps tests deterministic
# and bounds the worst-case work if a caller requests a huge Cartesian move.
_MAX_CARTESIAN_WAYPOINTS = 4096
# Servo-to-pose uses a bounded Cartesian progress search instead of clipping a
# full IK result joint-by-joint.  These limits are intentionally internal: the
# public tuning knobs remain the velocity/step caps in ``So101Config``.
_SERVO_ALPHA_SEARCH_ITERS = 14
_SERVO_ALPHA_MIN = 1.0 / 256.0
_SERVO_PROGRESS_EPS_MM = 1e-3
_SERVO_PROGRESS_EPS_DEG = 1e-3
# Treat a nominal control period as elapsed despite sub-microsecond floating
# point/scheduler jitter. Calls that are materially early remain throttled.
_SERVO_TIME_EPS_S = 1e-6
_GRIPPER_COMMAND_STEP = 5.0

# --------------------------------------------------------------------- constants
# Order is the LeRobot SO-101 feature naming (see SOFollower motor mapping).
# Kept at module top so ``config.py`` can import it without triggering LeRobot.
ARM_JOINT_ORDER: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
MOTOR_ORDER: tuple[str, ...] = (*ARM_JOINT_ORDER, "gripper")
_ARM_IDX = {name: i for i, name in enumerate(ARM_JOINT_ORDER)}


class So101PoseConvergenceError(RuntimeError):
    """The arm stopped in a safe state but did not reach a Cartesian target.

    This is deliberately distinct from a hardware/transport failure.  The
    command path has already rejected the unsafe compensation, so a recovery
    rail may report the failure without blindly homing a still-safe arm.
    """

    # RecoveryRail checks this opt-out marker without importing this hardware
    # module (which would make the generic rail depend on an adapter).
    skip_recovery = True

    def __init__(self, *, reason: str, residual_mm: float, tolerance_mm: float) -> None:
        self.reason = str(reason)
        self.residual_mm = float(residual_mm)
        self.tolerance_mm = float(tolerance_mm)
        super().__init__(
            f"SO-101 Cartesian target not reached: {self.reason}; "
            f"residual={self.residual_mm:.3f} mm > tolerance={self.tolerance_mm:.3f} mm."
        )


class So101CartesianServoError(ValueError):
    """Typed fail-closed Cartesian servo rejection."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {detail}")


class So101PreDispatchError(ValueError):
    """A motion request rejected before the first hardware command was sent."""

    skip_recovery = True


@dataclass
class _EndpointCompensationState:
    """Mutable joint-domain endpoint state shared by blocking and fast motion."""

    integral: np.ndarray
    last_command: np.ndarray
    previous_error: float
    drift_count: int = 0


@dataclass(frozen=True)
class _ServoSearchContext:
    """Immutable inputs shared by Cartesian servo candidate evaluations."""

    target: So101Pose
    target_matrix: np.ndarray
    planned_matrix: np.ndarray
    planned_q: np.ndarray
    planned_fk_pose: So101Pose
    max_step: float
    cartesian_step_cap: float
    alpha_limit: float
    goal_pos_tol: float
    goal_ori_tol: float | None
    planned_position_reached: bool
    planned_orientation_reached: bool
    planned_fk_pos_err: float
    planned_fk_ori_err: float


@dataclass(frozen=True)
class _ServoCandidate:
    """One evaluated Cartesian progress candidate."""

    q: np.ndarray | None
    code: str | None = None
    error: str | None = None


class _EndpointCompensationDrift(RuntimeError):
    """Internal signal mapped to each caller's existing public error type."""

    def __init__(self, *, count: int, previous_error: float, current_error: float) -> None:
        self.count = int(count)
        self.previous_error = float(previous_error)
        self.current_error = float(current_error)
        super().__init__(
            f"endpoint joint error grew {self.count} consecutive updates "
            f"({self.previous_error:.3f} -> {self.current_error:.3f} deg)"
        )


class _ZOnlyLiftUnavailable(RuntimeError):
    """A validated Z-only endpoint correction could not be generated."""


class So101HardwareSendMismatch(RuntimeError):
    """LeRobot changed a command already validated by the Jiuwen driver."""

    code = "hardware_send_mismatch"


# --------------------------------------------------------------------- helpers
def _is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _require_finite_so101_pose(pose: So101Pose, *, label: str) -> None:
    """Reject non-finite Cartesian pose values before any IK or dispatch."""
    for name, value in (
        ("x", pose.x),
        ("y", pose.y),
        ("z", pose.z),
        ("rx", pose.rx),
        ("ry", pose.ry),
        ("rz", pose.rz),
    ):
        if not _is_finite(value):
            raise ValueError(f"{label}: {name} must be finite, got {value!r}.")


def _arm_action(q: list[float] | np.ndarray) -> dict[str, float]:
    """Build an arm-only action dict ``{f"{j}.pos": q_i}`` (no ``gripper.pos``)."""
    if len(q) != len(ARM_JOINT_ORDER):
        raise ValueError(f"arm target must have {len(ARM_JOINT_ORDER)} joints, got {len(q)}.")
    return {f"{name}.pos": float(v) for name, v in zip(ARM_JOINT_ORDER, q, strict=True)}


def _interp_se3(start: np.ndarray, target: np.ndarray, t: float) -> np.ndarray:
    """Interpolate between two 4x4 SE(3) matrices at parameter ``t`` in [0, 1].

    Translation is linearly interpolated; rotation uses Slerp (via the relative
    rotation ``R_start.inv() @ R_target`` raised to the power ``t``). At ``t=0``
    returns ``start``; at ``t=1`` returns ``target`` (modulo float rounding).
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"interpolation parameter t must be in [0, 1], got {t}.")
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    if start.shape != (4, 4) or target.shape != (4, 4):
        raise ValueError(f"SE(3) matrices must be 4x4, got {start.shape} and {target.shape}.")

    r_start = Rotation.from_matrix(start[:3, :3])
    r_target = Rotation.from_matrix(target[:3, :3])
    # Relative rotation start->target, then Slerp by raising to power t.
    relative = r_start.inv() * r_target
    rvec = relative.as_rotvec()  # radians
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-12:
        # No rotation to interpolate (or near-identity); keep start orientation.
        r_interp = r_start
    else:
        axis = rvec / angle
        r_interp = r_start * Rotation.from_rotvec(axis * (angle * t))

    out = np.eye(4, dtype=float)
    out[:3, :3] = r_interp.as_matrix()
    out[:3, 3] = start[:3, 3] * (1.0 - t) + target[:3, 3] * t
    return out


def _lerobot_version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in version.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts) or (0,)


# --------------------------------------------------------------------- driver
class So101Driver:
    """Wraps ``SOFollower`` for the jiuwensymbiosis env/api contract.

    Satisfies the ``RobotDriver`` + ``JointDriver`` + ``GripperDriver`` protocols
    (see ``jiuwensymbiosis/env/protocol.py``). Construction is cheap and does not
    open the serial port; that happens in :meth:`connect`.
    """

    def __init__(
        self,
        cfg: So101Config,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] | None = None,
        so_follower_factory: Callable[..., Any] | None = None,
        kinematics_factory: Callable[..., Any] | None = None,
        lerobot_import: Callable[[], tuple[Any, Any, Any, str]] | None = None,
    ) -> None:
        from jiuwensymbiosis.adapters.so101.config import So101Config  # noqa: F401

        if not isinstance(cfg, So101Config):
            raise TypeError(f"cfg must be a So101Config, got {type(cfg).__name__}.")
        self._cfg = cfg
        self._sleep = sleep
        self._monotonic = monotonic or time.monotonic
        self._so_follower_factory = so_follower_factory
        self._kinematics_factory = kinematics_factory
        # Injection point for the LeRobot import tuple (SOFollower,
        # SOFollowerRobotConfig, RobotKinematics, version). ``None`` (default)
        # runs the real :meth:`_import_lerobot` so production behavior is
        # unchanged; tests inject a fake to exercise connect()'s control flow
        # without the optional ``so101`` extra installed.
        self._lerobot_import = lerobot_import

        # Hardware handles — populated by connect().
        self._robot: Any = None
        self._kin: Any = None
        self._connected: bool = False

        # Vision (milestone B): desktop-fixed eye-to-hand RealSense + hand-eye
        # calibration. The camera is NOT wrist-mounted, so ``tf_base_cam`` is a
        # constant (camera-in-base), and projection does NOT read the flange per
        # step (unlike piper's eye-in-hand ``tf_base_flange @ tf_flange_cam``).
        self._camera: Any = None  # RealSenseCamera once started
        self._calib: dict[str, Any] | None = None
        self._tf_base_cam: np.ndarray | None = None

        # Last action actually dispatched (LeRobot may clip it via
        # ``max_relative_target``). Recorded per plan §A3 so clipping is
        # observable — completion is still judged from real observation.
        self._last_sent_action: dict[str, float] | None = None
        self._last_gripper_result: dict[str, Any] | None = None
        self._last_motion_result: dict[str, Any] | None = None
        self._holding_payload = False

        # Real-time servo velocity gate: monotonic timestamp of the last
        # dispatched servo action. ``servo_to_pose`` enforces a minimum
        # inter-send interval and a deg/s cap derived from the real ``dt``,
        # so actual joint velocity is independent of the caller's tick rate.
        self._servo_last_send_t: float = 0.0
        # Streaming Cartesian plan state. The first servo command initializes
        # these from live FK/joints; each successful command advances them to
        # the accepted Cartesian waypoint and IK solution. Later ticks plan
        # from this command state instead of re-anchoring to encoder lag.
        self._servo_planned_matrix: np.ndarray | None = None
        self._servo_planned_q: np.ndarray | None = None
        # Endpoint-only software integral state. It is deliberately separate
        # from the Cartesian plan/IK seed: compensation changes only the motor
        # command used to make live joints reach ``_servo_planned_q``.
        self._servo_endpoint_state: _EndpointCompensationState | None = None

        # URDF resolution: explicit > packaged.
        self._urdf_path: str = self._resolve_urdf_path()

    # --- RobotDriver Protocol: required properties ---------------------------
    @property
    def home_pose(self) -> So101Pose:
        """FK(home_joints_deg) -> control-frame pose. Read-only report, no motion."""
        if self._kin is None:
            raise RuntimeError("So101Driver.home_pose called before connect().")
        matrix = self._kin.forward_kinematics(np.asarray(self._cfg.home_joints_deg, dtype=float))
        return matrix_m_to_pose_mm_deg(np.asarray(matrix, dtype=float))

    @property
    def z_min_safe(self) -> float:
        """Tip/control-frame Z floor in mm (config-driven)."""
        return float(self._cfg.z_min_safe_mm)

    @property
    def z_max_safe(self) -> float | None:
        """Tip/control-frame Z ceiling in mm (config-driven), or None if unset."""
        zmax = getattr(self._cfg, "z_max_safe_mm", None)
        return float(zmax) if zmax is not None else None

    @property
    def flange_z_min_safe(self) -> float:
        """Flange-frame Z floor in mm. Milestone A mirrors ``z_min_safe``."""
        return float(self._cfg.z_min_safe_mm)

    @property
    def tool_offset_mm(self) -> float:
        """Fixed 0.0 for milestone A (see plan §Decision 3)."""
        return 0.0

    # --- connection / cleanup (idempotent, one path) ------------------------
    def connect(self) -> None:
        """Open the serial port and validate calibration + kinematics.

        Follows the 9-step sequence in §A3. Any failure runs the idempotent
        cleanup so the driver is left disconnected.
        """
        if self._connected:
            return
        # Fail-closed: home/limits ship as unverified placeholders; refuse to open
        # hardware until the operator sets safety_validated: true after confirming
        # a safe home and tightened limits on the real robot. Runs before any serial
        # open so no torque is ever applied with an unvalidated config.
        # home_use_init_pose=True implicitly bypasses this gate (connecting is the
        # only way to read the current joints that become home), but the startup
        # pose is still checked against joint_limits in Step 8 below.
        if not self._cfg.safety_validated and not self._cfg.home_use_init_pose:
            raise RuntimeError(
                "SO-101 connect refused: config not safety-validated. "
                "home_joints_deg / joint_limits ship as UNVERIFIED placeholders. "
                "Manually confirm a safe home (teach pendant) and tighten joint_limits "
                "to measured safe ranges, then set `safety_validated: true` in the YAML, "
                "or set `home_use_init_pose: true` to use the startup pose as home."
            )
        # `robot` is the live follower once constructed; if a later validation
        # step (action_features, kinematics build, FK/home checks) fails BEFORE
        # the handle is assigned to self._robot, we must still tear down the
        # already-open serial bus — otherwise the port/torque stay open.
        robot: Any = None
        try:
            import_fn = self._lerobot_import or So101Driver._import_lerobot
            SOFollower, SOFollowerRobotConfig, RobotKinematics, lerobot_version = import_fn()

            # Step 2: build config with the correct SOFollowerRobotConfig class.
            robot_cfg = SOFollowerRobotConfig(
                port=self._cfg.port,
                id=self._cfg.robot_id,
                calibration_dir=(Path(self._cfg.calibration_dir) if self._cfg.calibration_dir else None),
                use_degrees=True,
                disable_torque_on_disconnect=self._cfg.disable_torque_on_disconnect,
                max_relative_target=self._cfg.max_relative_target,
                cameras={},
            )
            follower_factory = self._so_follower_factory or SOFollower
            robot = follower_factory(robot_cfg)

            # Step 3: calibration file preload (serial not yet open).
            calib_path = getattr(robot, "calibration_fpath", None)
            if not calib_path or not Path(calib_path).is_file():
                raise RuntimeError(
                    f"SO-101 calibration file not found: {calib_path}. "
                    f"Run `lerobot-calibrate --robot.id={self._cfg.robot_id}` first."
                )

            # Step 4: open the bus, no interactive calibration.
            robot.connect(calibrate=False)

            # Step 5: confirm calibration is available.
            if not getattr(robot, "is_calibrated", False):
                self._teardown(robot)
                raise RuntimeError(
                    "SO-101 is not calibrated after connect(calibrate=False). "
                    "Run `lerobot-calibrate --robot.id=" + self._cfg.robot_id + "` first."
                )

            # Step 6: validate action_features keys carry the .pos suffix.
            expected = {f"{name}.pos" for name in MOTOR_ORDER}
            actual = set(getattr(robot, "action_features", {}).keys())
            missing = expected - actual
            if missing:
                self._teardown(robot)
                raise RuntimeError(f"SOFollower action_features missing: {sorted(missing)}.")

            # Step 7: build the FK/IK backend over RobotKinematics (target_frame
            # from config). The reusable _common backend forwards FK/IK to the
            # solver; kinematics_factory injects the solver class (a fake in tests,
            # or a pre-imported RobotKinematics).
            kin = LerobotKinematicsBackend(
                self._urdf_path,
                target_frame_name=self._cfg.ik_target_frame,
                joint_names=list(ARM_JOINT_ORDER),
                robot_kinematics_cls=self._kinematics_factory or RobotKinematics,
            )

            # Step 8: validate current FK + home FK/limits.
            current = self._read_arm_angles(robot)
            _ = kin.forward_kinematics(np.asarray(current, dtype=float))  # raises if bad frame
            if self._cfg.home_use_init_pose:
                # Use the startup joint pose as home (mirrors piper's
                # home_use_init_pose): overwrite the runtime home_joints_deg so
                # home_pose / home() / SafetyRail all read the live pose. The
                # current angles are STILL checked against joint_limits right
                # below, so an operator who parked the arm outside its soft
                # limits is refused here rather than trusting an illegal pose.
                init_home = [float(v) for v in current]
                self._cfg.home_joints_deg = init_home
                _logger.info("[SO-101] home_use_init_pose=True; home=%s", init_home)
            home = np.asarray(self._cfg.home_joints_deg, dtype=float)
            _ = kin.forward_kinematics(home)
            self._check_joint_limits(home, label="home_joints_deg")

            self._robot = robot
            self._kin = kin
            self._connected = True
            self._reset_servo_plan()
            _logger.info("[SO-101] motion config: %s", self._cfg.motion_summary())

            # Vision (milestone B): when a camera is configured, opening it is
            # part of the connection contract. Agent tools are commonly built
            # before session.connect(), so silently degrading here would leave
            # already-emitted vision tools active. Calibration remains optional
            # and fail-closed at vision-call time.
            self._start_camera()
            self._load_calibration()
        except Exception:
            # Tear down the live follower even if it was never assigned to
            # self._robot (e.g. kinematics/FK/home check failed after the bus
            # opened). self._robot may still be None here.
            self._teardown(robot)
            self._robot = None
            self._kin = None
            self._connected = False
            raise

    # --- vision (milestone B): eye-to-hand camera + calibration --------------
    def _start_camera(self) -> None:
        """Start the desktop RealSense if ``camera_serial`` is configured.

        ``camera_serial=None`` explicitly disables vision. If a serial is
        configured, failure to open it raises so a session cannot continue with
        vision tools that were emitted before ``connect()``.
        """
        if self._camera is not None:
            return
        serial = getattr(self._cfg, "camera_serial", None)
        if not serial:
            return
        from jiuwensymbiosis.perception.camera import RealSenseCamera

        rw, rh = self._cfg.camera_resolution
        cam = RealSenseCamera(
            serial=serial,
            resolution=(int(rw), int(rh)),
            fps=int(self._cfg.camera_fps),
            log_prefix="[SO-101 vision]",
        )
        if not cam.start():
            raise RuntimeError(f"SO-101: configured camera {serial!r} failed to start.")
        self._camera = cam

    def _load_calibration(self) -> None:
        """Load the eye-to-hand hand-eye calibration (``T_base_cam``).

        Best-effort: a missing/malformed file logs a warning and leaves
        ``tf_base_cam`` None (vision tools raise at call, fail-closed like piper).
        """
        calib_path = getattr(self._cfg, "calib_path", None)
        if not calib_path:
            return
        from jiuwensymbiosis.perception.calibration import LegacyCalibrationError, load_calibration

        try:
            calib = load_calibration(
                calib_path,
                frame_field="T_base_cam",
                legacy_field="T_base_cam_legacy",
                env_var="JIUWEN_SO101_ALLOW_LEGACY_CALIB",
            )
        except (LegacyCalibrationError, ValueError, OSError) as exc:
            _logger.warning("SO-101: calibration load failed (%s); vision tools will raise at call.", exc)
            return
        self._calib = calib
        self._tf_base_cam = calib["T_base_cam"]["matrix_4x4"]
        _logger.info("SO-101: loaded eye-to-hand calibration from %s (T_base_cam).", calib_path)

    @property
    def tf_base_cam(self) -> np.ndarray | None:
        """4x4 eye-to-hand transform (camera-in-base, CONSTANT). None if uncalibrated."""
        return self._tf_base_cam

    @property
    def calibration(self) -> dict[str, Any] | None:
        """Loaded hand-eye calibration payload, or None."""
        return self._calib

    @property
    def has_calibration(self) -> bool:
        """True if a calibration JSON was loaded."""
        return self._calib is not None

    @property
    def intrinsics(self) -> np.ndarray | None:
        """3x3 camera intrinsics K from the live camera, or None."""
        return self._camera.intrinsics if self._camera is not None else None

    @property
    def camera_available(self) -> bool:
        """Whether the configured desktop camera is currently streaming."""
        return self._camera is not None

    def grab_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Grab one aligned (rgb, depth_m) pair from the desktop camera, or None."""
        return self._camera.grab_frames() if self._camera is not None else None

    def disconnect(self) -> None:
        """Idempotent teardown entry."""
        self._teardown(self._robot)
        self._robot = None
        self._kin = None
        self._connected = False
        self._reset_servo_plan()

    def close(self) -> None:
        """Alias of :meth:`disconnect` for callers expecting ``close()``."""
        self.disconnect()

    def _teardown(self, robot: Any) -> None:
        """Best-effort, idempotent cleanup safe for None / partial / repeat."""
        # Vision: stop the desktop camera (independent of the arm).
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception as exc:  # noqa: BLE001 - best-effort
                _logger.debug("SO-101 teardown: camera.stop() failed: %s", exc)
            self._camera = None
        if robot is None:
            return
        try:
            if self._cfg.disable_torque_on_disconnect:
                # SOFollower disconnect already disables torque; keep the flag's
                # intent honoured when a custom factory overrides the behavior.
                pass
            getattr(robot, "disconnect", lambda: None)()
        except Exception as exc:
            # Teardown must never raise; callers rely on idempotent close.
            _logger.debug("SO-101 teardown: robot.disconnect() failed: %s", exc)

    # --- JointDriver / GripperDriver / observation --------------------------
    def get_angles(self) -> list[float]:
        """Return the 5 arm joint angles in ``ARM_JOINT_ORDER`` (degrees)."""
        self._require_connected()
        return self._read_arm_angles(self._robot)

    def forward_kinematics_mm(self, joints_deg: list[float] | np.ndarray) -> np.ndarray:
        """Return flange FK as a finite 4x4 transform with translation in mm."""
        self._require_connected()
        joint_array = np.asarray(joints_deg, dtype=float)
        self._validate_joint_vector(joint_array.tolist(), label="forward_kinematics_mm joints")
        transform = np.asarray(self._kin.forward_kinematics(joint_array), dtype=float).copy()
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise RuntimeError(
                "forward_kinematics_mm produced an invalid transform: "
                f"shape={transform.shape}, finite={bool(np.all(np.isfinite(transform)))}."
            )
        transform[:3, 3] *= 1000.0
        return transform

    def disable_arm_torque(self) -> None:
        """Disable the five arm joints while leaving gripper torque unchanged."""
        self._require_connected()
        self._robot.bus.disable_torque(list(ARM_JOINT_ORDER))

    def enable_arm_torque(self) -> None:
        """Enable the five arm joints while leaving gripper torque unchanged."""
        self._require_connected()
        self._robot.bus.enable_torque(list(ARM_JOINT_ORDER))

    def preset_current_joint_goal(self) -> None:
        """Write the observed arm joints as goals without enabling torque."""
        self._require_connected()
        joint_array = np.asarray(self.get_angles(), dtype=float)
        self._validate_joint_vector(joint_array.tolist(), label="current arm joints")
        requested = _arm_action(joint_array)
        actual = self._send_action(requested)
        self._require_action_match(requested, actual, label="preset current joint goal")

    def restore_all_torque(self) -> None:
        """Enable torque for every motor, including the gripper."""
        self._require_connected()
        self._robot.bus.enable_torque()

    def get_gripper_position(self) -> float:
        """Return the gripper target in 0..100 % (native SO-101 units)."""
        self._require_connected()
        obs = self._robot.get_observation()
        return self._read_motor(obs, "gripper")

    @property
    def last_gripper_result(self) -> dict[str, Any] | None:
        """Last completed gripper outcome, copied to prevent external mutation."""
        return dict(self._last_gripper_result) if self._last_gripper_result is not None else None

    @property
    def last_motion_result(self) -> dict[str, Any] | None:
        """Last endpoint-settle outcome, copied to prevent external mutation."""
        return copy.deepcopy(self._last_motion_result)

    @property
    def holding_payload(self) -> bool:
        """Whether the last successful gripper close indicates a held object."""
        return bool(self._holding_payload)

    def get_pose(self) -> So101Pose:
        """FK(current 5 joints) -> control-frame pose (mm / XYZ-Euler deg)."""
        self._require_connected()
        q = np.asarray(self.get_angles(), dtype=float)
        matrix = self._kin.forward_kinematics(q)
        return matrix_m_to_pose_mm_deg(np.asarray(matrix, dtype=float))

    def set_gripper(self, on: bool) -> None:
        """Drive the gripper to the configured two-state target, blocking until settled.

        Sends ONLY ``{"gripper.pos": target}`` — never any arm joint keys — so an
        in-flight arm motion is not disturbed. The driver bounds each gripper
        command itself and polls the real observation until it converges within
        ``gripper_tolerance`` for ``settle_samples`` consecutive reads, bounded by
        ``gripper_timeout_s``. During CLOSE only, meaningful travel followed by
        consecutive stable observations away from the full-close target is
        accepted as object contact. A gripper that never moves still times out,
        and OPEN always has to reach its target. Completion is judged from real
        observation, not the ``send_action`` return; the actual (possibly clipped)
        target is recorded each send. Each poll waits one ``trajectory_hz`` period
        so the loop never saturates the serial bus. After convergence/contact an
        optional ``gripper_settle_s`` dwell is waited via the injectable sleep.
        """
        self._require_connected()
        target = float(self._cfg.gripper_close_pos if on else self._cfg.gripper_open_pos)
        self._last_gripper_result = None
        deadline = self._monotonic() + float(self._cfg.gripper_timeout_s)
        settle_needed = max(1, int(self._cfg.settle_samples))
        start = float(self.get_gripper_position())
        last_observed = start
        close_direction = 1.0 if target > start else -1.0
        contact_window: list[float] = []
        contact_position: float | None = None
        # One period between polls so the loop never hammers the serial bus at full
        # speed (reuses the arm trajectory_hz for a consistent dispatch rate).
        period = 1.0 / float(self._cfg.trajectory_hz) if self._cfg.trajectory_hz > 0 else 0.0
        while True:
            if self._monotonic() > deadline:
                self._last_gripper_result = {
                    "ok": False,
                    "state": "timeout",
                    "start": start,
                    "position": last_observed,
                    "target": target,
                }
                raise TimeoutError(
                    "SO-101 gripper did not settle within the gripper timeout "
                    f"({self._cfg.gripper_timeout_s}s): start={start:.3f}, "
                    f"observed={last_observed:.3f}, target={target:.3f}."
                )
            command_delta = float(np.clip(target - last_observed, -_GRIPPER_COMMAND_STEP, _GRIPPER_COMMAND_STEP))
            requested = {"gripper.pos": last_observed + command_delta}
            self._send_action(requested)
            observed = float(self.get_gripper_position())
            last_observed = observed
            if abs(observed - target) <= self._cfg.gripper_tolerance:
                settle_needed -= 1
                if settle_needed <= 0:
                    break
            else:
                settle_needed = max(1, int(self._cfg.settle_samples))

            # Contact inference is intentionally conservative and CLOSE-only:
            # 1) movement must be toward the requested close target;
            # 2) it must exceed a configured minimum, excluding stale/no-motion
            #    observations and immediate mechanical jams;
            # 3) it must then remain stable for consecutive samples while still
            #    outside the normal target tolerance.
            directed_progress = (observed - start) * close_direction
            if (
                on
                and abs(observed - target) > self._cfg.gripper_tolerance
                and directed_progress >= self._cfg.gripper_contact_min_travel
            ):
                contact_window.append(observed)
                contact_window = contact_window[-self._cfg.gripper_contact_stall_samples:]  # fmt: skip
                if (
                    len(contact_window) >= self._cfg.gripper_contact_stall_samples
                    and max(contact_window) - min(contact_window) <= self._cfg.gripper_contact_stall_tolerance
                ):
                    _logger.info(
                        "[SO-101] gripper contact inferred: start=%.3f observed=%.3f "
                        "target=%.3f progress=%.3f stable_samples=%d",
                        start,
                        observed,
                        target,
                        directed_progress,
                        len(contact_window),
                    )
                    contact_position = observed
                    break
            else:
                contact_window.clear()
            if period > 0:
                self._sleep(period)

        if contact_position is not None:
            remaining = abs(target - contact_position)
            preload = min(float(self._cfg.gripper_contact_hold_offset), remaining)
            hold_target = contact_position + close_direction * preload
            requested = {"gripper.pos": hold_target}
            self._send_action(requested)
            self._last_gripper_result = {
                "ok": True,
                "state": "contact",
                "start": start,
                "position": contact_position,
                "target": target,
                "hold_target": hold_target,
                "travel": (contact_position - start) * close_direction,
            }
            _logger.info(
                "[SO-101] gripper contact hold: position=%.3f hold_target=%.3f full_close_target=%.3f preload=%.3f",
                contact_position,
                hold_target,
                target,
                preload,
            )
            self._holding_payload = True
        else:
            self._last_gripper_result = {
                "ok": True,
                "state": "closed" if on else "open",
                "start": start,
                "position": last_observed,
                "target": target,
                "hold_target": target,
                "travel": (last_observed - start) * close_direction,
            }
            # A full close without an early contact stall means empty space.
            # Only the contact branch above is evidence of a held payload.
            self._holding_payload = False
        if self._cfg.gripper_settle_s > 0:
            self._sleep(float(self._cfg.gripper_settle_s))

    def home(self) -> None:
        """Return home through a fully FK-validated joint path."""
        self.safe_retreat_home(mode="home")

    def retreat_home(self) -> None:
        """Return home while retaining payload/table clearance checks."""
        self.safe_retreat_home(mode="payload")

    def recovery_home(self) -> None:
        """Return home after recovery using the same FK-validated planner."""
        self.safe_retreat_home(mode="recovery")

    def safe_retreat_home(self, *, mode: str) -> None:
        """Execute a direct joint-home path after validating every waypoint's FK.

        Joint interpolation is planned in full and every waypoint is checked
        against joint limits, Cartesian bounds, and (when configured) the table
        plus gripper/payload clearance before the first command is sent. Runtime
        encoder observations pass through the same checks while tracking.
        """
        self._require_connected()
        self._reset_servo_plan()
        start_q = np.asarray(self.get_angles(), dtype=float)
        target_q = np.asarray(self._cfg.home_joints_deg, dtype=float)
        waypoints = self._joint_waypoints(start_q, target_q)
        for index, waypoint in enumerate(waypoints, start=1):
            self._validate_joint_waypoint(
                waypoint,
                label=f"{mode}-home waypoint {index}/{len(waypoints)}",
            )
        self._dispatch_prevalidated_waypoints(
            waypoints,
            target_q,
            timeout_s=None,
        )

    def move_joint_blocking(
        self,
        q: list[float],
        *,
        timeout_s: float | None = None,
    ) -> None:
        """Interpolate in joint space to ``q`` and block until settled.

        Interpolation (§A3): ``steps = ceil(max|Δ| / max_joint_step_deg)`` (>=1),
        linear ``alpha_k = k/steps``. ALL waypoints are validated (finite +
        soft limits + FK Cartesian bounds) before the first ``send_action``;
        any failure raises ``ValueError`` and sends nothing. The settle loop
        re-sends the final target when the last waypoint was clipped, judging
        completion from real observation (not from the ``send_action`` return
        value).

        The dispatch + settle loop is shared with :meth:`move_to_pose_blocking`
        via :meth:`_dispatch_prevalidated_waypoints` so the dispatched path is
        exactly the pre-validated path (no re-read of the start, no divergent
        re-interpolation).
        """
        self._require_connected()
        self._reset_servo_plan()
        try:
            self._validate_joint_vector(q, label="move_joint_blocking target")
            self._check_joint_limits(np.asarray(q, dtype=float), label="move_joint_blocking target")
        except ValueError as exc:
            raise So101PreDispatchError(str(exc)) from exc

        current = np.asarray(self.get_angles(), dtype=float)
        target = np.asarray(q, dtype=float)
        try:
            waypoints = self._joint_waypoints(current, target)

            # Pre-validate every waypoint before issuing the first action.
            # Joint limits alone are insufficient: FK can put a legal joint
            # vector below the Z floor or outside the configured XY work area.
            for index, wp in enumerate(waypoints, start=1):
                self._validate_joint_waypoint(wp, label=f"joint waypoint {index}/{len(waypoints)}")
        except ValueError as exc:
            raise So101PreDispatchError(str(exc)) from exc

        self._dispatch_prevalidated_waypoints(waypoints, target, timeout_s=timeout_s)

    def _send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Dispatch ``action`` and record the *actual* target LeRobot applied.

        ``SOFollower.send_action`` returns the action actually sent to the motors
        (potentially clipped by ``max_relative_target``). Plan §A3 requires this
        be recorded so clipping is observable; completion is still judged from real
        observation (:meth:`_dispatch_prevalidated_waypoints` settle loop), not
        from this return value.
        """
        actual = self._robot.send_action(action)
        actual = dict(actual) if actual is not None else dict(action)
        self._last_sent_action = actual
        # Surface clipping at DEBUG so a silent clamp to a different pose is
        # traceable without rerunning with hardware.
        for key, req in action.items():
            act = actual.get(key)
            if act is None or not _is_finite(float(act)) or not _is_finite(float(req)):
                continue
            if abs(float(req) - float(act)) > 1e-9:
                _logger.debug("so101 send_action: %s clipped requested=%.6f actual=%.6f", key, req, act)
        return actual

    @staticmethod
    def _require_action_match(requested: dict[str, float], actual: dict[str, float], *, label: str) -> None:
        mismatches: list[str] = []
        for key, requested_value in requested.items():
            actual_value = actual.get(key)
            if actual_value is None or not _is_finite(actual_value):
                mismatches.append(f"{key}=missing")
            elif abs(float(actual_value) - float(requested_value)) > 1e-6:
                mismatches.append(f"{key} requested={requested_value:.4f} returned={float(actual_value):.4f}")
        if mismatches:
            raise So101HardwareSendMismatch(f"{label}: LeRobot modified validated action: {'; '.join(mismatches)}")

    def _settle_metrics(
        self,
        target: np.ndarray,
        actual: np.ndarray,
        *,
        cartesian_target_matrix: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float | None, float | None]:
        """Return joint errors plus actual-vs-target FK position/Z errors."""
        joint_errors = np.asarray(actual, dtype=float) - np.asarray(target, dtype=float)
        try:
            target_matrix = (
                np.asarray(self._kin.forward_kinematics(target), dtype=float)
                if cartesian_target_matrix is None
                else np.asarray(cartesian_target_matrix, dtype=float)
            )
            actual_matrix = np.asarray(self._kin.forward_kinematics(actual), dtype=float)
            if target_matrix.shape != (4, 4) or actual_matrix.shape != (4, 4):
                raise ValueError(f"unexpected FK shapes target={target_matrix.shape}, actual={actual_matrix.shape}")
            delta_mm = (actual_matrix[:3, 3] - target_matrix[:3, 3]) * 1000.0
            if not np.all(np.isfinite(delta_mm)):
                raise ValueError(f"non-finite FK delta {delta_mm!r}")
            return joint_errors, float(np.linalg.norm(delta_mm)), float(delta_mm[2])
        except Exception as exc:  # noqa: BLE001 - metrics must not mask the primary motion failure
            _logger.debug("SO-101 settle FK metrics unavailable: %s", exc)
            return joint_errors, None, None

    def _record_settle_result(
        self,
        classification: str,
        target: np.ndarray,
        actual: np.ndarray,
        *,
        ok: bool,
        cartesian_target_matrix: np.ndarray | None = None,
    ) -> tuple[float | None, float | None]:
        """Persist and log a strict/soft/hard endpoint-settle classification."""
        joint_errors, cartesian_error_mm, z_error_mm = self._settle_metrics(
            target,
            actual,
            cartesian_target_matrix=cartesian_target_matrix,
        )
        max_error_index = int(np.argmax(np.abs(joint_errors)))
        joint_details = "; ".join(
            (f"{name}(target={target[index]:.3f}, actual={actual[index]:.3f}, error={joint_errors[index]:+.3f})")
            for index, name in enumerate(ARM_JOINT_ORDER)
        )
        self._last_motion_result = {
            "ok": bool(ok),
            "classification": classification,
            "strict_tolerance_deg": float(self._cfg.joint_tolerance_deg),
            "soft_tolerance_deg": float(self._cfg.settle_soft_tolerance_deg),
            "max_z_undershoot_mm": float(self._cfg.settle_max_z_undershoot_mm),
            "z_requirement_met": (
                z_error_mm is not None and z_error_mm >= -float(self._cfg.settle_max_z_undershoot_mm) - 1e-6
            ),
            "max_joint": ARM_JOINT_ORDER[max_error_index],
            "max_abs_joint_error_deg": abs(float(joint_errors[max_error_index])),
            "joint_errors_deg": {name: float(joint_errors[index]) for index, name in enumerate(ARM_JOINT_ORDER)},
            "cartesian_position_error_mm": cartesian_error_mm,
            "cartesian_z_error_mm": z_error_mm,
        }
        log_level = logging.WARNING if classification != "strict" else logging.DEBUG
        display_classification = {
            "hard_timeout": "timeout (hard)",
            "hard_drift": "drift (hard)",
        }.get(classification, classification)
        _logger.log(
            log_level,
            "SO-101 final settle %s: strict=%.3f deg, soft=%.3f deg, "
            "max_joint=%s, max_abs_error=%.3f deg, cartesian_error=%s mm, z_error=%s mm; %s",
            display_classification,
            self._cfg.joint_tolerance_deg,
            self._cfg.settle_soft_tolerance_deg,
            ARM_JOINT_ORDER[max_error_index],
            abs(float(joint_errors[max_error_index])),
            "unavailable" if cartesian_error_mm is None else f"{cartesian_error_mm:.3f}",
            "unavailable" if z_error_mm is None else f"{z_error_mm:+.3f}",
            joint_details,
        )
        return cartesian_error_mm, z_error_mm

    def _next_endpoint_command(
        self,
        target_q: np.ndarray,
        actual_q: np.ndarray,
        state: _EndpointCompensationState,
        *,
        max_step_deg: float,
        integral_limit_deg: float,
        drift_abort_samples: int,
        context: str,
        max_command_offset_deg: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Return the next bounded endpoint command for blocking or fast motion.

        This is the shared SO-101 endpoint-control primitive. Callers retain
        their own lifecycle and success semantics; this method only owns drift
        observation, bounded integral compensation, step limiting, safety
        validation, and fail-closed fallback to the bare joint target.

        ``state.last_command`` must be updated by the caller from the action
        actually accepted by LeRobot after dispatch.
        """
        target = np.asarray(target_q, dtype=float)
        actual = np.asarray(actual_q, dtype=float)
        if target.shape != actual.shape or target.shape != state.integral.shape:
            raise ValueError(
                f"{context}: endpoint state shape mismatch: "
                f"target={target.shape}, actual={actual.shape}, integral={state.integral.shape}."
            )
        if state.last_command.shape != target.shape:
            raise ValueError(
                f"{context}: last command shape {state.last_command.shape} does not match target {target.shape}."
            )

        error_deg = float(np.max(np.abs(actual - target)))
        drift_cap = max(0, int(drift_abort_samples))
        previous_error = float(state.previous_error)
        if drift_cap > 0:
            if error_deg > previous_error + 1e-6:
                state.drift_count += 1
                if state.drift_count >= drift_cap:
                    raise _EndpointCompensationDrift(
                        count=state.drift_count,
                        previous_error=previous_error,
                        current_error=error_deg,
                    )
            else:
                state.drift_count = 0
        state.previous_error = error_deg

        command = target.copy()
        if not self._cfg.settle_overcompensate:
            return command, error_deg

        gain = float(self._cfg.settle_gain)
        effective_integral_limit = float(integral_limit_deg)
        if max_command_offset_deg is not None:
            effective_integral_limit = min(
                effective_integral_limit,
                float(max_command_offset_deg) / gain,
            )
        state.integral += target - actual
        state.integral = np.clip(
            state.integral,
            -effective_integral_limit,
            effective_integral_limit,
        )
        desired = target + gain * state.integral
        try:
            self._validate_joint_waypoint(
                desired,
                label=f"{context} over-compensate desired",
            )
            delta = desired - state.last_command
            command = state.last_command + np.clip(delta, -float(max_step_deg), float(max_step_deg))
            self._validate_joint_waypoint(
                command,
                label=f"{context} over-compensate waypoint",
            )
        except ValueError as exc:
            _logger.warning(
                "[SO-101] %s over-command %s rejected (%s); re-sending bare planned target (joint residual %.3f deg)",
                context,
                np.round(desired, 3).tolist(),
                exc,
                error_deg,
            )
            command = target.copy()
        return np.asarray(command, dtype=float), error_deg

    def _next_z_only_lift_command(
        self,
        target_q: np.ndarray,
        actual_q: np.ndarray,
        last_command_q: np.ndarray,
    ) -> np.ndarray:
        """Return one safe local command whose only Cartesian objective is +Z.

        This deliberately does not constrain Cartesian X/Y or orientation. It
        is used only for an upward Cartesian move while holding a payload, after
        the live joints have already entered the ordinary soft settle band.
        Joint limits, Cartesian workspace bounds, per-command joint steps and a
        total offset from the original IK endpoint remain hard constraints.
        """
        target = np.asarray(target_q, dtype=float)
        actual = np.asarray(actual_q, dtype=float)
        last_command = np.asarray(last_command_q, dtype=float)
        if target.shape != actual.shape or target.shape != last_command.shape:
            raise _ZOnlyLiftUnavailable(
                "Z-only lift state shape mismatch: "
                f"target={target.shape}, actual={actual.shape}, command={last_command.shape}."
            )

        # Central differences avoid selecting a direction from the original IK
        # joint residual. The resulting one-row Jacobian is under-constrained by
        # design: Jz.T / ||Jz||² is the minimum-joint-motion +Z solution.
        epsilon_deg = 0.1
        jz = np.zeros_like(actual)
        for index in range(actual.size):
            plus = actual.copy()
            minus = actual.copy()
            plus[index] += epsilon_deg
            minus[index] -= epsilon_deg
            plus_z = matrix_m_to_pose_mm_deg(np.asarray(self._kin.forward_kinematics(plus), dtype=float)).z
            minus_z = matrix_m_to_pose_mm_deg(np.asarray(self._kin.forward_kinematics(minus), dtype=float)).z
            jz[index] = (plus_z - minus_z) / (2.0 * epsilon_deg)

        norm_sq = float(np.dot(jz, jz))
        if not math.isfinite(norm_sq) or norm_sq <= 1e-8:
            raise _ZOnlyLiftUnavailable(f"Z-only lift Jacobian is singular or non-finite: Jz={jz.tolist()}.")
        delta = jz * (float(self._cfg.settle_z_only_lift_step_mm) / norm_sq)
        max_step = float(self._cfg.max_joint_step_deg)
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > max_step:
            delta *= max_step / max_delta

        max_offset = float(self._cfg.settle_z_only_lift_max_joint_offset_deg)
        lower = target - max_offset
        upper = target + max_offset
        command_pose = matrix_m_to_pose_mm_deg(np.asarray(self._kin.forward_kinematics(last_command), dtype=float))

        # FK is nonlinear and the Jacobian is evaluated at the live joints
        # rather than the over-command. Back off until the actual candidate is
        # a validated +Z command; never dispatch an assumed direction.
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = np.clip(last_command + scale * delta, lower, upper)
            if np.allclose(candidate, last_command, atol=1e-9, rtol=0.0):
                continue
            try:
                self._validate_joint_waypoint(candidate, label="Z-only lift command")
                candidate_pose = matrix_m_to_pose_mm_deg(
                    np.asarray(self._kin.forward_kinematics(candidate), dtype=float)
                )
                self._check_cartesian_bounds(candidate_pose, label="Z-only lift command FK")
            except ValueError:
                continue
            if candidate_pose.z > command_pose.z + 1e-4:
                _logger.info(
                    "[SO-101] Z-only lift command: live_z=%.3f command_z=%.3f->%.3f Jz=%s live_q=%s command_q=%s",
                    matrix_m_to_pose_mm_deg(np.asarray(self._kin.forward_kinematics(actual), dtype=float)).z,
                    command_pose.z,
                    candidate_pose.z,
                    np.round(jz, 3).tolist(),
                    np.round(actual, 3).tolist(),
                    np.round(candidate, 3).tolist(),
                )
                return candidate
        raise _ZOnlyLiftUnavailable(
            "Z-only lift found no safe +Z command within joint limits, workspace bounds, "
            f"step={max_step:g}deg and endpoint offset={max_offset:g}deg."
        )

    def _dispatch_prevalidated_waypoints(
        self,
        waypoints: list[np.ndarray],
        final_target: np.ndarray,
        *,
        timeout_s: float | None,
        cartesian_target_matrix: np.ndarray | None = None,
        cartesian_start_matrix: np.ndarray | None = None,
    ) -> None:
        """Stream pre-validated joint waypoints, then settle to ``final_target``.

        Shared by joint-space and Cartesian motion. Callers MUST have validated
        every waypoint (finiteness, soft limits, Cartesian bounds, residuals)
        BEFORE calling this — it begins sending immediately. The settle loop
        re-sends the final target when observation hasn't converged (LeRobot may
        clip via ``max_relative_target``), judging completion from real
        observation, not from the ``send_action`` return value.
        """
        period = 1.0 / float(self._cfg.trajectory_hz) if self._cfg.trajectory_hz > 0 else 0.0
        # Settle re-send throttle: cap the rate the final target is re-sent at.
        # 0 falls back to the interpolation period (legacy 30 Hz behavior).
        resend_period = float(self._cfg.settle_resend_period_s) if self._cfg.settle_resend_period_s > 0 else period
        drift_cap = max(0, int(self._cfg.settle_drift_abort_samples))
        settle_needed = max(1, int(self._cfg.settle_samples))
        soft_settle_needed = max(1, int(self._cfg.settle_soft_samples))
        soft_tolerance = float(self._cfg.settle_soft_tolerance_deg)
        strict_tolerance = float(self._cfg.joint_tolerance_deg)
        last_wp = waypoints[-1] if waypoints else np.asarray(final_target, dtype=float)
        self._last_motion_result = None
        # Keep the requested command path slew-limited even when LeRobot's
        # max_relative_target is disabled.  This is separate from the encoder
        # observation: a stalled servo must not make the next over-command jump
        # by the full position error.
        last_command = np.asarray(self.get_angles(), dtype=float)
        overcomp_integral_limit = max(2.0 * float(self._cfg.max_joint_step_deg), 4.0)
        # Intermediate waypoints form a time-parameterized command stream, not
        # a sequence of static settle targets.  Every command and every observed
        # pose is still safety-validated, but encoder lag does not pause and
        # repeatedly re-send an intermediate point.  The strict convergence
        # contract applies only to ``last_wp`` below.
        for wp in waypoints:
            requested = _arm_action(wp.tolist())
            actual_sent = self._send_action(requested)
            self._require_action_match(requested, actual_sent, label="waypoint dispatch")
            last_command = np.array(
                [actual_sent[f"{name}.pos"] for name in ARM_JOINT_ORDER],
                dtype=float,
            )
            if period > 0:
                self._sleep(period)
            observed = np.asarray(self.get_angles(), dtype=float)
            self._validate_joint_waypoint(observed, label="waypoint tracking")

        # Final settle gets its own complete timeout budget.  A long waypoint
        # stream must not consume the time reserved for endpoint convergence.
        settle_timeout_s = float(timeout_s if timeout_s is not None else self._cfg.move_timeout_s)
        settle_deadline = self._monotonic() + settle_timeout_s
        last_actual = np.asarray(self.get_angles(), dtype=float)
        endpoint_state = _EndpointCompensationState(
            integral=np.zeros_like(last_wp),
            last_command=last_command.copy(),
            previous_error=float(np.max(np.abs(last_actual - last_wp))),
        )
        z_only_lift_eligible = False
        lift_enabled = bool(self._cfg.settle_z_only_lift_enabled) and self._holding_payload
        if lift_enabled and cartesian_target_matrix is not None and cartesian_start_matrix is not None:
            target_z = matrix_m_to_pose_mm_deg(cartesian_target_matrix).z
            start_z = matrix_m_to_pose_mm_deg(cartesian_start_matrix).z
            z_only_lift_eligible = target_z > start_z + 1e-6
        z_only_lift_active = False
        z_only_settle_needed = max(1, int(self._cfg.settle_samples))
        while True:
            if self._monotonic() > settle_deadline:
                cartesian_error_mm, z_error_mm = self._record_settle_result(
                    "hard_timeout",
                    last_wp,
                    last_actual,
                    ok=False,
                    cartesian_target_matrix=cartesian_target_matrix,
                )
                last_joint_error = float(np.max(np.abs(last_actual - last_wp)))
                safe_cartesian_undershoot = (
                    cartesian_target_matrix is not None
                    and cartesian_error_mm is not None
                    and z_error_mm is not None
                    and last_joint_error <= soft_tolerance
                    and z_error_mm < -float(self._cfg.settle_max_z_undershoot_mm) - 1e-6
                )
                if safe_cartesian_undershoot:
                    if self._last_motion_result is not None:
                        self._last_motion_result["classification"] = "safe_z_undershoot"
                    raise So101PoseConvergenceError(
                        reason=(
                            "endpoint remained in the validated soft joint band but "
                            f"vertical undershoot was {abs(z_error_mm):.3f} mm"
                        ),
                        residual_mm=abs(z_error_mm),
                        tolerance_mm=float(self._cfg.settle_max_z_undershoot_mm),
                    )
                raise TimeoutError(f"SO-101 final target did not settle within the move timeout ({settle_timeout_s}s).")
            actual = np.asarray(self.get_angles(), dtype=float)
            last_actual = actual
            self._validate_joint_waypoint(actual, label="settle tracking")
            err = float(np.max(np.abs(actual - last_wp)))
            _, _, z_error_mm = self._settle_metrics(
                last_wp,
                actual,
                cartesian_target_matrix=cartesian_target_matrix,
            )
            z_acceptable = z_error_mm is not None and z_error_mm >= -float(self._cfg.settle_max_z_undershoot_mm) - 1e-6
            if z_only_lift_active:
                if z_acceptable:
                    z_only_settle_needed -= 1
                    if z_only_settle_needed <= 0:
                        self._record_settle_result(
                            "z_only_lift",
                            last_wp,
                            actual,
                            ok=True,
                            cartesian_target_matrix=cartesian_target_matrix,
                        )
                        return
                else:
                    z_only_settle_needed = max(1, int(self._cfg.settle_samples))
            elif err <= strict_tolerance and z_acceptable:
                settle_needed -= 1
                if settle_needed <= 0:
                    self._record_settle_result(
                        "strict",
                        last_wp,
                        actual,
                        ok=True,
                        cartesian_target_matrix=cartesian_target_matrix,
                    )
                    return
            else:
                settle_needed = max(1, int(self._cfg.settle_samples))
            # The full Cartesian norm remains diagnostic because SO-101 has
            # 5 DoF and load-dependent steady-state offsets. Vertical
            # undershoot is safety-relevant, however, so both strict and soft
            # arrival also require the one-sided Z threshold above.
            soft_eligible = (
                not z_only_lift_active and soft_tolerance > strict_tolerance and err <= soft_tolerance and z_acceptable
            )
            if soft_eligible:
                soft_settle_needed -= 1
                if soft_settle_needed <= 0:
                    self._record_settle_result(
                        "soft",
                        last_wp,
                        actual,
                        ok=True,
                        cartesian_target_matrix=cartesian_target_matrix,
                    )
                    return
            else:
                soft_settle_needed = max(1, int(self._cfg.settle_soft_samples))
            lift_eligible = not z_only_lift_active and z_only_lift_eligible
            lift_needed = not z_acceptable and err <= soft_tolerance
            if lift_eligible and lift_needed:
                z_only_lift_active = True
                _logger.warning(
                    "[SO-101] activating Z-only payload lift compensation: "
                    "joint_error=%.3fdeg <= soft %.3fdeg, z_error=%+.3fmm < -%.3fmm; "
                    "XY/orientation are no longer endpoint objectives",
                    err,
                    soft_tolerance,
                    z_error_mm,
                    float(self._cfg.settle_max_z_undershoot_mm),
                )
            # Re-send to drive convergence: observation polling alone will not
            # close LeRobot clipping or STS3215 gravity steady-state error.
            if z_only_lift_active:
                try:
                    cmd = self._next_z_only_lift_command(
                        last_wp,
                        actual,
                        endpoint_state.last_command,
                    )
                except (ValueError, _ZOnlyLiftUnavailable) as exc:
                    self._record_settle_result(
                        "hard_z_only_unavailable",
                        last_wp,
                        actual,
                        ok=False,
                        cartesian_target_matrix=cartesian_target_matrix,
                    )
                    raise So101PoseConvergenceError(
                        reason=f"Z-only payload lift compensation unavailable: {exc}",
                        residual_mm=abs(float(z_error_mm)),
                        tolerance_mm=float(self._cfg.settle_max_z_undershoot_mm),
                    ) from exc
            else:
                try:
                    cmd, _ = self._next_endpoint_command(
                        last_wp,
                        actual,
                        endpoint_state,
                        max_step_deg=float(self._cfg.max_joint_step_deg),
                        integral_limit_deg=overcomp_integral_limit,
                        drift_abort_samples=drift_cap,
                        context="settle",
                    )
                except _EndpointCompensationDrift as exc:
                    self._record_settle_result(
                        "hard_drift",
                        last_wp,
                        actual,
                        ok=False,
                        cartesian_target_matrix=cartesian_target_matrix,
                    )
                    raise RuntimeError(
                        f"SO-101 settle drift: max joint error grew {exc.count} consecutive "
                        f"re-sends (err {exc.previous_error:.3f} -> {exc.current_error:.3f} deg, target within "
                        f"{self._cfg.joint_tolerance_deg} deg). Aborting to avoid pushing the arm "
                        f"toward a limit — the servo likely cannot track under gravity load."
                    ) from exc
            requested = _arm_action(cmd.tolist())
            actual_sent = self._send_action(requested)
            self._require_action_match(requested, actual_sent, label="settle endpoint command")
            endpoint_state.last_command = np.array(
                [actual_sent[f"{name}.pos"] for name in ARM_JOINT_ORDER],
                dtype=float,
            )
            if resend_period > 0:
                self._sleep(resend_period)

    def move_to_pose_blocking(
        self,
        pose: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Move to a Cartesian pose (mm/deg XYZ-Euler) via IK, blocking until settled.

        ``pose`` is a :class:`So101Pose`. Generates SE(3) waypoints along the
        Cartesian path (translation lerp + rotation Slerp), solves IK for each
        seeded by the PREVIOUS step's solution (a continuous seed chain, matching
        lerobot's ``InverseKinematicsEEToJoints`` with
        ``initial_guess_current_joints=False``), and validates — BEFORE the first
        ``send_action`` — for every waypoint:

        - the commanded target's Z floor + XY bounds (driver second-layer check),
        - the IK FK position/orientation residual vs tolerances (5-DoF:
          orientation is best-effort; only rejected when a tolerance is set),
        - the FK pose Z/XY bounds (the arm may reach a safe target at an unsafe
          intermediate pose),
        - joint soft limits + finiteness.

        The continuous seed chain keeps placo inside its convergence basin so IK
        does not jump branches (verified ~0.005 mm residual on real IK for
        z +/-50 mm). A step whose IK fails validation (residual / joint limit /
        Cartesian bound) is rejected immediately rather than re-solved from a
        stale seed — at a fine enough ``cartesian_interp_step_mm`` this only
        happens when the target is genuinely unreachable.

        All waypoints are planned and validated before the first action; then
        the pre-validated joint waypoints are dispatched via the shared settle
        loop — the dispatched path is exactly the pre-validated one.

        When ``pose_convergence_max_iters > 0``, the first planned move is
        followed by a joint-space convergence trim (:meth:`_converge_to_pose`):
        STS3215 servos hold a pose-dependent PD steady-state error (firmware I
        term inert), so the arm settles at ``q_target - e`` instead of ``q_target``,
        leaving a Cartesian residual. The trim reads the encoder joint error and
        over-commands ``q_target + accum_e`` (re-solving NO IK), converging to the
        true target in ~2 iterations — a software integral term the firmware lacks.
        """
        self._require_connected()
        self._reset_servo_plan()
        if not isinstance(pose, So101Pose):
            raise TypeError(f"pose must be a So101Pose, got {type(pose).__name__}.")

        q_target = self._dispatch_pose_move(pose, timeout_s=kwargs.get("timeout_s"))
        if self._cfg.pose_convergence_max_iters > 0:
            self._converge_to_pose(pose, q_target, timeout_s=kwargs.get("timeout_s"))

    def servo_to_pose(self, pose: Any) -> bool:
        """Issue one non-blocking Cartesian servo command.

        Solve a bounded Cartesian progress step from the previous planned pose,
        seed IK from the previous planned joint solution, and issue one
        ``send_action``. The first call initializes both from live encoders.
        Live joints remain execution feedback and are checked against the
        configured following-error allowance. When encoder lag exceeds it,
        bounded compensation is dispatched toward the previous planned point
        while both the low-level and outer Cartesian plans remain unchanged.
        Encoder lag never redefines the next Cartesian waypoint. No per-axis/Z
        monotonic constraint is imposed; candidates outside the configured
        terminal deadband must still reduce Cartesian position error and pass
        residual, joint, velocity and safety checks. Orientation is a progress
        requirement only when ``ik_orientation_tolerance_deg`` is explicit.

        Returns ``True`` when a motor command was dispatched and ``False`` only
        when the minimum-period throttle deliberately skips this call.
        """
        self._require_connected()
        target = self._coerce_servo_pose(pose)
        _require_finite_so101_pose(target, label="servo_to_pose target")
        try:
            self._check_cartesian_bounds(target, label="servo_to_pose target")
        except ValueError as exc:
            raise So101CartesianServoError("cartesian_bounds_rejected", str(exc)) from exc

        # Min inter-send interval: a call within this window is a no-op
        # (non-blocking; skipped calls do not accumulate or catch up).
        now = self._monotonic()
        min_period = float(self._cfg.servo_min_send_period_s)
        first_send = self._servo_last_send_t == 0.0
        if not first_send and (now - self._servo_last_send_t) + _SERVO_TIME_EPS_S < min_period:
            return False

        q_actual = np.asarray(self.get_angles(), dtype=float)
        self._validate_joint_vector(q_actual.tolist(), label="servo_to_pose current")
        target_matrix = np.asarray(pose_mm_deg_to_matrix_m(target), dtype=float)
        actual_matrix = np.asarray(self._kin.forward_kinematics(q_actual), dtype=float)
        actual_pose = matrix_m_to_pose_mm_deg(actual_matrix)

        # First send uses one min_period (timestamp 0 is not a real instant).
        dt = min_period if first_send else now - self._servo_last_send_t
        vel_cap = float(self._cfg.servo_max_joint_vel_dps) * dt
        max_step = min(float(self._cfg.servo_max_joint_step_deg), vel_cap)
        if not _is_finite(max_step) or max_step <= 0.0:
            raise ValueError(f"servo_to_pose: computed joint step cap is invalid ({max_step!r}).")

        if self._servo_planned_q is None or self._servo_planned_matrix is None:
            planned_q = q_actual.copy()
            planned_matrix = actual_matrix.copy()
        else:
            planned_q = self._servo_planned_q.copy()
            planned_matrix = self._servo_planned_matrix.copy()
            tracking_errors = np.abs(q_actual - planned_q)
            tracking_error = float(np.max(tracking_errors))
            tracking_limit = float(self._cfg.tracking_error_deg)
            if tracking_error > tracking_limit + 1e-6:
                joint_index = int(np.argmax(tracking_errors))
                joint_name = ARM_JOINT_ORDER[joint_index]
                _logger.warning(
                    "[SO-101] servo catch-up hold: %s planned=%.3fdeg live=%.3fdeg "
                    "error=%.3fdeg exceeds %.3fdeg; dispatching bounded compensation "
                    "while plan remains unchanged",
                    joint_name,
                    float(planned_q[joint_index]),
                    float(q_actual[joint_index]),
                    tracking_error,
                    tracking_limit,
                )
                return self._dispatch_servo_catchup_hold(
                    planned_q=planned_q,
                    q_actual=q_actual,
                    max_step=max_step,
                )
        planned_pose = matrix_m_to_pose_mm_deg(planned_matrix)
        planned_fk_pose = matrix_m_to_pose_mm_deg(np.asarray(self._kin.forward_kinematics(planned_q), dtype=float))
        _logger.debug(
            "[SO-101] servo request q_actual=%s actual_pose=%s planned_q=%s planned_pose=%s requested_target=%s",
            np.round(q_actual, 4).tolist(),
            actual_pose,
            np.round(planned_q, 4).tolist(),
            planned_pose,
            target,
        )

        current_pos_err = position_error_mm(planned_pose, target)
        planned_fk_pos_err = position_error_mm(planned_fk_pose, target)
        planned_fk_ori_err = orientation_error_deg(planned_fk_pose, target)
        goal_pos_tol = float(self._cfg.servo_goal_tolerance_mm)
        goal_ori_tol = self._cfg.ik_orientation_tolerance_deg
        planned_position_reached = planned_fk_pos_err <= goal_pos_tol
        planned_orientation_reached = goal_ori_tol is None or planned_fk_ori_err <= float(goal_ori_tol)
        cartesian_step_cap = min(
            float(self._cfg.servo_max_cartesian_step_mm),
            float(self._cfg.servo_max_cartesian_vel_mm_s) * dt,
        )
        alpha_limit = min(1.0, cartesian_step_cap / current_pos_err) if current_pos_err > 0.0 else 1.0

        # The Cartesian plan may already be practically at the requested
        # target while the physical arm is still catching up. Do not demand
        # sub-millimetre IK progress inside the configured deadband: re-send
        # the last planned command without advancing or re-anchoring the plan.
        # A null orientation tolerance means orientation is best-effort and
        # therefore cannot prevent this hold state.
        if planned_position_reached and planned_orientation_reached:
            return self._dispatch_servo_endpoint_hold(
                target=target,
                planned_q=planned_q,
                q_actual=q_actual,
                actual_pose=actual_pose,
                max_step=max_step,
            )

        self._reset_servo_hold_state()
        search = _ServoSearchContext(
            target=target,
            target_matrix=target_matrix,
            planned_matrix=planned_matrix,
            planned_q=planned_q,
            planned_fk_pose=planned_fk_pose,
            max_step=max_step,
            cartesian_step_cap=cartesian_step_cap,
            alpha_limit=alpha_limit,
            goal_pos_tol=goal_pos_tol,
            goal_ori_tol=goal_ori_tol,
            planned_position_reached=planned_position_reached,
            planned_orientation_reached=planned_orientation_reached,
            planned_fk_pos_err=planned_fk_pos_err,
            planned_fk_ori_err=planned_fk_ori_err,
        )
        q_cmd, best_alpha = self._search_servo_progress(search)
        self._validate_joint_waypoint(q_cmd, label=f"servo_to_pose command t={best_alpha:.5f}")
        requested = _arm_action(q_cmd.tolist())
        actual = self._send_action(requested)
        self._require_action_match(requested, actual, label="Cartesian servo")
        _logger.debug(
            "[SO-101] servo dispatch alpha=%.5f dt=%.4fs q_actual=%s planned_q=%s requested=%s actual=%s",
            best_alpha,
            dt,
            np.round(q_actual, 4).tolist(),
            np.round(planned_q, 4).tolist(),
            np.round(q_cmd, 4).tolist(),
            actual,
        )
        # Keep the Cartesian plan origin coherent with its IK seed. This is FK
        # of the previous PLANNED command, not live encoder FK, so execution
        # lag still cannot re-anchor the trajectory.
        self._servo_planned_matrix = np.asarray(self._kin.forward_kinematics(q_cmd), dtype=float)
        self._servo_planned_q = q_cmd.copy()
        self._servo_last_send_t = self._monotonic()
        return True

    def _dispatch_servo_catchup_hold(
        self,
        *,
        planned_q: np.ndarray,
        q_actual: np.ndarray,
        max_step: float,
    ) -> bool:
        """Drive live joints toward the held fast plan without advancing it."""
        hold_error = float(np.max(np.abs(planned_q - q_actual)))
        drift_cap = max(0, int(self._cfg.settle_drift_abort_samples))
        if self._servo_endpoint_state is None:
            self._servo_endpoint_state = _EndpointCompensationState(
                integral=np.zeros_like(planned_q),
                last_command=planned_q.copy(),
                previous_error=hold_error,
            )
        hold_state = self._servo_endpoint_state
        try:
            command_q, hold_error = self._next_endpoint_command(
                planned_q,
                q_actual,
                hold_state,
                max_step_deg=max_step,
                integral_limit_deg=max(2.0 * float(self._cfg.servo_max_joint_step_deg), 4.0),
                drift_abort_samples=drift_cap,
                context="servo catch-up",
                max_command_offset_deg=0.9 * float(self._cfg.tracking_error_deg),
            )
        except _EndpointCompensationDrift as exc:
            raise So101CartesianServoError(
                "servo_catchup_drift",
                f"{exc}; aborting catch-up over-compensation",
            ) from exc

        requested = _arm_action(command_q.tolist())
        actual = self._send_action(requested)
        self._require_action_match(requested, actual, label="Cartesian servo catch-up hold")
        hold_state.last_command = np.array(
            [actual[f"{name}.pos"] for name in ARM_JOINT_ORDER],
            dtype=float,
        )
        _logger.info(
            "[SO-101] servo catch-up compensation live_q=%s planned_q=%s command_q=%s joint_err=%.3fdeg integral=%s",
            np.round(q_actual, 3).tolist(),
            np.round(planned_q, 3).tolist(),
            np.round(hold_state.last_command, 3).tolist(),
            hold_error,
            np.round(hold_state.integral, 3).tolist(),
        )
        self._servo_last_send_t = self._monotonic()
        return True

    def _dispatch_servo_endpoint_hold(
        self,
        *,
        target: So101Pose,
        planned_q: np.ndarray,
        q_actual: np.ndarray,
        actual_pose: So101Pose,
        max_step: float,
    ) -> bool:
        """Re-send a reached plan with bounded endpoint compensation."""
        self._validate_joint_waypoint(planned_q, label="servo_to_pose hold command")
        hold_error = float(np.max(np.abs(planned_q - q_actual)))
        drift_cap = max(0, int(self._cfg.settle_drift_abort_samples))
        if self._servo_endpoint_state is None:
            self._servo_endpoint_state = _EndpointCompensationState(
                integral=np.zeros_like(planned_q),
                last_command=planned_q.copy(),
                previous_error=hold_error,
            )
        endpoint_state = self._servo_endpoint_state
        try:
            command_q, hold_error = self._next_endpoint_command(
                planned_q,
                q_actual,
                endpoint_state,
                max_step_deg=max_step,
                integral_limit_deg=max(2.0 * float(self._cfg.servo_max_joint_step_deg), 4.0),
                drift_abort_samples=drift_cap,
                context="servo endpoint",
                # Keep compensation strictly inside the planned-vs-live
                # watchdog so it cannot create a permanent catch-up hold.
                max_command_offset_deg=0.9 * float(self._cfg.tracking_error_deg),
            )
        except _EndpointCompensationDrift as exc:
            raise So101CartesianServoError(
                "servo_settle_drift",
                f"{exc}; aborting endpoint over-compensation",
            ) from exc

        requested = _arm_action(command_q.tolist())
        actual = self._send_action(requested)
        self._require_action_match(requested, actual, label="Cartesian servo endpoint hold")
        endpoint_state.last_command = np.array(
            [actual[f"{name}.pos"] for name in ARM_JOINT_ORDER],
            dtype=float,
        )
        _logger.info(
            "[SO-101] servo endpoint hold live_q=%s planned_q=%s command_q=%s "
            "joint_err=%.3fdeg live_pos_err=%.3fmm integral=%s",
            np.round(q_actual, 3).tolist(),
            np.round(planned_q, 3).tolist(),
            np.round(endpoint_state.last_command, 3).tolist(),
            hold_error,
            position_error_mm(actual_pose, target),
            np.round(endpoint_state.integral, 3).tolist(),
        )
        self._servo_last_send_t = self._monotonic()
        return True

    def _evaluate_servo_candidate(self, alpha: float, ctx: _ServoSearchContext) -> _ServoCandidate:
        """Return a safe, progressing IK candidate or its failure reason."""
        matrix = _interp_se3(ctx.planned_matrix, ctx.target_matrix, alpha)
        label = f"servo_to_pose candidate t={alpha:.5f}"
        configured_orientation_weight = float(self._cfg.ik_orientation_weight)
        orientation_weights = (
            [0.0, configured_orientation_weight]
            if self._cfg.ik_orientation_tolerance_deg is None and configured_orientation_weight != 0.0
            else [configured_orientation_weight]
        )
        q_candidate: np.ndarray | None = None
        candidate_errors: list[str] = []
        for orientation_weight in orientation_weights:
            try:
                solved = np.asarray(
                    self._kin.inverse_kinematics(
                        ctx.planned_q,
                        matrix,
                        position_weight=1.0,
                        orientation_weight=orientation_weight,
                    ),
                    dtype=float,
                )
                self._validate_ik_solution(solved, matrix, label=label)
                q_candidate = solved
                break
            except Exception as exc:  # noqa: BLE001 - try position-only or a smaller alpha
                candidate_errors.append(f"orientation_weight={orientation_weight:g}: {exc}")
        if q_candidate is None:
            error = "; ".join(candidate_errors)
            code = (
                "cartesian_bounds_rejected"
                if any(token in error for token in ("soft limits", "z=", "workspace"))
                else "ik_unreachable"
            )
            return _ServoCandidate(None, code, error)

        q_delta = float(np.max(np.abs(q_candidate - ctx.planned_q)))
        if not _is_finite(q_delta) or q_delta > ctx.max_step + 1e-6:
            return _ServoCandidate(
                None,
                "joint_velocity_limited_no_progress",
                f"{label}: joint delta {q_delta:.4f} deg exceeds cap {ctx.max_step:.4f} deg",
            )

        candidate_pose = matrix_m_to_pose_mm_deg(np.asarray(self._kin.forward_kinematics(q_candidate)))
        candidate_pos_err = position_error_mm(candidate_pose, ctx.target)
        candidate_ori_err = orientation_error_deg(candidate_pose, ctx.target)
        candidate_step_mm = position_error_mm(ctx.planned_fk_pose, candidate_pose)
        if candidate_step_mm > ctx.cartesian_step_cap + _SERVO_PROGRESS_EPS_MM:
            return _ServoCandidate(
                None,
                "joint_velocity_limited_no_progress",
                f"{label}: Cartesian step {candidate_step_mm:.3f} mm exceeds "
                f"velocity cap {ctx.cartesian_step_cap:.3f} mm",
            )

        if not ctx.planned_position_reached:
            progressing = (
                candidate_pos_err <= ctx.goal_pos_tol
                or candidate_pos_err < ctx.planned_fk_pos_err - _SERVO_PROGRESS_EPS_MM
            )
        elif ctx.goal_ori_tol is not None and not ctx.planned_orientation_reached:
            progressing = (
                candidate_ori_err <= ctx.goal_ori_tol
                or candidate_ori_err < ctx.planned_fk_ori_err - _SERVO_PROGRESS_EPS_DEG
            )
        else:
            progressing = False
        if not progressing:
            return _ServoCandidate(
                None,
                "cartesian_progress_reversed",
                f"{label}: candidate does not reduce Cartesian error "
                f"(position {candidate_pos_err:.3f}/{ctx.planned_fk_pos_err:.3f} mm, "
                f"orientation {candidate_ori_err:.3f}/{ctx.planned_fk_ori_err:.3f} deg)",
            )
        _logger.debug(
            "[SO-101] servo candidate alpha=%.5f q_delta=%.4f pos_err=%.3f->%.3f z=%.3f->%.3f q=%s fk=%s",
            alpha,
            q_delta,
            ctx.planned_fk_pos_err,
            candidate_pos_err,
            ctx.planned_fk_pose.z,
            candidate_pose.z,
            np.round(q_candidate, 4).tolist(),
            {
                "x": round(candidate_pose.x, 3),
                "y": round(candidate_pose.y, 3),
                "z": round(candidate_pose.z, 3),
            },
        )
        return _ServoCandidate(q_candidate)

    def _search_servo_progress(self, ctx: _ServoSearchContext) -> tuple[np.ndarray, float]:
        """Grow along one local IK branch and return its furthest safe step."""
        best_q: np.ndarray | None = None
        best_alpha = 0.0
        failure: _ServoCandidate | None = None
        # Placo keeps internal solver state, so start small. A far failed solve
        # can poison subsequent local solves despite receiving the same seed.
        alpha = min(_SERVO_ALPHA_MIN, ctx.alpha_limit)
        for _ in range(_SERVO_ALPHA_SEARCH_ITERS):
            candidate = self._evaluate_servo_candidate(alpha, ctx)
            if candidate.q is None:
                failure = candidate
                break
            best_q = candidate.q
            best_alpha = alpha
            if alpha >= ctx.alpha_limit:
                break
            alpha = min(ctx.alpha_limit, alpha * 2.0)

        if best_q is not None:
            return np.asarray(best_q, dtype=float), best_alpha
        detail = f" Last candidate error: {failure.error}" if failure and failure.error else ""
        raise So101CartesianServoError(
            failure.code if failure and failure.code else "joint_velocity_limited_no_progress",
            "servo_to_pose found no safe Cartesian progress step satisfying IK, "
            f"residual, joint cap and safety checks.{detail}",
        )

    def _reset_servo_plan(self) -> None:
        """Discard streaming command state before a new/non-servo motion."""
        self._servo_last_send_t = 0.0
        self._servo_planned_matrix = None
        self._servo_planned_q = None
        self._reset_servo_hold_state()

    def _reset_servo_hold_state(self) -> None:
        """Discard endpoint compensation without changing the IK plan."""
        self._servo_endpoint_state = None

    @staticmethod
    def _coerce_servo_pose(pose: Any) -> So101Pose:
        """Coerce a complete mapping/attribute-bag into ``So101Pose``."""
        if isinstance(pose, So101Pose):
            return pose
        if isinstance(pose, Mapping):

            def get(key: str) -> Any:
                return pose.get(key)
        else:

            def get(key: str) -> Any:
                return getattr(pose, key, None)

        missing: list[str] = []
        values: dict[str, Any] = {}
        for name in ("x", "y", "z", "rx", "ry"):
            value = get(name)
            if value is None:
                missing.append(name)
            else:
                values[name] = value
        rz = get("rz")
        if rz is None:
            rz = get("r")
        if rz is None:
            missing.append("rz (or r)")
        else:
            values["rz"] = rz
        if missing:
            raise TypeError(f"SO-101 servo pose missing required fields: {', '.join(missing)}.")
        try:
            return So101Pose(**{name: float(value) for name, value in values.items()})
        except (TypeError, ValueError) as exc:
            raise TypeError(f"SO-101 servo pose contains non-numeric fields: {pose!r}.") from exc

    def _validate_ik_solution(self, q: np.ndarray, desired_matrix: np.ndarray, *, label: str) -> None:
        """Validate an IK solution against the requested pose and safety envelope."""
        self._validate_joint_vector(np.asarray(q, dtype=float).tolist(), label=label)
        self._check_joint_limits(np.asarray(q, dtype=float), label=label)
        fk_matrix = np.asarray(self._kin.forward_kinematics(q), dtype=float)
        fk_pose = matrix_m_to_pose_mm_deg(fk_matrix)
        desired_pose = matrix_m_to_pose_mm_deg(desired_matrix)
        pos_err = position_error_mm(fk_pose, desired_pose)
        if not _is_finite(pos_err):
            raise ValueError(f"{label}: IK position residual non-finite: {pos_err}.")
        if pos_err > self._cfg.ik_position_tolerance_mm:
            raise ValueError(
                f"{label}: IK position residual {pos_err:.3f} mm exceeds "
                f"tolerance {self._cfg.ik_position_tolerance_mm} mm."
            )
        if self._cfg.ik_orientation_tolerance_deg is not None:
            ori_err = orientation_error_deg(fk_pose, desired_pose)
            if not _is_finite(ori_err):
                raise ValueError(f"{label}: IK orientation residual non-finite: {ori_err}.")
            if ori_err > self._cfg.ik_orientation_tolerance_deg:
                raise ValueError(
                    f"{label}: IK orientation residual {ori_err:.3f} deg exceeds "
                    f"tolerance {self._cfg.ik_orientation_tolerance_deg} deg."
                )
        self._check_cartesian_bounds(fk_pose, label=f"{label} FK")

    def _dispatch_pose_move(
        self,
        pose: So101Pose,
        *,
        timeout_s: float | None,
    ) -> np.ndarray:
        """Plan + validate + dispatch ONE Cartesian move; return the IK endpoint q.

        Single planned move: commanded-target boundary check → SE(3) waypoint
        plan (seed-chain IK, all residuals/limits/bounds pre-validated) → shared
        settle dispatch. Returns ``ik_waypoints[-1]`` (the joint-space target the
        arm was commanded toward) so the convergence trim can over-command it
        without re-solving IK.
        """
        # 1. Commanded target boundary (driver repeats SafetyRail's check).
        # These checks happen before the live seed is read and before any
        # hardware command; mark their failures so RecoveryRail does not open
        # the gripper or home an arm that never moved.
        try:
            self._check_cartesian_bounds(pose, label="goto_pose target")
        except ValueError as exc:
            raise So101PreDispatchError(str(exc)) from exc

        desired_matrix = np.asarray(pose_mm_deg_to_matrix_m(pose), dtype=float)
        current_q = np.asarray(self.get_angles(), dtype=float)
        start_matrix = np.asarray(self._kin.forward_kinematics(current_q), dtype=float)

        # 2. Plan the SE(3) waypoint path via the seed chain (one IK per step,
        #    seeded by the previous step's solution). All residuals, limits and
        #    Cartesian bounds are checked here, before any send_action.
        try:
            ik_waypoints = self._plan_cartesian_waypoints(current_q, start_matrix, desired_matrix, pose)
        except ValueError as exc:
            raise So101PreDispatchError(str(exc)) from exc

        # 3. Dispatch the pre-validated joint waypoints (shared settle loop).
        self._dispatch_prevalidated_waypoints(
            ik_waypoints,
            ik_waypoints[-1],
            timeout_s=timeout_s,
            cartesian_target_matrix=desired_matrix,
            cartesian_start_matrix=start_matrix,
        )
        return np.asarray(ik_waypoints[-1], dtype=float)

    def _converge_to_pose(
        self,
        target: So101Pose,
        q_target: np.ndarray,
        *,
        timeout_s: float | None,
    ) -> None:
        """Joint-space integral trim to compensate STS3215 PD steady-state error.

        ``q_target`` is the IK endpoint from the first planned move (solved ONCE —
        this method re-solves NO IK). After the first move the arm settles at
        ``q_target - e`` (e = pose-dependent PD steady-state error; firmware I
        term is inert), leaving a Cartesian residual vs ``target``. This loop reads
        the encoder joint error and over-commands ``q_target + accum_e`` via
        :meth:`move_joint_blocking`, accumulating the steady-state offset like a
        software integral term. For a locally-constant e it converges in ~2
        iterations; a residual already within ``pose_convergence_tolerance_mm``
        stops on iteration 1 (no compensation).

        Safety: an over-command outside the soft limits or Cartesian envelope is
        rejected fail-closed and raises :class:`So101PoseConvergenceError`, so
        the arm stays at its current validated real pose — never breaking a
        limit.  RecoveryRail recognizes this typed failure and does not issue a
        home move that would lose the useful position.  Settle drift abort inside
        ``move_joint_blocking`` still propagates a real settle failure.
        """
        accum_e = np.zeros(len(ARM_JOINT_ORDER), dtype=float)
        final_residual = float("nan")  # set each iteration; read post-loop
        for n in range(1, self._cfg.pose_convergence_max_iters + 1):
            final_residual = position_error_mm(self.get_pose(), target)
            _logger.info(
                "SO-101 pose convergence iter %d/%d: residual %.3f mm",
                n,
                self._cfg.pose_convergence_max_iters,
                final_residual,
            )
            if final_residual <= self._cfg.pose_convergence_tolerance_mm:
                return
            q_actual = np.asarray(self.get_angles(), dtype=float)
            accum_e = accum_e + (q_target - q_actual)
            cmd_q = q_target + accum_e
            # Over-command must stay inside the soft limits — a compensation that
            # would break a limit means the target genuinely needs an
            # out-of-bounds joint; stop at the current safe real pose.
            try:
                self._validate_joint_waypoint(cmd_q, label=f"convergence iter {n} over-command")
            except ValueError as exc:
                _logger.warning(
                    "SO-101 pose convergence iter %d: over-command rejected (%s); "
                    "stopping at the current real pose (residual %.2f mm).",
                    n,
                    exc,
                    final_residual,
                )
                raise So101PoseConvergenceError(
                    reason=f"convergence iter {n} compensation rejected: {exc}",
                    residual_mm=final_residual,
                    tolerance_mm=self._cfg.pose_convergence_tolerance_mm,
                ) from exc
            # Reuse joint interpolation + settle + drift abort. A drift-abort
            # RuntimeError (real settle failure) propagates — the convergence loop
            # must not mask a genuine servo-under-load divergence.
            try:
                self.move_joint_blocking(cmd_q.tolist(), timeout_s=timeout_s)
            except TimeoutError as exc:
                settle_result = self._last_motion_result or {}
                stayed_in_soft_band = (
                    settle_result.get("classification") == "hard_timeout"
                    and isinstance(settle_result.get("max_abs_joint_error_deg"), (int, float))
                    and float(settle_result["max_abs_joint_error_deg"]) <= float(self._cfg.settle_soft_tolerance_deg)
                )
                if stayed_in_soft_band:
                    if self._last_motion_result is not None:
                        self._last_motion_result["classification"] = "safe_convergence_timeout"
                    raise So101PoseConvergenceError(
                        reason=(
                            f"convergence iter {n} compensation remained in the validated soft joint band until timeout"
                        ),
                        residual_mm=final_residual,
                        tolerance_mm=self._cfg.pose_convergence_tolerance_mm,
                    ) from exc
                raise
            except ValueError as exc:
                # A compensation path can also fail its intermediate FK safety
                # check even when the endpoint's joints are inside soft limits.
                # Surface that as the same explicit not-reached condition.
                raise So101PoseConvergenceError(
                    reason=f"convergence iter {n} compensation path rejected: {exc}",
                    residual_mm=final_residual,
                    tolerance_mm=self._cfg.pose_convergence_tolerance_mm,
                ) from exc
        # Re-read once more so the exhaustion message reflects the real final
        # state (the last compensation move may have actually converged).
        final_residual = position_error_mm(self.get_pose(), target)
        if final_residual > self._cfg.pose_convergence_tolerance_mm:
            _logger.warning(
                "SO-101 pose convergence: %d iterations did not converge "
                "(residual %.2f mm > %.1f mm); stopped at the current real pose.",
                self._cfg.pose_convergence_max_iters,
                final_residual,
                self._cfg.pose_convergence_tolerance_mm,
            )
            raise So101PoseConvergenceError(
                reason=(f"{self._cfg.pose_convergence_max_iters} convergence iterations exhausted"),
                residual_mm=final_residual,
                tolerance_mm=self._cfg.pose_convergence_tolerance_mm,
            )

    def _plan_cartesian_waypoints(
        self,
        start_q: np.ndarray,
        start_matrix: np.ndarray,
        target_matrix: np.ndarray,
        target_pose: So101Pose,
    ) -> list[np.ndarray]:
        """Plan a Cartesian SE(3) path as a list of joint-space IK solutions.

        Splits the SE(3) path ``start_matrix -> target_matrix`` into N evenly
        spaced interpolation steps (translation lerp + rotation Slerp via
        :func:`_interp_se3`), where ``N = ceil(max(translation_mm, rotation_deg)
        / cartesian_interp_step_mm)`` (>= 1). Solves IK per step, seeded by the
        PREVIOUS accepted solution. If a valid weighted-IK joint vector exceeds
        the position-residual tolerance while orientation is not a hard
        constraint, the same waypoint is retried once from the same seed with
        ``orientation_weight=0``. This matches lerobot's
        ``InverseKinematicsEEToJoints`` with
        ``initial_guess_current_joints=False`` and keeps placo inside its
        convergence basin so IK does not jump branches between steps.

        Every waypoint's residual, joint soft limits, finiteness and Cartesian
        bounds are validated before the first action. A step that fails
        validation raises immediately (the seed chain already gives the best
        seed; a stale-seed re-solve cannot help, so no bisection is attempted).
        """
        step_mm = float(self._cfg.cartesian_interp_step_mm)
        ease = bool(self._cfg.cartesian_ease_in_out)

        # Step count from the larger of translation (mm) and rotation (deg), so a
        # pure-rotation move still gets enough IK steps for the seed chain.
        start_pose = matrix_m_to_pose_mm_deg(start_matrix)
        tgt_pose = matrix_m_to_pose_mm_deg(target_matrix)
        trans_mm = position_error_mm(start_pose, tgt_pose)
        rot_deg = orientation_error_deg(start_pose, tgt_pose)
        magnitude = max(trans_mm, rot_deg)
        steps = max(1, int(math.ceil(magnitude / step_mm)))
        if steps > _MAX_CARTESIAN_WAYPOINTS:
            raise ValueError(
                f"goto_pose path: {steps} interpolation steps exceed the cap "
                f"({_MAX_CARTESIAN_WAYPOINTS}); the Cartesian move is too large for "
                f"cartesian_interp_step_mm={step_mm}. Split it or increase the step."
            )

        configured_orientation_weight = float(self._cfg.ik_orientation_weight)
        position_only_retry_enabled = (
            self._cfg.ik_orientation_tolerance_deg is None and configured_orientation_weight != 0.0
        )

        def ik(seed: np.ndarray, matrix: np.ndarray, *, orientation_weight: float) -> np.ndarray:
            return np.asarray(
                self._kin.inverse_kinematics(
                    seed,
                    matrix,
                    position_weight=1.0,
                    orientation_weight=orientation_weight,
                ),
                dtype=float,
            )

        def validate_waypoint(q: np.ndarray, matrix: np.ndarray, label: str) -> None:
            """Validate a waypoint; raise on failure (no silent skip)."""
            self._validate_ik_solution(q, matrix, label=label)

        def position_residual_for_valid_joints(q: np.ndarray, matrix: np.ndarray, label: str) -> float:
            """Return residual only after non-position joint safety checks pass."""
            self._validate_joint_vector(q.tolist(), label=label)
            self._check_joint_limits(q, label=label)
            fk_pose = matrix_m_to_pose_mm_deg(np.asarray(self._kin.forward_kinematics(q), dtype=float))
            desired_pose = matrix_m_to_pose_mm_deg(matrix)
            return position_error_mm(fk_pose, desired_pose)

        accepted: list[np.ndarray] = []
        # Seed chain: starts at the current joint config, then tracks each step's
        # IK solution so the next step solves from a nearby (converged) seed.
        seed = np.asarray(start_q, dtype=float)
        for k in range(1, steps + 1):
            t = k / steps
            if ease:
                # Ease-in-out (sin^2) so the first/last steps move least — better
                # IK convergence at path ends. Still monotonic in [0, 1].
                t = math.sin(t * math.pi / 2.0) ** 2
            matrix_k = _interp_se3(start_matrix, target_matrix, t)
            label = f"goto_pose waypoint t={t:.4f}"
            try:
                q_k = ik(
                    seed,
                    matrix_k,
                    orientation_weight=configured_orientation_weight,
                )
            except Exception as exc:  # noqa: BLE001 - placo may raise on singular seeds
                raise ValueError(
                    f"goto_pose waypoint t={t:.4f}: IK raised {exc!r}; target likely unreachable or on a singularity."
                ) from exc

            original_residual = position_residual_for_valid_joints(q_k, matrix_k, label)
            if (
                position_only_retry_enabled
                and math.isfinite(original_residual)
                and original_residual > float(self._cfg.ik_position_tolerance_mm)
            ):
                try:
                    position_only_q = ik(seed, matrix_k, orientation_weight=0.0)
                    validate_waypoint(position_only_q, matrix_k, label=f"{label} position-only retry")
                except Exception as exc:  # noqa: BLE001 - preserve both attempts in one pre-dispatch error
                    raise ValueError(
                        f"{label}: weighted IK position residual {original_residual:.3f} mm exceeds "
                        f"tolerance {self._cfg.ik_position_tolerance_mm} mm; "
                        f"position-only retry failed: {exc}"
                    ) from exc
                _logger.info(
                    "[SO-101] %s weighted IK residual %.3fmm exceeds %.3fmm; "
                    "accepted position-only retry from the same seed",
                    label,
                    original_residual,
                    float(self._cfg.ik_position_tolerance_mm),
                )
                q_k = position_only_q
            else:
                validate_waypoint(q_k, matrix_k, label=label)

            # Only an accepted solution advances the continuous seed chain.
            seed = q_k
            accepted.append(q_k)

        # The last step targets _interp_se3(..., 1) ~ target_matrix; guard against
        # float-lerp drift with an exact check against the commanded target.
        if not accepted:
            # steps == 0 is impossible (>= 1), but keep the guard defensive.
            raise ValueError("goto_pose path: produced no waypoints.")
        final_q = accepted[-1]
        validate_waypoint(final_q, target_matrix, label="goto_pose IK endpoint")
        return accepted

    # ----------------------------------------------------------------- internals
    def _resolve_urdf_path(self) -> str:
        if self._cfg.urdf_path:
            return str(self._cfg.urdf_path)
        # Packaged default alongside this module.
        here = Path(__file__).resolve().parent / "description" / "so101_new_calib.urdf"
        return str(here)

    def _require_connected(self) -> None:
        if not self._connected or self._robot is None or self._kin is None:
            raise RuntimeError("So101Driver method called before connect().")

    def _read_arm_angles(self, robot: Any) -> list[float]:
        """Read observation and extract the 5 arm joints in ``ARM_JOINT_ORDER``."""
        obs = robot.get_observation()
        return [self._read_motor(obs, name) for name in ARM_JOINT_ORDER]

    @staticmethod
    def _read_motor(obs: dict[str, Any], name: str) -> float:
        """Read a single motor value from observation, trying ``.pos`` then bare."""
        for key in (f"{name}.pos", name):
            if key in obs:
                val = obs[key]
                if isinstance(val, np.ndarray):
                    val = float(val.item()) if val.size == 1 else float(val.ravel()[0])
                else:
                    val = float(val)
                return val
        raise RuntimeError(f"SOFollower observation missing motor '{name}' (tried '{name}.pos', '{name}').")

    def _check_joint_limits(self, q: np.ndarray, *, label: str) -> None:
        limits = self._cfg.joint_limits
        for i, name in enumerate(ARM_JOINT_ORDER):
            lo, hi = limits[name]
            if not (lo <= float(q[i]) <= hi):
                raise ValueError(f"{label}: {name}={float(q[i])} out of soft limits [{lo}, {hi}].")

    def _validate_joint_waypoint(self, q: np.ndarray, *, label: str) -> None:
        """Validate one joint command, including its FK Cartesian envelope.

        This is intentionally called before dispatch for both normal
        interpolation waypoints and settle over-compensation waypoints.  The
        Cartesian target check in :meth:`move_to_pose_blocking` cannot protect a
        direct ``move_joint`` caller because a legal joint vector may have an
        unsafe FK pose.
        """
        q_arr = np.asarray(q, dtype=float)
        self._validate_joint_vector(q_arr.tolist(), label=label)
        self._check_joint_limits(q_arr, label=label)
        fk_matrix = np.asarray(self._kin.forward_kinematics(q_arr), dtype=float)
        fk_pose = matrix_m_to_pose_mm_deg(fk_matrix)
        self._check_cartesian_bounds(fk_pose, label=f"{label} FK")

    def _check_cartesian_bounds(self, pose: So101Pose, *, label: str) -> None:
        """Second-layer Z-floor/ceiling + XY-bound check the driver runs before sending.

        SafetyRail checks the *target* at the tool layer, but a caller can bypass
        the tool layer (direct driver use) or the 5-DoF IK may land the arm at a
        reachable-but-unsafe pose. Per plan §Decision 4 the driver repeats the
        boundary check before actually dispatching. Applied to both the commanded
        target and the IK solution's FK result, and to every interpolated
        waypoint along the path.
        """
        z_floor = self._cfg.z_min_safe_mm
        if pose.z < z_floor:
            raise ValueError(f"{label}: z={pose.z:.3f} mm below driver z_min_safe={z_floor} mm.")
        z_ceil = getattr(self._cfg, "z_max_safe_mm", None)
        if z_ceil is not None and pose.z > z_ceil:
            raise ValueError(f"{label}: z={pose.z:.3f} mm above driver z_max_safe={z_ceil} mm.")
        table_z = getattr(self._cfg, "table_z_mm", None)
        if table_z is not None:
            payload_offset = float(self._cfg.payload_protrusion_mm) if self._holding_payload else 0.0
            lowest_z = pose.z - float(self._cfg.gripper_lowest_offset_mm) - payload_offset
            lowest_floor = float(table_z) + float(self._cfg.minimum_floor_margin_mm)
            if lowest_z < lowest_floor:
                raise ValueError(
                    f"{label}: effective lowest z={lowest_z:.3f} mm below table clearance floor "
                    f"{lowest_floor:.3f} mm (control z={pose.z:.3f}, holding_payload={self._holding_payload})."
                )
        bounds = self._cfg.workspace_bounds
        if bounds is not None:
            xmin, ymin, xmax, ymax = bounds
            if not (xmin <= pose.x <= xmax):
                raise ValueError(f"{label}: x={pose.x:.3f} mm out of workspace x=[{xmin}, {xmax}].")
            if not (ymin <= pose.y <= ymax):
                raise ValueError(f"{label}: y={pose.y:.3f} mm out of workspace y=[{ymin}, {ymax}].")

    def _joint_waypoints(self, current: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
        """Linear joint interpolation; ``steps = ceil(max|Δ| / max_joint_step_deg)``."""
        delta = np.abs(target - current)
        max_delta = float(np.max(delta)) if delta.size else 0.0
        if max_delta <= 1e-12:
            return [target.copy()]
        steps = max(1, int(math.ceil(max_delta / float(self._cfg.max_joint_step_deg))))
        return [current + (target - current) * (k / steps) for k in range(1, steps + 1)]

    def _check_timeout(self, deadline: float) -> None:
        if self._monotonic() > deadline:
            raise TimeoutError(f"SO-101 motion did not settle within the move timeout ({self._cfg.move_timeout_s}s).")

    def _check_gripper_timeout(self, deadline: float) -> None:
        if self._monotonic() > deadline:
            raise TimeoutError(
                f"SO-101 gripper did not settle within the gripper timeout ({self._cfg.gripper_timeout_s}s)."
            )

    # --- staticmethod helpers (no instance state) --------------------------
    @staticmethod
    def _import_lerobot() -> tuple[Any, Any, Any, str]:
        try:
            import lerobot  # noqa: F401
        except ImportError as exc:  # pragma: no cover - hardware-only path
            raise RuntimeError(
                'LeRobot is required for the SO-101 driver. Install it with: pip install -e ".[so101]"'
            ) from exc
        import lerobot as _lerobot

        version = getattr(_lerobot, "__version__", "0.0.0")
        vt = _lerobot_version_tuple(version)
        if not (vt >= (0, 6, 0) and vt < (0, 7, 0)):
            raise RuntimeError(f"SO-101 driver requires LeRobot >=0.6.0,<0.7.0, got {version}.")
        from lerobot.model.kinematics import RobotKinematics  # noqa: F401
        from lerobot.robots.so_follower.config_so_follower import (  # noqa: F401
            SOFollowerRobotConfig,
        )
        from lerobot.robots.so_follower.so_follower import SOFollower  # noqa: F401

        return SOFollower, SOFollowerRobotConfig, RobotKinematics, version

    @staticmethod
    def _validate_joint_vector(q: list[float], *, label: str) -> None:
        if not isinstance(q, (list, tuple, np.ndarray)):
            raise ValueError(f"{label} must be a sequence, got {type(q).__name__}.")
        if len(q) != len(ARM_JOINT_ORDER):
            raise ValueError(f"{label} must have {len(ARM_JOINT_ORDER)} joints, got {len(q)}.")
        for i, v in enumerate(q):
            if not _is_finite(v):
                raise ValueError(f"{label}[{i}] must be finite, got {v!r}.")
