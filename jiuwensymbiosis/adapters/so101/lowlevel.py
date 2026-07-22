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

import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

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
    "So101Driver",
    "So101PoseConvergenceError",
]

_logger = get_logger(__name__)

# Cap on the number of SE(3) interpolation steps in Cartesian path planning. The
# planner splits start->target into N = ceil(max(translation_mm, rotation_deg) /
# cartesian_interp_step_mm) steps (one IK per step, seeded by the previous step's
# solution). A count cap — not a wall-clock timeout — keeps tests deterministic
# and bounds the worst-case work if a caller requests a huge Cartesian move.
_MAX_CARTESIAN_WAYPOINTS = 4096

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
        so_follower_factory: Callable[..., Any] | None = None,
        kinematics_factory: Callable[..., Any] | None = None,
        lerobot_import: Callable[[], tuple[Any, Any, Any, str]] | None = None,
    ) -> None:
        from jiuwensymbiosis.adapters.so101.config import So101Config  # noqa: F401

        if not isinstance(cfg, So101Config):
            raise TypeError(f"cfg must be a So101Config, got {type(cfg).__name__}.")
        self._cfg = cfg
        self._sleep = sleep
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

        # Real-time servo velocity gate: monotonic timestamp of the last
        # dispatched servo action. ``servo_to_pose`` enforces a minimum
        # inter-send interval and a deg/s cap derived from the real ``dt``,
        # so actual joint velocity is independent of the caller's tick rate.
        self._servo_last_send_t: float = 0.0

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
        if not self._cfg.safety_validated:
            raise RuntimeError(
                "SO-101 connect refused: config not safety-validated. "
                "home_joints_deg / joint_limits ship as UNVERIFIED placeholders. "
                "Manually confirm a safe home (teach pendant) and tighten joint_limits "
                "to measured safe ranges, then set `safety_validated: true` in the YAML."
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

            # Step 7: build RobotKinematics with target_frame_name from config.
            kin_factory = self._kinematics_factory or RobotKinematics
            kin = kin_factory(
                self._urdf_path,
                target_frame_name=self._cfg.ik_target_frame,
                joint_names=list(ARM_JOINT_ORDER),
            )

            # Step 8: validate current FK + home FK/limits.
            current = self._read_arm_angles(robot)
            _ = kin.forward_kinematics(np.asarray(current, dtype=float))  # raises if bad frame
            home = np.asarray(self._cfg.home_joints_deg, dtype=float)
            _ = kin.forward_kinematics(home)
            self._check_joint_limits(home, label="home_joints_deg")

            self._robot = robot
            self._kin = kin
            self._connected = True
            self._servo_last_send_t = 0.0

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
        self._servo_last_send_t = 0.0

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

    def get_gripper_position(self) -> float:
        """Return the gripper target in 0..100 % (native SO-101 units)."""
        self._require_connected()
        obs = self._robot.get_observation()
        return self._read_motor(obs, "gripper")

    def get_pose(self) -> So101Pose:
        """FK(current 5 joints) -> control-frame pose (mm / XYZ-Euler deg)."""
        self._require_connected()
        q = np.asarray(self.get_angles(), dtype=float)
        matrix = self._kin.forward_kinematics(q)
        return matrix_m_to_pose_mm_deg(np.asarray(matrix, dtype=float))

    def set_gripper(self, on: bool) -> None:
        """Drive the gripper to the configured two-state target, blocking until settled.

        Sends ONLY ``{"gripper.pos": target}`` — never any arm joint keys — so an
        in-flight arm motion is not disturbed. Under the default
        ``max_relative_target`` a single ``send_action`` cannot move 0->100 or
        100->0 in one step (LeRobot clips it), so this re-sends the target and
        polls the real gripper observation until it converges within
        ``gripper_tolerance`` for ``settle_samples`` consecutive reads, bounded by
        ``gripper_timeout_s`` (raises ``TimeoutError`` on stall). Completion is
        judged from real observation, not the ``send_action`` return; the actual
        (possibly clipped) target is recorded each send. Each poll waits one
        ``trajectory_hz`` period so the loop never saturates the serial bus. After
        convergence an optional ``gripper_settle_s`` dwell is waited via the
        injectable sleep.
        """
        self._require_connected()
        target = float(self._cfg.gripper_close_pos if on else self._cfg.gripper_open_pos)
        action = {"gripper.pos": target}
        deadline = time.monotonic() + float(self._cfg.gripper_timeout_s)
        settle_needed = max(1, int(self._cfg.settle_samples))
        # One period between polls so the loop never hammers the serial bus at full
        # speed (reuses the arm trajectory_hz for a consistent dispatch rate).
        period = 1.0 / float(self._cfg.trajectory_hz) if self._cfg.trajectory_hz > 0 else 0.0
        # Re-send the target (each send is itself clipped by max_relative_target)
        # until the observed gripper position converges. Observation polling alone
        # won't move the gripper toward the clipped goal.
        while True:
            self._check_gripper_timeout(deadline)
            self._send_action(action)
            observed = float(self.get_gripper_position())
            if abs(observed - target) <= self._cfg.gripper_tolerance:
                settle_needed -= 1
                if settle_needed <= 0:
                    break
            else:
                settle_needed = max(1, int(self._cfg.settle_samples))
            if period > 0:
                self._sleep(period)
        if self._cfg.gripper_settle_s > 0:
            self._sleep(float(self._cfg.gripper_settle_s))

    def home(self) -> None:
        """Move to the configured safe joint home (NOT a startup snapshot)."""
        self._require_connected()
        self.move_joint_blocking(list(self._cfg.home_joints_deg))

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
        self._validate_joint_vector(q, label="move_joint_blocking target")
        self._check_joint_limits(np.asarray(q, dtype=float), label="move_joint_blocking target")

        current = np.asarray(self.get_angles(), dtype=float)
        target = np.asarray(q, dtype=float)
        waypoints = self._joint_waypoints(current, target)

        # Pre-validate every waypoint before issuing the first action.  Joint
        # limits alone are insufficient: FK can put a legal joint vector below
        # the Z floor or outside the configured XY work area.
        for index, wp in enumerate(waypoints, start=1):
            self._validate_joint_waypoint(wp, label=f"joint waypoint {index}/{len(waypoints)}")

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

    def _dispatch_prevalidated_waypoints(
        self,
        waypoints: list[np.ndarray],
        final_target: np.ndarray,
        *,
        timeout_s: float | None,
    ) -> None:
        """Stream pre-validated joint waypoints, then settle to ``final_target``.

        Shared by joint-space and Cartesian motion. Callers MUST have validated
        every waypoint (finiteness, soft limits, Cartesian bounds, residuals)
        BEFORE calling this — it begins sending immediately. The settle loop
        re-sends the final target when observation hasn't converged (LeRobot may
        clip via ``max_relative_target``), judging completion from real
        observation, not from the ``send_action`` return value.
        """
        deadline = time.monotonic() + (timeout_s if timeout_s is not None else self._cfg.move_timeout_s)
        period = 1.0 / float(self._cfg.trajectory_hz) if self._cfg.trajectory_hz > 0 else 0.0
        # Settle re-send throttle: cap the rate the final target is re-sent at.
        # 0 falls back to the interpolation period (legacy 30 Hz behavior).
        resend_period = float(self._cfg.settle_resend_period_s) if self._cfg.settle_resend_period_s > 0 else period
        drift_cap = max(0, int(self._cfg.settle_drift_abort_samples))
        settle_needed = max(1, int(self._cfg.settle_samples))
        last_wp = waypoints[-1] if waypoints else np.asarray(final_target, dtype=float)
        # Keep the requested command path slew-limited even when LeRobot's
        # max_relative_target is disabled.  This is separate from the encoder
        # observation: a stalled servo must not make the next over-command jump
        # by the full position error.
        last_command = np.asarray(last_wp, dtype=float).copy()

        # Interpolation sweep: stream each pre-validated waypoint in turn.
        for wp in waypoints:
            self._check_timeout(deadline)
            self._send_action(_arm_action(wp.tolist()))
            last_command = np.asarray(wp, dtype=float).copy()
            if period > 0:
                self._sleep(period)

        # Settle loop: re-send the final target until observed joints converge.
        prev_err = float(np.max(np.abs(np.asarray(self.get_angles(), dtype=float) - last_wp)))
        drift_count = 0
        while True:
            self._check_timeout(deadline)
            actual = np.asarray(self.get_angles(), dtype=float)
            err = float(np.max(np.abs(actual - last_wp)))
            if err <= self._cfg.joint_tolerance_deg:
                settle_needed -= 1
                if settle_needed <= 0:
                    return
            else:
                settle_needed = max(1, int(self._cfg.settle_samples))
            # Drift abort: if the max joint error grew this round, the servo is
            # moving away from the target (gravity overcoming torque on a loaded
            # joint). Abort immediately rather than re-send toward a limit.
            if drift_cap > 0:
                if err > prev_err + 1e-6:
                    drift_count += 1
                    if drift_count >= drift_cap:
                        raise RuntimeError(
                            f"SO-101 settle drift: max joint error grew {drift_count} consecutive "
                            f"re-sends (err {prev_err:.3f} -> {err:.3f} deg, target within "
                            f"{self._cfg.joint_tolerance_deg} deg). Aborting to avoid pushing the arm "
                            f"toward a limit — the servo likely cannot track under gravity load."
                        )
                else:
                    drift_count = 0
                prev_err = err
            # Re-send to drive convergence: observation polling alone won't move
            # the arm if the last waypoint was clipped by max_relative_target.
            # With ``settle_overcompensate`` the re-send is ``target + e`` (e =
            # target - actual, fresh from the encoder each round) instead of the
            # bare ``target``: the STS3215 firmware I term is inert, so the servo
            # parks at ``target - e``; over-commanding ``target + e`` makes it
            # park AT ``target``, closing the PD steady-state error (~2.46 deg
            # elbow -> ~0). If the over-command would break a soft limit, fall
            # back to the bare target (fail-closed: keeps the residual but stays
            # in bounds) and log it.
            if self._cfg.settle_overcompensate:
                desired_cmd = last_wp + (last_wp - actual)  # = target + e
                try:
                    # Reject the desired over-command itself first.  A clipped
                    # intermediate command must not turn an out-of-limit target
                    # into a slow march toward the limit.
                    self._validate_joint_waypoint(
                        desired_cmd,
                        label="settle over-compensate desired",
                    )
                    delta = desired_cmd - last_command
                    max_step = float(self._cfg.max_joint_step_deg)
                    cmd = last_command + np.clip(delta, -max_step, max_step)
                    self._validate_joint_waypoint(
                        np.asarray(cmd, dtype=float),
                        label="settle over-compensate waypoint",
                    )
                    self._send_action(_arm_action(cmd.tolist()))
                    last_command = np.asarray(cmd, dtype=float)
                except ValueError as exc:
                    _logger.warning(
                        "SO-101 settle: over-command %s rejected (%s); re-sending bare target (residual %.3f deg).",
                        np.round(desired_cmd, 2).tolist(),
                        exc,
                        err,
                    )
                    self._send_action(_arm_action(last_wp.tolist()))
                    last_command = np.asarray(last_wp, dtype=float).copy()
            else:
                self._send_action(_arm_action(last_wp.tolist()))
                last_command = np.asarray(last_wp, dtype=float).copy()
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
        if not isinstance(pose, So101Pose):
            raise TypeError(f"pose must be a So101Pose, got {type(pose).__name__}.")

        q_target = self._dispatch_pose_move(pose, timeout_s=kwargs.get("timeout_s"))
        if self._cfg.pose_convergence_max_iters > 0:
            self._converge_to_pose(pose, q_target, timeout_s=kwargs.get("timeout_s"))

    def servo_to_pose(self, pose: Any) -> None:
        """Issue one non-blocking Cartesian servo command.

        Exactly one IK solve from the live encoder seed and one ``send_action``;
        the caller's real-time loop calls this again each tick. The IK solution
        is checked for residual/error and the dispatched waypoint is slew-limited
        and checked against the SO-101 safety envelope. Velocity (inter-send
        interval + deg/s cap) is enforced here, not by the caller.
        """
        self._require_connected()
        target = self._coerce_servo_pose(pose)
        _require_finite_so101_pose(target, label="servo_to_pose target")
        self._check_cartesian_bounds(target, label="servo_to_pose target")

        # Min inter-send interval: a call within this window is a no-op
        # (non-blocking; skipped calls do not accumulate or catch up).
        now = time.monotonic()
        min_period = float(self._cfg.servo_min_send_period_s)
        first_send = self._servo_last_send_t == 0.0
        if not first_send and (now - self._servo_last_send_t) < min_period:
            return

        q_current = np.asarray(self.get_angles(), dtype=float)
        self._validate_joint_vector(q_current.tolist(), label="servo_to_pose current")
        target_matrix = np.asarray(pose_mm_deg_to_matrix_m(target), dtype=float)
        try:
            q_ik = np.asarray(
                self._kin.inverse_kinematics(
                    q_current,
                    target_matrix,
                    position_weight=1.0,
                    orientation_weight=float(self._cfg.ik_orientation_weight),
                ),
                dtype=float,
            )
        except Exception as exc:  # noqa: BLE001 - normalize vendor IK failures
            raise ValueError(f"servo_to_pose: IK raised {exc!r}.") from exc

        self._validate_ik_solution(q_ik, target_matrix, label="servo_to_pose IK")
        # Re-clip the step against vel_cap = max_joint_vel_dps * dt; first send
        # uses one min_period (timestamp 0 is not a real instant).
        dt = min_period if first_send else now - self._servo_last_send_t
        vel_cap = float(self._cfg.servo_max_joint_vel_dps) * dt
        max_step = min(float(self._cfg.servo_max_joint_step_deg), vel_cap)
        q_cmd = q_current + np.clip(q_ik - q_current, -max_step, max_step)
        self._validate_joint_waypoint(np.asarray(q_cmd, dtype=float), label="servo_to_pose command")
        self._send_action(_arm_action(q_cmd.tolist()))
        self._servo_last_send_t = time.monotonic()

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
        self._check_cartesian_bounds(pose, label="goto_pose target")

        desired_matrix = np.asarray(pose_mm_deg_to_matrix_m(pose), dtype=float)
        current_q = np.asarray(self.get_angles(), dtype=float)
        start_matrix = np.asarray(self._kin.forward_kinematics(current_q), dtype=float)

        # 2. Plan the SE(3) waypoint path via the seed chain (one IK per step,
        #    seeded by the previous step's solution). All residuals, limits and
        #    Cartesian bounds are checked here, before any send_action.
        ik_waypoints = self._plan_cartesian_waypoints(current_q, start_matrix, desired_matrix, pose)

        # 3. Dispatch the pre-validated joint waypoints (shared settle loop).
        self._dispatch_prevalidated_waypoints(ik_waypoints, ik_waypoints[-1], timeout_s=timeout_s)
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
        / cartesian_interp_step_mm)`` (>= 1). Solves IK once per step, seeded by
        the PREVIOUS step's solution (the seed chain — ``seed`` starts at
        ``start_q`` and is updated to each step's IK result regardless of
        acceptance). This matches lerobot's ``InverseKinematicsEEToJoints`` with
        ``initial_guess_current_joints=False`` and keeps placo inside its
        convergence basin so IK does not jump branches between steps (verified
        ~0.005 mm residual on real IK for z +/-50 mm).

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

        def ik(seed: np.ndarray, matrix: np.ndarray) -> np.ndarray:
            return np.asarray(
                self._kin.inverse_kinematics(
                    seed,
                    matrix,
                    position_weight=1.0,
                    orientation_weight=float(self._cfg.ik_orientation_weight),
                ),
                dtype=float,
            )

        def validate_waypoint(q: np.ndarray, matrix: np.ndarray, label: str) -> None:
            """Validate a waypoint; raise on failure (no silent skip)."""
            self._validate_ik_solution(q, matrix, label=label)

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
            try:
                q_k = ik(seed, matrix_k)
            except Exception as exc:  # noqa: BLE001 - placo may raise on singular seeds
                raise ValueError(
                    f"goto_pose waypoint t={t:.4f}: IK raised {exc!r}; target likely unreachable or on a singularity."
                ) from exc
            # Update the seed to THIS step's solution before validating, so a
            # later step never re-solves from a stale seed even if this one is
            # the last accepted before a failure.
            seed = q_k
            validate_waypoint(q_k, matrix_k, label=f"goto_pose waypoint t={t:.4f}")
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
        """Second-layer Z-floor + XY-bound check the driver runs before sending.

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
        if time.monotonic() > deadline:
            raise TimeoutError(f"SO-101 motion did not settle within the move timeout ({self._cfg.move_timeout_s}s).")

    def _check_gripper_timeout(self, deadline: float) -> None:
        if time.monotonic() > deadline:
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
