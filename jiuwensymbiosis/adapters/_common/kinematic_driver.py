# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``KinematicArmDriver`` — a reusable ``CartesianDriver`` for joint-level arms.

For an arm whose SDK only takes joint targets, this driver supplies the entire
Cartesian layer the framework expects of a driver (``move_to_pose_blocking`` /
``move_joint_blocking`` / ``home`` / ``get_pose`` / safety floors), built once
above two injected seams:

  * a ``JointTransport`` — the vendor SDK glue (open / read / send / close);
  * an ``FkIkBackend`` — a URDF kinematics solver.

The driver is capability-complete: besides motion it structurally satisfies the
framework's optional driver protocols so a joint-level arm composes with the
*existing* capability machinery instead of a bespoke subset —
``GripperDriver`` + ``SuctionDriver`` (one two-state effector channel),
``ServoDriver`` (single-step IK, non-blocking), and ``CameraDriver`` +
``VisionDriver`` (forwarded from the transport when it carries a camera /
hand-eye calibration). An Env binds it as ``low_level`` exactly like any other
driver, and ``env.set_end_effector`` / ``env.servo_to_flange`` / the vision
helpers dispatch to it unchanged.

``sleep`` / ``clock`` are injectable so tests drive trajectories without wall time.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jiuwensymbiosis.adapters._common.geometry import (
    CartesianPose,
    matrix_to_pose,
    orientation_error_deg,
    pose_to_matrix,
    position_error_mm,
    read_pose_xyzrpy,
)
from jiuwensymbiosis.adapters._common.joint_transport import JointTransport
from jiuwensymbiosis.adapters._common.kinematics import (
    CartesianServoError,
    FkIkBackend,
    MotionLimits,
    _check_limits,
    interp_se3_at,
    plan_cartesian_move,
    plan_joint_move,
    solve_ik_checked,
)
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)

# Bounded Cartesian servo search: start at the minimum safe path fraction and
# grow along one continuous IK branch (a far failed target can poison Placo's
# internal solver state for subsequent small solves).
_SERVO_ALPHA_SEARCH_ITERS = 14
_SERVO_ALPHA_MIN = 1.0 / 256.0
_SERVO_PROGRESS_EPS_MM = 1e-3
_SERVO_PROGRESS_EPS_DEG = 1e-3


@dataclass(frozen=True, slots=True)
class KinematicSpec:
    """Everything ``KinematicArmDriver`` needs, mapped from an adapter's config.

    All fields are generic robot constants — no vendor or SDK specifics. The
    arm joints are the FK/IK joints only; the end effector is a separate channel.
    """

    arm_joint_names: tuple[str, ...]
    home_joints: tuple[float, ...]
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    # When True, ``home_joints`` may be empty and connect() captures the live
    # startup joints as home (an SDK that boots at a known safe pose).
    home_from_startup: bool = False

    # kinematics / IK
    ik_orientation_weight: float = 0.01
    ik_position_tolerance_mm: float = 3.0
    ik_orientation_tolerance_deg: float | None = None
    euler_axes: str = "xyz"

    # trajectory / arrival
    max_joint_step_deg: float = 2.0
    max_cartesian_step_mm: float = 5.0
    max_ik_jump_deg: float = 30.0
    trajectory_hz: float = 30.0
    joint_tolerance_deg: float = 1.5
    settle_samples: int = 3
    move_timeout_s: float = 30.0

    # safety
    z_min_safe_mm: float = 30.0
    z_max_safe_mm: float | None = None  # optional control-frame ceiling (None = no cap)
    tool_offset_mm: float = 0.0

    # end effector (two-state; engaged = holding, e.g. gripper closed / suction on)
    effector_engaged_pos: float = 0.0
    effector_released_pos: float = 100.0
    effector_settle_s: float = 0.4

    # Contact-aware close (position-feedback effector only). On an ENGAGE command,
    # once the effector has travelled >= gripper_contact_min_travel and then stalls
    # (spread <= stall_tolerance over stall_samples reads), object contact is
    # inferred and a small hold preload (contact + hold_offset toward target) is
    # applied instead of driving to the full closed position. 0 travel disables it.
    gripper_contact_min_travel: float = 0.0
    gripper_contact_stall_tolerance: float = 0.5
    gripper_contact_stall_samples: int = 5
    gripper_contact_hold_offset: float = 1.0

    # settle-loop hardening (real-servo safety). Defaults reproduce the bare
    # resend loop. A plant with steady-state PD droop (e.g. Feetech STS3215,
    # whose firmware position-loop I term is inert) sets an overcompensate gain
    # plus a throttled resend + drift abort so a gravity-loaded joint is not
    # overdriven toward a mechanical limit.
    settle_resend_period_s: float = 0.0  # 0 -> resend at 1/trajectory_hz
    settle_drift_abort_samples: int = 0  # 0 -> disabled
    settle_overcompensate_gain: float = 0.0  # 0 -> bare resend; 1.0 -> resend target + (target - observed)

    # Real-time servo velocity enforcement (hardware-boundary safety). The
    # caller's tick rate is untrusted (a busy loop / misconfigured control_hz can
    # call servo_to_pose far faster than the arm tracks), so the driver caps the
    # ACTUAL joint velocity itself: a call within servo_min_send_period_s is a
    # no-op (non-blocking, no catch-up), and the per-tick step is re-clipped
    # against vel_cap = servo_max_joint_vel_dps * dt so speed is tick-rate
    # independent.
    servo_min_send_period_s: float = 0.02
    servo_max_joint_vel_dps: float = 30.0
    servo_max_joint_step_deg: float = 1.0

    # End-effector arrival. When both are set AND the transport reports real
    # feedback, poll read_effector until converged (an SDK that clips per-command
    # deltas cannot move 0<->100 in one send); else send-once + effector_settle_s.
    effector_tolerance: float | None = None
    effector_timeout_s: float | None = None

    # Fail-closed safety gate. An adapter shipping placeholder home/limits sets
    # requires_safety_validation=True so connect() refuses until the operator
    # confirms real values on the robot with safety_validated=True.
    requires_safety_validation: bool = False
    safety_validated: bool = False

    def __post_init__(self) -> None:
        if not self.home_from_startup and len(self.home_joints) != len(self.arm_joint_names):
            raise ValueError(
                f"home_joints has {len(self.home_joints)} values but there are {len(self.arm_joint_names)} arm joints"
            )
        if self.settle_samples < 1:
            raise ValueError(f"settle_samples must be >= 1, got {self.settle_samples}")
        if self.settle_resend_period_s < 0:
            raise ValueError(f"settle_resend_period_s must be >= 0, got {self.settle_resend_period_s}")
        if self.settle_drift_abort_samples < 0:
            raise ValueError(f"settle_drift_abort_samples must be >= 0, got {self.settle_drift_abort_samples}")
        if self.settle_overcompensate_gain < 0:
            raise ValueError(f"settle_overcompensate_gain must be >= 0, got {self.settle_overcompensate_gain}")
        if self.effector_tolerance is not None and not (self.effector_tolerance > 0):
            raise ValueError(f"effector_tolerance must be > 0 or None, got {self.effector_tolerance}")
        if self.effector_timeout_s is not None and not (self.effector_timeout_s > 0):
            raise ValueError(f"effector_timeout_s must be > 0 or None, got {self.effector_timeout_s}")
        for name, val in (
            ("servo_min_send_period_s", self.servo_min_send_period_s),
            ("servo_max_joint_vel_dps", self.servo_max_joint_vel_dps),
            ("servo_max_joint_step_deg", self.servo_max_joint_step_deg),
        ):
            if not (val > 0):
                raise ValueError(f"{name} must be > 0, got {val}")

    def to_motion_limits(self) -> MotionLimits:
        return MotionLimits(
            joint_names=tuple(self.arm_joint_names),
            joint_limits=tuple(self.joint_limits.get(name) for name in self.arm_joint_names),
            max_joint_step_deg=self.max_joint_step_deg,
            max_cartesian_step_mm=self.max_cartesian_step_mm,
            max_ik_jump_deg=self.max_ik_jump_deg,
            ik_position_tolerance_mm=self.ik_position_tolerance_mm,
            ik_orientation_tolerance_deg=self.ik_orientation_tolerance_deg,
            ik_orientation_weight=self.ik_orientation_weight,
        )


class KinematicArmDriver:
    """Reusable, capability-complete driver for a joint-level arm with local FK/IK."""

    def __init__(
        self,
        transport: JointTransport,
        backend: FkIkBackend,
        spec: KinematicSpec,
        *,
        name: str = "arm",
        sleep: Callable[[float], Any] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._transport = transport
        self._backend = backend
        self._spec = spec
        self._name = name
        self._sleep = sleep
        self._clock = clock
        self._limits = spec.to_motion_limits()
        self._connected = False
        self._effector_engaged = False
        # Monotonic timestamp of the last dispatched servo action (None = none yet).
        self._servo_last_send_t: float | None = None
        # Home joints; when home_from_startup, connect() captures the live pose.
        self._home_joints: list[float] = [float(v) for v in spec.home_joints]
        # Structured result of the last effector command (contact inference, etc.).
        self._last_gripper_result: dict[str, Any] | None = None

    # ----------------------------------------------------------- lifecycle
    def connect(self) -> None:
        """Open the transport, run prechecks, verify FK, then mark connected.

        Atomic: any failure after ``open`` closes the transport and re-raises,
        leaving the driver unconnected.
        """
        if self._connected:
            return
        # Fail-closed: refuse to open the transport (no serial/torque) until the
        # operator has confirmed the shipped home/limits on the real robot.
        if self._spec.requires_safety_validation and not self._spec.safety_validated:
            raise RuntimeError(
                f"{self._name} connect refused: config not safety-validated. "
                "home_joints / joint_limits ship as unverified placeholders; confirm a "
                "safe home and tightened limits on the real robot, then set safety_validated=True."
            )
        self._transport.open()
        ok = False
        try:
            self._transport.precheck()
            current = self._read_arm_joints()
            self._backend.forward_kinematics(current)
            if self._spec.home_from_startup:
                # No shipped home: adopt the live startup pose as the home target.
                self._home_joints = [float(v) for v in current]
            self._backend.forward_kinematics(self._home_joints)
            ok = True
        finally:
            if not ok:
                self._best_effort_close()
        self._connected = True
        logger.info("%s connected (%d arm joints)", self._name, len(self._spec.arm_joint_names))

    def disconnect(self) -> None:
        """Release the transport. Idempotent and safe at any state."""
        self._connected = False
        self._best_effort_close()

    def close(self) -> None:
        """Alias of ``disconnect`` (RobotDriver contract); same idempotent path."""
        self.disconnect()

    def _best_effort_close(self) -> None:
        try:
            self._transport.close()
        except Exception as exc:  # best-effort cleanup: never mask the real error
            logger.warning("%s: transport close failed: %s", self._name, exc)

    # ----------------------------------------------------------- properties
    @property
    def home_pose(self) -> CartesianPose:
        """Control-frame pose at the home joints (FK, no motion)."""
        matrix = self._backend.forward_kinematics(self._home_joints)
        return matrix_to_pose(matrix, euler_axes=self._spec.euler_axes)

    @property
    def last_gripper_result(self) -> dict[str, Any] | None:
        """Structured result of the last effector command (e.g. contact inference)."""
        return dict(self._last_gripper_result) if self._last_gripper_result is not None else None

    @property
    def z_min_safe(self) -> float:
        return float(self._spec.z_min_safe_mm)

    @property
    def flange_z_min_safe(self) -> float:
        return float(self._spec.z_min_safe_mm + self._spec.tool_offset_mm)

    @property
    def tool_offset_mm(self) -> float:
        return float(self._spec.tool_offset_mm)

    # ----------------------------------------------------------- reads
    def get_angles(self) -> list[float]:
        """Current arm joints (no effector), in the transport's native order."""
        self._require_connected()
        return self._read_arm_joints()

    def get_effector_position(self) -> float:
        """Current end-effector value in the transport's native unit."""
        self._require_connected()
        return float(self._transport.read_effector())

    def get_pose(self) -> CartesianPose:
        """FK of the current arm joints → control-frame pose (mm/deg)."""
        self._require_connected()
        matrix = self._backend.forward_kinematics(self._read_arm_joints())
        return matrix_to_pose(matrix, euler_axes=self._spec.euler_axes)

    # ----------------------------------------------------------- motion
    def home(self) -> None:
        self.move_joint_blocking(list(self._home_joints))

    def recovery_home(self) -> None:
        """RecoveryRail entry: planned, joint-limit-validated path back to home."""
        self.home()

    def retreat_home(self) -> None:
        """Alias of recovery_home for callers expecting a payload-aware retreat."""
        self.home()

    def move_joint_blocking(self, q: list[float], *, timeout_s: float | None = None) -> None:
        """Move the arm joints to ``q`` (blocking). Rejects invalid ``q`` first."""
        self._require_connected()
        waypoints = plan_joint_move(self._read_arm_joints(), [float(v) for v in q], self._limits)
        self._run_trajectory(waypoints, timeout_s)

    def move_to_pose_blocking(self, pose: Any, *args: Any, timeout_s: float | None = None, **kwargs: Any) -> None:
        """Move the control frame to ``pose`` (mm/deg) via local IK (blocking).

        Pre-solves and validates the whole path before commanding; an
        unreachable / out-of-limit / out-of-floor target raises ``ValueError``.
        """
        self._require_connected()
        _, _, z, *_ = read_pose_xyzrpy(pose)
        self._check_flange_z(z)
        target_matrix = pose_to_matrix(pose, euler_axes=self._spec.euler_axes)
        plan = plan_cartesian_move(self._backend, self._read_arm_joints(), target_matrix, self._limits)
        self._run_trajectory([wp.q for wp in plan], timeout_s)

    def servo_to_pose(self, pose: Any) -> None:
        """Non-blocking bounded-Cartesian servo step (``ServoDriver``).

        The caller's real-time loop calls this each tick. Velocity is enforced
        HERE, not by the caller: a call within ``servo_min_send_period_s`` of the
        last dispatch is a no-op (non-blocking, no catch-up). Rather than solving
        the final pose and clipping each joint independently (which on a coupled
        arm can turn an upward Cartesian move into a downward FK move), it
        interpolates the Cartesian path, solves IK per candidate, and dispatches
        the largest fraction that satisfies the joint-velocity cap, the Cartesian
        velocity cap, IK residual checks and a Cartesian-progress check. Raises
        :class:`CartesianServoError` (typed code) when no safe progress step exists.
        """
        self._require_connected()
        _, _, z, *_ = read_pose_xyzrpy(pose)
        try:
            self._check_flange_z(z)
        except ValueError as exc:
            raise CartesianServoError("cartesian_bounds_rejected", str(exc)) from exc

        now = self._clock()
        min_period = self._spec.servo_min_send_period_s
        last_send_t = self._servo_last_send_t
        if last_send_t is not None and (now - last_send_t) < min_period:
            return

        dt = min_period if last_send_t is None else now - last_send_t
        max_step = min(self._spec.servo_max_joint_step_deg, self._spec.servo_max_joint_vel_dps * dt)
        if not math.isfinite(max_step) or max_step <= 0.0:
            raise ValueError(f"{self._name}: servo joint step cap is invalid ({max_step!r}).")

        current = self._read_arm_joints()
        _check_limits(current, self._limits)
        target_matrix = pose_to_matrix(pose, euler_axes=self._spec.euler_axes)
        current_matrix = self._backend.forward_kinematics(current)
        # Cartesian velocity cap for this tick = per-interp-step distance * control rate * dt.
        cartesian_step_cap = self._spec.max_cartesian_step_mm * self._spec.trajectory_hz * dt

        best_q, first_err = self._servo_search(current, current_matrix, target_matrix, max_step, cartesian_step_cap)
        if best_q is None:
            code, detail = first_err or ("joint_velocity_limited_no_progress", "no safe Cartesian progress step")
            raise CartesianServoError(code, f"{self._name} servo_to_pose: {detail}")

        _check_limits(best_q, self._limits)
        self._transport.send_arm_joints(best_q)
        self._servo_last_send_t = self._clock()

    def _servo_search(
        self,
        current: list[float],
        current_matrix: np.ndarray,
        target_matrix: np.ndarray,
        max_step: float,
        cartesian_step_cap: float,
    ) -> tuple[list[float] | None, tuple[str, str] | None]:
        """Grow the path fraction along one IK branch; return (best_q, first_error).

        Position-only IK weight is tried first on an underactuated arm, then the
        configured weight. Each candidate must satisfy the joint step cap, the
        Cartesian velocity cap, and reduce Cartesian error (position first, else
        orientation). The largest passing fraction (up to the velocity-limited
        ``alpha_limit``) wins; failures are recorded with a typed reason.
        """
        current_pos_err = position_error_mm(current_matrix, target_matrix)
        current_ori_err = orientation_error_deg(current_matrix, target_matrix)
        alpha_limit = min(1.0, cartesian_step_cap / current_pos_err) if current_pos_err > 0.0 else 1.0
        weights = self._servo_orientation_weights()
        best_q: list[float] | None = None
        first_err: tuple[str, str] | None = None

        def fail(code: str, detail: str) -> None:
            nonlocal first_err
            if first_err is None:
                first_err = (code, detail)

        def evaluate(alpha: float) -> list[float] | None:
            matrix = interp_se3_at(current_matrix, target_matrix, alpha)
            q_cand: list[float] | None = None
            for weight in weights:
                try:
                    q_cand = solve_ik_checked(self._backend, current, matrix, self._limits, orientation_weight=weight).q
                    break
                except ValueError as exc:
                    fail("joint_velocity_limited_no_progress", str(exc))
            if q_cand is None:
                return None
            q_delta = max((abs(a - b) for a, b in zip(q_cand, current, strict=True)), default=0.0)
            if q_delta > max_step + 1e-6:
                fail("joint_velocity_limited_no_progress", f"joint delta {q_delta:.3f}deg > cap {max_step:.3f}")
                return None
            cand_matrix = self._backend.forward_kinematics(q_cand)
            if position_error_mm(current_matrix, cand_matrix) > cartesian_step_cap + _SERVO_PROGRESS_EPS_MM:
                fail("joint_velocity_limited_no_progress", "Cartesian step exceeds velocity cap")
                return None
            cand_pos_err = position_error_mm(cand_matrix, target_matrix)
            cand_ori_err = orientation_error_deg(cand_matrix, target_matrix)
            if current_pos_err > _SERVO_PROGRESS_EPS_MM:
                progressing = cand_pos_err < current_pos_err - _SERVO_PROGRESS_EPS_MM
            elif current_ori_err > _SERVO_PROGRESS_EPS_DEG:
                progressing = cand_ori_err < current_ori_err - _SERVO_PROGRESS_EPS_DEG
            else:
                progressing = False
            if not progressing:
                fail("cartesian_progress_reversed", "candidate does not reduce Cartesian error")
                return None
            return q_cand

        alpha = min(_SERVO_ALPHA_MIN, alpha_limit)
        for _ in range(_SERVO_ALPHA_SEARCH_ITERS):
            q_cand = evaluate(alpha)
            if q_cand is None:
                break
            best_q = q_cand
            if alpha >= alpha_limit:
                break
            alpha = min(alpha_limit, alpha * 2.0)
        return best_q, first_err

    def _servo_orientation_weights(self) -> list[float]:
        """Position-only weight first (underactuated arm) then the configured weight."""
        weight = self._spec.ik_orientation_weight
        if self._spec.ik_orientation_tolerance_deg is None and weight != 0.0:
            return [0.0, weight]
        return [weight]

    # ----------------------------------------------------------- end effector
    # One two-state channel serves both a parallel gripper and a suction cup;
    # engaged = holding (gripper closed / suction on). The driver therefore
    # satisfies GripperDriver and SuctionDriver, and env.set_end_effector picks
    # the right verb by the declared grasp capability.
    def _set_effector(self, engaged: bool) -> None:
        self._require_connected()
        target = float(self._spec.effector_engaged_pos if engaged else self._spec.effector_released_pos)
        self._effector_engaged = bool(engaged)
        tol = self._spec.effector_tolerance
        timeout = self._spec.effector_timeout_s
        default_state = "closed" if engaged else "open"
        if tol is None or timeout is None:
            # Open-loop (no position feedback): send once + optional dwell.
            self._transport.send_effector(target)
            if self._spec.effector_settle_s > 0:
                self._sleep(self._spec.effector_settle_s)
            self._last_gripper_result = {"state": default_state, "target": target}
            return
        # Closed-loop: an SDK that clips per-command deltas cannot move the full
        # range in one send, so poll real feedback until converged. On an ENGAGE
        # command with contact inference enabled, a mid-travel stall is read as
        # object contact and a small hold preload is applied instead of driving
        # to the full closed position.
        infer = engaged and self._spec.gripper_contact_min_travel > 0
        contact = self._converge_effector(target, tol, timeout, infer_contact=infer)
        if not contact:
            self._last_gripper_result = {"state": default_state, "target": target}

    def _converge_effector(self, target: float, tol: float, timeout: float, *, infer_contact: bool) -> bool:
        """Resend/read the effector until observed holds within ``tol``, or time out.

        Returns True if object contact was inferred (a hold preload was applied and
        ``last_gripper_result`` set), else False (the caller records the state).

        Completion is judged from ``read_effector``, never the commanded value.
        Throttled by ``settle_resend_period_s`` (else the trajectory period) so the
        loop never saturates the bus; an optional ``effector_settle_s`` dwell runs
        after convergence. When ``infer_contact`` and the effector has travelled
        ``gripper_contact_min_travel`` then stalls (spread <= stall_tolerance over
        stall_samples reads), object contact is inferred: a small hold preload is
        applied and the loop returns with a ``"contact"`` result instead of
        driving to the full closed position.
        """
        period = self._spec.settle_resend_period_s if self._spec.settle_resend_period_s > 0 else self._settle_period()
        deadline = self._clock() + timeout
        start = float(self._transport.read_effector())
        direction = 1.0 if target >= start else -1.0
        window: list[float] = []
        stall_samples = self._spec.gripper_contact_stall_samples
        settle = 0
        observed = start
        while True:
            self._transport.send_effector(target)
            observed = float(self._transport.read_effector())
            if infer_contact and (observed - start) * direction >= self._spec.gripper_contact_min_travel:
                window.append(observed)
                window = window[-stall_samples:]
                if (
                    len(window) >= stall_samples
                    and max(window) - min(window) <= self._spec.gripper_contact_stall_tolerance
                ):
                    self._apply_contact_hold(observed, target, direction)
                    return True
            if abs(observed - target) <= tol:
                settle += 1
                if settle >= self._spec.settle_samples:
                    break
            else:
                settle = 0
            if self._clock() > deadline:
                raise TimeoutError(
                    f"{self._name}: effector did not reach {target} within {timeout:.1f}s "
                    f"(observed {observed}, tol {tol})"
                )
            if period:
                self._sleep(period)
        if self._spec.effector_settle_s > 0:
            self._sleep(self._spec.effector_settle_s)
        return False

    def _apply_contact_hold(self, contact: float, target: float, direction: float) -> None:
        """Object contact inferred mid-close: hold at contact + a small preload."""
        remaining = abs(target - contact)
        preload = min(self._spec.gripper_contact_hold_offset, remaining)
        hold_target = contact + direction * preload
        self._transport.send_effector(hold_target)
        if self._spec.effector_settle_s > 0:
            self._sleep(self._spec.effector_settle_s)
        logger.info("%s gripper contact inferred at %.3f; holding at %.3f", self._name, contact, hold_target)
        self._last_gripper_result = {
            "state": "contact",
            "target": target,
            "contact_position": contact,
            "hold_target": hold_target,
        }

    def _settle_period(self) -> float:
        return 1.0 / self._spec.trajectory_hz if self._spec.trajectory_hz > 0 else 0.0

    def set_gripper(self, engaged: bool) -> None:
        """Two-state parallel gripper (True = close / grip). ``GripperDriver``."""
        self._set_effector(engaged)

    def set_suction(self, on: bool) -> None:
        """Two-state suction (True = on / grip). ``SuctionDriver``."""
        self._set_effector(on)

    @property
    def gripper_state(self) -> bool:
        return self._effector_engaged

    @property
    def suction_state(self) -> bool:
        return self._effector_engaged

    @property
    def suction_di_last(self) -> int | None:
        return None

    # ----------------------------------------------------------- vision (forwarded)
    # Camera / hand-eye calibration live on the transport (they are not part of
    # the kinematics). The driver forwards the CameraDriver + VisionDriver surface
    # so VisionMixin's shared grasp pipeline (via the _project_pixel_to_base_raw
    # seam) works when the transport carries a camera, and reads as "absent"
    # (None) otherwise.
    def grab_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        grab = getattr(self._transport, "grab_frames", None)
        return grab() if callable(grab) else None

    @property
    def camera_available(self) -> bool:
        """Whether the transport is currently streaming a camera (else False)."""
        return bool(getattr(self._transport, "camera_available", False))

    @property
    def intrinsics(self) -> np.ndarray | None:
        return getattr(self._transport, "intrinsics", None)

    @property
    def tf_flange_cam(self) -> np.ndarray | None:
        """Eye-in-hand transform (camera->flange); None for eye-to-hand / no camera."""
        return getattr(self._transport, "tf_flange_cam", None)

    @property
    def tf_base_cam(self) -> np.ndarray | None:
        """Eye-to-hand transform (camera-in-base, constant); None for eye-in-hand / no camera."""
        return getattr(self._transport, "tf_base_cam", None)

    @property
    def calibration(self) -> dict | None:
        return getattr(self._transport, "calibration", None)

    # ----------------------------------------------------------- internals
    def _read_arm_joints(self) -> list[float]:
        return [float(v) for v in self._transport.read_arm_joints()]

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(f"{self._name}: driver not connected")

    def _check_flange_z(self, z: float) -> None:
        floor = self._spec.z_min_safe_mm + self._spec.tool_offset_mm
        if not math.isfinite(z):
            raise ValueError(f"{self._name}: move blocked, non-finite z={z}")
        if z < floor:
            raise ValueError(f"{self._name}: move blocked, flange z={z:.1f}mm below floor {floor:.1f}mm")
        if self._spec.z_max_safe_mm is not None:
            ceiling = self._spec.z_max_safe_mm + self._spec.tool_offset_mm
            if z > ceiling:
                raise ValueError(f"{self._name}: move blocked, flange z={z:.1f}mm above ceiling {ceiling:.1f}mm")

    def _run_trajectory(self, waypoints: list[list[float]], timeout_s: float | None) -> None:
        """Stream pre-solved joint waypoints, then poll to settle on the last one.

        Two phases: an interpolation sweep that sends each pre-solved waypoint
        once (the plan already respects ``max_joint_step_deg``, so no
        overcompensation), then a settle phase that resends the final target —
        optionally overcompensated, rate-throttled, and drift-aborted — until the
        OBSERVED joints hold within tolerance. Arrival is judged only from
        observation, never the commanded value (an SDK may clip per-command
        deltas).
        """
        if not waypoints:
            return
        final = [float(v) for v in waypoints[-1]]
        timeout = timeout_s if timeout_s is not None else self._spec.move_timeout_s
        period = self._settle_period()
        resend_period = self._spec.settle_resend_period_s if self._spec.settle_resend_period_s > 0 else period
        tol = self._spec.joint_tolerance_deg
        drift_cap = self._spec.settle_drift_abort_samples
        gain = self._spec.settle_overcompensate_gain
        deadline = self._clock() + timeout

        # Interpolation sweep — bare targets, no overcompensation.
        last_command = final
        for wp in waypoints:
            if self._clock() > deadline:
                raise TimeoutError(f"{self._name}: move exceeded {timeout:.1f}s during trajectory dispatch")
            cmd = [float(v) for v in wp]
            self._transport.send_arm_joints(list(cmd))
            last_command = cmd
            if period:
                self._sleep(period)

        # Settle phase — resend final until observed converges.
        settle = 0
        drift_count = 0
        prev_err: float | None = None
        while True:
            observed = self._read_arm_joints()
            err = max((abs(obs - tgt) for obs, tgt in zip(observed, final, strict=True)), default=0.0)
            if err <= tol:
                settle += 1
                if settle >= self._spec.settle_samples:
                    return
            else:
                settle = 0
            if self._clock() > deadline:
                raise TimeoutError(
                    f"{self._name}: move did not settle within {timeout:.1f}s (joint error {err:.2f}deg > tol {tol}deg)"
                )
            if drift_cap > 0 and prev_err is not None:
                if err > prev_err + 1e-9:
                    drift_count += 1
                    if drift_count >= drift_cap:
                        raise RuntimeError(
                            f"{self._name}: settle drift — joint error grew for {drift_count} consecutive "
                            f"resends (servo diverging under load); aborted at {err:.2f}deg"
                        )
                else:
                    drift_count = 0
            prev_err = err
            last_command = self._settle_resend(final, observed, last_command, gain)
            if resend_period:
                self._sleep(resend_period)

    def _settle_resend(
        self, final: list[float], observed: list[float], last_command: list[float], gain: float
    ) -> list[float]:
        """Resend toward the arm target during settle; return the command sent.

        ``gain <= 0`` resends the bare target. Otherwise aim past the target by
        ``gain * (target - observed)`` to close a steady-state PD droop, clamp the
        step from the LAST command to ``max_joint_step_deg`` (so a stalled joint
        moves at most one step per resend, never a full-error jump toward a
        limit), and fall back to the bare target when the over-command is
        non-finite or breaks a soft limit (fail-closed).
        """
        if gain > 0.0:
            max_step = self._spec.max_joint_step_deg
            command = [
                last + max(-max_step, min(max_step, (tgt + gain * (tgt - obs)) - last))
                for last, tgt, obs in zip(last_command, final, observed, strict=True)
            ]
            if self._within_soft_limits(command):
                self._transport.send_arm_joints(list(command))
                return command
            logger.warning("%s: settle over-command out of soft limits; resending bare target", self._name)
        self._transport.send_arm_joints(list(final))
        return list(final)

    def _within_soft_limits(self, q: list[float]) -> bool:
        """True if every joint value is finite and inside its soft limit (if any)."""
        for value, lim in zip(q, self._limits.joint_limits, strict=True):
            if not math.isfinite(value):
                return False
            if lim is not None and not (lim[0] <= value <= lim[1]):
                return False
        return True
