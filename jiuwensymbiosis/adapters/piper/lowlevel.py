# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Low-level Piper driver — 6-DoF AgileX arm over CAN.
Prerequisites:
  - ``piper_sdk`` installed; the CAN interface (e.g. ``can_left``) up at 1 Mbps.
  - Arm powered and not in an error state.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jiuwensymbiosis.adapters._common.safety import WorkspaceBounds
from jiuwensymbiosis.adapters.piper._calibration import load_calibration
from jiuwensymbiosis.adapters.piper.geometry import FlangePose
from jiuwensymbiosis.perception.camera import RealSenseCamera

logger = logging.getLogger(__name__)

# SDK boundary scale: piper speaks 0.001 mm and 0.001 deg.
_FACTOR = 1000.0

# MotionCtrl_2 move modes.
_MOVE_P = 0x00  # point-to-point (reach-friendly; used for transit / home)
_MOVE_J = 0x01  # joint
_MOVE_L = 0x02  # linear cartesian (straight line; used for pick/place strokes)
_CTRL_CAN = 0x01
_GRIPPER_ENABLE = 0x01


# =============================================================================
# Per-run command log. Persists every ``[Piper]``
# motion line to ``logs/motion/<run-stamp>/commands.log`` so a real-run
# failure leaves a motion trace on disk. Disable with ``JIUWEN_PIPER_CMD_LOG=0``.
# =============================================================================
_CMD_LOG_RUN_STAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
_CMD_LOG_ATTACHED = False
_CMD_LOG_PATH: Path | None = None


def _attach_cmd_log_handler() -> Path | None:
    global _CMD_LOG_ATTACHED, _CMD_LOG_PATH
    if _CMD_LOG_ATTACHED:
        return _CMD_LOG_PATH
    _CMD_LOG_ATTACHED = True
    if os.environ.get("JIUWEN_PIPER_CMD_LOG", "") == "0":
        return None
    root = Path(os.environ.get("JIUWEN_PIPER_CMD_LOG_DIR") or os.environ.get("JIUWEN_CMD_LOG_DIR", "./logs/motion"))
    try:
        log_dir = root / _CMD_LOG_RUN_STAMP
        log_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("JIUWEN_MOTION_LOG_RUN_DIR", str(log_dir))
        path = log_dir / "commands.log"
        # Centralised logging: uniform format + console, DEBUG so motion lines land.
        from jiuwensymbiosis.utils.logging import _OWNED_TAG, DEFAULT_FMT, configure_logging

        configure_logging(level="DEBUG", log_dir=None)
        # Per-run Piper file handler (same uniform format), tagged as owned.
        handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(DEFAULT_FMT, datefmt="%H:%M:%S"))
        setattr(handler, _OWNED_TAG, True)
        logger.addHandler(handler)
        if logger.level == logging.NOTSET or logger.level > logging.DEBUG:
            logger.setLevel(logging.DEBUG)
        _CMD_LOG_PATH = path
        logger.info(
            "[Piper] cmd log started: pid=%d log=%s run-stamp=%s",
            os.getpid(),
            path,
            _CMD_LOG_RUN_STAMP,
        )
        return path
    except Exception as exc:  # noqa: BLE001 - logging is best-effort
        logger.warning("[Piper] cmd log setup failed: %s", exc)
        return None


def _ang_diff_deg(a: float, b: float) -> float:
    """Shortest signed angular difference a-b in degrees, wrapped to [-180,180]."""
    return (a - b + 180.0) % 360.0 - 180.0


# =============================================================================
# Pose / joints
# =============================================================================
@dataclass
class PiperPose:
    x: float = 0.0  # mm
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0  # deg
    ry: float = 0.0
    rz: float = 0.0

    def as_tuple(self):
        return astuple(self)


@dataclass
class PiperJointAngles:
    """The 6 joint angles in degrees."""

    j1: float = 0.0
    j2: float = 0.0
    j3: float = 0.0
    j4: float = 0.0
    j5: float = 0.0
    j6: float = 0.0

    def as_tuple(self):
        return astuple(self)


# =============================================================================
# Driver
# =============================================================================
class PiperLowLevel:
    """CAN wrapper around an AgileX Piper (6-DoF) + parallel gripper + RealSense.

    Frame conventions:
      - When calibration is loaded, ``home_pose`` / ``calib_object_pose`` /
        ``z_min_safe`` are in TIP frame.
      - ``move_to_pose_blocking`` always speaks FLANGE frame.
      - ``flange_z = tip_z + tool_offset_mm`` (tool extends along base -Z).
      - Without calibration, poses stay in flange frame.
    """

    def __init__(
        self,
        *,
        can_port: str = "can_left",
        move_speed: int = 50,
        tool_offset_mm: float = 0.0,
        # Calibration (preferred): tf_flange_cam + intrinsics + object anchor.
        calib_path: str | None = None,
        home_lift_mm: float = 250.0,
        z_safe_margin_mm: float = -10.0,
        # Fallback when no calibration is provided.
        home_pose_xyzrxryrz_mm_deg: list[float] | None = None,
        calib_object_xyzrxryrz_mm_deg: list[float] | None = None,
        z_min_safe_mm: float | None = None,
        home_use_init_pose: bool = False,
        # Cartesian workspace box (mm). None disables that bound.
        x_min_mm: float | None = None,
        x_max_mm: float | None = None,
        y_min_mm: float | None = None,
        y_max_mm: float | None = None,
        z_max_mm: float | None = None,
        # Camera (optional).
        camera_serial: str | None = None,
        camera_resolution: tuple[int, int] = (640, 480),
        camera_fps: int = 30,
        # Gripper.
        gripper_open_mm: float = 70.0,
        gripper_effort: int = 1000,
        gripper_settle_s: float = 0.8,
        # Motion completion polling.
        reach_tol_mm: float = 3.0,
        reach_tol_deg: float = 2.0,
        move_timeout_s: float = 30.0,
        enable_timeout_s: float = 8.0,
        # Fast abort: if the arm hasn't started moving AND the controller flags
        # the target as unreachable (motion_status REACH_TARGET_POS_FAILED) for
        # this long, abort instead of polling until move_timeout_s.
        unreachable_grace_s: float = 2.0,
    ) -> None:
        """Connect to CAN, enable arm/gripper, load calibration or fallback, init camera."""
        self._cmd_log_path = _attach_cmd_log_handler()
        self._lock = threading.RLock()
        self._move_speed = max(1, min(100, int(move_speed)))
        self._gripper_open_mm = float(gripper_open_mm)
        self._gripper_effort = int(gripper_effort)
        self._gripper_settle_s = float(gripper_settle_s)
        self._gripper_state: bool = False  # False=open, True=closed
        self._reach_tol_mm = float(reach_tol_mm)
        self._reach_tol_deg = float(reach_tol_deg)
        self._move_timeout_s = float(move_timeout_s)
        self._unreachable_grace_s = float(unreachable_grace_s)
        self._xy_box = (x_min_mm, x_max_mm, y_min_mm, y_max_mm, z_max_mm)

        # --- connect + enable
        try:
            from piper_sdk import C_PiperInterface_V2
        except ImportError as exc:
            raise RuntimeError(
                "[Piper] piper_sdk not installed. `pip install piper_sdk` "
                "(and bring the CAN interface up) to use the real arm."
            ) from exc

        logger.info("[Piper] Connecting CAN %s ...", can_port)
        self._arm = C_PiperInterface_V2(can_port)
        self._arm.ConnectPort()
        t0 = time.time()
        while not self._arm.EnablePiper():
            if time.time() - t0 > float(enable_timeout_s):
                raise RuntimeError(
                    f"[Piper] EnablePiper() did not succeed within {enable_timeout_s}s "
                    f"on {can_port} — check power, E-stop, and CAN bitrate."
                )
            time.sleep(0.01)
        time.sleep(0.5)
        # Establish CAN command control mode (MOVE P). The arm powers up / exits
        # teaching in STANDBY and otherwise IGNORES EndPoseCtrl/JointCtrl/
        # GripperCtrl. IMPORTANT: on this firmware MotionCtrl_2 ALONE does not
        # actually flip STANDBY→CAN_CTRL — the mode only changes when the FIRST
        # real control command (EndPoseCtrl/JointCtrl) is sent, so every move
        # re-asserts the mode right before its command (see move_to_pose_blocking).
        # The gripper therefore only responds once the arm has been moved at
        # least once. Mirrors rlinf PiperController._set_control_mode(1, 0).
        self._arm.MotionCtrl_2(_CTRL_CAN, _MOVE_P, self._move_speed, 0x00)
        time.sleep(0.1)
        # Enable the gripper driver.
        self._arm.GripperCtrl(0, self._gripper_effort, _GRIPPER_ENABLE, 0)
        logger.info("[Piper] CAN %s connected + arm/gripper enabled.", can_port)

        # --- calibration / workspace anchor
        self._calib: dict[str, Any] | None = None
        self._tf_flange_cam: np.ndarray | None = None
        if calib_path is not None:
            self._calib = load_calibration(calib_path)
            self._tf_flange_cam = self._calib["T_flange_cam"]["matrix_4x4"]
            calib_object_xyz = self._calib["object"]["xyz_base_mm"]
            self._calib_object_pose = PiperPose(
                x=float(calib_object_xyz[0]),
                y=float(calib_object_xyz[1]),
                z=float(calib_object_xyz[2]),
            )
            self._home_pose = PiperPose(
                x=float(calib_object_xyz[0]),
                y=float(calib_object_xyz[1]),
                z=float(calib_object_xyz[2]) + float(home_lift_mm),
            )
            z_min_safe = float(calib_object_xyz[2]) + float(z_safe_margin_mm)
            poses_are_tip_frame = True
            logger.info(
                "[Piper] loaded calibration from %s: object_tip=%s, home_lift=%smm, "
                "z_safe_margin=%smm, tool_offset=%smm",
                calib_path,
                np.asarray(calib_object_xyz).round(2).tolist(),
                home_lift_mm,
                z_safe_margin_mm,
                tool_offset_mm,
            )
        else:
            if home_pose_xyzrxryrz_mm_deg is None or len(home_pose_xyzrxryrz_mm_deg) != 6:
                raise ValueError("[Piper] either calib_path or home_pose_xyzrxryrz_mm_deg (6-tuple) must be set")
            if z_min_safe_mm is None:
                z_min_safe_mm = 50.0
            self._home_pose = PiperPose(*[float(v) for v in home_pose_xyzrxryrz_mm_deg])
            if calib_object_xyzrxryrz_mm_deg is not None:
                if len(calib_object_xyzrxryrz_mm_deg) != 6:
                    raise ValueError("[Piper] calib_object_xyzrxryrz_mm_deg must be 6-tuple if given")
                self._calib_object_pose = PiperPose(*[float(v) for v in calib_object_xyzrxryrz_mm_deg])
            else:
                self._calib_object_pose = PiperPose()
            z_min_safe = float(z_min_safe_mm)
            poses_are_tip_frame = False

        self._bounds = WorkspaceBounds(
            z_min_safe=z_min_safe,
            tool_offset_mm=float(tool_offset_mm),
            poses_are_tip_frame=poses_are_tip_frame,
            log_prefix="[Piper]",
        )

        if poses_are_tip_frame and self._calib_object_pose.z < self._bounds.z_min_safe:
            raise ValueError(
                f"[Piper] calibration object z ({self._calib_object_pose.z:.2f}) is below "
                f"z_min_safe ({self._bounds.z_min_safe:.2f}); refusing to start."
            )

        # --- snapshot starting pose
        init_pose = self._read_pose()
        if init_pose is None:
            raise RuntimeError("[Piper] GetArmEndPoseMsgs() failed during init.")
        self._init_pose = init_pose  # FLANGE frame
        if self._calib is not None:
            # Inherit live orientation into the home / calibration-object poses.
            self._home_pose = PiperPose(
                self._home_pose.x,
                self._home_pose.y,
                self._home_pose.z,
                init_pose.rx,
                init_pose.ry,
                init_pose.rz,
            )
            self._calib_object_pose = PiperPose(
                self._calib_object_pose.x,
                self._calib_object_pose.y,
                self._calib_object_pose.z,
                init_pose.rx,
                init_pose.ry,
                init_pose.rz,
            )
        if home_use_init_pose:
            home_z_local = (
                init_pose.z - self._bounds.tool_offset_mm if self._bounds.poses_are_tip_frame else init_pose.z
            )
            self._home_pose = PiperPose(
                init_pose.x,
                init_pose.y,
                home_z_local,
                init_pose.rx,
                init_pose.ry,
                init_pose.rz,
            )
            logger.info("[Piper] home_use_init_pose=True; home=%s", self._home_pose.as_tuple())
        logger.info(
            "[Piper] init=%s home=%s z_min_safe=%.2fmm",
            init_pose.as_tuple(),
            self._home_pose.as_tuple(),
            self._bounds.z_min_safe,
        )

        # --- camera (optional)
        self._camera_serial = camera_serial
        self._camera: RealSenseCamera | None = None
        if camera_serial:
            self._camera = RealSenseCamera(
                serial=camera_serial,
                resolution=camera_resolution,
                fps=camera_fps,
                log_prefix="[Piper]",
            )
            self._camera.start()

        self._closed = False

    # ============================================================== special methods
    def __del__(self) -> None:
        """Ensure close() is called on garbage collection."""
        try:
            self.close()
        except Exception:  # noqa: BLE001 - __del__ cleanup must never raise
            pass

    # ============================================================== properties
    @property
    def home_pose(self) -> PiperPose:
        """Configured home pose (TIP or flange frame depending on calibration)."""
        return self._home_pose

    @property
    def calib_object_pose(self) -> PiperPose:
        """Calibration anchor object pose (TIP frame when calibrated)."""
        return self._calib_object_pose

    @property
    def init_pose(self) -> PiperPose:
        """Starting flange pose captured at init."""
        return self._init_pose

    @property
    def z_min_safe(self) -> float:
        """Lowest allowed TIP Z in mm (from WorkspaceBounds)."""
        return self._bounds.z_min_safe

    @property
    def flange_z_min_safe(self) -> float:
        """Lowest allowed flange Z in mm (TIP Z + tool_offset)."""
        return self._bounds.flange_z_min_safe

    @property
    def tool_offset_mm(self) -> float:
        """Flange-to-tip offset along base -Z in mm."""
        return self._bounds.tool_offset_mm

    @property
    def tf_flange_cam(self) -> np.ndarray | None:
        """4x4 calibration matrix: camera pose in flange frame, or None."""
        return self._tf_flange_cam

    @property
    def has_calibration(self) -> bool:
        """True if a calibration JSON was loaded."""
        return self._calib is not None

    @property
    def calibration(self) -> dict[str, Any] | None:
        """Raw calibration dict, or None."""
        return self._calib

    @property
    def intrinsics(self) -> np.ndarray | None:
        """Camera intrinsics matrix (3x3) from live camera or None."""
        return self._camera.intrinsics if self._camera is not None else None

    @property
    def camera_serial(self) -> str | None:
        """Configured camera serial number, or None."""
        return self._camera_serial

    @property
    def gripper_state(self) -> bool:
        """True=closed, False=open."""
        return self._gripper_state

    # ============================================================== arm pose
    def get_pose(self) -> PiperPose:
        """Read the current flange pose; raises on failure."""
        p = self._read_pose()
        if p is None:
            raise RuntimeError("[Piper] failed to read end pose")
        return p

    def get_angles(self) -> PiperJointAngles:
        """Read the current joint angles; raises on failure."""
        a = self._read_angles()
        if a is None:
            raise RuntimeError("[Piper] failed to read joint angles")
        return a

    def grab_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Grab (rgb, depth_m) from the camera, or None if no camera."""
        return self._camera.grab_frames() if self._camera is not None else None

    # =================================================================== move
    def move_to_pose_blocking(
        self,
        pose: FlangePose,
        sync_timeout_s: float | None = None,
        joint: bool = False,
    ) -> None:
        """Move to absolute pose in mm/deg (FLANGE) via EndPoseCtrl.

        ``joint=False`` → MOVE L (straight line, for short pick/place strokes).
        ``joint=True``  → MOVE P (point-to-point, reach-friendly, for transit/home).
        Blocks (polls GetArmEndPoseMsgs) until within tolerance or timeout.
        """
        x, y, z = pose.x_mm, pose.y_mm, pose.z_mm
        rx, ry, rz = pose.rx_deg, pose.ry_deg, pose.rz_deg
        x, y, z = self._clamp_xy_box(x, y, z)
        self._bounds.check_flange_z(z)
        for v, name in ((x, "x"), (y, "y"), (z, "z"), (rx, "rx"), (ry, "ry"), (rz, "rz")):
            if not math.isfinite(v):
                raise ValueError(f"[Piper] non-finite {name}={v}")
        x_int, y_int, z_int = round(x * _FACTOR), round(y * _FACTOR), round(z * _FACTOR)
        rx_int, ry_int, rz_int = round(rx * _FACTOR), round(ry * _FACTOR), round(rz * _FACTOR)
        with self._lock:
            logger.info(
                "[Piper] EndPoseCtrl(MOVE_P, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f) ...",
                x,
                y,
                z,
                rx,
                ry,
                rz,
            )
            self._arm.MotionCtrl_2(_CTRL_CAN, _MOVE_P, self._move_speed, 0x00)
            self._arm.EndPoseCtrl(x_int, y_int, z_int, rx_int, ry_int, rz_int)
        self._wait_pose_reached(
            (x, y, z, rx, ry, rz),
            sync_timeout_s or self._move_timeout_s,
            label="EndPose",
        )

    def servo_to_pose(self, pose: Any) -> None:
        """NON-BLOCKING FLANGE-frame pose command for the real-time servo loop.

        Sends one ``EndPoseCtrl`` (MOVE_P) toward the target and returns
        immediately — no ``_wait_pose_reached`` poll. The control loop is
        responsible for slew-limiting and for re-issuing toward the latest
        target each tick. Workspace clamp + Z floor are still enforced (a servo
        loop must not be able to drive the tip through the floor).

        ``pose`` is a mapping (``x/y/z`` mm + optional ``rx/ry/rz`` deg) or any
        object exposing those attributes; missing orientation falls back to the
        live flange orientation.
        """

        def _get(key: str, default: float) -> float:
            if isinstance(pose, dict):
                v = pose.get(key)
            else:
                v = getattr(pose, key, None)
            return float(v) if v is not None else float(default)

        cur = self._read_pose()
        rx0, ry0, rz0 = (cur.rx, cur.ry, cur.rz) if cur is not None else (0.0, 0.0, 0.0)
        x = _get("x", cur.x if cur is not None else 0.0)
        y = _get("y", cur.y if cur is not None else 0.0)
        z = _get("z", cur.z if cur is not None else 0.0)
        rx = _get("rx", rx0)
        ry = _get("ry", ry0)
        # accept either rz or r for the wrist angle
        rz = _get("rz", _get("r", rz0))

        x, y, z = self._clamp_xy_box(x, y, z)
        self._bounds.check_flange_z(z)
        for v, name in ((x, "x"), (y, "y"), (z, "z"), (rx, "rx"), (ry, "ry"), (rz, "rz")):
            if not math.isfinite(v):
                raise ValueError(f"[Piper] servo non-finite {name}={v}")
        x_int, y_int, z_int = round(x * _FACTOR), round(y * _FACTOR), round(z * _FACTOR)
        rx_int, ry_int, rz_int = round(rx * _FACTOR), round(ry * _FACTOR), round(rz * _FACTOR)
        with self._lock:
            self._arm.MotionCtrl_2(_CTRL_CAN, _MOVE_P, self._move_speed, 0x00)
            self._arm.EndPoseCtrl(x_int, y_int, z_int, rx_int, ry_int, rz_int)
        # Intentionally NO wait: return immediately so the servo loop keeps its rate.

    def move_joint_blocking(
        self,
        q: list[float],
        sync_timeout_s: float | None = None,
    ) -> None:
        """Move to joint configuration ``q`` (6 angles in deg) via MOVE J + JointCtrl."""
        if len(q) != 6:
            raise ValueError(f"[Piper] expected 6 joint angles, got {len(q)}")
        for i, v in enumerate(q):
            if not math.isfinite(v):
                raise ValueError(f"[Piper] non-finite q[{i}]={v}")
        q_int = [round(float(v) * _FACTOR) for v in q]
        with self._lock:
            logger.info("[Piper] JointCtrl(%s) ...", [round(v, 2) for v in q])
            self._arm.MotionCtrl_2(_CTRL_CAN, _MOVE_J, self._move_speed, 0x00)
            self._arm.JointCtrl(*q_int)
        self._wait_joints_reached(q, sync_timeout_s or self._move_timeout_s)

    def home(self) -> None:
        """Move to the configured home pose (TIP→FLANGE Z conversion when needed),
        via MOVE P (point-to-point) for a reach-friendly path.
        """
        p = self._home_pose
        flange_z = p.z + self._bounds.tool_offset_mm if self._bounds.poses_are_tip_frame else p.z
        self.move_to_pose_blocking(FlangePose(p.x, p.y, flange_z, p.rx, p.ry, p.rz), joint=True)

    # ================================================================ gripper
    def set_gripper(self, closed: bool) -> None:
        """Two-state convenience over ``GripperCtrl``: closed → 0mm, open → open_mm."""
        self.set_gripper_width(0.0 if closed else self._gripper_open_mm)
        self._gripper_state = bool(closed)

    def set_gripper_width(self, width_mm: float, effort: int | None = None) -> None:
        """Command an explicit gripper width (mm) + force (0.001 N·m units)."""
        width_int = max(0, round(float(width_mm) * _FACTOR))
        eff = int(effort) if effort is not None else self._gripper_effort
        with self._lock:
            logger.info("[Piper] GripperCtrl(width=%.1fmm, effort=%d)", width_mm, eff)
            self._arm.GripperCtrl(width_int, eff, _GRIPPER_ENABLE, 0)
        time.sleep(self._gripper_settle_s)

    # ============================================================== teardown
    def close(self) -> None:
        """Stop camera, disconnect CAN; leave arm energized holding pose."""
        if self._closed:
            return
        try:
            if self._camera is not None:
                self._camera.stop()
        except Exception:  # noqa: BLE001 - best-effort camera teardown
            pass
        # Leave the arm ENERGIZED and holding its pose: do NOT DisableArm
        # (that makes it go limp and drop whatever it is holding) and do NOT
        # force standby. Just drop the CAN connection — the firmware keeps the
        # last enabled state, so the arm holds position until power-cycled or
        # explicitly disabled.
        try:
            self._arm.DisconnectPort()
        except Exception:  # noqa: BLE001 - best-effort CAN disconnect
            pass
        self._closed = True
        logger.info("[Piper] Closed.")

    # ============================================================== private helpers
    def _read_pose(self) -> PiperPose | None:
        """Thread-safe read of the arm end-pose from CAN; returns None on failure."""
        try:
            with self._lock:
                msg = self._arm.GetArmEndPoseMsgs().end_pose
            return PiperPose(
                x=msg.X_axis / _FACTOR,
                y=msg.Y_axis / _FACTOR,
                z=msg.Z_axis / _FACTOR,
                rx=msg.RX_axis / _FACTOR,
                ry=msg.RY_axis / _FACTOR,
                rz=msg.RZ_axis / _FACTOR,
            )
        except Exception as exc:  # noqa: BLE001 - CAN read may fail; return None
            logger.warning("[Piper] GetArmEndPoseMsgs failed: %s", exc)
            return None

    def _read_angles(self) -> PiperJointAngles | None:
        """Thread-safe read of joint angles from CAN; returns None on failure."""
        try:
            with self._lock:
                js = self._arm.GetArmJointMsgs().joint_state
            return PiperJointAngles(
                j1=js.joint_1 / _FACTOR,
                j2=js.joint_2 / _FACTOR,
                j3=js.joint_3 / _FACTOR,
                j4=js.joint_4 / _FACTOR,
                j5=js.joint_5 / _FACTOR,
                j6=js.joint_6 / _FACTOR,
            )
        except Exception as exc:  # noqa: BLE001 - CAN read may fail; return None
            logger.warning("[Piper] GetArmJointMsgs failed: %s", exc)
            return None

    def _clamp_xy_box(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Clamp a FLANGE target into the configured XY box + Z ceiling (warn on clamp).

        The Z floor is enforced separately (and hard) by ``check_flange_z``. X/Y are
        frame-invariant to the Z tool offset, so the same box applies to tip/flange.
        """
        x_min, x_max, y_min, y_max, z_max = self._xy_box
        cx, cy, cz = x, y, z
        reasons = []
        if x_min is not None and cx < x_min:
            reasons.append(f"x<{x_min}")
            cx = x_min
        if x_max is not None and cx > x_max:
            reasons.append(f"x>{x_max}")
            cx = x_max
        if y_min is not None and cy < y_min:
            reasons.append(f"y<{y_min}")
            cy = y_min
        if y_max is not None and cy > y_max:
            reasons.append(f"y>{y_max}")
            cy = y_max
        if z_max is not None and cz > z_max:
            reasons.append(f"z>{z_max}")
            cz = z_max
        if reasons:
            logger.warning(
                "[Piper] target (%.1f,%.1f,%.1f) clamped to (%.1f,%.1f,%.1f) [%s]",
                x,
                y,
                z,
                cx,
                cy,
                cz,
                ", ".join(reasons),
            )
        return cx, cy, cz

    def _wait_pose_reached(
        self,
        target: tuple[float, float, float, float, float, float],
        timeout_s: float,
        *,
        label: str = "motion",
        poll_s: float = 0.05,
        settle_polls: int = 3,
    ) -> None:
        """Poll the end pose until within tolerance of ``target`` and settled.

        Fast-aborts an UNREACHABLE target: if the arm hasn't started moving and
        the controller reports motion_status=REACH_TARGET_POS_FAILED for longer
        than ``unreachable_grace_s``, raise immediately instead of polling for
        the full ``timeout_s`` (the firmware silently refuses out-of-envelope
        EndPose targets — the arm just doesn't move).
        """
        tx, ty, tz, trx, try_, trz = target
        start = self._read_pose()
        sx, sy, sz = (start.x, start.y, start.z) if start is not None else (None, None, None)
        # Is the target a non-trivial distance from where we start? If so, a
        # reachable command will visibly move the arm within a second or two;
        # if the arm never moves, the firmware silently refused it (unreachable
        # / out of envelope). motion_status is NOT a reliable signal here — the
        # controller often keeps the prior REACH_TARGET_POS_SUCCESSFULLY flag —
        # so we key the fast-abort on "the pose never changed".
        # sx/sy/sz come from one tuple; the `sx is not None` short-circuit
        # guards them, but mypy can't track same-source narrowing, so the
        # operator error below is suppressed.
        start_far = (
            sx is not None
            and math.sqrt(
                (tx - sx) ** 2 + (ty - sy) ** 2 + (tz - sz) ** 2  # type: ignore[operator]
            )
            > self._reach_tol_mm + 5.0
        )
        t0 = time.time()
        in_tol_count = 0
        moved = False
        while True:
            p = self._read_pose()
            if p is not None:
                pos_err = math.sqrt((p.x - tx) ** 2 + (p.y - ty) ** 2 + (p.z - tz) ** 2)
                rot_err = max(
                    abs(_ang_diff_deg(p.rx, trx)),
                    abs(_ang_diff_deg(p.ry, try_)),
                    abs(_ang_diff_deg(p.rz, trz)),
                )
                if pos_err <= self._reach_tol_mm and rot_err <= self._reach_tol_deg:
                    in_tol_count += 1
                    if in_tol_count >= settle_polls:
                        return
                else:
                    in_tol_count = 0
                if not moved and sx is not None:
                    # sx/sy/sz guarded by the `sx is not None` check above
                    # (same-source tuple); mypy can't track it — see line ~630.
                    if math.sqrt((p.x - sx) ** 2 + (p.y - sy) ** 2 + (p.z - sz) ** 2) > 3.0:  # type: ignore[operator]
                        moved = True
            elapsed = time.time() - t0
            # PRIMARY unreachable signal: the firmware sets Arm Status =
            # TARGET_POS_EXCEEDS_LIMIT when the commanded EndPose is out of the
            # reachable envelope. (motion_status is NOT reliable — it keeps the
            # prior REACH_TARGET_POS_SUCCESSFULLY flag.) Give the command ~0.5s
            # to register the new target before trusting the flag.
            if elapsed > 0.5 and self._target_exceeds_limit():
                last = p.as_tuple() if p is not None else None
                raise RuntimeError(
                    f"[Piper] {label} target OUT OF REACH — Arm Status="
                    f"TARGET_POS_EXCEEDS_LIMIT after {elapsed:.1f}s "
                    f"(target={tuple(round(v, 1) for v in target)}, last={last}). "
                    "Aborted; arm held in place."
                )
            # Backstop: a far target the arm never started executing.
            if start_far and (not moved) and elapsed > self._unreachable_grace_s:
                last = p.as_tuple() if p is not None else None
                raise RuntimeError(
                    f"[Piper] {label} not executing — arm did not move from start after "
                    f"{elapsed:.1f}s; target likely UNREACHABLE / out of envelope "
                    f"(target={tuple(round(v, 1) for v in target)}, last={last}). "
                    "Aborted; arm held in place."
                )
            if timeout_s and elapsed > timeout_s:
                last = p.as_tuple() if p is not None else None
                raise RuntimeError(
                    f"[Piper] {label} did not reach target within {timeout_s:.1f}s "
                    f"(target={tuple(round(v, 1) for v in target)}, last={last})"
                )
            time.sleep(poll_s)

    def _target_exceeds_limit(self) -> bool:
        """True when the firmware flags the commanded EndPose as out of reach.

        The reliable "unreachable" signal is ``Arm Status`` (arm_status field) ==
        ``TARGET_POS_EXCEEDS_LIMIT(0x4)`` — NOT ``motion_status`` (which keeps the
        prior REACH_TARGET_POS_SUCCESSFULLY flag and is misleading here).
        """
        try:
            return "EXCEEDS_LIMIT" in str(self._arm.GetArmStatus().arm_status.arm_status)
        except Exception:  # noqa: BLE001 - status read best-effort; assume not exceeded
            return False

    def _wait_joints_reached(
        self,
        target_q: list[float],
        timeout_s: float,
        *,
        poll_s: float = 0.05,
        settle_polls: int = 3,
        tol_deg: float = 1.0,
    ) -> None:
        t0 = time.time()
        in_tol_count = 0
        while True:
            a = self._read_angles()
            if a is not None:
                err = max(abs(_ang_diff_deg(c, t)) for c, t in zip(a.as_tuple(), target_q, strict=True))
                if err <= tol_deg:
                    in_tol_count += 1
                    if in_tol_count >= settle_polls:
                        return
                else:
                    in_tol_count = 0
            if timeout_s and (time.time() - t0) > timeout_s:
                raise RuntimeError(
                    f"[Piper] JointCtrl did not reach target within {timeout_s:.1f}s "
                    f"(target={[round(v, 1) for v in target_q]})"
                )
            time.sleep(poll_s)
