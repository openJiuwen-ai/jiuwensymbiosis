#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""SO-101 eye-to-hand hand-eye calibration — board tied to the gripper.

Camera fixed on the desk (looking down). The ChArUco board is **tied to the arm
gripper**, so it moves with the arm — this satisfies the OpenCV eye-to-hand
AX=XB rigid-coupling precondition. Each station = (flange FK pose in base,
board PnP in camera); OpenCV ``calibrateHandEye`` solves the constant
``T_base_cam``.

Four modes (exactly one is required, mutually exclusive):

``--auto --bounds-file <npz>`` (collection + solve): the arm drives itself
along the taught joint path under torque, settles, then motion-gates N
candidate frames per pose (joint stillness, corner count, reprojection,
board-facing-camera). No manual posing or torque toggle is required.

``--collect-bounds --bounds-file <npz>``: interactive hand-pose the arm to an
ordered sequence of reachable + board-visible waypoints; ``--auto`` later
follows those points and inserts intermediate capture stations.

``--replay <npz>``: offline re-solve from a previously dumped stations archive
(no arm / no camera / no board spec).

``--release-torque``: release the five arm joints for manual repositioning,
then safely restore torque (preset current goal → enable all). No camera, no
board, no capture. Any exit (Enter / q / EOF / exception) goes through the same
safe restore; a non-confirmed restore exits non-zero.

Usage (operator-taught path — recommended):
  # 1) hand-pose the arm to several reachable + board-visible waypoints:
  python scripts/calibrate/so101_eye_calib.py \
      --config scripts/calibrate/so101_calibrate.yaml \
      --board charuco --squares-x 5 --squares-y 7 \
      --square-size-mm 15.28 --marker-size-mm 11 \
      --collect-bounds --bounds-file tmp/so101_bounds.npz
  # 2) auto-sweep through those points; intermediate stations are inserted:
  python scripts/calibrate/so101_eye_calib.py \
      --config scripts/calibrate/so101_calibrate.yaml \
      --board charuco --squares-x 5 --squares-y 7 \
      --square-size-mm 15.28 --marker-size-mm 11 \
      --auto --bounds-file tmp/so101_bounds.npz --n-stations 24 \
      --dump tmp/so101_eye_stations.npz --cross-check

Usage (offline re-solve from a previous dump):
  python scripts/calibrate/so101_eye_calib.py --replay tmp/so101_eye_stations.npz \
      --drop-outliers --method HORAUD

Usage (release torque for manual repositioning):
  python scripts/calibrate/so101_eye_calib.py \
      --config scripts/calibrate/so101_calibrate.yaml --release-torque
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

if __package__:
    from .handeye_board import BoardSpec, detect_board
    from .handeye_core import (
        MIN_STATIONS,
        Station,
        ViewDetection,
        _cv2_methods,
        _rotation_angle_deg,
        orthonormalize,
        rotation_spread_deg,
        save_calibration,
    )
else:  # Support direct execution: python scripts/calibrate/so101_eye_calib.py
    from handeye_board import BoardSpec, detect_board
    from handeye_core import (
        MIN_STATIONS,
        Station,
        ViewDetection,
        _cv2_methods,
        _rotation_angle_deg,
        orthonormalize,
        rotation_spread_deg,
        save_calibration,
    )

from jiuwensymbiosis.adapters.so101.config import So101Config
from jiuwensymbiosis.adapters.so101.env import So101Env
from jiuwensymbiosis.adapters.so101.lowlevel import ARM_JOINT_ORDER
from jiuwensymbiosis.utils.geometry import invert_transform, make_transform

logger = logging.getLogger("so101_eye_calib")


def _configure_logging(debug: bool = False) -> None:
    """Attach a stdout handler emitting only this script's records."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.WARNING)


def _solve_eye_to_hand(cv2, stations: list[Station], method_const: int) -> np.ndarray:
    """Solve T_base_cam (cam pose in base, constant) for eye-to-hand.

    Convention verified by synthetic self-test (ground-truth known): feed
    R_base2gripper (inv of gripper2base = inv of FK) and R_target2cam (the
    solvePnP result AS-IS, not inverted); the returned H is T_base_cam directly.
    NOT the 'invert both' rule sometimes cited — that diverges (cross-check
    max_deg ~180deg).
    """
    r_b2g, t_b2g, r_t2c, t_t2c = [], [], [], []
    for s in stations:
        b2g = invert_transform(s.tf_base_flange)  # base2gripper (inv of FK)
        t2c = s.detection.tf_cam_target  # target2cam (as-is)
        r_b2g.append(b2g[:3, :3])
        t_b2g.append(b2g[:3, 3].reshape(3, 1))
        r_t2c.append(t2c[:3, :3])
        t_t2c.append(t2c[:3, 3].reshape(3, 1))
    r_x, t_x = cv2.calibrateHandEye(r_b2g, t_b2g, r_t2c, t_t2c, method=method_const)
    return make_transform(orthonormalize(np.asarray(r_x)), np.asarray(t_x).reshape(3))


def _axxb_eye_to_hand(stations: list[Station], tf_base_cam: np.ndarray) -> tuple[list[float], list[float]]:
    """AX=XB consistency for eye-to-hand: board is rigidly on the arm, camera fixed.

    For each station the board-in-base is T_base_cam @ T_cam_target (camera is
    static, so board-in-base == board-in-flange-chain, must be CONSTANT since
    the board is clamped to the gripper). Per-pair residual = how far the
    relative transform deviates from identity under the solved T_base_cam.
    """
    rot_errs: list[float] = []
    trans_errs: list[float] = []
    n = len(stations)
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = stations[i].tf_base_flange, stations[j].tf_base_flange
            ci, cj = stations[i].detection.tf_cam_target, stations[j].detection.tf_cam_target
            # board-in-flange (constant): T_flange_board = inv(gripper2base) @ T_base_cam @ T_cam_target
            bi = invert_transform(gi) @ tf_base_cam @ ci
            bj = invert_transform(gj) @ tf_base_cam @ cj
            m = invert_transform(bi) @ bj  # should be identity (board rigid in flange)
            c = np.clip((np.trace(m[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            rot_errs.append(float(np.degrees(np.arccos(c))))
            trans_errs.append(float(np.linalg.norm(m[:3, 3])))
    return rot_errs, trans_errs


def _board_in_flange_spread(stations: list[Station], tf_base_cam: np.ndarray) -> np.ndarray:
    """Board origin in the flange frame per station (mm). Board clamped rigid =>
    constant. Spread measures rigidity / calibration quality."""
    pts = []
    for s in stations:
        b_in_f = invert_transform(s.tf_base_flange) @ tf_base_cam @ s.detection.tf_cam_target
        pts.append(b_in_f[:3, 3])
    return np.asarray(pts)


def _station_rigidity_errors(
    stations: list[Station],
    tf_base_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-station deviation from the mean constant board-in-control transform."""
    transforms = [invert_transform(s.tf_base_flange) @ tf_base_cam @ s.detection.tf_cam_target for s in stations]
    translations = np.stack([tf[:3, 3] for tf in transforms])
    mean_translation = translations.mean(axis=0)
    mean_rotation = Rotation.from_matrix(np.stack([tf[:3, :3] for tf in transforms])).mean().as_matrix()
    rot_errors = np.array([_rotation_angle_deg(mean_rotation, tf[:3, :3]) for tf in transforms])
    trans_errors = np.linalg.norm(translations - mean_translation, axis=1)
    return rot_errors, trans_errors


def _mad_threshold(values: np.ndarray, *, floor: float) -> float:
    """Median + 3 robust sigmas, never tighter than the physical floor."""
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    return max(float(floor), median + 3.0 * robust_sigma)


def _conservative_outlier_filter(
    stations: list[Station],
    cv2,
    method_const: int,
) -> tuple[list[Station], list[int]]:
    """Drop at most 20% gross rigidity outliers, one at a time.

    This is deliberately opt-in. A station is removable only when it exceeds a
    robust threshold and an absolute physical floor. Rotation diversity must
    remain >=20 degrees, so filtering cannot manufacture a numerically weak fit.
    """
    kept = list(range(len(stations)))
    dropped: list[int] = []
    max_drop = max(1, int(np.floor(len(stations) * 0.20)))
    while len(dropped) < max_drop and len(kept) > max(MIN_STATIONS, 7):
        current = [stations[i] for i in kept]
        solved = _solve_eye_to_hand(cv2, current, method_const)
        rot_err, trans_err = _station_rigidity_errors(current, solved)
        rot_limit = _mad_threshold(rot_err, floor=2.0)
        trans_limit = _mad_threshold(trans_err, floor=10.0)
        severity = np.maximum(rot_err / rot_limit, trans_err / trans_limit)
        local_idx = int(np.argmax(severity))
        if severity[local_idx] <= 1.0:
            break
        candidate_kept = kept[:local_idx] + kept[local_idx + 1:]  # fmt: skip
        candidate = [stations[i] for i in candidate_kept]
        if rotation_spread_deg(candidate) < 20.0:
            logger.info("  outlier filtering stopped: removal would leave <20deg rotation spread")
            break
        original_idx = kept.pop(local_idx)
        dropped.append(original_idx)
        logger.info(
            f"  outlier drop station {original_idx}: "
            f"rot={rot_err[local_idx]:.2f}deg (limit {rot_limit:.2f}), "
            f"trans={trans_err[local_idx]:.2f}mm (limit {trans_limit:.2f})"
        )
    return [stations[i] for i in kept], dropped


def _dump_stations(
    path: Path,
    stations: list[Station],
    camera_matrix: np.ndarray,
    capture_meta: list[dict],
) -> None:
    """Persist transforms plus capture diagnostics and selected RGB frames."""
    arr = {
        "n": len(stations),
        "intrinsics": np.asarray(camera_matrix, dtype=np.float64),
        "flange": np.stack([np.asarray(s.tf_base_flange, dtype=np.float64) for s in stations]),
        "board_cam": np.stack([np.asarray(s.detection.tf_cam_target, dtype=np.float64) for s in stations]),
        "reproj": np.array([float(s.detection.reproj_rms_px or 0.0) for s in stations]),
        "q_before_deg": np.stack([np.asarray(m["q_before"], dtype=np.float64) for m in capture_meta]),
        "q_after_deg": np.stack([np.asarray(m["q_after"], dtype=np.float64) for m in capture_meta]),
        "capture_joint_motion_deg": np.array(
            [float(m["joint_motion_deg"]) for m in capture_meta],
            dtype=np.float64,
        ),
        "corner_count": np.array([int(m["corners"]) for m in capture_meta], dtype=np.int64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **arr)
    frames_dir = path.parent / f"{path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        for i, meta in enumerate(capture_meta):
            bgr = cv2.cvtColor(np.asarray(meta["rgb"]), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frames_dir / f"station_{i:03d}.jpg"), bgr)
    except Exception as exc:
        logger.warning("failed to save diagnostic RGB frames: %s", exc)
    logger.info(f"dumped {len(stations)} stations -> {path}")


def _load_stations(path: Path) -> tuple[list[Station], np.ndarray]:
    """Reload stations dumped by _dump_stations for offline re-solve."""
    d = np.load(str(path))
    camera_matrix = d["intrinsics"]
    stations = []
    for i in range(int(d["n"])):
        stations.append(
            Station(
                tf_base_flange=d["flange"][i],
                detection=ViewDetection(
                    ok=True,
                    tf_cam_target=d["board_cam"][i],
                    reproj_rms_px=float(d["reproj"][i]),
                ),
            )
        )
    logger.info(f"loaded {len(stations)} stations from {path}")
    return stations, camera_matrix


def _standalone_calibration_config(args) -> So101Config:
    """Build the live SO-101 connection config without reading a YAML file."""
    broad_joint_limits = dict.fromkeys(ARM_JOINT_ORDER, (-180.0, 180.0))
    return So101Config(
        port=str(args.port),
        robot_id=str(args.robot_id),
        calibration_dir=args.calibration_dir,
        urdf_path=args.urdf_path,
        camera_serial=str(args.camera_serial),
        home_use_init_pose=True,
        joint_limits=broad_joint_limits,
        safety_validated=True,
        disable_torque_on_disconnect=False,
        z_min_safe_mm=-1_000_000.0,
        z_max_safe_mm=None,
        workspace_bounds=None,
        joint_tolerance_deg=3.5,
        move_timeout_s=5.0 if args.move_timeout_s is None else float(args.move_timeout_s),
        settle_overcompensate=False,
    )


def _release_torque_config(cfg: So101Config) -> So101Config:
    """Return a config copy for the release-torque command: no camera, no torque-off on disconnect.

    ``camera_serial=None`` skips ``_start_camera`` so a pure torque command is not held
    hostage by a missing desktop camera (fail-closed at connect time). With
    ``disable_torque_on_disconnect=False`` the teardown does not re-disable torque, so
    whatever the release recovery restored survives ``disconnect``. The original config
    is not mutated.
    """
    return dataclasses.replace(
        cfg,
        camera_serial=None,
        disable_torque_on_disconnect=False,
    )


def _release_torque(cfg: So101Config) -> bool:
    """Release the five arm joints for manual repositioning, then safely restore torque.

    Any exit path (Enter / q / EOF / exception) goes through the same
    :func:`_safe_restore_torque` chain: preset the current joint goal, then re-enable
    all motors. Returns True only when torque was confirmed restored; ``main`` turns a
    False into a non-zero exit. Exceptions (including ``KeyboardInterrupt``) propagate
    unchanged after best-effort recovery — the finally block only cleans up resources.
    """
    env = So101Env(_release_torque_config(cfg))
    ll = None
    restored = False
    try:
        env.connect()
        ll = env.low_level
        disabled = _disable_arm_torque(ll)
        if disabled:
            logger.info("!!! 五臂力矩已松;夹爪力矩保留。手动挪臂后回车恢复;q=中止并恢复。")
            _prompt("  回车恢复力矩并断开;q=中止并恢复 > ")
        # disable succeeded or failed, the operator has not moved the arm in the
        # failure case, so presetting the current pose is safe either way.
        restored = _safe_restore_torque(ll)
    except BaseException:
        if ll is not None and not restored:
            _safe_restore_torque(ll)
        raise
    finally:
        env.disconnect()
        logger.info("disconnected.")
    if not restored:
        logger.error("力矩未确认恢复;请人工托臂并检查硬件后再离开。")
    return restored


def _prompt(prompt: str) -> str | None:
    """Return stripped user input, or ``None`` on EOF."""
    try:
        return input(prompt).strip()
    except EOFError:
        return None


def _disable_arm_torque(ll) -> bool:
    """Disable only the five arm joints; keep gripper holding torque enabled."""
    try:
        ll.disable_arm_torque()
    except Exception as exc:
        logger.error("selective arm torque disable failed: %s", exc)
        return False
    logger.info("!!! ARM JOINT TORQUE DISABLED; GRIPPER TORQUE REMAINS ON — support the elbow! !!!")
    return True


def _enable_arm_torque(ll) -> bool:
    """Re-enable only the five arm joints; best-effort."""
    try:
        ll.enable_arm_torque()
    except Exception as exc:
        logger.warning("selective arm torque enable failed: %s", exc)
        return False
    logger.info("arm joint torque re-enabled.")
    return True


def _enable_all_torque(ll) -> bool:
    """Safety cleanup: re-enable every motor, including the gripper."""
    try:
        ll.restore_all_torque()
    except Exception as exc:
        logger.warning("enable-all torque failed: %s", exc)
        return False
    return True


def _joint_delta_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest absolute angular difference for degree-valued joints."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def _preset_and_lock_current_pose(ll) -> bool:
    """Preset current joint positions while torque is off, then enable arm torque."""
    if not _preset_current_pose_goal(ll):
        return False
    return _enable_arm_torque(ll)


def _preset_current_pose_goal(ll) -> bool:
    """Read current joint positions and write them as the goal; NEVER enable torque.

    Unlike :func:`_preset_and_lock_current_pose`, a failure here enables nothing, so a
    release-torque recovery cannot accidentally arm-torque a stale goal. Returns True
    only when the current joint vector is finite, correctly shaped, and written.
    """
    try:
        ll.preset_current_joint_goal()
        return True
    except Exception as exc:
        logger.error("  failed to preset current joint goal: %s", exc)
        return False


def _safe_restore_torque(ll) -> bool:
    """Totalised safe recovery for release-torque: preset current goal, then enable all.

    Does not raise ordinary ``Exception`` (both helpers swallow ``Exception``
    internally); ``KeyboardInterrupt`` / other ``BaseException`` may still propagate
    from the underlying calls. Returns True only when both preset AND enable succeeded;
    on failure a high-visibility warning is logged and torque is NOT falsely reported
    as restored (caller surfaces a non-zero exit when it cannot confirm restoration).
    """
    if not _preset_current_pose_goal(ll):
        logger.error("预写当前位置失败;五臂保持松开;请持续托臂直至手动归位。")
        return False
    if not _enable_all_torque(ll):
        logger.error("已预写当前位置,但恢复力矩失败;请持续托臂并检查总线。")
        return False
    logger.info("力矩已恢复(已预写当前关节 goal)。")
    return True


def _fk_from_joint_midpoint(ll, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """FK at the midpoint of the two encoder samples; output uses mm."""
    q_mid = (
        np.asarray(q0, dtype=np.float64)
        + ((np.asarray(q1, dtype=np.float64) - np.asarray(q0, dtype=np.float64) + 180.0) % 360.0 - 180.0) * 0.5
    )
    return np.asarray(ll.forward_kinematics_mm(q_mid), dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class _CaptureContext:
    """Inputs shared by live board capture operations."""

    board: BoardSpec
    camera_matrix: np.ndarray
    distortion_coeffs: object | None
    args: argparse.Namespace


# --------------------------------------------------------------------------- auto
def _collect_bounds(ll, context: _CaptureContext) -> None:
    """Interactively collect ordered joint waypoints for later auto capture.

    Per point: Enter to disable arm torque -> hand-pose the arm (board visible,
    no collision, comfortable) -> lock and settle -> verify ChArUco corners ->
    record the actual joint pose. Auto mode later follows the recorded points
    in order and inserts intermediate joint-space stations between them.
    """
    board = context.board
    camera_matrix = context.camera_matrix
    distortion_coeffs = context.distortion_coeffs
    args = context.args
    home = np.asarray(ll.get_angles(), dtype=np.float64) if ll is not None else None
    points: list[np.ndarray] = []
    flanges: list[np.ndarray] = []
    corner_counts: list[int] = []
    reprojection_errors: list[float] = []
    logger.info("\n=== bounds collection (ordered joint waypoints for auto capture) ===")
    logger.info("collect >=4 reachable points in the order the arm should revisit them;")
    logger.info("the later --auto sweep follows this path and inserts intermediate stations.")
    if home is not None:
        logger.info(
            "home (current) joints: %s",
            {n: round(float(v), 1) for n, v in zip(ARM_JOINT_ORDER, home, strict=True)},
        )
    idx = 0
    while True:
        logger.info(f"\n--- bounds point {idx} ---")
        ans = _prompt("  托好肘,回车关手臂力矩；q=结束采集 > ")
        if ans is None or ans.lower() == "q":
            break
        if not _disable_arm_torque(ll):
            break
        try:
            ans = _prompt(
                "  掰到边界点(板可见、不碰、安全),回车锁定并检查角点；s=跳过/q=结束 > ",
            )
            if ans is None or ans.lower() in {"s", "q"}:
                if ans is not None and ans.lower() == "q":
                    break
                continue
            if not _preset_and_lock_current_pose(ll):
                logger.info("  point rejected: failed to lock the taught pose")
                continue
            time.sleep(float(args.settle_dwell_s))
            if not _wait_until_settled(
                ll,
                max_delta_deg=float(args.settle_max_delta_deg),
                samples=int(args.settle_samples),
                period_s=float(args.settle_period_s),
                timeout_s=float(args.settle_timeout_s),
            ):
                logger.info("  point rejected: encoder did not settle")
                continue
            q_before = np.asarray(ll.get_angles(), dtype=np.float64)
            frames = ll.grab_frames()
            q_after = np.asarray(ll.get_angles(), dtype=np.float64)
            if frames is None:
                logger.info("  point rejected: camera returned no frame")
                continue
            joint_motion = float(np.max(_joint_delta_deg(q_after, q_before)))
            if joint_motion > float(args.max_capture_joint_delta_deg):
                logger.info(
                    f"  point rejected: encoder motion {joint_motion:.3f}deg "
                    f"> {args.max_capture_joint_delta_deg:.3f}deg"
                )
                continue
            rgb, _depth = frames
            det = detect_board(
                rgb,
                board,
                intrinsics=camera_matrix,
                dist=distortion_coeffs,
                min_corners=6,
            )
            if not det.ok or det.tf_cam_target is None:
                logger.info(f"  point rejected: {det.reason}")
                continue
            corners = 0 if det.object_points is None else len(det.object_points)
            if corners < int(args.min_corners):
                logger.info(f"  point rejected: ChArUco corners {corners} < required {args.min_corners}")
                continue
            q = _joint_midpoint_deg(q_before, q_after)
            try:
                tf = _fk_from_joint_midpoint(ll, q_before, q_after)
            except Exception as exc:  # reject unusable taught poses
                logger.info(f"  point rejected: FK failed: {exc}")
                continue
            points.append(q)
            flanges.append(tf)
            corner_counts.append(corners)
            reprojection_errors.append(float(det.reproj_rms_px or 0.0))
            from jiuwensymbiosis.adapters.so101.geometry import matrix_m_to_pose_mm_deg

            tf_m = tf.copy()
            tf_m[:3, 3] /= 1000.0
            pose = matrix_m_to_pose_mm_deg(tf_m)
            joints = ", ".join(f"{n}={round(float(v), 1)}" for n, v in zip(ARM_JOINT_ORDER, q, strict=True))
            logger.info(
                f"  ✓ point {idx}: joints={{{joints}}} "
                f"| corners={corners} reproj={float(det.reproj_rms_px or 0.0):.2f}px "
                f"| flange t=({pose.x:.1f},{pose.y:.1f},{pose.z:.1f})mm "
                f"rpy=({pose.rx:.1f},{pose.ry:.1f},{pose.rz:.1f})"
            )
            idx += 1
        finally:
            _enable_arm_torque(ll)

    if len(points) < 2:
        logger.info("need >= 2 points to form a box; nothing saved.")
        return

    joint_points = np.stack(points)
    flange_transforms = np.stack(flanges) if flanges else np.zeros((0, 4, 4))
    lo = joint_points.min(axis=0)
    hi = joint_points.max(axis=0)
    out = {
        "n": len(points),
        "home": home if home is not None else np.zeros(5),
        "points": joint_points,
        "flange": flange_transforms,
        "corner_count": np.asarray(corner_counts, dtype=np.int32),
        "reproj": np.asarray(reprojection_errors, dtype=np.float64),
        "lo": lo,
        "hi": hi,
    }
    path = Path(args.bounds_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **out)
    logger.info(f"\nsaved {len(points)} bound point(s) -> {path}")
    logger.info("joint range summary (deg):")
    for j, name in enumerate(ARM_JOINT_ORDER):
        logger.info(f"  {name:13s} [{lo[j]:7.2f}, {hi[j]:7.2f}] span={hi[j] - lo[j]:6.2f}")
    logger.info(f"re-run --auto --bounds-file {path} to follow this taught path.")


def _load_bound_points(path: Path) -> np.ndarray:
    """Load the ordered, operator-taught joint poses from a bounds archive."""
    with np.load(str(path)) as data:
        if "points" not in data.files:
            raise ValueError(f"bounds file {path} has no 'points' array; re-run --collect-bounds")
        points = np.asarray(data["points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != len(ARM_JOINT_ORDER) or len(points) < 2:
        raise ValueError(
            f"bounds file {path} points must have shape (N, {len(ARM_JOINT_ORDER)}) with N>=2; got {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"bounds file {path} points contain non-finite joint values")
    return points


def _interpolate_taught_poses(points: np.ndarray, target: int, *, min_gap_deg: float) -> list[np.ndarray]:
    """Densify an ordered taught path while retaining its distinct vertices.

    Near-duplicate consecutive taught poses are removed first.  Additional
    stations are then allocated by repeatedly splitting the segment with the
    largest remaining max-joint step, so long moves receive more intermediate
    captures than short moves.  Interpolation follows the shortest angular arc.
    """
    if target < 1:
        raise ValueError(f"target station count must be >=1, got {target}")
    taught = [np.asarray(points[0], dtype=np.float64)]
    duplicate_threshold = max(1e-6, float(min_gap_deg))
    for point in points[1:]:
        q = np.asarray(point, dtype=np.float64)
        if _joint_angle_deg(q, taught[-1]) >= duplicate_threshold:
            taught.append(q)
    if len(taught) < 2:
        raise ValueError(
            "bounds points collapse to fewer than 2 distinct poses; re-collect bounds or lower --min-pose-gap-deg"
        )
    if target <= len(taught):
        indices = np.linspace(0, len(taught) - 1, target).round().astype(int)
        return [taught[int(index)].copy() for index in indices]

    segment_lengths = np.asarray(
        [_joint_angle_deg(taught[index + 1], taught[index]) for index in range(len(taught) - 1)],
        dtype=np.float64,
    )
    intervals = np.ones(len(segment_lengths), dtype=int)
    while int(intervals.sum()) < target - 1:
        # Split the segment whose current sub-interval is widest.
        split_index = int(np.argmax(segment_lengths / intervals))
        intervals[split_index] += 1

    poses = [taught[0].copy()]
    for index, count in enumerate(intervals):
        start = taught[index]
        signed_delta = (taught[index + 1] - start + 180.0) % 360.0 - 180.0
        for step in range(1, int(count) + 1):
            poses.append(start + signed_delta * (step / int(count)))
    return poses


def _generate_joint_poses(args) -> list[np.ndarray]:
    """Build auto-capture poses from a taught joint path via segment interpolation.

    Loads the ordered ``points`` saved by ``--collect-bounds`` and inserts
    intermediate stations along the taught joint-space path: long moves receive more
    intermediate captures than short moves, while the distinct taught vertices are
    retained. No random sampling, no FK pre-filter, no config home/amplitudes — the
    driver safety layer validates every commanded path at runtime.
    """
    points = _load_bound_points(Path(args.bounds_file))
    poses = _interpolate_taught_poses(
        points,
        int(args.n_stations),
        min_gap_deg=float(args.min_pose_gap_deg),
    )
    logger.info(
        f"  sampling source: ordered taught points from {args.bounds_file}; "
        f"recorded={len(points)}, generated={len(poses)} by segment interpolation"
    )
    return poses


def _joint_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Max absolute per-joint difference (deg) between two joint vectors."""
    return float(np.max(_joint_delta_deg(a, b)))


def _joint_midpoint_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular midpoint of two encoder joint vectors in degrees."""
    a_arr = np.asarray(a, dtype=np.float64)
    signed_delta = (np.asarray(b, dtype=np.float64) - a_arr + 180.0) % 360.0 - 180.0
    return a_arr + signed_delta * 0.5


def _wait_until_settled(
    ll,
    *,
    max_delta_deg: float,
    samples: int,
    period_s: float,
    timeout_s: float,
) -> bool:
    """Block until the arm stops creeping (encoder-verified), or timeout.

    ``move_joint_blocking`` returns once the joints are within
    ``joint_tolerance_deg`` for ``settle_samples`` reads, but STS3215 servos
    with ``settle_overcompensate`` keep creeping for a few hundred ms after
    that as the software I-term finishes trimming. Polling the encoder until
    ``samples`` consecutive reads all differ by < ``max_delta_deg`` is a far
    stronger "actually still" check than a blind fixed dwell. Returns False on
    timeout so the caller can reject the moving station.
    """
    deadline = time.monotonic() + timeout_s
    consecutive = 0
    prev = np.asarray(ll.get_angles(), dtype=np.float64)
    while time.monotonic() < deadline:
        time.sleep(period_s)
        cur = np.asarray(ll.get_angles(), dtype=np.float64)
        if _joint_angle_deg(cur, prev) < max_delta_deg:
            consecutive += 1
            if consecutive >= samples:
                return True
        else:
            consecutive = 0
        prev = cur
    return False


def _board_tilt_deg(tf_cam_target: np.ndarray) -> float:
    """Angle (deg) between the board normal and the camera optical axis.

    Board normal == the board's z-axis expressed in the camera frame ==
    column 3 of R_cam_target. Camera looks down +Z_cam. A small angle means
    the board faces the camera squarely; near 90 deg the board is edge-on
    (degenerate corners, unstable PnP) and must be rejected.
    """
    r = tf_cam_target[:3, :3]
    normal = r[:, 2]
    cos_tilt = float(np.clip(normal @ np.asarray([0.0, 0.0, 1.0]), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_tilt)))


def _capture_station_frames(
    ll,
    context: _CaptureContext,
    *,
    label: str,
) -> tuple[Station, dict] | None:
    """Capture and select motion-, reprojection-, and tilt-gated frames."""
    board = context.board
    camera_matrix = context.camera_matrix
    distortion_coeffs = context.distortion_coeffs
    args = context.args
    candidates: list[tuple[Station, dict]] = []
    rejected = {
        "settle": 0,
        "camera": 0,
        "motion": 0,
        "detection": 0,
        "reproj": 0,
        "tilt": 0,
    }
    relaxed = bool(getattr(args, "relaxed_gates", False))
    last_detection_reason = ""
    for frame_idx in range(args.frames_per_station):
        if not _wait_until_settled(
            ll,
            max_delta_deg=float(args.settle_max_delta_deg),
            samples=max(2, int(args.settle_samples) // 2),
            period_s=float(args.settle_period_s),
            timeout_s=float(args.settle_timeout_s),
        ):
            rejected["settle"] += 1
            continue
        q_before = np.asarray(ll.get_angles(), dtype=np.float64)
        frames = ll.grab_frames()
        q_after = np.asarray(ll.get_angles(), dtype=np.float64)
        if frames is None:
            rejected["camera"] += 1
            continue
        joint_motion = float(np.max(_joint_delta_deg(q_after, q_before)))
        if joint_motion > args.max_capture_joint_delta_deg:
            rejected["motion"] += 1
            continue
        rgb, _depth = frames
        det = detect_board(
            rgb,
            board,
            intrinsics=camera_matrix,
            dist=distortion_coeffs,
            min_corners=int(args.min_corners),
        )
        if not det.ok or det.tf_cam_target is None:
            rejected["detection"] += 1
            last_detection_reason = det.reason
            continue
        corners = 0 if det.object_points is None else len(det.object_points)
        reproj = float(det.reproj_rms_px or 0.0)
        if not relaxed and reproj > args.max_reproj_px:
            rejected["reproj"] += 1
            continue
        tilt = _board_tilt_deg(det.tf_cam_target)
        if not relaxed and tilt > args.max_board_tilt_deg:
            rejected["tilt"] += 1
            continue
        candidates.append(
            (
                Station(
                    tf_base_flange=_fk_from_joint_midpoint(ll, q_before, q_after),
                    detection=det,
                ),
                {
                    "q_before": q_before,
                    "q_after": q_after,
                    "joint_motion_deg": joint_motion,
                    "corners": corners,
                    "rgb": rgb,
                    "frame_index": frame_idx,
                },
            )
        )

    prefix = f"{label} " if label else ""
    if not candidates:
        counts = ", ".join(f"{name}={count}" for name, count in rejected.items())
        suffix = f"; last detection: {last_detection_reason}" if last_detection_reason else ""
        logger.info(f"  {prefix}station skipped: no usable frame ({counts}){suffix}")
        return None

    station, meta = max(
        candidates,
        key=lambda item: (int(item[1]["corners"]), -float(item[0].detection.reproj_rms_px or 0.0)),
    )
    tf_bf = station.tf_base_flange
    target = station.detection.tf_cam_target
    reproj = float(station.detection.reproj_rms_px or 0.0)
    tilt = _board_tilt_deg(target)
    logger.info(
        f"  {prefix}✓ flange t=({tf_bf[0, 3]:.1f},{tf_bf[1, 3]:.1f},{tf_bf[2, 3]:.1f})mm "
        f"board t_cam=({target[0, 3]:.1f},{target[1, 3]:.1f},{target[2, 3]:.1f})mm "
        f"reproj={reproj:.2f}px corners={meta['corners']} tilt={tilt:.0f}deg "
        f"motion={meta['joint_motion_deg']:.3f}deg"
    )
    return station, meta


def _collect_auto_station(
    ll,
    context: _CaptureContext,
    q_target: np.ndarray,
    station_idx: int,
    total: int,
) -> tuple[Station, dict] | None:
    """Drive to one joint pose under torque, settle, then motion-gate N frames."""
    args = context.args
    move_timed_out = False
    try:
        ll.move_joint_blocking(q_target.tolist())
    except TimeoutError as exc:
        # Reaching the requested joint target is not a hand-eye requirement.  A
        # loaded servo may stop at a repeatable offset outside the driver's
        # target tolerance; if the encoder subsequently proves that the arm is
        # still, capture its *actual* joint pose below.
        move_timed_out = True
        logger.warning(f"  [{station_idx}/{total}] WARN: move target timeout; checking actual pose stability: {exc}")
    except ValueError as exc:
        # Joint/FK/workspace safety violations are fail-closed.  Never turn a
        # rejected path into a capture attempt.
        logger.info(f"  [{station_idx}/{total}] SKIPPED unsafe/invalid move_joint: {exc}")
        return None
    except RuntimeError as exc:
        # In particular, the driver's settle-drift RuntimeError means the arm is
        # moving away from its target under load, not merely resting off-target.
        logger.info(f"  [{station_idx}/{total}] SKIPPED unstable move_joint: {exc}")
        return None
    except Exception as exc:  # isolate hardware/transport failures
        logger.info(f"  [{station_idx}/{total}] SKIPPED move_joint hardware error: {exc}")
        return None
    time.sleep(float(args.settle_dwell_s))
    is_settled = _wait_until_settled(
        ll,
        max_delta_deg=float(args.settle_max_delta_deg),
        samples=int(args.settle_samples),
        period_s=float(args.settle_period_s),
        timeout_s=float(args.settle_timeout_s),
    )
    if not is_settled:
        context = " after move target timeout" if move_timed_out else ""
        logger.info(f"  [{station_idx}/{total}] SKIPPED: encoder did not settle{context}; no capture")
        return None
    if move_timed_out:
        actual = np.asarray(ll.get_angles(), dtype=np.float64)
        target_error = _joint_angle_deg(actual, q_target)
        logger.info(
            f"  [{station_idx}/{total}] INFO: stable off-target pose accepted for capture "
            f"(max target error={target_error:.2f}deg)"
        )

    return _capture_station_frames(
        ll,
        context,
        label=f"[{station_idx}/{total}]",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the calibration CLI without running hardware actions."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=None,
        help="optional SO-101 YAML; omit it to use standalone calibration CLI defaults",
    )
    ap.add_argument("--port", default="/dev/ttyACM0", help="serial port when --config is omitted")
    ap.add_argument("--robot-id", default="so101_left", help="LeRobot calibration ID when --config is omitted")
    ap.add_argument(
        "--camera-serial",
        default="260322272909",
        help="RealSense serial when --config is omitted",
    )
    ap.add_argument("--calibration-dir", default=None, help="optional LeRobot motor calibration directory")
    ap.add_argument("--urdf-path", default=None, help="optional URDF path; default uses the packaged SO-101 URDF")
    ap.add_argument(
        "--move-timeout-s",
        type=float,
        default=None,
        help="whole joint-move timeout; standalone default=5s, YAML value unchanged when omitted",
    )
    ap.add_argument("--board", default="charuco")
    ap.add_argument("--squares-x", type=int, default=5)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument(
        "--square-size-mm",
        type=float,
        default=None,
        help="MEASURED board square edge (mm). Required for --auto / --collect-bounds "
        "(live detection); not needed with --replay or --release-torque.",
    )
    ap.add_argument("--marker-size-mm", type=float, default=None)
    ap.add_argument("--aruco-dict", default="DICT_4X4_50")
    mode = ap.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--auto",
        action="store_true",
        help="auto collect: the arm drives itself along a taught joint path "
        "(requires --bounds-file), motion-gating each frame without manual posing.",
    )
    mode.add_argument(
        "--collect-bounds",
        action="store_true",
        help="interactive mode: hand-pose the arm to an ordered sequence of "
        "reachable+board-visible points; --auto later follows those points and "
        "inserts intermediate capture stations.",
    )
    mode.add_argument(
        "--release-torque",
        action="store_true",
        help="release the five arm joints for manual repositioning, then safely restore "
        "torque. No camera, no board, no capture.",
    )
    mode.add_argument("--replay", default=None, help="re-solve from a previously dumped .npz (skip live collection)")
    ap.add_argument(
        "--n-stations",
        type=int,
        default=24,
        help="target station count for --auto (default 24; raise to 25-30 for a noisy arm).",
    )
    ap.add_argument("--min-pose-gap-deg", type=float, default=5.0, help="min joint-vector angle between poses (deg)")
    ap.add_argument(
        "--settle-dwell-s",
        type=float,
        default=0.3,
        help="initial blind dwell after move_joint returns, before encoder-verified settle (s)",
    )
    ap.add_argument(
        "--settle-samples",
        type=int,
        default=4,
        help="consecutive still encoder reads required before capture (each < --settle-max-delta-deg)",
    )
    ap.add_argument(
        "--settle-max-delta-deg",
        type=float,
        default=0.05,
        help="max inter-read joint motion (deg) that counts as 'still' during settle verification",
    )
    ap.add_argument(
        "--settle-period-s",
        type=float,
        default=0.05,
        help="encoder sampling period during settle verification (s)",
    )
    ap.add_argument(
        "--settle-timeout-s",
        type=float,
        default=3.0,
        help="max wait for the arm to go still per pose; an unsettled pose is not captured",
    )
    ap.add_argument(
        "--max-board-tilt-deg",
        type=float,
        default=45.0,
        help="reject a frame if the board normal is tilted more than this from the camera optical axis",
    )
    ap.add_argument("--method", default="PARK", choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"])
    ap.add_argument("--cross-check", action="store_true", help="multi-algorithm cross-check")
    ap.add_argument("--out", default=None, help="output calib JSON path")
    ap.add_argument(
        "--save-review",
        action="store_true",
        help="also save a failed/REVIEW result (unsafe; default saves ACCEPT only)",
    )
    ap.add_argument(
        "--drop-outliers",
        action="store_true",
        help="conservatively remove at most 20%% gross rigidity outliers, then re-solve",
    )
    ap.add_argument(
        "--bounds-file",
        default=None,
        help=".npz with ordered taught points (from --collect-bounds). Required for --auto; "
        "used by --collect-bounds to save the taught joint box.",
    )
    ap.add_argument(
        "--dump", default=None, help="dump collected stations (with rotations) to this .npz for offline replay"
    )
    ap.add_argument(
        "--min-corners", type=int, default=18, help="minimum ChArUco corners per accepted frame (5x7 board has 24)"
    )
    ap.add_argument("--max-reproj-px", type=float, default=0.8)
    ap.add_argument(
        "--relaxed-gates",
        action="store_true",
        help="collection fallback: retain --min-corners, stationary-frame, and valid-PnP "
        "requirements; do not reject on reprojection error or board tilt",
    )
    ap.add_argument("--frames-per-station", type=int, default=5)
    ap.add_argument(
        "--max-capture-joint-delta-deg",
        type=float,
        default=0.10,
        help="reject a frame if any joint moves this much across frame acquisition",
    )
    return ap


def _validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject inconsistent modes and unsafe collection settings.

    The mode group is already required + mutually exclusive at parse time; here we
    apply per-mode business rules (board-spec availability, bounds-file presence,
    n-stations floor). Only the live board-detection modes (--auto, --collect-bounds)
    require --square-size-mm; --replay and --release-torque do not.
    """
    if args.move_timeout_s is not None and args.move_timeout_s <= 0:
        ap.error("--move-timeout-s must be >0")

    if args.release_torque:
        # No board, no camera, no n-stations check; nothing else to validate.
        return

    if args.replay:
        # Offline re-solve: board spec and n-stations are unused.
        return

    # Live board-detection modes (--auto / --collect-bounds) from here.
    max_board_corners = (args.squares_x - 1) * (args.squares_y - 1)
    if not 6 <= args.min_corners <= max_board_corners:
        ap.error(f"--min-corners must be in [6, {max_board_corners}] for this board")
    if args.frames_per_station < 1:
        ap.error("--frames-per-station must be >=1")
    if args.square_size_mm is None:
        ap.error("--square-size-mm is required for --auto / --collect-bounds (measure the board)")

    if args.collect_bounds:
        if not args.bounds_file:
            ap.error("--collect-bounds requires --bounds-file <path.npz> to save the joint box")
        return

    # --auto
    if args.n_stations < MIN_STATIONS:
        ap.error(f"--n-stations must be >={MIN_STATIONS}")
    if not args.bounds_file:
        ap.error(
            "--auto requires --bounds-file <path.npz> (run --collect-bounds first); "
            "random around-home sampling was removed"
        )
    if not Path(args.bounds_file).is_file():
        ap.error(f"--bounds-file not found: {args.bounds_file}; run --collect-bounds first")


def _mode_label(args: argparse.Namespace) -> str:
    """Return the concise mode description printed at startup."""
    if args.collect_bounds:
        return "COLLECT-BOUNDS (hand-pose envelope)"
    if args.release_torque:
        return "RELEASE-TORQUE (release five arm joints, then restore)"
    if args.auto:
        return f"AUTO ({args.n_stations} joint poses)"
    if args.replay:
        return f"REPLAY ({args.replay})"
    return "UNKNOWN"


def _build_board(args: argparse.Namespace) -> tuple[BoardSpec, float]:
    """Build the live board specification and resolved marker size."""
    marker_size_mm = args.marker_size_mm or args.square_size_mm * 0.72
    return (
        BoardSpec(
            kind=args.board,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_size_mm=float(args.square_size_mm),
            marker_size_mm=float(marker_size_mm),
            aruco_dict=args.aruco_dict,
        ),
        float(marker_size_mm),
    )


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()
    _validate_args(ap, args)
    _configure_logging(getattr(args, "debug", False))
    cfg = So101Config.from_yaml(args.config) if args.config else _standalone_calibration_config(args)
    if args.config and args.move_timeout_s is not None:
        cfg.move_timeout_s = float(args.move_timeout_s)
    logger.info("=== SO-101 eye-to-hand hand-eye calibration (board clamped in gripper) ===")
    config_label = args.config if args.config else "standalone CLI defaults (no YAML)"
    logger.info(f"    config: {config_label}  port={cfg.port}")
    logger.info(f"    mode:   {_mode_label(args)}")

    # --replay: offline re-solve from a dumped .npz (no arm / no camera / no board spec).
    if args.replay:
        stations, camera_matrix = _load_stations(Path(args.replay))
        _solve_and_report(stations, camera_matrix, args)
        return

    # --release-torque: pure torque command; no board, no camera, no capture.
    if args.release_torque:
        if not _release_torque(cfg):
            raise SystemExit(1)
        return

    # Live board-detection modes (--auto / --collect-bounds) need the board spec.
    board, mk = _build_board(args)
    logger.info(
        f"    board:  {args.squares_x}x{args.squares_y} sq={args.square_size_mm}mm mk={mk:.1f}mm {args.aruco_dict}"
    )
    _run_live_collection(cfg, board, args)


def _run_live_collection(cfg: So101Config, board: BoardSpec, args: argparse.Namespace) -> None:
    """Connect hardware, collect stations, and always restore torque."""
    if cfg.camera_serial is None:
        logger.error("config has no camera_serial")
        return
    env = So101Env(cfg)
    ll = None
    stations: list[Station] = []
    capture_meta: list[dict] = []
    try:
        env.connect()
        ll = env.low_level
        if ll is None:
            raise RuntimeError("SO-101 env connected without a low-level driver.")
        camera_matrix = ll.intrinsics
        if camera_matrix is None:
            logger.error("no camera intrinsics")
            return
        logger.info(
            f"K: fx={camera_matrix[0, 0]:.1f} fy={camera_matrix[1, 1]:.1f} "
            f"ppx={camera_matrix[0, 2]:.1f} ppy={camera_matrix[1, 2]:.1f}"
        )
        context = _CaptureContext(
            board=board,
            camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
            distortion_coeffs=None,  # D405 factory intrinsics; detect_board handles dist=None.
            args=args,
        )
        if args.collect_bounds:
            _collect_bounds(ll, context)
            return

        limit = args.n_stations
        poses = _generate_joint_poses(args)
        source_label = f"along taught path {args.bounds_file}"
        logger.info(
            f"\n=== AUTO collection ({len(poses)} joint pose{'s' if len(poses) != 1 else ''} {source_label}) ==="
        )
        logger.info("the arm drives itself under torque; board stays tied+rigid; no hand vibration.")
        ack = _prompt("  回车启动自动采集；q=退出 > ")
        if ack is None or ack.lower() == "q":
            logger.info("aborted before auto start.")
            return
        for qi, q_target in enumerate(poses, start=1):
            if len(stations) >= limit:
                break
            result = _collect_auto_station(ll, context, q_target, qi, len(poses))
            if result is not None:
                st, meta = result
                q_actual = _joint_midpoint_deg(meta["q_before"], meta["q_after"])
                accepted_q = [
                    _joint_midpoint_deg(previous["q_before"], previous["q_after"]) for previous in capture_meta
                ]
                if any(_joint_angle_deg(q_actual, previous) < float(args.min_pose_gap_deg) for previous in accepted_q):
                    logger.info(
                        f"  [{qi}/{len(poses)}] station rejected: actual pose is within "
                        f"{args.min_pose_gap_deg:.1f}deg of an accepted station"
                    )
                    continue
                stations.append(st)
                capture_meta.append(meta)
                if len(stations) % 5 == 0 or len(stations) == limit:
                    spread = rotation_spread_deg(stations)
                    logger.info(f"  collected {len(stations)}/{limit}, rotation_spread={spread:.0f}deg")

        logger.info(f"\n=== collected {len(stations)} station(s) ===")
        for i, s in enumerate(stations):
            t = s.detection.tf_cam_target
            bf = s.tf_base_flange
            reproj = s.detection.reproj_rms_px if s.detection.reproj_rms_px is not None else float("nan")
            logger.info(
                f"  [{i}] flange_t=({bf[0, 3]:.1f},{bf[1, 3]:.1f},{bf[2, 3]:.1f})mm "
                f"board_t_cam=({t[0, 3]:.1f},{t[1, 3]:.1f},{t[2, 3]:.1f})mm reproj={reproj:.2f}px"
            )

        if not stations:
            logger.info("no accepted stations; nothing to dump or solve.")
            return
        if args.dump:
            _dump_stations(Path(args.dump), stations, camera_matrix, capture_meta)
        _solve_and_report(stations, camera_matrix, args)
        return
    finally:
        # Safety: never leave torque disabled. Re-enable even if collection aborted.
        if ll is not None:
            _enable_all_torque(ll)
        env.disconnect()
        logger.info("disconnected.")


def _solve_and_report(stations: list[Station], camera_matrix: np.ndarray, args) -> None:
    """Solve T_base_cam with the corrected eye-to-hand convention + log metrics."""
    if len(stations) < MIN_STATIONS:
        logger.info(f"need >= {MIN_STATIONS} stations to solve, have {len(stations)} — not solved.")
        return
    import cv2

    methods = _cv2_methods(cv2)
    if args.method not in methods:
        logger.info(f"unknown method {args.method}; choose {list(methods)}")
        return
    logger.info(f"\n=== solving T_base_cam (eye-to-hand, method={args.method}) ===")
    good = [s for s in stations if s.detection.ok and s.detection.tf_cam_target is not None]
    raw_solution = _solve_eye_to_hand(cv2, good, methods[args.method])
    raw_rot, raw_trans = _station_rigidity_errors(good, raw_solution)
    logger.info(
        f"  raw rigidity per-station: rot mean={raw_rot.mean():.2f}deg "
        f"max={raw_rot.max():.2f}deg; trans mean={raw_trans.mean():.2f}mm "
        f"max={raw_trans.max():.2f}mm"
    )
    dropped: list[int] = []
    if args.drop_outliers:
        good, dropped = _conservative_outlier_filter(good, cv2, methods[args.method])
        if dropped:
            logger.info(f"  retained {len(good)}/{len(stations)} stations after dropping {dropped}")
        else:
            logger.info("  no station met the conservative outlier criteria")
    tf_base_cam = _solve_eye_to_hand(cv2, good, methods[args.method])

    # Cross-check: all 5 OpenCV methods should agree if the convention is right.
    cc_deg = cc_mm = 0.0
    if args.cross_check:
        sols = {name: _solve_eye_to_hand(cv2, good, const) for name, const in methods.items()}
        for xi, xj in itertools.combinations(sols.values(), 2):
            cc_deg = max(cc_deg, _rotation_angle_deg(xi[:3, :3], xj[:3, :3]))
            cc_mm = max(cc_mm, float(np.linalg.norm(xi[:3, 3] - xj[:3, 3])))

    # Metrics with the CORRECT eye-to-hand convention (board-in-flange constant).
    rot_errs, trans_errs = _axxb_eye_to_hand(good, tf_base_cam)
    bf_pts = _board_in_flange_spread(good, tf_base_cam)
    bf_dists = np.linalg.norm(bf_pts - bf_pts.mean(axis=0), axis=1)
    reproj = [s.detection.reproj_rms_px for s in good if s.detection.reproj_rms_px is not None]
    spread = rotation_spread_deg(good)

    rpy = Rotation.from_matrix(tf_base_cam[:3, :3]).as_euler("xyz", degrees=True)
    t = tf_base_cam[:3, 3]
    logger.info(f"  T_base_cam t_mm = ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")
    logger.info(f"  rpy_deg        = ({rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f})")
    logger.info(f"  n_stations={len(good)}  rotation_spread={spread:.0f}deg")
    logger.info(f"  reproj      mean={np.mean(reproj):.2f}px  max={np.max(reproj):.2f}px")
    logger.info(f"  AX=XB rot   mean={np.mean(rot_errs):.3f}deg  max={np.max(rot_errs):.3f}deg")
    logger.info(f"  AX=XB trans mean={np.mean(trans_errs):.2f}mm  max={np.max(trans_errs):.2f}mm")
    logger.info(f"  board-in-flange spread mean={np.mean(bf_dists):.2f}mm  max={np.max(bf_dists):.2f}mm")
    if args.cross_check:
        logger.info(f"  cross-check max_deg={cc_deg:.2f}  max_mm={cc_mm:.2f}")
    # A calibration is publishable only when data quality, rigidity, excitation,
    # and (when requested) solver agreement all pass.
    solver_agrees = not args.cross_check or (cc_deg < 2.0 and cc_mm < 10.0)
    ok = (
        len(good) >= 8
        and spread >= 20.0
        and np.max(reproj) <= args.max_reproj_px
        and np.max(rot_errs) < 1.0
        and np.max(trans_errs) < 5.0
        and np.max(bf_dists) < 5.0
        and solver_agrees
    )
    logger.info(f"  => {'ACCEPT' if ok else 'REVIEW (some metric over threshold)'}")
    if dropped:
        logger.info(f"  NOTE: final solution excluded original station(s) {dropped}")
    # Leave-one-out board-in-flange spread to spot which station(s) break rigidity.
    if np.max(bf_dists) > 5.0 and len(good) > 3:
        per_drop = _leave_one_out_spreads(good, cv2, methods, base_max=float(np.max(bf_dists)))
        logger.info("  leave-one-out (drop each station -> board-in-flange max spread mm):")
        for k, m in per_drop:
            mark = "  <— most improved" if k == min(per_drop, key=lambda x: x[1])[0] else ""
            logger.info(f"    drop {k}: {m:.2f}mm{mark}")
        # Per-station board-in-flange position to eyeball outliers.
        logger.info("  per-station board-in-flange position (mm):")
        for i, p in enumerate(bf_pts):
            logger.info(f"    [{i}] ({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})")
    if args.out:
        out = Path(args.out)
    elif args.config:
        config_path = Path(args.config)
        out = config_path.with_name(config_path.stem + "_calib.json")
    else:
        out = Path("tmp/so101_eye_calib.json")
    if ok or args.save_review:
        save_calibration(
            out,
            tf_base_cam,
            camera_matrix,
            [float(v) for v in bf_pts.mean(axis=0)],
            frame_field="T_base_cam",
            frame_comment="camera pose in base frame; translation in mm",
            top_comment="SO-101 eye-to-hand hand-eye calibration (board clamped in gripper).",
        )
        logger.info(f"saved -> {out}")
    else:
        logger.info(f"NOT SAVED: REVIEW result would have been written to {out}")


def _leave_one_out_spreads(stations, cv2, methods, *, base_max: float) -> list[tuple[int, float]]:
    """Drop each station in turn, re-solve, report board-in-flange max spread mm."""
    out: list[tuple[int, float]] = []
    for k in range(len(stations)):
        sub = [s for j, s in enumerate(stations) if j != k]
        if len(sub) < MIN_STATIONS:
            continue
        solution_tf = _solve_eye_to_hand(cv2, sub, methods["PARK"])
        pts = _board_in_flange_spread(sub, solution_tf)
        m = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
        out.append((k, m))
    return out


if __name__ == "__main__":
    main()
