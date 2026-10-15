# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Live controlled-trajectory eye-to-hand calibration workflow."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from jiuwensymbiosis.calibration.artifacts.archives import dump_station_archive, load_waypoint_archive
from jiuwensymbiosis.calibration.domain.models import Station, ViewDetection
from jiuwensymbiosis.calibration.domain.ports import (
    CalibrationCameraFrame,
    CalibrationCaptureSource,
    CalibrationPoseSource,
    CartesianCalibrationMotion,
    JointCalibrationMotion,
)
from jiuwensymbiosis.calibration.domain.trajectory import interpolate_cartesian_path, interpolate_joint_sequence
from jiuwensymbiosis.calibration.domain.validation import validate_intrinsics
from jiuwensymbiosis.calibration.workflows.capture import CaptureResult
from jiuwensymbiosis.calibration.workflows.preflight import (
    PreflightError,
    validate_archive_ownership,
    validate_cartesian_trajectory,
    validate_joint_metadata,
    validate_motion_capabilities,
    validate_mount,
    validate_mount_consistency,
)
from jiuwensymbiosis.calibration.workflows.profile import load_profile, resolve_out_path, resolve_space
from jiuwensymbiosis.calibration.workflows.publication import solve_and_publish
from jiuwensymbiosis.calibration.workflows.workflows import CalibrationRunOptions, RunOutcome, WorkflowDependencies
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger("jiuwensymbiosis.calibration.execute")


@dataclass(frozen=True)
class CaptureGateThresholds:
    """Reach and exposure stability thresholds in degrees and millimetres."""

    reach_rotation_deg: float = 0.5
    reach_translation_mm: float = 0.5
    exposure_rotation_deg: float = 0.5
    exposure_translation_mm: float = 0.5


def execute_calibration(
    device,
    waypoint_archive: str | Path,
    options: CalibrationRunOptions | None = None,
    *,
    mount: str | None = None,
    dependencies: WorkflowDependencies | None = None,
) -> RunOutcome:
    """Execute a validated waypoint archive, capture stations, solve and publish."""
    deps = dependencies or WorkflowDependencies(options=options or CalibrationRunOptions())
    run_options = deps.options
    waypoints = load_waypoint_archive(waypoint_archive)
    space = resolve_space({"trajectory": {"space": waypoints["space"]}})
    raw_camera_mount = getattr(device, "camera_mount", None)
    if raw_camera_mount is None:
        raise PreflightError(
            f"{type(device).__name__} does not declare camera_mount; "
            "execution cannot determine whether the camera is eye_in_hand or eye_to_hand."
        )
    camera_mount = validate_mount(str(raw_camera_mount), source="device.camera_mount")
    if mount is not None:
        validate_mount(mount, source="caller mount")
        validate_mount_consistency(mount, camera_mount)
    validate_motion_capabilities(device, space)
    if deps.adapter_package is not None:
        validate_archive_ownership(
            waypoints.get("metadata") or {},
            deps.adapter_package,
            camera_mount,
            space,
            waypoints["space"],
        )
    if space == "cartesian":
        validate_cartesian_trajectory(np.asarray(waypoints["cartesian_tfs"]))
    # Expand before the dry-run return so joint-space archives are checked
    # against the live order/unit/periodic metadata even when no motion is
    # requested.  Expansion itself is pure and never commands the device.
    dense = _expand_trajectory(device, waypoints, space, run_options)
    if run_options.dry_run:
        logger.info("dry-run: trajectory and data contract validated; no motion, solve or publication.")
        return RunOutcome(result=None, decision=None, artifact_path=None, candidate=False)
    _dynamic_camera_preflight(device)
    capture = capture_stations(
        device,
        waypoints,
        dense,
        space,
        options=run_options,
        detect_fn=deps.detect_fn,
    )
    if not capture.stations:
        logger.error("no valid stations captured; nothing to solve.")
        return RunOutcome(result=None, decision=None, artifact_path=None, candidate=True)
    if capture.intrinsics is None:
        raise PreflightError("capture returned stations without camera intrinsics")
    out_path = resolve_out_path(run_options.out, run_options.config)
    station_path = out_path.with_name(out_path.stem + ".stations.npz")
    dump_station_archive(
        station_path,
        capture.stations,
        capture.intrinsics,
        adapter_module=deps.adapter_package or "unknown",
        mount=camera_mount,
        capture_meta=capture.capture_meta,
        joint_meta=capture.joint_meta,
        distortion=capture.distortion,
        trajectory_space=space,
        config=None,
    )
    logger.info("wrote station archive (%d stations) -> %s", len(capture.stations), station_path)
    return solve_and_publish(
        capture.stations,
        capture.intrinsics,
        mount=camera_mount,
        out_path=out_path,
        dependencies=deps,
    )


def _dynamic_camera_preflight(device) -> None:
    """Probe the live camera before any trajectory motion is commanded.

    Static preflight can verify the archive and motion protocols, but only a
    connected device can prove that the camera port is available and exposes a
    usable intrinsic matrix.  The frame is intentionally discarded; capture
    still takes fresh frames at the selected stations.
    """
    if not isinstance(device, CalibrationCaptureSource):
        raise PreflightError(f"{type(device).__name__} lacks CalibrationCaptureSource; cannot preflight camera.")
    try:
        frame = device.capture_calibration_frame()
    except Exception as exc:
        raise PreflightError(f"camera preflight failed: capture_calibration_frame() raised {exc}") from exc
    try:
        validate_intrinsics(frame.intrinsics, source="camera preflight frame")
    except Exception as exc:
        raise PreflightError(f"camera preflight failed: {exc}") from exc
    if frame.distortion is not None:
        try:
            distortion = np.asarray(frame.distortion, dtype=np.float64)
        except Exception as exc:
            raise PreflightError(f"camera preflight failed: distortion is not numeric ({exc})") from exc
        if not np.isfinite(distortion).all():
            raise PreflightError("camera preflight failed: distortion contains non-finite values")


def capture_stations(
    device,
    waypoints: dict[str, Any],
    dense: list[np.ndarray],
    space: str,
    *,
    options: CalibrationRunOptions,
    detect_fn: Callable[[CalibrationCameraFrame], ViewDetection] | None = None,
) -> CaptureResult:
    """Drive the complete dense path and capture only the selected station indices."""
    if not isinstance(device, CalibrationCaptureSource):
        raise PreflightError(f"{type(device).__name__} lacks CalibrationCaptureSource; cannot capture frames.")
    pose_source = cast(CalibrationPoseSource, device)
    joint_motion = cast(JointCalibrationMotion, device)
    cartesian_motion = cast(CartesianCalibrationMotion, device)
    gate_thresholds = _capture_gate_thresholds_from(load_profile(options.config))
    unit = str(waypoints.get("joint_unit", "deg")) if space == "joint" else "deg"
    capture_at = set(_capture_indices(len(dense), options.n_stations))
    if len(capture_at) < min(len(dense), options.n_stations):
        logger.warning(
            "dense path (%d) shorter than requested stations (%d); capturing at %d targets.",
            len(dense),
            options.n_stations,
            len(capture_at),
        )
    joint_meta = None
    if space == "joint":
        joint_state = joint_motion.get_joint_state()
        joint_meta = {
            "unit": joint_state.unit,
            "order": list(joint_state.order),
            "periodic": [bool(value) for value in joint_state.periodic],
        }

    def read_pose() -> np.ndarray:
        if space == "joint":
            return np.asarray(joint_motion.get_joint_state().values)
        return np.asarray(pose_source.get_flange_transform_mm())

    def move_to(target: np.ndarray) -> None:
        if space == "joint":
            joint_motion.move_joint_vector(np.asarray(target, dtype=np.float64))
        else:
            cartesian_motion.move_to_flange_transform_mm(np.asarray(target, dtype=np.float64))

    # Stream a dense joint sub-path and settle only at its end.  When the device
    # exposes ``move_joint_path`` (SO-101), intermediate slew-limit points are
    # dispatched at trajectory_hz without per-point settling — otherwise fall
    # back to per-point ``move_to``.  Either way, a settle ``TimeoutError`` at
    # the segment's final target is swallowed: SO-101's STS3215 servos have no
    # integral term, so under gravity load the steady-state offset can exceed
    # ``joint_tolerance_deg`` even though the arm has settled at a stable, real
    # pose.  Calibration records FK of the *real* joint angles, so a segment
    # that did not reach the command is still a valid observation as long as the
    # reach/exposure gates accept the actual pose.  ``So101PoseConvergenceError``
    # (genuine drift / overcomp-fail-closed) still propagates.
    path_move = getattr(device, "move_joint_path", None)

    def run_segment(segment: list[np.ndarray], end_index: int) -> dict[str, Any]:
        settle_timeout_exc: TimeoutError | None = None
        try:
            if space == "joint" and callable(path_move) and len(segment) > 1:
                path_move(np.stack(segment, axis=0))
            else:
                for wp in segment:
                    move_to(np.asarray(wp, dtype=np.float64))
        except TimeoutError as exc:
            settle_timeout_exc = exc
            logger.warning(
                "segment ending at dense %d: move settle timeout (%s); "
                "continuing to the reach gate at the settled pose.",
                end_index,
                exc,
            )
        settle_info = _settle(device)
        if settle_timeout_exc is not None:
            settle_info = {**settle_info, "settle_timeout": str(settle_timeout_exc)}
        return settle_info

    stations: list[Station] = []
    capture_meta: list[dict[str, Any]] = []
    intrinsics: np.ndarray | None = None
    distortion: np.ndarray | None = None
    # Iterate only the capture indices; each segment runs from the previous
    # capture point (or 0) through this capture point inclusive.  Non-capture
    # dense points are streamed through, not settled on.
    capture_indices = sorted(capture_at)
    prev_boundary = 0
    for capture_count, index in enumerate(capture_indices):
        target = dense[index]
        segment_end = index + 1
        segment = list(dense[prev_boundary:segment_end])
        pre_move = read_pose()
        settle_info = run_segment(segment, index)
        logger.info(
            "capture %d/%d (dense %d/%d, segment %d pts)",
            capture_count + 1,
            len(capture_indices),
            index + 1,
            len(dense),
            len(segment),
        )
        pre_capture = read_pose()
        reached, reach_rotation, reach_translation = _reached(
            target,
            pre_capture,
            space=space,
            thresholds=gate_thresholds,
            unit=unit,
        )
        # ``reach`` (command vs real pose) is DIAGNOSTIC ONLY, not a skip gate.
        # Calibration records ``Station.tf_base_flange`` = FK of the *real* joint
        # angles (``device.get_flange_transform_mm()``), so a station that did
        # not reach the command is still a geometrically self-consistent
        # observation as long as the board is visible and the pose is stable
        # during exposure.  SO-101 STS3215 servos have no integral term; under
        # gravity load the steady-state offset can exceed any reasonable
        # ``reach_rotation_deg`` — gating on it would discard valid stations.
        # Genuine drift / overcomp failure still fails closed at the driver via
        # ``So101PoseConvergenceError`` (propagated, not swallowed here).
        if not reached:
            logger.info(
                "station %d: reach gate diagnostic (rot=%.3fdeg trans=%.3fmm); "
                "continuing — calibration uses FK of the real pose.",
                index,
                reach_rotation,
                reach_translation,
            )
        try:
            frame = device.capture_calibration_frame()
        except Exception as exc:
            logger.warning("station %d: capture_calibration_frame failed (%s); skipping.", index, exc)
            capture_meta.append(
                _capture_meta(
                    index,
                    reached=reached,
                    reason=f"capture_error:{exc}",
                    settle=settle_info,
                    pre_move=_pose_list(pre_move),
                    pre_capture=_pose_list(pre_capture),
                )
            )
            prev_boundary = index + 1
            continue
        if intrinsics is None:
            intrinsics = validate_intrinsics(frame.intrinsics, source="live camera frame")
        if distortion is None:
            distortion = frame.distortion
        detection = _detect_board(frame, detect_fn)
        if not detection.ok or detection.tf_cam_target is None:
            logger.warning("station %d: board not detected (%s); skipping.", index, detection.reason)
            capture_meta.append(
                _capture_meta(
                    index,
                    reached=reached,
                    reason=f"detect:{detection.reason}",
                    settle=settle_info,
                    pre_move=_pose_list(pre_move),
                    pre_capture=_pose_list(pre_capture),
                )
            )
            prev_boundary = index + 1
            continue
        post_capture = read_pose()
        stable, exposure_rotation, exposure_translation = _exposure_stable(
            pre_capture,
            post_capture,
            space=space,
            thresholds=gate_thresholds,
            unit=unit,
        )
        if not stable:
            logger.warning(
                "station %d: exposure drift (rot=%.3fdeg trans=%.3fmm); skipping.",
                index,
                exposure_rotation,
                exposure_translation,
            )
            capture_meta.append(
                _capture_meta(
                    index,
                    reached=reached,
                    reason="exposure_drift",
                    exposure=(exposure_rotation, exposure_translation),
                    settle=settle_info,
                    pre_move=_pose_list(pre_move),
                    pre_capture=_pose_list(pre_capture),
                    post_capture=_pose_list(post_capture),
                )
            )
            prev_boundary = index + 1
            continue
        stations.append(Station(tf_base_flange=pose_source.get_flange_transform_mm(), detection=detection))
        capture_meta.append(
            _capture_meta(
                index,
                reached=reached,
                reason="ok" if reached else "ok_reach_diagnostic",
                accepted_index=len(stations) - 1,
                reach=(reach_rotation, reach_translation),
                exposure=(exposure_rotation, exposure_translation),
                settle=settle_info,
                pre_move=_pose_list(pre_move),
                pre_capture=_pose_list(pre_capture),
                post_capture=_pose_list(post_capture),
            )
        )
        prev_boundary = index + 1
    return CaptureResult(
        stations=stations,
        intrinsics=intrinsics,
        capture_meta=capture_meta,
        joint_meta=joint_meta,
        distortion=distortion,
    )


def _expand_trajectory(device, waypoints: dict[str, Any], space: str, options: CalibrationRunOptions):
    if space == "joint":
        joint_motion = cast(JointCalibrationMotion, device)
        validate_joint_metadata(
            tuple(waypoints["joint_order"]),
            str(waypoints["joint_unit"]),
            tuple(bool(value) for value in waypoints["joint_periodic"]),
            joint_motion.get_joint_state(),
        )
        return interpolate_joint_sequence(
            waypoints["joint_values"],
            unit=waypoints["joint_unit"],
            order=tuple(waypoints["joint_order"]),
            periodic=tuple(waypoints["joint_periodic"]),
            max_step_deg=options.max_joint_step_deg,
        )
    return _interpolate_cart_sequence(
        np.asarray(waypoints["cartesian_tfs"]),
        max_step_mm=options.max_cartesian_step_mm,
        max_step_deg=options.max_cartesian_step_deg,
    )


def _detect_board(
    frame: CalibrationCameraFrame,
    detect_fn: Callable[[CalibrationCameraFrame], ViewDetection] | None,
) -> ViewDetection:
    if frame.tf_cam_target is not None:
        return ViewDetection(ok=True, tf_cam_target=frame.tf_cam_target, reproj_rms_px=frame.reproj_rms_px)
    if detect_fn is not None:
        return detect_fn(frame)
    raise PreflightError(
        "board detection is not wired on this device; expose tf_cam_target on "
        "CalibrationCameraFrame or inject a detector into the workflow."
    )


def _capture_gate_thresholds_from(profile: dict[str, Any]) -> CaptureGateThresholds:
    values = profile.get("capture_gate") or {}
    defaults = CaptureGateThresholds()
    return CaptureGateThresholds(
        reach_rotation_deg=float(values.get("reach_rotation_deg", defaults.reach_rotation_deg)),
        reach_translation_mm=float(values.get("reach_translation_mm", defaults.reach_translation_mm)),
        exposure_rotation_deg=float(values.get("exposure_rotation_deg", defaults.exposure_rotation_deg)),
        exposure_translation_mm=float(values.get("exposure_translation_mm", defaults.exposure_translation_mm)),
    )


def _interpolate_cart_sequence(tfs: np.ndarray, *, max_step_mm: float, max_step_deg: float) -> list[np.ndarray]:
    dense: list[np.ndarray] = []
    for index in range(len(tfs) - 1):
        segment = interpolate_cartesian_path(
            tfs[index],
            tfs[index + 1],
            max_step_mm=max_step_mm,
            max_step_deg=max_step_deg,
        )
        dense.extend(segment if index == 0 else segment[1:])
    return dense


def _capture_indices(n_dense: int, n_stations: int) -> list[int]:
    if n_dense <= 0:
        return []
    if n_dense <= n_stations:
        return list(range(n_dense))
    if n_stations == 1:
        return [0]
    return [round(index * (n_dense - 1) / (n_stations - 1)) for index in range(n_stations)]


# Polling-based settle parameters.  Devices that block inside their move
# (SO-101's ``move_joint_blocking`` / ``move_joint_path_blocking`` already
# settle internally) will converge on the first 1-2 samples, so ``_settle``
# adds only a brief residual-creep confirmation before the camera exposes.
# Fire-and-forget devices without a ``wait_until_settled`` hook rely on this
# poll loop as their primary settle criterion.  ``_SETTLE_DEADLINE_S`` is a hard
# upper bound so a stalled servo cannot hang capture indefinitely.
_SETTLE_POLL_PERIOD_S = 0.08
_SETTLE_STABLE_DEG = 0.1
_SETTLE_STABLE_SAMPLES = 3
_SETTLE_DEADLINE_S = 5.0


def _settle(device) -> dict[str, Any]:
    settle = getattr(device, "wait_until_settled", None)
    if callable(settle):
        try:
            return {"hook": "wait_until_settled", "ok": bool(settle())}
        except Exception as exc:
            logger.warning("wait_until_settled failed; continuing to the reach gate: %s", exc)
            return {"hook": "wait_until_settled", "error": str(exc)}
    # Poll the joint state (or flange pose) until it stops creeping.  Unlike a
    # fixed ``time.sleep`` this returns as soon as the pose is genuinely still,
    # which is near-instant for devices whose move already blocks to a settle
    # (SO-101) and bounded by ``_SETTLE_DEADLINE_S`` for fire-and-forget ones.
    # It does NOT eliminate the gravity steady-state offset (no integral term);
    # it only confirms residual creep has damped before the camera exposes.
    joint_fn = getattr(device, "get_joint_state", None)
    flange_fn = getattr(device, "get_flange_transform_mm", None)
    if not callable(joint_fn) and not callable(flange_fn):
        time.sleep(0.1)
        return {"hook": "sleep", "seconds": 0.1, "reason": "no pose reader"}

    joint_reader = cast(Callable[[], Any] | None, joint_fn if callable(joint_fn) else None)
    flange_reader = cast(Callable[[], Any] | None, flange_fn if callable(flange_fn) else None)

    def _read_pose_vec() -> np.ndarray:
        if joint_reader is not None:
            return np.asarray(joint_reader().values, dtype=np.float64).reshape(-1)
        if flange_reader is None:  # guarded above; keeps the closure type-safe
            raise RuntimeError("settle polling has no pose reader")
        return np.asarray(flange_reader(), dtype=np.float64).reshape(-1)

    deadline = time.monotonic() + _SETTLE_DEADLINE_S
    prev = None
    stable_run = 0
    last = None
    while True:
        last = _read_pose_vec()
        if prev is not None:
            if float(np.max(np.abs(last - prev))) <= _SETTLE_STABLE_DEG:
                stable_run += 1
                if stable_run >= _SETTLE_STABLE_SAMPLES:
                    return {"hook": "poll", "samples": stable_run}
            else:
                stable_run = 0
        prev = last
        if time.monotonic() >= deadline:
            return {"hook": "poll", "timeout": True, "deadline_s": _SETTLE_DEADLINE_S}
        time.sleep(_SETTLE_POLL_PERIOD_S)


def _reached(
    target,
    actual,
    *,
    space: str,
    thresholds: CaptureGateThresholds,
    unit: str = "deg",
) -> tuple[bool, float, float]:
    if space == "joint":
        rotation = _joint_delta_deg(np.asarray(target), np.asarray(actual), unit=unit)
        translation = 0.0
    else:
        rotation = _rot_delta_deg(np.asarray(target), np.asarray(actual))
        translation = _trans_delta_mm(np.asarray(target), np.asarray(actual))
    return (
        rotation <= thresholds.reach_rotation_deg and translation <= thresholds.reach_translation_mm,
        rotation,
        translation,
    )


def _exposure_stable(
    pre,
    post,
    *,
    space: str,
    thresholds: CaptureGateThresholds,
    unit: str = "deg",
) -> tuple[bool, float, float]:
    if space == "joint":
        rotation = _joint_delta_deg(np.asarray(pre), np.asarray(post), unit=unit)
        translation = 0.0
    else:
        rotation = _rot_delta_deg(np.asarray(pre), np.asarray(post))
        translation = _trans_delta_mm(np.asarray(pre), np.asarray(post))
    return (
        rotation <= thresholds.exposure_rotation_deg and translation <= thresholds.exposure_translation_mm,
        rotation,
        translation,
    )


def _rot_delta_deg(tf_a: np.ndarray, tf_b: np.ndarray) -> float:
    relative = np.linalg.inv(tf_a) @ tf_b
    cosine = float(np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _trans_delta_mm(tf_a: np.ndarray, tf_b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(tf_b)[:3, 3] - np.asarray(tf_a)[:3, 3]))


def _joint_delta_deg(q_a: np.ndarray, q_b: np.ndarray, unit: str) -> float:
    delta = np.abs(np.asarray(q_b, dtype=np.float64) - np.asarray(q_a, dtype=np.float64))
    if unit == "rad":
        delta = np.degrees(delta)
    return float(np.max(delta)) if delta.size else 0.0


def _pose_list(pose) -> list[float]:
    return [float(value) for value in np.asarray(pose, dtype=np.float64).reshape(-1)]


def _capture_meta(
    target_index: int,
    *,
    reached: bool,
    reason: str,
    reach: tuple[float, float] | None = None,
    exposure: tuple[float, float] | None = None,
    accepted_index: int | None = None,
    settle: dict[str, Any] | None = None,
    pre_move: list[float] | None = None,
    pre_capture: list[float] | None = None,
    post_capture: list[float] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"target_index": int(target_index), "reached": bool(reached), "reason": reason}
    if reach is not None:
        record["reach_rot_deg"] = float(reach[0])
        record["reach_trans_mm"] = float(reach[1])
    if exposure is not None:
        record["exposure_rot_deg"] = float(exposure[0])
        record["exposure_trans_mm"] = float(exposure[1])
    if accepted_index is not None:
        record["accepted_index"] = int(accepted_index)
    if settle is not None:
        record["settle"] = settle
    if pre_move is not None:
        record["pre_move"] = pre_move
    if pre_capture is not None:
        record["pre_capture"] = pre_capture
    if post_capture is not None:
        record["post_capture"] = post_capture
    return record


__all__ = ["CaptureGateThresholds", "execute_calibration"]
