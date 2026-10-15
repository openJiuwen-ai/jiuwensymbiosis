# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Operator-guided calibration waypoint collection workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from jiuwensymbiosis.calibration.artifacts.archives import dump_waypoint_archive, load_waypoint_archive
from jiuwensymbiosis.calibration.workflows.preflight import (
    PreflightError,
    resolve_teaching_mode,
    validate_mount,
    validate_mount_consistency,
)
from jiuwensymbiosis.calibration.workflows.profile import load_profile, resolve_space
from jiuwensymbiosis.calibration.workflows.workflows import CalibrationRunOptions
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger("jiuwensymbiosis.calibration.collect")


def collect_waypoints(
    device,
    out_path: str | Path,
    options: CalibrationRunOptions | None = None,
    *,
    adapter_package: str | None = None,
    mount: str | None = None,
    prompt_fn: Callable[[str], str | None] | None = None,
) -> Path:
    """Collect waypoints through manual guidance or external-teach snapshots."""
    run_options = options or CalibrationRunOptions()
    out = Path(out_path)
    space = resolve_space(load_profile(run_options.config))
    raw_camera_mount = getattr(device, "camera_mount", None)
    if raw_camera_mount is None:
        raise PreflightError(
            f"{type(device).__name__} does not declare camera_mount; "
            "collection cannot determine whether the camera is eye_in_hand or eye_to_hand."
        )
    camera_mount = validate_mount(str(raw_camera_mount), source="device.camera_mount")
    if mount is not None:
        validate_mount(mount, source="caller mount")
        validate_mount_consistency(mount, camera_mount)
    mode = resolve_teaching_mode(device)
    logger.info("collect-waypoints: mode=%s, space=%s", mode, space)

    def snapshot() -> np.ndarray:
        if space == "joint":
            return np.asarray(device.get_joint_state().values, dtype=np.float64)
        return np.asarray(device.get_flange_transform_mm(), dtype=np.float64)

    prompt = prompt_fn or _default_prompt
    if mode == "manual_guidance":
        with device.manual_guidance():
            waypoints = _collect_interactive_snapshots(snapshot, prompt)
    else:
        waypoints = _collect_interactive_snapshots(snapshot, prompt)
    values = np.stack(waypoints)
    if space == "joint":
        joint_state = device.get_joint_state()
        dump_waypoint_archive(
            out,
            space="joint",
            joint_values=values,
            joint_unit=joint_state.unit,
            joint_order=joint_state.order,
            joint_periodic=joint_state.periodic,
            adapter_module=adapter_package or "unknown",
            mount=camera_mount,
            config=None,
        )
    else:
        dump_waypoint_archive(
            out,
            space="cartesian",
            cartesian_tfs=values,
            adapter_module=adapter_package or "unknown",
            mount=camera_mount,
            config=None,
        )
    logger.info("collected %d waypoints -> %s", len(waypoints), out)
    return out


def import_waypoints(
    in_archive: str | Path,
    out_path: str | Path,
    *,
    expected_mount: str | None = None,
) -> Path:
    """Normalize an existing waypoint archive without connecting hardware.

    ``expected_mount`` is an optional entry-point assertion. The archive is
    loaded once, then its self-describing mount is checked before writing the
    normalized copy.
    """
    loaded = load_waypoint_archive(in_archive)
    out = Path(out_path)
    metadata = loaded["metadata"]
    camera_mount = validate_mount(str(metadata["mount"]), source=f"{in_archive}:metadata.mount")
    if expected_mount is not None:
        validate_mount_consistency(expected_mount, camera_mount)
    logger.info("imported %d waypoints (space=%s) from %s", loaded["n"], loaded["space"], in_archive)
    if loaded["space"] == "joint":
        dump_waypoint_archive(
            out,
            space="joint",
            joint_values=loaded["joint_values"],
            joint_unit=loaded["joint_unit"],
            joint_order=loaded["joint_order"],
            joint_periodic=loaded["joint_periodic"],
            adapter_module=metadata.get("adapter_module", "unknown"),
            mount=camera_mount,
            config=None,
        )
    else:
        dump_waypoint_archive(
            out,
            space="cartesian",
            cartesian_tfs=loaded["cartesian_tfs"],
            adapter_module=metadata.get("adapter_module", "unknown"),
            mount=camera_mount,
            config=None,
        )
    logger.info("normalized waypoints -> %s", out)
    return out


def _collect_interactive_snapshots(
    snapshot_fn: Callable[[], np.ndarray],
    prompt_fn: Callable[[str], str | None],
) -> list[np.ndarray]:
    waypoints: list[np.ndarray] = []
    try:
        while True:
            answer = prompt_fn(f"  move to waypoint {len(waypoints)} then Enter to record (q=finish) > ")
            if answer is None or answer.lower() == "q":
                break
            waypoints.append(snapshot_fn())
    except KeyboardInterrupt:
        logger.info("waypoint collection interrupted by operator")
    except EOFError:
        logger.info("waypoint collection input reached EOF")
    if len(waypoints) < 2:
        raise ValueError("need at least 2 waypoints; re-run --collect-poses.")
    return waypoints


def _default_prompt(prompt: str) -> str | None:
    try:
        return input(prompt).strip()
    except EOFError:
        logger.info("waypoint collection input reached EOF")
        return None


__all__ = ["collect_waypoints", "import_waypoints"]
