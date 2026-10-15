# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Offline station-archive replay workflow (mount-neutral)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jiuwensymbiosis.calibration.artifacts.archives import load_station_archive
from jiuwensymbiosis.calibration.artifacts.artifacts import save_candidate_report
from jiuwensymbiosis.calibration.domain.models import EyeInHandResult, EyeToHandResult
from jiuwensymbiosis.calibration.domain.solver import MIN_STATIONS
from jiuwensymbiosis.calibration.domain.validation import validate_intrinsics
from jiuwensymbiosis.calibration.workflows.preflight import (
    validate_mount,
    validate_mount_consistency,
)
from jiuwensymbiosis.calibration.workflows.profile import resolve_out_path
from jiuwensymbiosis.calibration.workflows.publication import publish_with_reload, solve_stations
from jiuwensymbiosis.calibration.workflows.workflows import CalibrationRunOptions, RunOutcome, WorkflowDependencies
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger("jiuwensymbiosis.calibration.replay")


def replay_calibration(
    station_archive: str | Path,
    options: CalibrationRunOptions | None = None,
    *,
    mount: str | None = None,
    dependencies: WorkflowDependencies | None = None,
) -> RunOutcome:
    """Re-solve a station archive and emit a candidate or validated artifact.

    The archive metadata is the single mount authority; a caller-supplied
    ``mount`` is a consistency assertion only and is rejected when it disagrees.
    """
    deps = dependencies or WorkflowDependencies(options=options or CalibrationRunOptions())
    stations, intrinsics, metadata = load_station_archive(station_archive)
    intrinsics = validate_intrinsics(intrinsics, source="station archive")
    raw_mount = metadata.get("mount")
    if raw_mount is None:
        raise ValueError(f"{station_archive}: station archive metadata.mount is required; frame is ambiguous.")
    archive_mount = validate_mount(str(raw_mount), source="station archive")
    logger.info(
        "replay: %d stations, mount=%s, adapter=%s",
        len(stations),
        archive_mount,
        metadata.get("adapter_module"),
    )
    out_path = resolve_out_path(deps.options.out, deps.options.config)
    has_config = bool(deps.options.config)
    if mount is not None:
        validate_mount_consistency(mount, archive_mount)
    if has_config:
        if deps.adapter_package is None:
            raise ValueError("--config integration requires an adapter package identity")
        archive_package = metadata.get("adapter_module")
        if archive_package and archive_package != "unknown" and archive_package != deps.adapter_package:
            raise ValueError(f"archive adapter={archive_package!r} != this robot={deps.adapter_package!r}")
    if len(stations) < MIN_STATIONS:
        # Same REVIEW/candidate path as solve_and_publish. Live capture writes
        # archives with >=1 station, so the solver's hard lower bound would be
        # hit by an expected thin archive, not a real quality failure.
        logger.error(
            "archive has %d station(s) (need >= %d); not solving.",
            len(stations),
            MIN_STATIONS,
        )
        if has_config:
            return publish_with_reload(
                stations,
                intrinsics,
                None,
                None,
                mount=archive_mount,
                out_path=out_path,
                dependencies=deps,
            )
        # No solve ran, so there is no camera pose to write into a candidate
        # report. Report the REVIEW outcome without an artifact — same shape as
        # publish_with_reload's insufficient-station return.
        return RunOutcome(result=None, decision=None, artifact_path=None, candidate=True)
    result, decision = solve_stations(stations, intrinsics, mount=archive_mount, dependencies=deps)
    if has_config:
        return publish_with_reload(
            stations,
            intrinsics,
            result,
            decision,
            mount=archive_mount,
            out_path=out_path,
            dependencies=deps,
        )
    candidate_path = out_path.with_name(out_path.stem + ".candidate.json")
    save_candidate_report(
        candidate_path,
        _camera_pose(result),
        intrinsics,
        mount=archive_mount,
        observability=result.quality.observability,
        n_stations=len(stations),
        method=result.method,
        reasons=list(decision.reasons) or [f"offline re-solve (no --config), mount={archive_mount}"],
        solution_quality=result.quality.target_consistency,
        top_comment="--replay without --config: NOT a loadable calibration.",
    )
    logger.warning("replay (no --config): REVIEW/candidate -> %s", candidate_path)
    return RunOutcome(result=result, decision=decision, artifact_path=candidate_path, candidate=True)


def _camera_pose(result: EyeInHandResult | EyeToHandResult) -> np.ndarray:
    """Return the solved camera pose in the mount frame."""
    if isinstance(result, EyeToHandResult):
        return result.tf_base_cam
    return result.tf_flange_cam


__all__ = ["replay_calibration"]
