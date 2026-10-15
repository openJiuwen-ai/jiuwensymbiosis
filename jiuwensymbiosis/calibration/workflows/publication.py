# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Solve, acceptance and artifact publication workflow services.

This module is the single mount-neutral publication path shared by the
eye-in-hand and eye-to-hand workflows.  ``solve_stations`` dispatches to the
frame-explicit solver + the matching acceptance policy for the mount;
``publish_with_reload`` writes the correct schema-2 artifact (``T_base_cam``
for eye-to-hand, ``T_flange_cam`` for eye-in-hand) after the reload smoke.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from jiuwensymbiosis.calibration.artifacts.artifacts import save_candidate_report, save_eye_to_hand_calibration
from jiuwensymbiosis.calibration.domain.models import (
    AcceptancePolicy,
    CalibrationDecision,
    EyeInHandResult,
    EyeToHandResult,
)
from jiuwensymbiosis.calibration.domain.quality import (
    EyeInHandAcceptancePolicy,
    EyeToHandAcceptancePolicy,
    t_flange_target_consistency,
)
from jiuwensymbiosis.calibration.domain.solver import (
    MIN_STATIONS,
    save_calibration,
    solve_eye_in_hand,
    solve_eye_to_hand,
)
from jiuwensymbiosis.calibration.workflows.preflight import validate_mount
from jiuwensymbiosis.calibration.workflows.workflows import RunOutcome, WorkflowDependencies
from jiuwensymbiosis.calibration_schema import EYE_IN_HAND_FRAME_FIELD
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger("jiuwensymbiosis.calibration.publication")


def acceptance_policy(mount: str, dependencies: WorkflowDependencies) -> AcceptancePolicy:
    """Build the mount-matching acceptance policy from workflow thresholds.

    Public so a caller that only *reports* the outcome (the GUI result page) can read
    the very thresholds this run was judged against, instead of keeping its own copy.
    """
    if mount == "eye_in_hand":
        return EyeInHandAcceptancePolicy()
    return EyeToHandAcceptancePolicy(
        observability_thresholds=(
            dependencies.observability_thresholds or EyeToHandAcceptancePolicy().observability_thresholds
        ),
        target_consistency_thresholds=(
            dependencies.target_consistency_thresholds or EyeToHandAcceptancePolicy().target_consistency_thresholds
        ),
    )


def solve_stations(
    stations: list,
    intrinsics,
    *,
    mount: str,
    dependencies: WorkflowDependencies,
) -> tuple[EyeInHandResult | EyeToHandResult, CalibrationDecision]:
    """Solve a station set under one mount and apply its acceptance policy.

    This is the single solve dispatch entry: ``mount`` selects the frame-explicit
    solver (``solve_eye_in_hand`` / ``solve_eye_to_hand``) and the matching
    acceptance policy.  Returns ``(result, decision)`` where ``result`` is an
    :class:`EyeInHandResult` or :class:`EyeToHandResult`.
    """
    mount = validate_mount(mount, source="solve mount")
    thresholds = dependencies.target_consistency_thresholds
    result: EyeInHandResult | EyeToHandResult
    if mount == "eye_in_hand":
        result = solve_eye_in_hand(
            stations,
            intrinsics,
            method=dependencies.options.method,
            cross_check=dependencies.options.cross_check,
            observability_thresholds=dependencies.observability_thresholds,
            target_consistency_thresholds=thresholds,
        )
    else:
        result = solve_eye_to_hand(
            stations,
            intrinsics,
            method=dependencies.options.method,
            cross_check=dependencies.options.cross_check,
            observability_thresholds=dependencies.observability_thresholds,
            target_consistency_thresholds=thresholds,
        )
    policy = acceptance_policy(mount, dependencies)
    _log_reprojection_advice(result.quality.reprojection, mount=mount)
    return result, policy.decide(result.quality)


# Advisory reprojection bands (px). See _log_reprojection_advice: never a gate.
_REPROJ_ADVICE_GOOD_PX = 1.0
_REPROJ_ADVICE_WARN_PX = 2.0


def _is_usable_per_view(values: Any) -> bool:
    """Whether per-view reprojection RMS is a non-empty finite 1-D vector."""
    if values is None or values.ndim != 1 or values.size == 0:
        return False
    return bool(np.all(np.isfinite(values)))


def _log_reprojection_advice(reprojection, *, mount: str) -> None:
    """Report reprojection magnitude and, when large, what to check.

    Diagnostic only: never affects acceptance. The eye-in-hand policy still
    gates on its own legacy thresholds; this adds visibility for eye-to-hand,
    where nothing else reports reprojection magnitude.
    """
    per_view = getattr(reprojection, "per_view_rms_px", None)
    try:
        values = np.asarray(per_view, dtype=np.float64) if per_view else None
    except (TypeError, ValueError):
        values = None
    if not _is_usable_per_view(values):
        logger.warning("reprojection: missing or non-finite; cannot assess detection quality.")
        return
    mean = float(np.mean(values))
    worst = float(np.max(values))
    if mean <= _REPROJ_ADVICE_GOOD_PX:
        logger.info("reprojection: mean=%.2fpx max=%.2fpx over %d views (good).", mean, worst, values.size)
        return
    level = logger.info if mean <= _REPROJ_ADVICE_WARN_PX else logger.warning
    level(
        "reprojection: mean=%.2fpx max=%.2fpx over %d views (>%.1fpx). "
        "Not a rejection — check board flatness and printed square/marker size, "
        "camera focus and exposure (motion blur), the intrinsics source, and drop "
        "views where the board is near the frame edge or steeply tilted.",
        mean,
        worst,
        values.size,
        _REPROJ_ADVICE_GOOD_PX,
    )
    if mount == "eye_to_hand":
        logger.info("eye-to-hand acceptance gates observability + rigidity only; this advice is not a gate.")


def solve_and_publish(
    stations: list,
    intrinsics,
    *,
    mount: str,
    out_path: Path,
    dependencies: WorkflowDependencies,
) -> RunOutcome:
    """Solve the captured stations and publish a formal or candidate artifact."""
    mount = validate_mount(mount, source="publication mount")
    if not stations:
        logger.error("no valid stations captured; nothing to solve.")
        return RunOutcome(result=None, decision=None, artifact_path=None, candidate=True)
    if len(stations) < MIN_STATIONS:
        # Keep the insufficient-station outcome on the REVIEW/candidate path.
        # Calling the numerical solver here would only turn an expected quality
        # failure into an uncaught ``ValueError`` from OpenCV's hard lower bound.
        logger.error(
            "only %d station(s) captured (need >= %d); nothing to solve.",
            len(stations),
            MIN_STATIONS,
        )
        return publish_with_reload(
            stations,
            intrinsics,
            None,
            None,
            mount=mount,
            out_path=out_path,
            dependencies=dependencies,
        )
    result, decision = solve_stations(stations, intrinsics, mount=mount, dependencies=dependencies)
    return publish_with_reload(
        stations,
        intrinsics,
        result,
        decision,
        mount=mount,
        out_path=out_path,
        dependencies=dependencies,
    )


def _solved_camera_pose(result: EyeInHandResult | EyeToHandResult | Any) -> np.ndarray:
    """Narrow the generic result to the mount frame for reload validation."""
    if hasattr(result, "tf_base_cam"):
        return np.asarray(result.tf_base_cam, dtype=np.float64)
    return np.asarray(result.tf_flange_cam, dtype=np.float64)


def publish_with_reload(
    stations: list,
    intrinsics,
    result,
    decision,
    *,
    mount: str,
    out_path: Path,
    dependencies: WorkflowDependencies,
) -> RunOutcome:
    """Validate an accepted artifact through the integration bridge before publication."""
    mount = validate_mount(mount, source="publication mount")
    if len(stations) < MIN_STATIONS:
        logger.error(
            "only %d station(s) (need >= %d); not solving. A station archive was still written.",
            len(stations),
            MIN_STATIONS,
        )
        return RunOutcome(result=result, decision=decision, artifact_path=None, candidate=True)
    if result is None or decision is None:
        logger.error("nothing to publish; no solve ran.")
        return RunOutcome(result=result, decision=decision, artifact_path=None, candidate=True)
    reasons = list(decision.reasons)
    reload_ok = False
    solved_pose = _solved_camera_pose(result)
    if decision.accept:
        # The validator is called with a private, unique path.  A deterministic
        # sibling such as ``result.reload.json`` can collide with another run,
        # and a cleanup in ``finally`` could then delete that other run's file.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{out_path.stem}.reload-",
            suffix=".json",
            dir=out_path.parent,
        )
        os.close(fd)
        temporary_json = Path(temporary_name)
        try:
            if dependencies.reload_validator is None:
                raise RuntimeError("framework integration did not provide an artifact reload validator")
            dependencies.reload_validator(
                temporary_json,
                solved_pose,
                intrinsics,
                mount,
                stations,
                t_flange_target=(
                    t_flange_target_consistency(stations, solved_pose) if mount == "eye_to_hand" else None
                ),
            )
            reload_ok = True
        except Exception as exc:
            logger.warning("artifact reload validation failed: %s", exc)
            reasons.append(f"adapter reload smoke test failed: {exc}")
        finally:
            temporary_json.unlink(missing_ok=True)
    if decision.accept and reload_ok:
        if mount == "eye_to_hand":
            save_eye_to_hand_calibration(
                out_path,
                solved_pose,
                intrinsics,
                mount=mount,
                t_flange_target=t_flange_target_consistency(stations, solved_pose),
                method=result.method,
                n_stations=len(stations),
                top_comment="Body-agnostic eye-to-hand calibration (board clamped in arm).",
            )
        else:
            save_calibration(
                out_path,
                solved_pose,
                intrinsics,
                None,
                frame_field=EYE_IN_HAND_FRAME_FIELD,
                top_comment="Body-agnostic eye-in-hand calibration (wrist camera).",
            )
        logger.info("ACCEPT + reload OK: published formal calibration -> %s", out_path)
        return RunOutcome(result=result, decision=decision, artifact_path=out_path, candidate=False)
    candidate_path = out_path.with_name(out_path.stem + ".candidate.json")
    save_candidate_report(
        candidate_path,
        solved_pose,
        intrinsics,
        mount=mount,
        observability=result.quality.observability,
        n_stations=len(stations),
        method=result.method,
        reasons=reasons,
        solution_quality=result.quality.target_consistency,
        top_comment="REVIEW/candidate report — NOT a loadable calibration.",
    )
    logger.warning("REVIEW: candidate report -> %s (reasons: %s)", candidate_path, reasons)
    return RunOutcome(result=result, decision=decision, artifact_path=candidate_path, candidate=True)


__all__ = ["acceptance_policy", "publish_with_reload", "solve_and_publish", "solve_stations"]
