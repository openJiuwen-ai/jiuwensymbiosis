# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Framework-neutral calibration artifact publication.

Two publication paths:

* :func:`save_eye_to_hand_calibration` — schema-2 formal calibration with a
  top-level ``T_base_cam`` loadable by the runtime loader. Published only after
  the adapter reload smoke passes.
* :func:`save_candidate_report` — a REVIEW report whose solved matrix lives
  under ``candidate.T_base_cam`` or ``candidate.T_flange_cam`` (never at the
  top level), so the runtime loader cannot treat it as a calibration.

Runtime adapter reload validation belongs to the JiuwenSymbiosis integration
bridge, not this framework-neutral artifact module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from jiuwensymbiosis.calibration._mount import MOUNT_VALUES, Mount, canonical_mount
from jiuwensymbiosis.calibration.artifacts.archives import _atomic_write_json
from jiuwensymbiosis.calibration.domain.models import require_se3_mm
from jiuwensymbiosis.calibration.domain.quality import ObservabilityReport
from jiuwensymbiosis.calibration_schema import (
    CURRENT_SCHEMA_VERSION,
    EYE_TO_HAND_FRAME_FIELD,
    frame_field_for_mount,
)
from jiuwensymbiosis.utils.geometry import orthonormalize
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger("jiuwensymbiosis.calibration.artifacts")

# Candidate reports remain separate from the loadable schema-2 camera fields.
REPORT_KIND = "eye_to_hand_solve_report"
EYE_IN_HAND_REPORT_KIND = "eye_in_hand_solve_report"
REPORT_SCHEMA_VERSION = "1.0"


def _validate_mount(value: Any, *, source: str) -> Mount:
    """Validate artifact metadata without depending on workflow/integration errors."""
    try:
        return canonical_mount(value)
    except ValueError as exc:
        raise ValueError(f"{source}={value!r} is not a valid camera mount (expected one of {MOUNT_VALUES}).") from exc


def _validate_eye_to_hand_mount(value: Any, *, source: str) -> Mount:
    mount = _validate_mount(value, source=source)
    if mount != "eye_to_hand":
        raise ValueError(f"{source}={mount!r} is not valid for an eye-to-hand calibration artifact.")
    return mount


def save_eye_to_hand_calibration(
    path: str | Path,
    tf_base_cam: np.ndarray,
    intrinsics: np.ndarray,
    *,
    mount: str,
    object_xyz_base_mm: np.ndarray | None = None,
    t_flange_target: np.ndarray | None = None,
    method: str = "PARK",
    n_stations: int = 0,
    top_comment: str | None = None,
) -> Path:
    """Write a schema-2 calibration JSON loadable by the runtime loader.

    Top-level ``T_base_cam`` carries the solved matrix (object optional for
    eye-to-hand). ``T_flange_target`` (board-in-flange constant) is stored as a
    full SE(3) for downstream rigidity checks; it is optional and ignored by
    older loaders.
    """
    path = Path(path)
    mount = _validate_eye_to_hand_mount(mount, source="artifact mount")
    tf = require_se3_mm(tf_base_cam, field=EYE_TO_HAND_FRAME_FIELD).copy()
    tf[:3, :3] = orthonormalize(tf[:3, :3])
    payload: dict[str, Any] = {"schema_version": CURRENT_SCHEMA_VERSION}
    if top_comment:
        payload["_comment"] = top_comment
    payload[EYE_TO_HAND_FRAME_FIELD] = {
        "_frame": "camera pose in base frame; translation in mm",
        "matrix_4x4": np.round(tf, 6).tolist(),
        "mount": mount,
    }
    payload["intrinsics"] = np.round(np.asarray(intrinsics, dtype=np.float64), 6).tolist()
    if object_xyz_base_mm is not None:
        payload["object"] = {
            "xyz_base_mm": [round(float(v), 4) for v in np.asarray(object_xyz_base_mm).reshape(-1)[:3]],
        }
    if t_flange_target is not None:
        tt = require_se3_mm(t_flange_target, field="T_flange_target").copy()
        tt[:3, :3] = orthonormalize(tt[:3, :3])
        payload["T_flange_target"] = {
            "_frame": "board origin in flange frame (constant); translation in mm",
            "matrix_4x4": np.round(tt, 6).tolist(),
        }
    payload["method"] = method
    payload["n_stations"] = int(n_stations)
    _atomic_write_json(path, payload)
    logger.info("published eye-to-hand calibration -> %s", path)
    return path


def _candidate_field(mount: Mount) -> str:
    """The candidate-report camera-pose field for a mount (base or flange)."""
    return frame_field_for_mount(mount)


def _candidate_kind(mount: str) -> str:
    return REPORT_KIND if mount == "eye_to_hand" else EYE_IN_HAND_REPORT_KIND


def save_candidate_report(
    path: str | Path,
    tf_cam: np.ndarray,
    intrinsics: np.ndarray,
    *,
    mount: str,
    observability: ObservabilityReport,
    n_stations: int,
    method: str = "PARK",
    reasons: list[str] | None = None,
    solution_quality: Any | None = None,
    top_comment: str | None = None,
) -> Path:
    """Write a REVIEW/candidate report JSON.

    The solved matrix lives under ``candidate.T_base_cam`` (eye-to-hand) or
    ``candidate.T_flange_cam`` (eye-in-hand) — there is NO top-level camera-pose
    field, so the runtime calibration loader (which requires schema-2 + a
    top-level camera-pose field) cannot load this as a calibration.
    ``--out result.json`` writes this to ``result.candidate.json`` so the formal
    path is never overwritten. Both the observability (A/B excitation) and the
    post-solve solution-quality rigidity metrics are serialised so REVIEW has
    enough to localise the failure.
    """
    path = Path(path)
    mount = _validate_mount(mount, source="artifact mount")
    field = _candidate_field(mount)
    tf = require_se3_mm(tf_cam, field=f"candidate.{field}").copy()
    tf[:3, :3] = orthonormalize(tf[:3, :3])
    payload: dict[str, Any] = {
        "artifact_kind": _candidate_kind(mount),
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "REVIEW",
        "mount": mount,
        "method": method,
        "n_stations": int(n_stations),
    }
    if top_comment:
        payload["_comment"] = top_comment
    payload["candidate"] = {
        field: {
            "_frame": f"candidate camera pose in {'base' if mount == 'eye_to_hand' else 'flange'} frame; "
            "NOT a published calibration",
            "matrix_4x4": np.round(tf, 6).tolist(),
        },
        "intrinsics": np.round(np.asarray(intrinsics, dtype=np.float64), 6).tolist(),
    }
    payload["observability"] = {
        "n_valid_pairs": observability.n_valid_pairs,
        "n_relative_axes": observability.n_relative_axes,
        "max_relative_rotation_deg": round(observability.max_relative_rotation_deg, 4),
        "max_relative_translation_mm": round(observability.max_relative_translation_mm, 4),
        "max_axis_separation_deg": round(observability.max_axis_separation_deg, 4),
        "n_relative_axes_camera": observability.n_relative_axes_camera,
        "max_relative_rotation_camera_deg": round(observability.max_relative_rotation_camera_deg, 4),
        "max_relative_translation_camera_mm": round(observability.max_relative_translation_camera_mm, 4),
        "max_axis_separation_camera_deg": round(observability.max_axis_separation_camera_deg, 4),
        "n_duplicates": observability.n_duplicates,
        "svd_condition_min": (
            round(observability.svd_condition_min, 4) if math.isfinite(observability.svd_condition_min) else None
        ),
    }
    if solution_quality is not None:
        # solution_quality is a TargetConsistencyReport (flange_target invariant).
        payload["solution_quality"] = {
            "invariant_frame": getattr(solution_quality, "invariant_frame", "flange_target"),
            "translation_std_mm": round(solution_quality.translation_residual_mm.std, 4),
            "rotation_spread_deg": round(solution_quality.rotation_residual_deg.max, 4),
            "max_translation_residual_mm": round(solution_quality.translation_residual_mm.max, 4),
            "max_rotation_residual_deg": round(solution_quality.rotation_residual_deg.max, 4),
        }
    if reasons:
        payload["extra_reasons"] = reasons
    _atomic_write_json(path, payload)
    logger.info("wrote REVIEW/candidate report -> %s", path)
    return path


def is_candidate_report(path: str | Path) -> bool:
    """True if a JSON file is a REVIEW/candidate report (not a calibration)."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact = payload.get("artifact_kind") if isinstance(payload, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.debug("candidate report probe failed for %s: %s", path, exc)
        return False
    return bool(artifact in (REPORT_KIND, EYE_IN_HAND_REPORT_KIND))


__all__ = [
    "EYE_IN_HAND_REPORT_KIND",
    "REPORT_KIND",
    "REPORT_SCHEMA_VERSION",
    "is_candidate_report",
    "save_candidate_report",
    "save_eye_to_hand_calibration",
]
