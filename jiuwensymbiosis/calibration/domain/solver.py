# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Hand-eye solver: OpenCV ``calibrateHandEye`` behind a private numeric core.

Public entry points:

* :func:`solve_eye_in_hand` -> :class:`EyeInHandResult` (``tf_flange_cam``)
* :func:`solve_eye_to_hand` -> :class:`EyeToHandResult` (``tf_base_cam``)

Both share a private numeric core :func:`_solve_hand_eye_transform` that takes a
``mode`` literal selecting the OpenCV input-transform convention; ``mode`` lives
only in the private layer — a public result type can no longer carry two
coordinate-frame meanings behind one ``frame_field`` string. Callers must select
the frame-explicit entry directly; there is no boolean mode dispatcher.

The solver populates the unified :class:`CalibrationQualityReport` via
:func:`evaluate_calibration_quality` so neither mode discards the reprojection /
AX=XB / cross-check information the shared numeric core already computed — the
old eye-to-hand wrapper dropped exactly those fields.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from jiuwensymbiosis.calibration.domain.models import (
    CalibrationQualityReport,
    EyeInHandResult,
    EyeToHandResult,
    Station,
    require_se3_mm,
)
from jiuwensymbiosis.calibration.domain.quality import (
    ObservabilityThresholds,
    TargetConsistencyThresholds,
    evaluate_calibration_quality,
)
from jiuwensymbiosis.calibration_schema import CURRENT_SCHEMA_VERSION, EYE_IN_HAND_FRAME_FIELD
from jiuwensymbiosis.utils.geometry import (
    invert_transform,
    make_transform,
    orthonormalize,
    rotation_angle_deg,
)
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger("jiuwensymbiosis.calibration.solver")

MIN_STATIONS = 3  # calibrateHandEye hard lower bound (10~15 recommended)


def _require_cv2():
    """Lazy import cv2; raise with a precise install hint when missing."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            '手眼标定需要 OpenCV：请安装 `pip install -e ".[calib]"`（ChArUco 需 opencv-python>=4.8）。'
        ) from exc
    if not hasattr(cv2, "calibrateHandEye"):
        raise RuntimeError(
            "OpenCV 已安装但缺少 cv2.calibrateHandEye（OpenCV 5.x 移除了该顶层名）；"
            "请降级到 opencv-python>=4.8,<5.0：`pip install 'opencv-python>=4.8,<5.0'`。"
        )
    return cv2


def rotation_spread_deg(stations: list[Station]) -> float:
    """All-station flange rotation pairwise max angle (excitation diversity)."""
    rs = [s.tf_base_flange[:3, :3] for s in stations]
    if len(rs) < 2:
        return 0.0
    return max(rotation_angle_deg(rs[i], rs[j]) for i in range(len(rs)) for j in range(i + 1, len(rs)))


def _cv2_methods(cv2) -> dict[str, int]:
    return {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }


@dataclass(frozen=True)
class _RawHandEyeSolution:
    """Neutral numeric output of the private core: transform + diagnostics only.

    Holds the solved transform, per-method cross-check transforms and the
    per-view reprojection list. No accept/reasons — those are produced by the
    acceptance policy from the assembled quality report.
    """

    tf_handeye: np.ndarray
    method: str
    n_stations: int
    per_view_reproj_rms_px: list[float]
    cross_check_transforms: dict[str, np.ndarray] | None = None


def _calibrate_once(cv2, stations: list[Station], method_const: int, eye_to_hand: bool) -> np.ndarray:
    r_g2b, t_g2b, r_t2c, t_t2c = [], [], [], []
    for s in stations:
        c = s.detection.tf_cam_target  # solvePnP gives target-in-cam directly, no inverse
        if c is None:
            continue  # caller filtered; defensive narrowing, keep the four lists equal-length
        t = s.tf_base_flange
        if eye_to_hand:
            t = invert_transform(t)  # eye-to-hand: feed base->gripper, get T_base_cam
        r_g2b.append(t[:3, :3])
        t_g2b.append(t[:3, 3].reshape(3, 1))
        r_t2c.append(c[:3, :3])
        t_t2c.append(c[:3, 3].reshape(3, 1))
    r_x, t_x = cv2.calibrateHandEye(r_g2b, t_g2b, r_t2c, t_t2c, method=method_const)
    return make_transform(orthonormalize(np.asarray(r_x)), np.asarray(t_x).reshape(3))


def _solve_hand_eye_transform(
    stations: list[Station],
    *,
    mode: Literal["eye_in_hand", "eye_to_hand"],
    method: str = "PARK",
    cross_check: bool = False,
) -> _RawHandEyeSolution:
    """Private numeric core shared by both public solve entries.

    ``mode`` selects the OpenCV input-transform convention (eye-to-hand inverts
    the flange transform). Validates station transforms via :func:`require_se3_mm`
    before solving. Returns the neutral :class:`_RawHandEyeSolution`; the public
    entries bind it to a frame-explicit result with the unified quality report.
    """
    cv2 = _require_cv2()
    good = [s for s in stations if s.detection.ok and s.detection.tf_cam_target is not None]
    if len(good) < MIN_STATIONS:
        raise ValueError(f"有效视图 {len(good)} < {MIN_STATIONS}，无法标定（请多采不同姿态的视图）")
    # Validate the SE(3) inputs at the boundary.
    for i, s in enumerate(good):
        require_se3_mm(s.tf_base_flange, field=f"station[{i}].tf_base_flange")
        tf_cam_target = s.detection.tf_cam_target
        if tf_cam_target is None:  # defensive: ``good`` already filters this case
            raise ValueError(f"station[{i}].tf_cam_target is missing")
        require_se3_mm(tf_cam_target, field=f"station[{i}].tf_cam_target")

    methods = _cv2_methods(cv2)
    if method not in methods:
        raise ValueError(f"未知方法 {method}，可选：{list(methods)}")
    method_const = methods[method]
    eye_to_hand = mode == "eye_to_hand"
    tf_handeye = _calibrate_once(cv2, good, method_const, eye_to_hand)

    cc_transforms: dict[str, np.ndarray] | None = None
    if cross_check:
        cc_transforms = {}
        for name, const in methods.items():
            try:
                cc_transforms[name] = _calibrate_once(cv2, good, const, eye_to_hand)
            except Exception as exc:
                logger.warning("交叉校验方法 %s 失败：%s", name, exc)

    reproj_list = [s.detection.reproj_rms_px for s in good if s.detection.reproj_rms_px is not None]
    return _RawHandEyeSolution(
        tf_handeye=tf_handeye,
        method=method,
        n_stations=len(good),
        per_view_reproj_rms_px=[float(x) for x in reproj_list],
        cross_check_transforms=cc_transforms,
    )


def _assemble_quality(
    raw: _RawHandEyeSolution,
    stations: list[Station],
    intrinsics: np.ndarray,
    *,
    mode: Literal["eye_in_hand", "eye_to_hand"],
    observability_thresholds: ObservabilityThresholds | None = None,
    target_consistency_thresholds: TargetConsistencyThresholds | None = None,
) -> CalibrationQualityReport:
    """Bind the raw numeric solution to a unified quality report."""
    good = [s for s in stations if s.detection.ok and s.detection.tf_cam_target is not None]
    return evaluate_calibration_quality(
        good,
        raw.tf_handeye,
        mode=mode,
        per_view_reproj_rms_px=raw.per_view_reproj_rms_px,
        cross_check_transforms=raw.cross_check_transforms,
        observability_thresholds=observability_thresholds,
        target_consistency_thresholds=target_consistency_thresholds,
    )


def solve_eye_in_hand(
    stations: list[Station],
    intrinsics: np.ndarray,
    *,
    method: str = "PARK",
    cross_check: bool = False,
    observability_thresholds: ObservabilityThresholds | None = None,
    target_consistency_thresholds: TargetConsistencyThresholds | None = None,
) -> EyeInHandResult:
    """Solve eye-in-hand ``T_flange_cam`` (camera-in-flange, mm)."""
    raw = _solve_hand_eye_transform(stations, mode="eye_in_hand", method=method, cross_check=cross_check)
    quality = _assemble_quality(
        raw,
        stations,
        intrinsics,
        mode="eye_in_hand",
        observability_thresholds=observability_thresholds,
        target_consistency_thresholds=target_consistency_thresholds,
    )
    return EyeInHandResult(
        tf_flange_cam=raw.tf_handeye,
        method=raw.method,
        n_stations=raw.n_stations,
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        quality=quality,
    )


def solve_eye_to_hand(
    stations: list[Station],
    intrinsics: np.ndarray,
    *,
    method: str = "PARK",
    cross_check: bool = False,
    observability_thresholds: ObservabilityThresholds | None = None,
    target_consistency_thresholds: TargetConsistencyThresholds | None = None,
) -> EyeToHandResult:
    """Solve eye-to-hand ``T_base_cam`` (camera-in-base, mm)."""
    raw = _solve_hand_eye_transform(stations, mode="eye_to_hand", method=method, cross_check=cross_check)
    quality = _assemble_quality(
        raw,
        stations,
        intrinsics,
        mode="eye_to_hand",
        observability_thresholds=observability_thresholds,
        target_consistency_thresholds=target_consistency_thresholds,
    )
    return EyeToHandResult(
        tf_base_cam=raw.tf_handeye,
        method=raw.method,
        n_stations=raw.n_stations,
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        quality=quality,
    )


# =============================================================================
# save_calibration — schema-2 publication (eye-in-hand, T_flange_cam)
# =============================================================================
def save_calibration(
    path: str | Path,
    tf_flange_cam: np.ndarray,
    intrinsics: np.ndarray,
    object_xyz_base_mm: np.ndarray | Sequence[float] | None = None,
    *,
    frame_field: str = EYE_IN_HAND_FRAME_FIELD,
    frame_comment: str = "camera pose in flange frame; translation in mm",
    top_comment: str | None = None,
    intrinsics_comment: str | None = None,
    object_comment: str | None = None,
) -> Path:
    """Write a schema-2 calibration JSON loadable verbatim by ``load_calibration``.

    ``object_xyz_base_mm`` is an optional workspace anchor.  It remains
    positional for compatibility with the legacy eye-in-hand CLI, while new
    callers can omit it when no measured base-frame anchor exists.
    """
    tf_handeye = require_se3_mm(tf_flange_cam, field=frame_field).copy()
    tf_handeye[:3, :3] = orthonormalize(tf_handeye[:3, :3])  # defence in depth
    payload: dict[str, Any] = {"schema_version": CURRENT_SCHEMA_VERSION}
    if top_comment:
        payload["_comment"] = top_comment
    payload[frame_field] = {"_frame": frame_comment, "matrix_4x4": np.round(tf_handeye, 6).tolist()}
    if intrinsics_comment:
        payload["_intrinsics_comment"] = intrinsics_comment
    payload["intrinsics"] = np.round(np.asarray(intrinsics, dtype=np.float64), 6).tolist()
    # A missing workspace anchor must stay absent rather than becoming a fake measurement.
    if object_xyz_base_mm is not None:
        obj: dict[str, Any] = {}
        if object_comment:
            obj["_comment"] = object_comment
        obj["xyz_base_mm"] = [round(float(v), 4) for v in np.asarray(object_xyz_base_mm).reshape(-1)[:3]]
        payload["object"] = obj
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "MIN_STATIONS",
    "_RawHandEyeSolution",
    "_calibrate_once",
    "_cv2_methods",
    "_require_cv2",
    "_solve_hand_eye_transform",
    "rotation_spread_deg",
    "save_calibration",
    "solve_eye_in_hand",
    "solve_eye_to_hand",
]
