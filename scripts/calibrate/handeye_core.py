# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""手眼标定核心引擎（本体无关）.

只依赖 numpy / scipy + 延迟加载的 OpenCV，不涉及任何具体机器人本体：
  * 几何与位姿：``pose_to_tf_base_flange`` / ``orthonormalize`` / 旋转角度；
  * 数据结构：``ViewDetection`` / ``Station`` / ``HandEyeResult``；
  * 精度评估：``axxb_residuals`` / ``board_origin_in_base``（纯 numpy）；
  * 求解：``solve_hand_eye``（OpenCV ``calibrateHandEye``）；
  * 写出：``save_calibration``（与 ``piper_calib.json`` 逐键一致的 schema-2）。

标定板检测/生成在 ``handeye_board.py``；CLI / 采集 / 真机流程在 ``calibrate_hand_eye.py``。
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation  # scipy 是 core 依赖，可顶层 import

# 复用本体无关的纯数学原语（只读，不修改）
from jiuwensymbiosis.adapters._common.geometry import invert_transform, make_transform

logger = logging.getLogger("calibrate_hand_eye")

MIN_STATIONS = 3  # calibrateHandEye 的硬下限（推荐 10~15）


def _require_cv2():
    """延迟 import cv2；缺包时给出精确安装提示。"""
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise RuntimeError(
            '手眼标定需要 OpenCV：请安装 `pip install -e ".[calib]"`'
            "（ChArUco 需 opencv-python>=4.8）。"
        ) from exc


# =============================================================================
# 几何 / 位姿（本体无关，纯函数）
# =============================================================================
def rpy_deg_to_rot(rx_deg: float, ry_deg: float, rz_deg: float, axes: str = "xyz") -> np.ndarray:
    """RPY (degrees) -> 3x3 rotation, matching the runtime FlangePose convention."""
    return Rotation.from_euler(axes, [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()


def _first_attr(obj: Any, *names: str) -> Any:
    """返回第一个存在的属性值；都不存在则抛 AttributeError（明确错误）。"""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"pose 缺少属性 {names}")


def _opt_attr(obj: Any, *names: str) -> Optional[float]:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def pose_to_tf_base_flange(pose: Any, *, axes: str = "xyz") -> np.ndarray:
    """Vendor pose object -> 4x4 base<-flange SE(3) (mm/deg). Duck-typed, cross-body.

    Reads ``x/y/z`` (mm; ``*_mm`` variant preferred) and ``rx/ry/rz`` (deg). A
    4-DoF SCARA exposing only ``r`` is mapped to ``rz`` with ``rx=ry=0``. The
    default ``axes="xyz"`` reproduces piper ``FlangePose.to_tf_base_flange()``.
    """
    x = float(_first_attr(pose, "x_mm", "x"))
    y = float(_first_attr(pose, "y_mm", "y"))
    z = float(_first_attr(pose, "z_mm", "z"))
    rx = _opt_attr(pose, "rx_deg", "rx")
    ry = _opt_attr(pose, "ry_deg", "ry")
    rz = _opt_attr(pose, "rz_deg", "rz")
    if rx is None and ry is None:
        # 4-DoF SCARA：只有绕基座 Z 的 r
        r = _opt_attr(pose, "r_deg", "r")
        if r is None:
            raise ValueError("pose 缺少旋转字段（需要 rx/ry/rz 或 r）")
        rx, ry, rz = 0.0, 0.0, float(r)
    else:
        rx = float(rx) if rx is not None else 0.0
        ry = float(ry) if ry is not None else 0.0
        rz = float(rz) if rz is not None else 0.0
    return make_transform(
        rpy_deg_to_rot(rx, ry, rz, axes=axes), np.array([x, y, z], dtype=np.float64)
    )


def orthonormalize(rot: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3) via SVD (det=+1)."""
    u, _, vt = np.linalg.svd(np.asarray(rot, dtype=np.float64))
    rn = u @ vt
    if np.linalg.det(rn) < 0:
        u = u.copy()
        u[:, -1] *= -1.0
        rn = u @ vt
    return rn


def _rotation_angle_deg(ra: np.ndarray, rb: np.ndarray) -> float:
    """两旋转之间的测地角（度）。"""
    m = ra.T @ rb
    c = np.clip((np.trace(m) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def rotation_spread_deg(stations: list["Station"]) -> float:
    """所有 station 法兰旋转两两最大角差（衡量姿态多样性；手眼求解需要它够大）。"""
    rs = [s.tf_base_flange[:3, :3] for s in stations]
    if len(rs) < 2:
        return 0.0
    return max(
        _rotation_angle_deg(rs[i], rs[j]) for i in range(len(rs)) for j in range(i + 1, len(rs))
    )


# =============================================================================
# 数据结构
# =============================================================================
@dataclass(frozen=True)
class VerifyStat:
    """一组误差的均值/最大/标准差。"""

    mean: float
    max: float
    std: float


def _stat(vals: list[float]) -> VerifyStat:
    if not vals:
        return VerifyStat(0.0, 0.0, 0.0)
    a = np.asarray(vals, dtype=np.float64)
    return VerifyStat(float(a.mean()), float(a.max()), float(a.std()))


@dataclass
class ViewDetection:
    """一帧图像的标定板检测结果（不含机器人信息）。"""

    ok: bool
    image_points: Optional[np.ndarray] = None  # (N,2)
    object_points: Optional[np.ndarray] = None  # (N,3) mm
    tf_cam_target: Optional[np.ndarray] = None  # (4,4) board-in-camera, mm
    reproj_rms_px: Optional[float] = None
    reason: str = ""


@dataclass
class Station:
    """机器人位姿 + 视觉检测的唯一耦合点。"""

    tf_base_flange: np.ndarray  # (4,4) flange-in-base, mm
    detection: ViewDetection


@dataclass
class HandEyeResult:
    """标定结果 + 精度评估报告。"""

    tf_flange_cam: np.ndarray
    frame_field: str
    method: str
    n_stations: int
    intrinsics: np.ndarray
    per_view_reproj_rms_px: list[float]
    reproj: VerifyStat
    axxb_rot_deg: VerifyStat
    axxb_trans_mm: VerifyStat
    board_origin_base_mm: np.ndarray
    board_origin_spread_mm: VerifyStat
    rotation_spread_deg: float
    cross_check: Optional[dict[str, np.ndarray]] = None
    cross_check_max_deg: Optional[float] = None
    cross_check_max_mm: Optional[float] = None


# =============================================================================
# 精度评估（纯 numpy）
# =============================================================================
def axxb_residuals(
    stations: list[Station], tf_handeye: np.ndarray
) -> tuple[VerifyStat, VerifyStat]:
    """AX=XB consistency across station pairs -> (rot_deg stat, trans_mm stat)."""
    rot_errs: list[float] = []
    trans_errs: list[float] = []
    n = len(stations)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = stations[i].tf_base_flange, stations[j].tf_base_flange
            ci, cj = stations[i].detection.tf_cam_target, stations[j].detection.tf_cam_target
            a = invert_transform(ti) @ tj  # 法兰运动
            b = ci @ invert_transform(cj)  # 相机运动
            m = (a @ tf_handeye) @ invert_transform(tf_handeye @ b)  # 理想为单位阵
            c = np.clip((np.trace(m[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            rot_errs.append(float(np.degrees(np.arccos(c))))
            trans_errs.append(float(np.linalg.norm(m[:3, 3])))
    return _stat(rot_errs), _stat(trans_errs)


def board_origin_points(stations: list[Station], tf_handeye: np.ndarray) -> np.ndarray:
    """Per-station board origin in base frame, shape (N, 3) mm."""
    pts = []
    for s in stations:
        t_base_target = (s.tf_base_flange @ tf_handeye) @ s.detection.tf_cam_target
        pts.append(t_base_target[:3, 3])
    return np.asarray(pts)


def board_origin_in_base(
    stations: list[Station], tf_handeye: np.ndarray
) -> tuple[np.ndarray, VerifyStat]:
    """Recompute the board origin in base from every station -> (mean, spread)."""
    pts = board_origin_points(stations, tf_handeye)
    mean = pts.mean(axis=0)
    dists = np.linalg.norm(pts - mean, axis=1)
    return mean, _stat([float(d) for d in dists])


def outlier_mask(points: np.ndarray, *, k: float = 3.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Robust outlier flags via median + MAD (NOT mean/std — mean is itself skewed by outliers).

    Returns ``(mask, dists, median_dist)``:
      * ``dists``: each point's distance to the **median** center (mm);
      * ``median_dist``: median of ``dists`` — if this is already large, the error
        is *systematic* (all frames off), not a few outliers, so the caller should
        refuse to drop and suspect scale/axes instead;
      * ``mask``: True where ``dist > median_dist + k*MAD`` (a point that stands out
        from the group). When MAD≈0 (frames agree) a 1mm floor avoids false drops.
    """
    center = np.median(points, axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    med = float(np.median(dists))
    mad = float(np.median(np.abs(dists - med))) * 1.4826  # normal-consistent MAD
    thr = med + (k * mad if mad > 1e-9 else 1.0)
    return dists > thr, dists, med


# =============================================================================
# 手眼求解（OpenCV calibrateHandEye）
# =============================================================================
def _cv2_methods(cv2) -> dict[str, int]:
    return {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }


def _calibrate_once(
    cv2, stations: list[Station], method_const: int, eye_to_hand: bool
) -> np.ndarray:
    r_g2b, t_g2b, r_t2c, t_t2c = [], [], [], []
    for s in stations:
        t = s.tf_base_flange
        if eye_to_hand:
            t = invert_transform(t)  # eye-to-hand：喂 base->gripper，得 T_base_cam
        r_g2b.append(t[:3, :3])
        t_g2b.append(t[:3, 3].reshape(3, 1))
        c = s.detection.tf_cam_target  # solvePnP 直接给 target-in-cam，不取逆
        r_t2c.append(c[:3, :3])
        t_t2c.append(c[:3, 3].reshape(3, 1))
    r_x, t_x = cv2.calibrateHandEye(r_g2b, t_g2b, r_t2c, t_t2c, method=method_const)
    return make_transform(orthonormalize(np.asarray(r_x)), np.asarray(t_x).reshape(3))


def solve_hand_eye(
    stations: list[Station],
    intrinsics: np.ndarray,
    *,
    method: str = "PARK",
    eye_to_hand: bool = False,
    cross_check: bool = False,
) -> HandEyeResult:
    """Solve eye-in-hand T_flange_cam (or eye-to-hand T_base_cam) + fill the report."""
    cv2 = _require_cv2()
    good = [s for s in stations if s.detection.ok and s.detection.tf_cam_target is not None]
    if len(good) < MIN_STATIONS:
        raise ValueError(f"有效视图 {len(good)} < {MIN_STATIONS}，无法标定（请多采不同姿态的视图）")
    methods = _cv2_methods(cv2)
    if method not in methods:
        raise ValueError(f"未知方法 {method}，可选：{list(methods)}")

    method_const = methods.get(method)
    tf_handeye = _calibrate_once(cv2, good, method_const, eye_to_hand)
    frame_field = "T_base_cam" if eye_to_hand else "T_flange_cam"
    reproj_list = [s.detection.reproj_rms_px for s in good if s.detection.reproj_rms_px is not None]
    rot_stat, trans_stat = axxb_residuals(good, tf_handeye)
    origin_mean, origin_spread = board_origin_in_base(good, tf_handeye)

    cc = cc_deg = cc_mm = None
    if cross_check:
        cc = {}
        for name, const in methods.items():
            try:
                cc[name] = _calibrate_once(cv2, good, const, eye_to_hand)
            except Exception as exc:
                logger.warning("交叉校验方法 %s 失败：%s", name, exc)
        cc_deg = cc_mm = 0.0
        for xi, xj in itertools.combinations(cc.values(), 2):
            cc_deg = max(cc_deg, _rotation_angle_deg(xi[:3, :3], xj[:3, :3]))
            cc_mm = max(cc_mm, float(np.linalg.norm(xi[:3, 3] - xj[:3, 3])))

    return HandEyeResult(
        tf_flange_cam=tf_handeye,
        frame_field=frame_field,
        method=method,
        n_stations=len(good),
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        per_view_reproj_rms_px=[float(x) for x in reproj_list],
        reproj=_stat(reproj_list),
        axxb_rot_deg=rot_stat,
        axxb_trans_mm=trans_stat,
        board_origin_base_mm=origin_mean,
        board_origin_spread_mm=origin_spread,
        rotation_spread_deg=rotation_spread_deg(good),
        cross_check=cc,
        cross_check_max_deg=cc_deg,
        cross_check_max_mm=cc_mm,
    )


# =============================================================================
# 标定文件写出（与 piper_calib.json 逐键一致的 schema-2）
# =============================================================================
def save_calibration(
    path: str | Path,
    tf_flange_cam: np.ndarray,
    intrinsics: np.ndarray,
    object_xyz_base_mm,
    *,
    frame_field: str = "T_flange_cam",
    frame_comment: str = "camera pose in flange frame; translation in mm",
    top_comment: Optional[str] = None,
    intrinsics_comment: Optional[str] = None,
    object_comment: Optional[str] = None,
) -> Path:
    """Write a schema-2 calibration JSON loadable verbatim by ``load_calibration``."""
    tf_handeye = np.asarray(tf_flange_cam, dtype=np.float64).copy()
    tf_handeye[:3, :3] = orthonormalize(tf_handeye[:3, :3])  # 纵深防御：写出前再次正交化
    payload: dict[str, Any] = {"schema_version": 2}
    if top_comment:
        payload["_comment"] = top_comment
    payload[frame_field] = {"_frame": frame_comment, "matrix_4x4": np.round(tf_handeye, 6).tolist()}
    if intrinsics_comment:
        payload["_intrinsics_comment"] = intrinsics_comment
    payload["intrinsics"] = np.round(np.asarray(intrinsics, dtype=np.float64), 6).tolist()
    obj: dict[str, Any] = {}
    if object_comment:
        obj["_comment"] = object_comment
    obj["xyz_base_mm"] = [
        round(float(v), 4) for v in np.asarray(object_xyz_base_mm).reshape(-1)[:3]
    ]
    payload["object"] = obj
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
