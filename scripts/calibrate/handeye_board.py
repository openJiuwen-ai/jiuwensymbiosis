# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""标定板检测与生成（ChArUco / 棋盘格）+ 相机内参标定.

依赖 OpenCV（延迟加载，缺包时由 ``handeye_core._require_cv2`` 抛安装提示）。
``detect_board`` 永不因坏帧抛异常，而是返回 ``ok=False`` + 中文原因。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from handeye_core import ViewDetection, _require_cv2

from jiuwensymbiosis.utils.geometry import make_transform

logger = logging.getLogger("calibrate_hand_eye")


@dataclass(frozen=True)
class BoardSpec:
    """标定板规格（ChArUco 或棋盘格）。square_size 强制 mm，决定输出平移单位。"""

    kind: str  # "charuco" | "chessboard"
    squares_x: int
    squares_y: int
    square_size_mm: float
    marker_size_mm: float | None = None  # ChArUco 专用；None 时取 0.75*square
    aruco_dict: str = "DICT_4X4_50"

    def inner_corners(self) -> tuple[int, int]:
        """棋盘格内角点数 = (squares_x-1, squares_y-1)。"""
        return self.squares_x - 1, self.squares_y - 1

    def chessboard_object_points(self) -> np.ndarray:
        cols, rows = self.inner_corners()
        objp = np.zeros((cols * rows, 3), dtype=np.float64)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= float(self.square_size_mm)
        return objp


# =============================================================================
# 标定板生成（可打印图）
# =============================================================================
def _aruco_dictionary(cv2, name: str):
    const = getattr(cv2.aruco, name, None)
    if const is None:
        raise ValueError(f"未知 aruco 字典：{name}")
    return cv2.aruco.getPredefinedDictionary(const)


def _charuco_board(cv2, board: BoardSpec):
    d = _aruco_dictionary(cv2, board.aruco_dict)
    marker = float(board.marker_size_mm or board.square_size_mm * 0.75)
    # CharucoBoard((squaresX, squaresY), squareLength, markerLength, dictionary)；单位 mm
    return cv2.aruco.CharucoBoard((board.squares_x, board.squares_y), float(board.square_size_mm), marker, d)


def generate_board_image(board: BoardSpec, path: str | Path, *, dpi: int = 300, margin_px: int = 40) -> Path:
    """Generate a printable calibration-board PNG (ChArUco or chessboard)."""
    cv2 = _require_cv2()
    px_per_mm = dpi / 25.4
    if board.kind == "charuco":
        cb = _charuco_board(cv2, board)
        w = int(round(board.squares_x * board.square_size_mm * px_per_mm)) + 2 * margin_px
        h = int(round(board.squares_y * board.square_size_mm * px_per_mm)) + 2 * margin_px
        img = cb.generateImage((w, h), marginSize=margin_px)
    else:
        img = _chessboard_image(board, dpi=dpi, margin_px=margin_px)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def _chessboard_image(board: BoardSpec, *, dpi: int = 300, margin_px: int = 40) -> np.ndarray:
    px = int(round(board.square_size_mm * dpi / 25.4))
    cols, rows = board.squares_x, board.squares_y
    w = cols * px + 2 * margin_px
    h = rows * px + 2 * margin_px
    img = np.full((h, w), 255, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y0, x0 = margin_px + r * px, margin_px + c * px
                y1, x1 = y0 + px, x0 + px
                img[y0:y1, x0:x1] = 0
    return img


def _imread_rgb(path) -> np.ndarray:
    cv2 = _require_cv2()
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像（文件不存在或格式不支持）：{path}")
    return np.asarray(bgr[:, :, ::-1]).copy()


# =============================================================================
# 标定板检测 + 单帧 solvePnP
# =============================================================================
def detect_board(
    rgb: np.ndarray,
    board: BoardSpec,
    intrinsics: np.ndarray | None = None,
    dist: np.ndarray | None = None,
    *,
    min_corners: int = 6,
) -> ViewDetection:
    """Detect the board corners; if K is given, also solvePnP for T_cam_target.

    Never raises on a bad frame — returns ``ok=False`` with a Chinese reason.
    When ``K is None`` only corners are stored (pose filled later, after intrinsics).
    ``min_corners`` is the floor for a usable ChArUco view; it must be >= 6 because
    ``cv2.solvePnP``'s default (DLT/ITERATIVE) needs at least 6 point pairs.
    """
    cv2 = _require_cv2()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if board.kind == "charuco":
        det = _detect_charuco(cv2, gray, board, max(6, int(min_corners)))
    else:
        det = _detect_chessboard(cv2, gray, board)
    if det.ok and intrinsics is not None:
        _fill_pose(cv2, det, np.asarray(intrinsics, dtype=np.float64), dist)
    return det


def _detect_chessboard(cv2, gray, board: BoardSpec) -> ViewDetection:
    cols, rows = board.inner_corners()
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
    if not found:
        return ViewDetection(ok=False, reason="未检测到棋盘格（板不在视野/太斜/反光/行列数不符？）")
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return ViewDetection(ok=True, image_points=corners.reshape(-1, 2), object_points=board.chessboard_object_points())


def _detect_charuco(cv2, gray, board: BoardSpec, min_corners: int = 6) -> ViewDetection:
    cb = _charuco_board(cv2, board)
    detector = cv2.aruco.CharucoDetector(cb)
    ch_corners, ch_ids, _m_corners, _m_ids = detector.detectBoard(gray)
    n = 0 if ch_ids is None else len(ch_ids)
    if n < min_corners:
        return ViewDetection(
            ok=False,
            reason=f"ChArUco 角点过少（{n}<{min_corners}，板太斜/太远/反光/被夹爪遮挡？）",
        )
    objp, imgp = cb.matchImagePoints(ch_corners, ch_ids)
    if objp is None or len(objp) < min_corners:
        m = 0 if objp is None else len(objp)
        return ViewDetection(ok=False, reason=f"matchImagePoints 角点不足（{m}<{min_corners}）")
    return ViewDetection(ok=True, image_points=imgp.reshape(-1, 2), object_points=objp.reshape(-1, 3))


def _fill_pose(cv2, det: ViewDetection, intrinsics: np.ndarray, dist: np.ndarray | None) -> None:
    """对已检测到角点的 view，用 solvePnP 填 T_cam_target 与重投影误差。"""
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)
    if det.object_points is None or det.image_points is None:
        det.ok = False
        det.reason = "缺少角点（object_points / image_points 为空）"
        return
    objp = det.object_points.astype(np.float64)
    imgp = det.image_points.astype(np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, intrinsics, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    except Exception as exc:  # 坏帧（点太少/退化）降级为未采纳，绝不中断采集
        det.ok = False
        det.reason = f"solvePnP 异常（{exc.__class__.__name__}：角点太少或退化）"
        return
    if not ok:
        det.ok = False
        det.reason = "solvePnP 失败"
        return
    rot, _ = cv2.Rodrigues(rvec)
    det.tf_cam_target = make_transform(rot, tvec.reshape(3))
    proj, _ = cv2.projectPoints(objp, rvec, tvec, intrinsics, dist)
    det.reproj_rms_px = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp) ** 2, axis=1))))


def _fill_poses(detections: list[ViewDetection], intrinsics: np.ndarray, dist, board: BoardSpec) -> None:
    cv2 = _require_cv2()
    for d in detections:
        if d.ok and d.image_points is not None:
            _fill_pose(cv2, d, np.asarray(intrinsics, dtype=np.float64), dist)


def calibrate_intrinsics_from_views(
    detections: list[ViewDetection], image_size: tuple[int, int], board: BoardSpec
) -> tuple[np.ndarray, np.ndarray, float]:
    """cv2.calibrateCamera over accepted views -> (K 3x3, dist, rms_px)."""
    cv2 = _require_cv2()
    objpoints = [d.object_points.astype(np.float32) for d in detections if d.ok and d.object_points is not None]
    imgpoints = [d.image_points.astype(np.float32) for d in detections if d.ok and d.image_points is not None]
    if len(objpoints) < 3:
        raise ValueError("内参标定需要 ≥3 个有效视图")
    # 不带 CALIB_USE_INTRINSIC_GUESS 时 cameraMatrix/distCoeffs 会被忽略并从零重估，
    # 传占位空矩阵等价于传 None（opencv 存根把这两个入参误标为必填非空）。
    init_k = np.eye(3, dtype=np.float64)
    init_d = np.zeros((5, 1), dtype=np.float64)
    rms, intrinsics, dist, _r, _t = cv2.calibrateCamera(objpoints, imgpoints, image_size, init_k, init_d)
    return (
        np.asarray(intrinsics, dtype=np.float64),
        np.asarray(dist, dtype=np.float64).reshape(-1),
        float(rms),
    )
