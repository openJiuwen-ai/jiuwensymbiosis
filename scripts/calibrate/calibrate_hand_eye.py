#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""jiuwensymbiosis 手眼标定工具（eye-in-hand，腕部相机）—— CLI / 向导 / 采集 / 真机流程.

在真机上采集「机械臂法兰位姿 + 相机看到标定板」的多组数据，用 OpenCV
``calibrateHandEye`` 解出 ``T_flange_cam``（相机在法兰坐标系下的 4x4 位姿），
写出与 ``configs/piper/piper_calib.json`` 完全一致的 schema-2 标定文件，并给出
重投影 / AX=XB 残差 / 板原点聚集度等多维精度评估。

本目录三个文件：
  * ``handeye_core.py``  —— 本体无关引擎（几何/求解/残差/写文件）；
  * ``handeye_board.py`` —— 标定板检测/生成 + 内参标定（OpenCV）；
  * ``calibrate_hand_eye.py``（本文件）—— 向导式 CLI、采集、真机流程、离线自检。

跨本体（generalization）
------------------------
脚本只依赖各 adapter 已经满足的「固定命名接口」，不依赖任何本体具体类型：
  * builder 命名约定 ``build_<vendor>_session``（``.from_yaml`` / ``.from_dict``）
  * Env 中立动词 ``env.get_flange_pose()`` / ``env.move_to_flange()``
  * Driver 协议 ``env.low_level.grab_frames()`` / ``env.low_level.intrinsics``
唯一与本体相关的「厂商 pose → 4x4」转换由通用 ``pose_to_tf_base_flange`` 用鸭子
类型 + 欧拉轴序处理，默认 ``xyz`` 度恰好等于 piper 运行时 ``FlangePose`` 的约定。

依赖
----
需要 ``opencv-python>=4.8``（ChArUco）+（真机）``pyrealsense2``，独立于核心包：
  pip install -e ".[calib,piper]"
``--selftest`` 的纯 numpy 部分仅需 core 依赖即可运行。

用法::

    # 没有标定板？先生成可打印图（含打印须知）
    python scripts/calibrate/calibrate_hand_eye.py --generate-board board.png \
        --board charuco --squares-x 5 --squares-y 7 --square-size-mm 30 --marker-size-mm 22

    # 真机手动标定（向导式；回车采集，s 求解）
    python scripts/calibrate/calibrate_hand_eye.py --config scripts/calibrate/calibrate.yaml \
        --board charuco --squares-x 5 --squares-y 7 --square-size-mm 30 --marker-size-mm 22

    # 离线自检（无需硬件/相机）
    python scripts/calibrate/calibrate_hand_eye.py --selftest
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

# 直接运行 `python scripts/calibrate/calibrate_hand_eye.py` 时，脚本所在目录已位于
# sys.path[0]，故同目录的 handeye_core / handeye_board 可直接 import。
from handeye_core import (
    MIN_STATIONS,
    HandEyeResult,
    Station,
    VerifyStat,
    ViewDetection,
    _first_attr,
    _opt_attr,
    _require_cv2,
    _rotation_angle_deg,
    axxb_residuals,
    board_origin_in_base,
    board_origin_points,
    invert_transform,
    make_transform,
    orthonormalize,
    outlier_mask,
    pose_to_tf_base_flange,
    rotation_spread_deg,
    rpy_deg_to_rot,
    save_calibration,
    solve_hand_eye,
)
from handeye_board import (
    BoardSpec,
    _fill_poses,
    _imread_rgb,
    calibrate_intrinsics_from_views,
    detect_board,
    generate_board_image,
)

from jiuwensymbiosis.utils.proxy import clear_proxy_env

logger = logging.getLogger("calibrate_hand_eye")


# =============================================================================
# 终端 UX 辅助
# =============================================================================
def _hr(char: str = "─", n: int = 64) -> str:
    return char * n


def _banner(lines: list[str]) -> None:
    logger.info(_hr("="))
    for ln in lines:
        logger.info(ln)
    logger.info(_hr("="))


def _ask_yes_no(prompt: str, *, default: bool = True, assume_yes: bool = False) -> bool:
    if assume_yes:
        return True
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        ans = input(prompt + suffix).strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans in ("y", "yes", "是", "好", "ok")


def _print_board_print_notice(board: BoardSpec) -> None:
    logger.info(_hr())
    logger.info("📄 打印须知（请逐条照做，否则标定会系统性偏差）：")
    logger.info("  1. 用 A4 纸 100% 原始比例打印，务必关闭“适应页面/缩放打印”。")
    logger.info("  2. 打印后用尺子量【一个方格】的实际边长，把真实毫米数传给 --square-size-mm。")
    logger.info("     —— 打印缩放是手眼标定最常见的尺度错误来源！")
    logger.info("  3. 平整裱在硬质平板（KT 板/亚克力/铝板）上，不可弯曲、起翘或反光。")
    logger.info("  4. 标定时让整块板尽量充满视野，并在不同距离/角度多次摆放。")
    logger.info("  在线生成器备选：https://calib.io/pages/camera-calibration-pattern-generator")
    logger.info(_hr())


def _judge(value: float, good_thr: float, warn_thr: float) -> str:
    """越小越好的指标 -> ✅/⚠️/❌。"""
    if value <= good_thr:
        return "✅"
    if value <= warn_thr:
        return "⚠️"
    return "❌"


def _maybe_save_debug(args, rgb, det: ViewDetection, idx: int) -> None:
    if not args.debug_dir:
        return
    try:
        cv2 = _require_cv2()
        d = Path(args.debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if det.ok and det.image_points is not None:
            for p in det.image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
        cv2.imwrite(str(d / f"view_{idx:03d}.png"), img)
    except Exception as exc:
        logger.debug("保存调试图失败：%s", exc)


def _maybe_show(args, rgb, det: Optional[ViewDetection]) -> None:
    if not args.show:
        return
    try:
        cv2 = _require_cv2()
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if det is not None and det.ok and det.image_points is not None:
            for p in det.image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
        cv2.imshow("calibrate_hand_eye", img)
        cv2.waitKey(1)
    except Exception as exc:
        logger.debug("实时预览失败（无显示器？）：%s", exc)


# =============================================================================
# adapter 解析（沿用 smoke_test_adapter 的固定命名约定）
# =============================================================================
def _resolve_module(module_str: Optional[str], path_str: Optional[str]) -> str:
    if module_str:
        return module_str
    if path_str:
        p = Path(path_str).resolve()
        parts = list(p.parts)
        try:
            idx = parts.index("jiuwensymbiosis")
        except ValueError:
            idx = len(parts) - 1
            for i, part in enumerate(parts):
                if part == "adapters":
                    idx = i - 1 if i > 0 else len(parts) - 1
                    break
        return ".".join(parts[idx:]).replace(".py", "")
    return ""


def _load_builder(module_str: str):
    """import 适配器包并返回其 ``build_*_session`` 可调用对象。"""
    import importlib

    module = importlib.import_module(module_str)
    candidate = getattr(module, "build_session", None)
    if callable(candidate):
        return candidate
    for attr_name in dir(module):
        if attr_name.startswith("build_") and attr_name.endswith("_session"):
            candidate = getattr(module, attr_name)
            if callable(candidate):
                return candidate
    raise AttributeError(f"{module_str} 中找不到 build_*_session 构造器")


def _board_from_args(args) -> BoardSpec:
    return BoardSpec(
        kind=args.board,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_size_mm=args.square_size_mm,
        marker_size_mm=args.marker_size_mm,
        aruco_dict=args.aruco_dict,
    )


def _env_self_check(env) -> tuple[bool, list[str]]:
    """环境自检：相机内参 / 取帧 / 位姿。返回 (是否全 OK, 问题列表)。"""
    issues: list[str] = []
    ll = getattr(env, "low_level", None)
    if ll is None:
        return False, ["env.low_level 为空——env 未连接（connect 失败？）"]
    try:
        env.get_flange_pose()
    except Exception as exc:
        issues.append(f"读取法兰位姿失败：{exc}")
    try:
        frames = ll.grab_frames()
        if frames is None:
            issues.append("相机取帧返回 None——相机未启动或 YAML 未配置 camera_serial？")
    except Exception as exc:
        issues.append(f"相机取帧异常：{exc}")
    if getattr(ll, "intrinsics", None) is None:
        issues.append(
            "相机内参不可用（intrinsics=None）——可用 --intrinsics 手动指定或 --calibrate-intrinsics"
        )
    return (len(issues) == 0), issues


def _resolve_intrinsics_pre(args, env) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """采集前确定内参来源。返回 (K 或 None, dist 或 None)。"""
    if args.intrinsics:
        fx, fy, ppx, ppy = args.intrinsics
        intrinsics = np.array([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]], dtype=np.float64)
        logger.info("内参来源：--intrinsics 手动指定。")
        return intrinsics, None
    if args.calibrate_intrinsics:
        logger.info("内参来源：采集后用视图标定（--calibrate-intrinsics）。")
        return None, None
    ll = getattr(env, "low_level", None)
    intrinsics = getattr(ll, "intrinsics", None) if ll is not None else None
    if intrinsics is None:
        logger.warning(
            "⚠️ 相机内参不可用，将自动改用采集后视图标定（等价 --calibrate-intrinsics）。"
        )
        return None, None
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    logger.info(
        "内参来源：相机（RealSense 出厂） fx=%.1f fy=%.1f ppx=%.1f ppy=%.1f",
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )
    return intrinsics, None


def _print_config_card(args, board: BoardSpec, intrinsics: Optional[np.ndarray]) -> None:
    logger.info("")
    logger.info(_hr())
    logger.info("📋 本次标定配置确认：")
    logger.info("   采集模式 : %s", "自动移动(--auto)" if args.auto else "手动交互")
    logger.info(
        "   手眼几何 : %s", "eye-to-hand" if args.eye_to_hand else "eye-in-hand（相机在腕部）"
    )
    marker = f"/marker {board.marker_size_mm}mm" if board.kind == "charuco" else ""
    logger.info(
        "   标定板   : %s %dx%d，方格 %.1fmm%s（务必=实测打印尺寸）",
        board.kind,
        board.squares_x,
        board.squares_y,
        board.square_size_mm,
        marker,
    )
    logger.info("   内参来源 : %s", "待视图标定" if intrinsics is None else "已就绪")
    logger.info(
        "   求解算法 : %s%s", args.method, "（+多算法交叉校验）" if args.cross_check else ""
    )
    logger.info("   欧拉轴序 : %s（须与本体运行时一致）", args.euler_axes)
    logger.info("   输出文件 : %s", args.out)
    logger.info(_hr())


# =============================================================================
# 采集
# =============================================================================
def _collect_manual(
    env, board: BoardSpec, intrinsics, dist, args
) -> tuple[list[Station], Optional[tuple[int, int]]]:
    stations: list[Station] = []
    image_size: Optional[tuple[int, int]] = None
    logger.info("")
    _banner(
        [
            "阶段 1 · 手动采集",
            "把机械臂摆到不同姿态（多倾斜手腕、绕 Z 旋转，别只平移），让相机看清整块板。",
            "回车=采集一帧 | s=求解 | u=撤销上一帧 | q=放弃 | ?=帮助",
        ]
    )
    while True:
        ok_stations = [s for s in stations if s.detection.ok]
        n_ok = len(ok_stations)
        spread = rotation_spread_deg(ok_stations)
        status = (
            f"[手动 | 已采 {n_ok}/推荐≥10 | 旋转跨度 {spread:.0f}° | 输出 {Path(args.out).name}]"
        )
        try:
            cmd = input(f"{status} > ").strip().lower()
        except EOFError:
            cmd = "q"
        if cmd == "q":
            logger.info("已放弃采集。")
            return stations, image_size
        if cmd == "?":
            logger.info("回车=采集当前帧；s=用已采视图求解；u=删除最近一帧；q=放弃退出。")
            continue
        if cmd == "u":
            if stations:
                stations.pop()
                logger.info("已撤销最近一帧。")
            else:
                logger.info("没有可撤销的帧。")
            continue
        if cmd == "s":
            if n_ok >= MIN_STATIONS:
                return stations, image_size
            logger.info("有效视图不足 %d，请继续采集。", MIN_STATIONS)
            continue
        # 默认：采集一帧
        try:
            pose = env.get_flange_pose()
            frames = env.low_level.grab_frames()
        except Exception as exc:
            logger.error("✗ 读取位姿/相机失败：%s", exc)
            continue
        if frames is None:
            logger.error("✗ 相机无帧，请检查相机连接。")
            continue
        rgb, _depth = frames
        image_size = (rgb.shape[1], rgb.shape[0])
        try:
            det = detect_board(rgb, board, intrinsics, dist, min_corners=args.min_corners)
        except Exception as exc:  # 单帧检测异常不丢已采数据
            logger.info("✗ 未采纳：检测异常（%s）", exc)
            continue
        _maybe_show(args, rgb, det)
        if not det.ok:
            logger.info("✗ 未采纳：%s", det.reason)
            _maybe_save_debug(args, rgb, det, n_ok)
            continue
        if (
            det.reproj_rms_px is not None
            and det.reproj_rms_px > args.max_reproj_px
            and not args.lax
        ):
            logger.info(
                "✗ 未采纳：重投影 %.2fpx > 阈值 %.2fpx（板太斜/模糊/反光？可加 --lax 放宽）",
                det.reproj_rms_px,
                args.max_reproj_px,
            )
            continue
        stations.append(Station(pose_to_tf_base_flange(pose, axes=args.euler_axes), det))
        ok_stations = [s for s in stations if s.detection.ok]
        n_ok2 = len(ok_stations)
        spread2 = rotation_spread_deg(ok_stations)
        msg = f"✓ 第 {n_ok2} 帧已采纳"
        if det.reproj_rms_px is not None:
            msg += f"，重投影 {det.reproj_rms_px:.2f}px"
        logger.info(msg)
        if n_ok2 >= 2 and spread2 < 30:
            logger.info(
                "  提示：旋转跨度仅 %.0f°，请多倾斜手腕/绕 Z 旋转以增加姿态多样性。", spread2
            )
        if n_ok2 < 10:
            logger.info("  建议继续采集至 ≥10 帧（当前 %d）。", n_ok2)
        _maybe_save_debug(args, rgb, det, n_ok2)


def _perturb_target(base_pose, drx, dry, drz, dz):
    """以读到的位姿为锚构造一个扰动后的法兰目标（piper 专属：构造 FlangePose）。"""
    from jiuwensymbiosis.adapters.piper.geometry import FlangePose

    bx = float(_first_attr(base_pose, "x_mm", "x"))
    by = float(_first_attr(base_pose, "y_mm", "y"))
    bz = float(_first_attr(base_pose, "z_mm", "z"))
    brx = float(_opt_attr(base_pose, "rx_deg", "rx") or 0.0)
    bry = float(_opt_attr(base_pose, "ry_deg", "ry") or 0.0)
    brz = float(_opt_attr(base_pose, "rz_deg", "rz") or 0.0)
    return FlangePose(bx, by, bz + dz, brx + drx, bry + dry, brz + drz)


def _collect_auto(
    env, board: BoardSpec, intrinsics, dist, args
) -> tuple[list[Station], Optional[tuple[int, int]]]:
    stations: list[Station] = []
    image_size: Optional[tuple[int, int]] = None
    base = env.get_flange_pose()
    tilts = sorted({-args.auto_tilt_deg, 0.0, args.auto_tilt_deg})
    yaws = sorted({-args.auto_yaw_deg, 0.0, args.auto_yaw_deg})
    dzs = sorted({-args.auto_dz_mm, 0.0, args.auto_dz_mm})
    targets = list(itertools.product(tilts, tilts, yaws, dzs))
    _banner([f"阶段 1 · 自动采集（共 {len(targets)} 个扰动位姿）", "随时按硬件 E-stop 可急停。"])
    if args.auto_dry_run:
        for drx, dry, drz, dz in targets:
            logger.info("  目标扰动 rx%+.0f ry%+.0f rz%+.0f z%+.0f", drx, dry, drz, dz)
        logger.info("（--auto-dry-run：仅打印目标，不移动机械臂）")
        return stations, image_size
    for drx, dry, drz, dz in targets:
        try:
            env.move_to_flange(_perturb_target(base, drx, dry, drz, dz))
        except Exception as exc:
            logger.warning("移动失败（可能不可达），跳过：%s", exc)
            continue
        frames = env.low_level.grab_frames()
        if frames is None:
            continue
        rgb, _depth = frames
        image_size = (rgb.shape[1], rgb.shape[0])
        try:
            det = detect_board(rgb, board, intrinsics, dist, min_corners=args.min_corners)
        except Exception as exc:  # 单帧检测异常不丢已采数据
            logger.info("✗ 未采纳：检测异常（%s）", exc)
            continue
        _maybe_show(args, rgb, det)
        pose = env.get_flange_pose()
        accept = det.ok and (
            det.reproj_rms_px is None or det.reproj_rms_px <= args.max_reproj_px or args.lax
        )
        if accept:
            stations.append(Station(pose_to_tf_base_flange(pose, axes=args.euler_axes), det))
            logger.info("✓ 采纳（累计 %d）", len([s for s in stations if s.detection.ok]))
        else:
            why = det.reason if not det.ok else f"重投影 {det.reproj_rms_px:.2f}px 超阈值"
            logger.info("✗ 跳过：%s", why)
    return stations, image_size


# =============================================================================
# object 锚点 / 报告 / 复验
# =============================================================================
def _resolve_object_xyz(args, res: HandEyeResult) -> list[float]:
    if args.object_xyz is not None:
        logger.info("object.xyz_base_mm 使用 --object-xyz：%s", list(args.object_xyz))
        return [float(v) for v in args.object_xyz]
    if args.base:
        from jiuwensymbiosis.adapters._common.calibration import load_calibration

        loaded = load_calibration(
            args.base,
            frame_field=res.frame_field,
            legacy_field="T_TCP_cam",
            env_var="JIUWEN_PIPER_ALLOW_LEGACY_CALIB",
        )
        xyz = np.asarray(loaded["object"]["xyz_base_mm"]).reshape(-1)[:3]
        logger.info("object.xyz_base_mm 沿用 --base 文件：%s", np.round(xyz, 2).tolist())
        return [float(v) for v in xyz]
    xyz = np.asarray(res.board_origin_base_mm).reshape(-1)[:3]
    logger.warning(
        "⚠️ object.xyz_base_mm 取自解算的标定板原点 %s。它决定 z_min_safe 与 fallback home，"
        "请确认这就是工作面高度锚点；若标定板不在工作面上，请改用 --object-xyz 指定。",
        np.round(xyz, 2).tolist(),
    )
    return [float(v) for v in xyz]


def _report_dict(res: HandEyeResult) -> dict[str, Any]:
    def _s(st: VerifyStat) -> dict[str, float]:
        return {"mean": st.mean, "max": st.max, "std": st.std}

    return {
        "frame_field": res.frame_field,
        "method": res.method,
        "n_stations": res.n_stations,
        "T_flange_cam": np.round(res.tf_flange_cam, 6).tolist(),
        "intrinsics": np.round(res.intrinsics, 6).tolist(),
        "rotation_spread_deg": res.rotation_spread_deg,
        "reproj_rms_px": _s(res.reproj),
        "per_view_reproj_rms_px": res.per_view_reproj_rms_px,
        "axxb_rot_deg": _s(res.axxb_rot_deg),
        "axxb_trans_mm": _s(res.axxb_trans_mm),
        "board_origin_base_mm": np.round(res.board_origin_base_mm, 3).tolist(),
        "board_origin_spread_mm": _s(res.board_origin_spread_mm),
        "cross_check_max_deg": res.cross_check_max_deg,
        "cross_check_max_mm": res.cross_check_max_mm,
    }


def _print_report(res: HandEyeResult, args, *, title: str = "") -> None:
    logger.info("")
    _banner(["阶段 2 · 标定结果与精度评估" + (f"  {title}" if title else "")])
    logger.info(
        "方法=%s | 有效视图=%d | 旋转跨度=%.0f°",
        res.method,
        res.n_stations,
        res.rotation_spread_deg,
    )
    j = _judge(res.reproj.mean, 1.0, 2.0)
    logger.info(
        "① 重投影 RMS    ：均值 %.2fpx，最大 %.2fpx  %s  （板检测质量；>2px 多为板太斜/模糊/尺寸不符）",
        res.reproj.mean,
        res.reproj.max,
        j,
    )
    jr = _judge(res.axxb_rot_deg.mean, 0.5, 1.0)
    jt = _judge(res.axxb_trans_mm.mean, 2.0, 4.0)
    logger.info(
        "② 手眼一致性    ：旋转 均值%.3f°/最大%.3f° %s ；平移 均值%.2f/最大%.2fmm %s",
        res.axxb_rot_deg.mean,
        res.axxb_rot_deg.max,
        jr,
        res.axxb_trans_mm.mean,
        res.axxb_trans_mm.max,
        jt,
    )
    logger.info("   （AX=XB 残差；旋转应≲0.5°、平移应≲2~3mm；偏大=某帧异常或旋转多样性不足）")
    jo = _judge(res.board_origin_spread_mm.std, 2.0, 4.0)
    logger.info(
        "③ 板原点一致性  ：std %.2fmm，最大 %.2fmm  %s  （臂与相机对同一点的一致性；>3mm 建议复采）",
        res.board_origin_spread_mm.std,
        res.board_origin_spread_mm.max,
        jo,
    )
    if res.cross_check is not None:
        logger.info(
            "④ 多算法交叉校验：最大旋转分歧 %.3f°，最大平移分歧 %.2fmm（越小越说明数据健康）",
            res.cross_check_max_deg,
            res.cross_check_max_mm,
        )
    logger.info(
        "T_flange_cam =\n%s", np.array2string(np.round(res.tf_flange_cam, 4), suppress_small=True)
    )
    if args.report:
        rp = Path(str(args.out) + ".report.json")
        rp.write_text(json.dumps(_report_dict(res), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("详细报告已存：%s", rp)


def _maybe_drop_outliers(res: HandEyeResult, stations, intrinsics, args) -> HandEyeResult:
    """剔除个别离群帧后重解；用三道闸门防止掩盖系统性偏差（尺度/轴序/板不平）。

    返回最终采用的结果（剔除成功则为重解结果，否则原结果）。
    """
    if not args.drop_outliers:
        return res
    good = [s for s in stations if s.detection.ok and s.detection.tf_cam_target is not None]
    if len(good) < 6:
        return res  # 帧太少，剔除不稳，跳过

    pts = board_origin_points(good, res.tf_flange_cam)
    mask, dists, med = outlier_mask(pts, k=args.outlier_k)
    logger.info("")
    logger.info(
        "离群检测：各帧板原点偏差 中位 %.2fmm / 最大 %.2fmm（k=%.1f）",
        med,
        float(dists.max()),
        args.outlier_k,
    )

    # 闸门①：偏差中位数就大 → 各帧【普遍】偏，而非个别离群 → 系统性问题，拒绝剔除
    if med > 3.0:
        logger.warning(
            "⚠️ 偏差中位数已达 %.2fmm（各帧普遍偏大，不是个别离群）——通常是系统性问题："
            "首查 --square-size-mm 是否量准，其次板平整度/欧拉轴序。离群剔除救不了，已跳过。",
            med,
        )
        return res
    n_out = int(mask.sum())
    if n_out == 0:
        logger.info("未发现离群帧，无需剔除。")
        return res
    # 闸门②：剔除占比过高 → 数据整体差，疑似系统问题 → 拒绝剔除（避免粉饰）
    max_drop = max(1, int(len(good) * 0.3))
    if n_out > max_drop:
        logger.warning(
            "⚠️ 识别出 %d 帧离群，超过上限 %d（>30%%）——更像数据整体偏差而非个别离群；"
            "为避免掩盖系统问题，已跳过剔除。建议重采并核对 --square-size-mm。",
            n_out,
            max_drop,
        )
        return res

    keep = [s for s, m in zip(good, mask) if not m]
    dropped = [(i, round(float(dists[i]), 1)) for i in range(len(good)) if mask[i]]
    logger.info(
        "剔除 %d 个离群帧后用剩余 %d 帧重解；剔除帧 (索引,偏差mm)=%s", n_out, len(keep), dropped
    )
    try:
        res2 = solve_hand_eye(
            keep,
            intrinsics,
            method=args.method,
            eye_to_hand=args.eye_to_hand,
            cross_check=args.cross_check,
        )
    except Exception as exc:
        logger.warning("剔除后重解失败（%s），保留全量结果。", exc)
        return res
    _print_report(res2, args, title="（剔除离群后 %d 帧）" % res2.n_stations)

    # 闸门③：剔除后仍不达标 → 剩余是系统性误差，别被"剔除"粉饰
    if res2.axxb_trans_mm.mean > 3.0 or res2.axxb_rot_deg.mean > 0.5:
        logger.warning(
            "⚠️ 剔除离群后残差仍未达标——剩余误差更可能是【系统性】的：请优先把 --square-size-mm "
            "用多格平均量准，再查板平整度、欧拉轴序、以及采集时每帧是否都停稳。"
        )
    return res2


def _print_next_steps(args, res: HandEyeResult) -> None:
    logger.info("")
    logger.info("👉 下一步：")
    logger.info("   - 真机反投影复验：重跑本脚本加 --verify（或 --verify-touch 靠近悬停目视）。")
    logger.info("   - 跑演示验证：piper-pick-demo --config configs/piper/pick_box.yaml")
    logger.info("   - 标定正确后，可逐步把 pick_box.yaml 的 z_correction_mm 调向 0。")


def _verify_live(env, res: HandEyeResult, intrinsics: np.ndarray, args) -> None:
    logger.info("")
    _banner(["阶段 3 · 真机反投影复验（与 demo 完全相同的反投影路径）"])
    try:
        from jiuwensymbiosis.adapters.piper.geometry import FlangePose, pixel_and_depth_to_base_xyz
    except Exception as exc:
        logger.warning("无法导入 piper 反投影（--verify 目前仅支持 piper）：%s", exc)
        return
    frames = env.low_level.grab_frames()
    if frames is None:
        logger.error("无相机帧，无法复验。")
        return
    rgb, depth_m = frames
    det = detect_board(rgb, _board_from_args(args), intrinsics, None, min_corners=args.min_corners)
    if not det.ok or det.image_points is None:
        logger.error("复验时未检测到标定板：%s", det.reason)
        return
    uv = det.image_points.mean(axis=0)
    u, v = int(round(uv[0])), int(round(uv[1]))
    d = float(depth_m[v, u])  # 注意：行=v、列=u
    if not np.isfinite(d) or d <= 0:
        logger.error("板中心像素 (%d,%d) 深度无效（%.3f）。", u, v, d)
        return
    pose = env.get_flange_pose()
    fp = FlangePose(
        float(pose.x), float(pose.y), float(pose.z), float(pose.rx), float(pose.ry), float(pose.rz)
    )
    p_base = pixel_and_depth_to_base_xyz((u, v), d, fp, res.tf_flange_cam, intrinsics)
    logger.info(
        "板中心像素(%d,%d) 深度%.3fm → 基座坐标 %s mm", u, v, d, np.round(p_base, 1).tolist()
    )
    if not args.verify_touch:
        logger.info("（仅反投影；如需机械臂靠近板面目视复验请加 --verify-touch）")
        return

    # verify-touch：让【指尖】悬停在板面上方 hover（默认不接触）。
    # 工具沿 base -Z 伸出 tool_offset，故法兰目标 z = 板面 z + tool_offset + hover。
    tool = args.verify_tool_offset_mm
    if tool is None:
        tool = float(getattr(env, "tool_offset_mm", 0.0) or 0.0)
    hover = float(args.verify_hover_mm)

    if tool <= 0.0:
        for ln in (
            _hr(),
            "⚠️ 强警告：tool_offset=0（未提供工具长度）。",
            "   verify-touch 会把【法兰】当指尖来定位、按法兰计算悬停高度。",
            "   若末端实际装了工具/夹爪（在法兰下方若干 mm），其尖端会比预期更低，",
            "   可能撞向甚至穿过标定板！",
            "   安全做法：用 --verify-tool-offset-mm <真实法兰→指尖 mm> 再跑 verify-touch。",
            _hr(),
        ):
            logger.warning(ln)
        # 此确认刻意【不受 --yes 跳过】：几何不可信，必须人工确认末端确实无工具
        if not _ask_yes_no(
            "确认末端【无工具、法兰裸露】可安全继续？", default=False, assume_yes=False
        ):
            logger.info("已跳过 verify-touch。")
            return

    if not _ask_yes_no(
        f"将移动机械臂：指尖悬停在板中心上方约 {hover:.0f}mm（不接触，tool_offset={tool:.0f}mm）。现场安全？",
        default=False,
        assume_yes=args.yes,
    ):
        logger.info("已跳过 verify-touch。")
        return

    # 指尖悬停高度换算为法兰目标 z；先到更高处再下降，避免侧向扫过标定板
    flange_hover_z = p_base[2] + tool + hover
    approach_z = flange_hover_z + 40.0
    env.move_to_flange(
        FlangePose(p_base[0], p_base[1], approach_z, fp.rx_deg, fp.ry_deg, fp.rz_deg)
    )
    env.move_to_flange(
        FlangePose(p_base[0], p_base[1], flange_hover_z, fp.rx_deg, fp.ry_deg, fp.rz_deg)
    )
    logger.info(
        "指尖应悬停在板中心正上方约 %.0fmm（未接触）。请肉眼确认指尖是否对准板中心：xy 对齐即标定良好。",
        hover,
    )


# =============================================================================
# 三个入口子命令
# =============================================================================
def do_generate_board(args) -> int:
    board = _board_from_args(args)
    path = generate_board_image(board, args.generate_board)
    logger.info(
        "✅ 已生成标定板图：%s （%s %dx%d，方格 %.1fmm）",
        path,
        board.kind,
        board.squares_x,
        board.squares_y,
        board.square_size_mm,
    )
    _print_board_print_notice(board)
    return 0


def do_calibrate(args) -> int:
    _banner(
        [
            "🤖 jiuwensymbiosis 手眼标定（eye-in-hand，腕部相机）",
            "目标：解出 T_flange_cam，写入标定 JSON，并给出精度评估。",
        ]
    )
    board = _board_from_args(args)
    interactive = not args.non_interactive

    # 阶段0：标定板确认
    if interactive and not args.yes:
        if not _ask_yes_no("是否已备好标定板并平整固定？", default=True):
            logger.info("请先生成并打印标定板，例如：")
            logger.info(
                "  python scripts/calibrate/calibrate_hand_eye.py --generate-board board.png "
                "--board %s --squares-x %d --squares-y %d --square-size-mm %g%s",
                board.kind,
                board.squares_x,
                board.squares_y,
                board.square_size_mm,
                f" --marker-size-mm {board.marker_size_mm}" if board.marker_size_mm else "",
            )
            _print_board_print_notice(board)
            return 1

    # 构建 session（固定命名 builder 约定）
    module = args.module or "jiuwensymbiosis.adapters.piper"
    logger.info("正在连接机器人（%s）...", module)
    try:
        builder = _load_builder(_resolve_module(module, args.path))
        session = builder.from_yaml(args.config) if args.config else builder.from_dict({})
    except Exception as exc:
        logger.error("❌ 构建 session 失败：%s", exc)
        logger.error("   请检查 --config / --module，以及硬件（CAN、相机）是否就绪。")
        return 2

    # 标定只需 CAN 读位姿 + 相机取帧，不需要 detector/VLM 等 sidecar。某些 adapter
    # （如 piper）的 detector.spawn 默认 True，会在 connect 时启动检测服务（加载
    # GroundingDINO+SAM2，慢且吃 GPU）。这里统一禁用，避免标定误启动它们。
    if getattr(session, "sidecar_starters", None):
        n = len(session.sidecar_starters)
        session.sidecar_starters = []
        logger.info("标定无需检测服务，已跳过 %d 个 sidecar（不会加载 GroundingDINO 等）。", n)

    with session:
        env = session.env
        # 阶段0：环境自检
        ok, issues = _env_self_check(env)
        if not ok:
            logger.error("❌ 环境自检发现问题：")
            for it in issues:
                logger.error("   - %s", it)
            if not _ask_yes_no(
                "仍要继续吗？", default=False, assume_yes=args.yes if interactive else False
            ):
                return 2
        else:
            logger.info("✅ 环境自检通过（位姿/相机/内参就绪）。")

        intrinsics, dist = _resolve_intrinsics_pre(args, env)
        _print_config_card(args, board, intrinsics)
        if interactive and not args.yes:
            if not _ask_yes_no("以上配置是否正确，开始采集？", default=True):
                logger.info("已取消。")
                return 1
        if args.auto and interactive and not args.yes:
            if not _ask_yes_no(
                "自动模式将驱动机械臂运动，请确认 E-stop 在手边，继续？", default=False
            ):
                logger.info("已取消。")
                return 1

        # 阶段1：采集
        if args.auto:
            stations, image_size = _collect_auto(env, board, intrinsics, dist, args)
        else:
            stations, image_size = _collect_manual(env, board, intrinsics, dist, args)
        n_ok = len([s for s in stations if s.detection.ok])
        if n_ok < MIN_STATIONS:
            logger.error("❌ 有效视图 %d < %d，已放弃。", n_ok, MIN_STATIONS)
            return 2

        # 内参标定（如需）
        if args.calibrate_intrinsics or intrinsics is None:
            if image_size is None:
                logger.error("❌ 无图像尺寸，无法标定内参。")
                return 2
            intrinsics, dist, rms = calibrate_intrinsics_from_views(
                [s.detection for s in stations], image_size, board
            )
            logger.info(
                "✅ 内参由视图标定：RMS=%.3fpx，K=%s", rms, np.round(intrinsics, 2).tolist()
            )
            if float(np.linalg.norm(dist)) > 1e-3:
                logger.warning(
                    "⚠️ 畸变系数较大(||dist||=%.4f)，但 schema 无畸变字段，运行时假定已校正帧。",
                    float(np.linalg.norm(dist)),
                )
            _fill_poses([s.detection for s in stations], intrinsics, dist, board)

        # 阶段2：求解 + 评估
        try:
            res = solve_hand_eye(
                stations,
                intrinsics,
                method=args.method,
                eye_to_hand=args.eye_to_hand,
                cross_check=args.cross_check,
            )
        except Exception as exc:
            logger.error("❌ 求解失败：%s", exc)
            return 2
        _print_report(res, args, title="（全部 %d 帧）" % res.n_stations)
        res = _maybe_drop_outliers(res, stations, intrinsics, args)

        # 写文件（确认 + 备份）
        obj_xyz = _resolve_object_xyz(args, res)
        out = Path(args.out)
        if out.exists() and interactive and not args.yes:
            if not _ask_yes_no(f"{out} 已存在，覆盖？（原文件备份为 .bak）", default=True):
                logger.info("未写入。")
                return 1
        if out.exists():
            bak = out.with_suffix(out.suffix + ".bak")
            bak.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("已备份旧文件 → %s", bak)
        save_calibration(
            out,
            res.tf_flange_cam,
            intrinsics,
            obj_xyz,
            frame_field=res.frame_field,
            top_comment=f"hand-eye calibrated by scripts/calibrate/calibrate_hand_eye.py "
            f"(views={res.n_stations}, method={res.method})",
            intrinsics_comment="measured/used by scripts/calibrate/calibrate_hand_eye.py",
            object_comment="workspace anchor (sets z_min_safe + fallback home); verify it is the work surface",
        )
        logger.info("✅ 已写入标定文件：%s", out)
        _print_next_steps(args, res)

        if args.verify or args.verify_touch:
            _verify_live(env, res, intrinsics, args)
    return 0


# =============================================================================
# 离线自检（合成数据，无需硬件）
# =============================================================================
def _selftest_assert(cond: bool, msg: str = "") -> None:
    """selftest 断言：用显式 raise 取代 assert（assert 在优化字节码下会被移除）。"""
    if not cond:
        raise AssertionError(f"selftest 失败: {msg}")


def _check_pose_convention() -> None:
    try:
        from jiuwensymbiosis.adapters.piper.geometry import FlangePose
    except Exception as exc:
        logger.info("（无法导入 piper geometry，跳过 pose→tf 一致性检查：%s）", exc)
        return
    rng = np.random.default_rng(1)
    for _ in range(50):
        x, y, z = rng.uniform(-500, 500, 3)
        rx, ry, rz = rng.uniform(-180, 180, 3)
        p = SimpleNamespace(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)
        a = pose_to_tf_base_flange(p)
        b = FlangePose(x, y, z, rx, ry, rz).to_tf_base_flange()
        _selftest_assert(
            np.allclose(a, b, atol=1e-12), f"pose→tf≠FlangePose: {np.max(np.abs(a - b))}"
        )
    logger.info("✅ pose_to_tf_base_flange ≡ piper FlangePose.to_tf_base_flange（逐位一致）")


def _check_save_load_roundtrip(tf_handeye: np.ndarray) -> None:
    from jiuwensymbiosis.adapters._common.calibration import load_calibration

    intrinsics = np.array([[603.678, 0.0, 326.699], [0.0, 603.188, 248.428], [0.0, 0.0, 1.0]])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        save_calibration(p, tf_handeye, intrinsics, [300.0, 0.0, 30.0], frame_field="T_flange_cam")
        loaded = load_calibration(
            p, frame_field="T_flange_cam", legacy_field="T_TCP_cam", env_var="JIUWEN_SELFTEST_X"
        )
        tf_check = np.asarray(tf_handeye, dtype=np.float64).copy()
        tf_check[:3, :3] = orthonormalize(tf_check[:3, :3])
        _selftest_assert(
            np.allclose(loaded["T_flange_cam"]["matrix_4x4"], np.round(tf_check, 6), atol=1e-5),
            "matrix_4x4 round-trip 不一致",
        )
        _selftest_assert(
            np.allclose(loaded["intrinsics"], intrinsics, atol=1e-6), "intrinsics 不一致"
        )
        _selftest_assert(
            np.allclose(loaded["object"]["xyz_base_mm"], [300.0, 0.0, 30.0], atol=1e-6),
            "object.xyz_base_mm 不一致",
        )
        raw = json.loads(p.read_text())
        _selftest_assert(raw["schema_version"] == 2, "schema_version != 2")
        _selftest_assert(
            "_frame" in raw["T_flange_cam"] and "matrix_4x4" in raw["T_flange_cam"],
            "T_flange_cam 缺字段",
        )
    logger.info("✅ save_calibration → load_calibration round-trip 通过（schema 一致）")


def do_selftest(args) -> int:
    _banner(["🧪 离线自检（合成数据，无需硬件/相机）"])
    rng = np.random.default_rng(0)
    # 已知真值
    tf_gt = make_transform(rpy_deg_to_rot(20.0, 95.0, -30.0), np.array([-80.0, -0.3, -114.0]))
    tf_base_target = make_transform(rpy_deg_to_rot(2.0, 1.0, 5.0), np.array([300.0, 0.0, 30.0]))
    stations: list[Station] = []
    for _ in range(12):
        rx, ry, rz = rng.uniform(-30, 30, 3)
        t = rng.uniform(-300, 300, 3)
        tf_bf = make_transform(rpy_deg_to_rot(rx, ry, rz), t)
        tf_base_cam = tf_bf @ tf_gt
        tf_cam_target = invert_transform(tf_base_cam) @ tf_base_target
        stations.append(
            Station(tf_bf, ViewDetection(ok=True, tf_cam_target=tf_cam_target, reproj_rms_px=0.0))
        )

    # 纯 numpy：用真值 X 的残差应 ~0（阈值留出浮点累积裕度：真值噪声 ~1e-6，逻辑错会差几个数量级）
    rot, trans = axxb_residuals(stations, tf_gt)
    _selftest_assert(rot.max < 1e-3 and trans.max < 1e-3, f"AX=XB 残差偏大: {rot}, {trans}")
    mean, spread = board_origin_in_base(stations, tf_gt)
    _selftest_assert(
        spread.max < 1e-3 and np.allclose(mean, tf_base_target[:3, 3], atol=1e-3),
        "板原点聚集度/均值异常",
    )
    logger.info("✅ AX=XB 残差与板原点聚集度在真值下 ≈0")

    _check_pose_convention()
    _check_save_load_roundtrip(tf_gt)

    # 离群检测：篡改一帧应被 outlier_mask 单独标出，且不误伤其余（中位偏差仍≈0，即非系统性）
    bad = list(stations)
    s0 = bad[0]
    t_bad = s0.detection.tf_cam_target.copy()
    t_bad[:3, 3] += np.array([50.0, 0.0, 0.0])
    bad[0] = Station(
        s0.tf_base_flange, ViewDetection(ok=True, tf_cam_target=t_bad, reproj_rms_px=0.0)
    )
    mask, _dists, med = outlier_mask(board_origin_points(bad, tf_gt), k=3.0)
    _selftest_assert(
        bool(mask[0]) and int(mask.sum()) == 1 and med < 1.0,
        f"离群检测异常: {mask.tolist()}, {med}",
    )
    logger.info("✅ 离群检测：篡改帧被单独标出，其余中位偏差 %.2emm（非系统性）", med)

    # 有 cv2 才测 calibrateHandEye 求解恢复
    try:
        _require_cv2()
        have_cv2 = True
    except RuntimeError:
        have_cv2 = False
    if have_cv2:
        intrinsics = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        res = solve_hand_eye(stations, intrinsics, method="PARK", cross_check=True)
        ang = _rotation_angle_deg(res.tf_flange_cam[:3, :3], tf_gt[:3, :3])
        dmm = float(np.linalg.norm(res.tf_flange_cam[:3, 3] - tf_gt[:3, 3]))
        _selftest_assert(ang < 1e-3 and dmm < 1e-2, f"求解未恢复真值: {ang}°, {dmm}mm")
        logger.info("✅ calibrateHandEye 恢复真值：旋转误差 %.2e°，平移误差 %.2e mm", ang, dmm)
        logger.info(
            "   多算法交叉校验最大分歧：%.2e°，%.2e mm",
            res.cross_check_max_deg,
            res.cross_check_max_mm,
        )
        # 检测链路：生成标定板图 → 检测 → solvePnP，重投影应很小（验证 cv2 检测路径不崩）
        for kind, msz in (("charuco", 22.0), ("chessboard", None)):
            bspec = BoardSpec(kind, 5, 7, 30.0, marker_size_mm=msz)
            with tempfile.TemporaryDirectory() as bd:
                bp = Path(bd) / "b.png"
                generate_board_image(bspec, bp)
                img = _imread_rgb(bp)
            det = detect_board(img, bspec, intrinsics)
            _selftest_assert(det.ok, f"{kind} 检测失败：{det.reason}")
            _selftest_assert(
                det.reproj_rms_px is not None and det.reproj_rms_px < 1.0,
                f"{kind} 重投影偏大: {det.reproj_rms_px}",
            )
            logger.info("✅ %s 检测链路通过（重投影 %.3fpx）", kind, det.reproj_rms_px)
    else:
        logger.info("（未装 cv2，跳过 calibrateHandEye 求解恢复测试；纯 numpy 部分已全过）")

    logger.info("✅ 全部离线自检通过。")
    return 0


# =============================================================================
# CLI
# =============================================================================
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="jiuwensymbiosis 手眼标定工具（eye-in-hand）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 入口
    p.add_argument("--selftest", action="store_true", help="离线自检（合成数据，无需硬件）")
    p.add_argument(
        "--generate-board", metavar="PATH", default=None, help="生成可打印标定板图后退出"
    )
    # 机器人 / 适配器
    p.add_argument(
        "--config", "-c", default=None, help="YAML 配置（如 scripts/calibrate/calibrate.yaml）"
    )
    p.add_argument(
        "--module", "-m", default=None, help="适配器模块（默认 jiuwensymbiosis.adapters.piper）"
    )
    p.add_argument("--path", default=None, help="适配器目录路径（自动推导模块）")
    # 标定板
    p.add_argument(
        "--board", choices=["charuco", "chessboard"], default="charuco", help="标定板类型"
    )
    p.add_argument("--squares-x", type=int, default=5, help="板 X 方向方格数")
    p.add_argument("--squares-y", type=int, default=7, help="板 Y 方向方格数")
    p.add_argument(
        "--square-size-mm", type=float, default=30.0, help="方格边长（mm，务必=实测打印尺寸）"
    )
    p.add_argument("--marker-size-mm", type=float, default=None, help="ChArUco marker 边长（mm）")
    p.add_argument("--aruco-dict", default="DICT_4X4_50", help="ChArUco aruco 字典名")
    p.add_argument(
        "--min-corners",
        type=int,
        default=6,
        help="ChArUco 单帧最少角点数（少于此数的帧丢弃；下限 6，solvePnP 要求；越大质量越稳）",
    )
    # 采集
    p.add_argument("--auto", action="store_true", help="自动移动采集（piper）；默认手动交互")
    p.add_argument(
        "--auto-tilt-deg", type=float, default=20.0, help="自动模式 rx/ry 倾斜幅度（度）"
    )
    p.add_argument("--auto-yaw-deg", type=float, default=25.0, help="自动模式 rz 旋转幅度（度）")
    p.add_argument("--auto-dz-mm", type=float, default=40.0, help="自动模式 z 升降幅度（mm）")
    p.add_argument("--auto-dry-run", action="store_true", help="自动模式只打印目标，不移动")
    p.add_argument("--max-reproj-px", type=float, default=1.0, help="单帧重投影门控阈值（px）")
    p.add_argument("--lax", action="store_true", help="放宽：采纳超过重投影阈值的视图")
    # 内参
    p.add_argument("--calibrate-intrinsics", action="store_true", help="用采集视图标定相机内参")
    p.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "PPX", "PPY"),
        default=None,
        help="手动指定内参 fx fy ppx ppy",
    )
    # 锚点 / 几何 / 求解
    p.add_argument(
        "--object-xyz",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="object.xyz_base_mm 锚点（mm）",
    )
    p.add_argument("--base", default=None, help="从已有标定文件沿用 object.xyz_base_mm")
    p.add_argument(
        "--method",
        default="PARK",
        choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"],
        help="calibrateHandEye 算法",
    )
    p.add_argument("--cross-check", action="store_true", help="多算法交叉校验一致性")
    p.add_argument(
        "--drop-outliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="自动剔除离群帧后重解（默认开；--no-drop-outliers 关闭）",
    )
    p.add_argument("--outlier-k", type=float, default=3.0, help="离群判别的 MAD 倍数（越小越激进）")
    p.add_argument("--eye-to-hand", action="store_true", help="eye-to-hand（默认 eye-in-hand）")
    p.add_argument("--euler-axes", default="xyz", help="厂商位姿欧拉轴序（须与本体运行时一致）")
    # 输出 / 复验 / UX
    p.add_argument("--out", "-o", default="configs/piper/piper_calib.json", help="输出标定文件")
    p.add_argument("--verify", action="store_true", help="标定后真机反投影复验（仅打印坐标）")
    p.add_argument(
        "--verify-touch", action="store_true", help="复验时让指尖靠近板面悬停目视（默认不接触）"
    )
    p.add_argument(
        "--verify-tool-offset-mm",
        type=float,
        default=None,
        help="verify-touch 用的法兰→指尖长度（mm）；默认取配置的 tool_offset_mm",
    )
    p.add_argument(
        "--verify-hover-mm",
        type=float,
        default=30.0,
        help="verify-touch 指尖悬停在板面上方的余量（mm，不接触）",
    )
    p.add_argument("--report", action="store_true", help="另存 <out>.report.json 详细报告")
    p.add_argument("--show", action="store_true", help="实时叠加角点预览（需显示器）")
    p.add_argument("--debug-dir", default=None, help="保存每帧检测叠加图的目录")
    p.add_argument("--yes", "-y", action="store_true", help="跳过所有确认（老手）")
    p.add_argument("--non-interactive", action="store_true", help="非交互（脚本化；不读 stdin）")
    p.add_argument("--debug", action="store_true", help="输出调试日志")
    return p.parse_args(argv)


def _configure_logging(debug: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.WARNING)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.debug)
    clear_proxy_env()  # 在构建 session / 触发 openjiuwen 之前清理代理环境
    try:
        if args.selftest:
            return do_selftest(args)
        if args.generate_board:
            return do_generate_board(args)
        return do_calibrate(args)
    except KeyboardInterrupt:
        logger.info("\n已中断。")
        return 130
    except RuntimeError as exc:
        logger.error("❌ %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
