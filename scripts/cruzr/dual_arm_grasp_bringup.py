# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Cruzr 双臂夹箱演示。M2: --single-arm/--dry-run；M3: --full（见 Task 11）。

用法（真机）:
    /usr/bin/python3 scripts/cruzr/dual_arm_grasp_bringup.py --dry-run
    /usr/bin/python3 scripts/cruzr/dual_arm_grasp_bringup.py --single-arm

注意：
    - M2 worker ramp 从 0.0 起步，仅低速运动。
    - --single-arm 硬件运行已 DEFER 至用户手工执行。
    - ROS 2 / rclpy 仅在 env.connect() 时懒加载，不在模块顶层导入。
"""

from __future__ import annotations

import argparse
import logging

from jiuwensymbiosis.adapters.cruzr.grasp_planner import solve_grasp

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv
from jiuwensymbiosis.kinematics.urdf_chain import parse_chain
from jiuwensymbiosis.perception.object_geometry import ObjectGeometry3D

logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cruzr box-grasp demo: detect → IK → (single-arm) reach."
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Detect and solve IK but do NOT move any joints.")
    ap.add_argument("--single-arm", action="store_true",
                    help="After IK, execute a low-speed left-arm reach (implies real hardware).")
    ap.add_argument("--full", action="store_true",
                    help="Run full orchestrated dual_arm_grasp pipeline: detect->IK->clamp->FT->lift.")
    ap.add_argument("--object", default="box",
                    help="Open-vocab detection prompt (e.g. 'box', 'white bin').")
    ap.add_argument("--config", default="configs/cruzr/cruzr.yaml",
                    help="Path to CruzrConfig YAML.")
    args = ap.parse_args()

    cfg = CruzrConfig.from_yaml(args.config)
    env = CruzrEnv(cfg)
    env.connect()
    api = CruzrApi(
        env,
        detector_service_url=cfg.detector.url,
        camera_calib_path=cfg.camera_calib_path,
    )

    # --- Full orchestrated pipeline (M3): grasp, then place back + home ---
    if args.full:
        det = api.locate_for_grasp(args.object)
        logger.info("locate_for_grasp: ok=%s reason=%s", det.get("ok"), det.get("reason"))
        if not det.get("ok"):
            logger.warning("detection failed; safe retreat ...")
            logger.info("home: %s", api.home())
            return 1
        result = api.dual_arm_grasp(object_name=args.object)  # uses cached detection
        logger.info("dual_arm_grasp result: %s", result)
        if result.get("ok") and result.get("box"):
            logger.info("placing box back ...")
            logger.info("dual_arm_place: %s", api.dual_arm_place())  # uses last grasped box
        # dual_arm_place no longer returns to zero — home owns that.
        logger.info("home: %s", api.home())
        return 0 if result.get("ok") else 3

    # --- Step 1: detect box 3D geometry ---
    d = api.locate_for_grasp(args.object)
    logger.info("locate_for_grasp: %s", d)
    if not d["ok"]:
        logger.error("detection failed: %s", d.get("reason"))
        return 1

    box = ObjectGeometry3D(
        ok=True,
        reason="",
        center_mm=tuple(d["center_mm"]),
        width_mm=d["width_mm"],
        height_mm=d["height_mm"],
        front_x_mm=d["front_x_mm"],
        top_z_mm=d["top_z_mm"],
        n_points=d["n_points"],
        back_x_mm=d.get("back_x_mm", 0.0),
    )

    # --- Step 2: parse kinematic chains ---
    left = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf)
    right = parse_chain(cfg.urdf_path, "base_link", cfg.right_arm_leaf)

    # --- Step 3: read lifter/waist q_fixed from live hardware ---
    q = env.low_level.get_joint_positions()
    _fixed_names = ("lifter_pitch_1_joint", "lifter_pitch_2_joint",
                    "lifter_pitch_3_joint", "waist_yaw_joint")
    q_fixed = {k: q.get(k, 0.0) for k in _fixed_names}

    # --- Step 4: solve dual-arm grasp IK ---
    plan = solve_grasp(box, left, right, q_fixed)
    logger.info("plan.ok=%s reason=%s", plan.ok, plan.reason)
    for arm in ("left", "right"):
        logger.info("%s ik converged=%s pos_err_m=%s",
                    arm, plan.ik[arm].converged, round(plan.ik[arm].pos_err_m, 4))

    # --- Step 5: dry-run exits here ---
    if args.dry_run or not plan.ok:
        return 0 if plan.ok else 2

    # --- Step 6: single-arm low-speed reach (hardware, DEFERRED to user) ---
    if args.single_arm:
        env.low_level.move_joints_blocking(plan.ik["left"].q)
        logger.info("single-arm reach done; readback: %s", env.low_level.get_joint_positions())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
