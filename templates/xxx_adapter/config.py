# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""XxxConfig — hardware configuration dataclass.

Fields are annotated with:
  [必填]       — all adapters must provide
  [选填]       — has a sensible default
  [选填-仅 motion.joint]  — only needed when adapter declares that capability
  [选填-仅 grasp.*]       — only needed for gripper/suction adapters
  [选填-仅 vision.*]      — only needed for vision-capable adapters
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class XxxConfig:
    """Hardware configuration for the Xxx robot.

    Use ``from_yaml(path)`` to load from a YAML file, or construct directly
    with keyword arguments.
    """

    # ==================== 基本信息 [必填] ====================
    name: str = "xxx"

    # ==================== 硬件连接 [必填] ====================
    can_port: str = "can0"  # CAN / 串口 / 网络 端口
    # TODO: Replace with your actual connection parameters
    # Examples: serial_port: str, ip_address: str, usb_vid_pid: str
    move_speed: int = 50  # [选填] 运动速度百分比 (0-100)

    # ==================== 运动学 [选填] ====================
    tool_offset_mm: float = 0.0  # 法兰 → 工具末端 的 Z 向偏移 (mm)
    home_pose_xyzrxryrz_mm_deg: list[float] = field(
        default_factory=lambda: [200.0, 0.0, 400.0, 0.0, 90.0, 0.0]
    )  # [选填] Home 位姿 (x,y,z,rx,ry,rz)
    home_use_init_pose: bool = False  # [选填] 是否用当前位置作为 home 位姿

    # ==================== 安全边界 [选填] ====================
    z_min_safe_mm: float = 50.0  # Z 向安全下限 (SafetyRail 读取)
    x_min_mm: float | None = 0.0  # X 向工作空间下界 (None=不限制)
    x_max_mm: float | None = 700.0  # X 向工作空间上界
    y_min_mm: float | None = -500.0  # Y 向工作空间下界
    y_max_mm: float | None = 500.0  # Y 向工作空间上界
    z_max_mm: float | None = 800.0  # Z 向工作空间上界 (None=不限制)
    # [选填-仅 motion.joint] 关节软限位；单位须与 move_joint(q) 一致。
    joint_limits: dict[str, tuple[float, float]] | None = None

    # ==================== 夹爪/吸盘 [选填-仅 grasp.*] ====================
    gripper_open_mm: float = 70.0  # [选填-仅 grasp.parallel] 打开宽度 (mm)
    gripper_effort: int = 1000  # [选填-仅 grasp.parallel] 夹持力 (驱动单位)

    # ==================== 相机 [选填-仅 vision.*] ====================
    camera_serial: str | None = None  # [选填-仅 vision.camera] 相机序列号
    camera_resolution: tuple[int, int] = (640, 480)  # [选填-仅 vision.camera]
    camera_fps: int = 30  # [选填-仅 vision.camera]

    # ==================== 检测校正 [选填-仅 vision.detection] ====================
    z_correction_mm: float = 0.0  # Z 方向常值校正 (添加到手眼反投影结果)
    grasp_z_offset_mm: float = -25.0  # 抓取点相对于检测物体顶面的偏移 (负数=下方)
    chip_thickness_mm: float = 75.0  # 堆叠放置偏移 (被放置物体的 tip→bottom 距离)

    # ==================== 检测服务 [选填-仅 vision.detection] ====================
    detector_spawn: bool = True  # 是否自动启动检测子进程
    detector_url: str = "http://127.0.0.1:8114"
    detector_host: str = "127.0.0.1"
    detector_port: int = 8114

    # ==================== 标定 [选填-仅 vision.detection] ====================
    calib_path: str | None = None  # 手眼标定文件路径 (JSON)

    # ==================== 任务 [选填] ====================
    task_prompt: str | None = None  # 自定义任务提示词

    # ========================================================================
    #  Loaders — do NOT modify (framework contract)
    # ========================================================================

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> XxxConfig:
        """Construct config from a flat dictionary.

        Only keys matching dataclass field names are used; extra keys
        are silently ignored.
        """
        valid = {f.name for f in dataclasses.fields(cls)}
        clean: dict[str, Any] = {k: v for k, v in data.items() if k in valid}
        if "camera_resolution" in clean and isinstance(clean["camera_resolution"], list):
            clean["camera_resolution"] = tuple(clean["camera_resolution"])
        if "joint_limits" in clean:
            raw_limits = clean["joint_limits"]
            if not isinstance(raw_limits, dict):
                clean["joint_limits"] = None
            else:
                normalised: dict[str, tuple[float, float]] = {}
                for k, v in raw_limits.items():
                    if not isinstance(v, (list, tuple)) or len(v) != 2:
                        continue
                    try:
                        normalised[str(k)] = (float(v[0]), float(v[1]))
                    except (TypeError, ValueError):
                        continue
                clean["joint_limits"] = normalised if normalised else None
        return cls(**clean)

    @classmethod
    def from_yaml(cls, path: str | Path) -> XxxConfig:
        """Load config from a YAML file."""
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls.from_dict(data)
        # Resolve relative calib_path
        if cfg.calib_path and not Path(cfg.calib_path).is_absolute():
            candidate = (path.parent / cfg.calib_path).resolve()
            if candidate.exists():
                cfg.calib_path = str(candidate)
        return cfg
