# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""界面跨页共享状态(框架无关,无 Qt / 无 nicegui)。

持有工作区、各任务的 ``ConfigModel`` 缓存、当前任务、模拟开关与正在运行的 ``RunEngine``。
配置装载/默认值填充逻辑与框架无关,可独立单测。同一时刻只允许一个运行。
"""

from __future__ import annotations

import os
from typing import Any

from jiuwensymbiosis.gui import local_models, registry
from jiuwensymbiosis.gui.config_model import ConfigModel
from jiuwensymbiosis.gui.run_engine import RunEngine, default_workspace
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["AppState"]


class AppState:
    """一个进程内单用户的界面状态容器。"""

    def __init__(self, workspace: str | None = None) -> None:
        self.workspace = workspace or default_workspace()
        self.current_task: str | None = None
        self.mock = False
        self.engine: RunEngine | None = None
        self._configs: dict[str, ConfigModel] = {}

    def config_for_task(self, task_key: str) -> ConfigModel:
        """取任务配置模型:优先缓存,否则从 YAML 载入(缺失则用默认指令起步)。"""
        if task_key in self._configs:
            return self._configs[task_key]
        task = registry.get_task(task_key)
        try:
            model = ConfigModel.from_yaml_text(task.config_path().read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.debug("load config for %s failed, using default prompt: %s", task_key, exc)
            model = ConfigModel.from_dict({"env": {"cfg": {"prompt": task.default_query}}})
        # 任务级默认(如 pick_box 意图开启技能工作流):配置未显式设置时填入。
        for name, val in task.agent_defaults.items():
            if model.get(f"agent.{name}") is None:
                model.set(f"agent.{name}", val)
        # 默认开启轨迹记录,让「历史」页开箱即用。
        if model.get("agent.enable_tracing") is None:
            model.set("agent.enable_tracing", True)
        # 默认用快速模式(fast):真机运行更快、可重复;模拟模式下 run_engine 会强制回逐步。
        if model.get("agent.exec_mode") is None:
            model.set("agent.exec_mode", "fast")
        # 任务指令:配置未提供 prompt 时(piper.yaml 任务无关化后不含 prompt),用任务的默认
        # 指令预填,让「配置 → 基础」的「任务指令」框开箱即有内容(用户可改;不改就用它)。
        if not model.get("env.cfg.prompt"):
            model.set("env.cfg.prompt", task.default_query)
        self._configs[task_key] = model
        return model

    def set_config(self, task_key: str, model: ConfigModel) -> None:
        self._configs[task_key] = model

    def apply_fix(self, patch: dict[str, Any]) -> None:
        """把运行页的一键修复(本地模型 / 镜像)沉淀进当前任务配置,便于导出/另存。"""
        if self.current_task is None or not isinstance(patch, dict):
            return
        self.config_for_task(self.current_task).patch_detector(**patch)

    def is_busy(self) -> bool:
        return self.engine is not None and self.engine.is_running()

    def prime_detector_models(self, task_key: str) -> list[str]:
        """真机运行前把已下好的本地视觉模型目录喂给检测器,返回仍缺失的模型名。

        检测器的 ``gdino_model_id`` / ``sam2_model_id`` 优先读同名环境变量;指向本地快照目录
        可直接离线加载,绕过「联网下载 / 已缓存却仍在线校验」的卡顿。任务不含视觉检测器、或
        用户已自行设过环境变量(如经诊断页)则不干预。
        """
        servers = self.config_for_task(task_key).data.get("api_servers")
        if not isinstance(servers, list):
            return []
        detector = next(
            (s for s in servers if isinstance(s, dict) and "grounding_dino" in str(s.get("_target_", "")).lower()),
            None,
        )
        if detector is None:
            return []  # 该任务不使用视觉检测器
        needed = [("GroundingDINO", "GDINO_MODEL_ID", local_models.GDINO_REPO, local_models.looks_like_gdino_dir)]
        if detector.get("use_sam2", True):
            needed.append(("SAM2", "SAM2_MODEL_ID", local_models.SAM2_REPO, local_models.looks_like_sam2_dir))
        missing: list[str] = []
        for name, env_var, repo_id, validator in needed:
            if os.environ.get(env_var):
                continue  # 用户已指定,尊重
            found = local_models.detect_local_model(repo_id, validator)
            if found is not None:
                os.environ[env_var] = str(found)
            else:
                missing.append(name)
        return missing
