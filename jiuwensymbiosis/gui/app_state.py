# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""界面跨页共享状态(框架无关,无 Qt / 无 nicegui)。

持有工作区、各任务的 ``ConfigModel`` 缓存、当前任务与正在运行的 ``RunEngine``。
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
        self.current_body: str | None = None
        self.engine: RunEngine | None = None
        # 配置属**本体**(与任务无关),按 (本体, 任务) 缓存:同一本体+任务共享一份可编辑配置,
        # 换本体则各自独立(本体无关任务在不同本体下用各自本体的配置)。
        self._configs: dict[tuple[str, str], ConfigModel] = {}

    def config_for(self, body_key: str, task_key: str) -> ConfigModel:
        """取(本体, 任务)的配置模型。

        优先缓存,否则从**本体**配置 YAML 载入,套上任务的 agent 默认与默认指令
        (本体配置缺失则用默认指令起步)。
        """
        cache_key = (body_key, task_key)
        if cache_key in self._configs:
            return self._configs[cache_key]
        body = registry.get_body(body_key)
        task = registry.get_task(task_key)
        try:
            model = ConfigModel.from_yaml_text(body.config_path().read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.debug("load config for body %s failed, using default prompt: %s", body_key, exc)
            model = ConfigModel.from_dict({"env": {"cfg": {"prompt": task.default_query}}})
        # 任务级默认(如 pick_banana 的 fast/技能/步数):配置未显式设置时填入。
        for name, val in task.agent_defaults.items():
            if model.get(f"agent.{name}") is None:
                model.set(f"agent.{name}", val)
        # 默认开启轨迹记录,让「历史」页开箱即用。
        if model.get("agent.enable_tracing") is None:
            model.set("agent.enable_tracing", True)
        # 默认用快速模式(fast):真机运行更快、可重复。
        if model.get("agent.exec_mode") is None:
            model.set("agent.exec_mode", "fast")
        # 任务指令:本体配置不含 prompt(与任务无关),用任务默认指令预填「配置 → 任务指令」框
        # (用户可改;不改就用它)。
        if not model.get("env.cfg.prompt"):
            model.set("env.cfg.prompt", task.default_query)
        self._configs[cache_key] = model
        return model

    def set_config(self, body_key: str, task_key: str, model: ConfigModel) -> None:
        self._configs[(body_key, task_key)] = model

    def current_config(self) -> ConfigModel | None:
        """当前(选中本体, 选中任务)的配置模型;任一未选则 None。"""
        if self.current_body is None or self.current_task is None:
            return None
        return self.config_for(self.current_body, self.current_task)

    def apply_fix(self, patch: dict[str, Any]) -> None:
        """把运行页的一键修复(本地模型 / 镜像)沉淀进当前配置,便于导出/另存。"""
        model = self.current_config()
        if model is None or not isinstance(patch, dict):
            return
        model.patch_detector(**patch)

    def is_busy(self) -> bool:
        return self.engine is not None and self.engine.is_running()

    def prime_detector_models(self, body_key: str, task_key: str) -> list[str]:
        """真机运行前把已下好的本地视觉模型目录喂给检测器,返回仍缺失的模型名。

        检测器的 ``gdino_model_id`` / ``sam2_model_id`` 优先读同名环境变量;指向本地快照目录
        可直接离线加载,绕过「联网下载 / 已缓存却仍在线校验」的卡顿。任务不含视觉检测器、或
        用户已自行设过环境变量(如经诊断页)则不干预。「禁用视觉服务」开关打开时直接跳过。
        """
        config = self.config_for(body_key, task_key)
        if config.get("gui.disable_vision"):
            return []
        servers = config.data.get("api_servers")
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
