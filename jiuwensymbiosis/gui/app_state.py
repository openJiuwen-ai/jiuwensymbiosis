# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""界面跨页共享状态(框架无关,无 Qt / 无 nicegui)。

持有工作区、各任务的 ``ConfigModel`` 缓存、当前任务、模拟开关与正在运行的 ``RunEngine``。
配置装载/默认值填充逻辑与框架无关,可独立单测。同一时刻只允许一个运行。
"""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.gui import registry
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
        self.mock = True
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
