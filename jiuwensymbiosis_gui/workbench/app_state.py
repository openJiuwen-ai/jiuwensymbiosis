# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""界面跨页共享状态(框架无关,无 Qt / 无 nicegui)。

持有工作区、各任务的 ``ConfigModel`` 缓存、当前任务与正在运行的 ``RunEngine``。
配置装载/默认值填充逻辑与框架无关,可独立单测。同一时刻只允许一个运行。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jiuwensymbiosis.utils.logging import get_logger
from jiuwensymbiosis_gui.workbench import local_models, registry
from jiuwensymbiosis_gui.workbench.app_host import AppHost
from jiuwensymbiosis_gui.workbench.config_model import ConfigModel
from jiuwensymbiosis_gui.workbench.run_engine import RunEngine, default_workspace

logger = get_logger(__name__)

__all__ = ["AppState"]


class AppState:
    """一个进程内单用户的界面状态容器。"""

    def __init__(self, workspace: str | None = None, *, host: AppHost | None = None) -> None:
        self.host = host or AppHost()
        self.workspace = workspace or default_workspace()
        self.current_task: str | None = None
        self.current_body: str | None = None
        # 主页「配置文件」下拉选中的本体配置绝对路径;None = 用本体的默认配置。
        self.current_config_file: str | None = None
        self._engine: RunEngine | None = None
        self._engine_template = None
        # 上次开跑时机械臂所在的关节角,连同它属于哪个 (本体, 配置文件) 一起记。
        # 供运行页「回到起始位」;换本体或换配置即失效——拿 A 配置的姿态喂 B 配置,
        # 关节数与限位都可能对不上。
        self._start_pose: dict[str, Any] | None = None
        # 配置属**本体**(与任务无关),按 (本体, 任务, 配置文件) 缓存:同一组合共享一份可编辑
        # 配置,换本体/换配置文件则各自独立(本体无关任务在不同本体下用各自本体的配置)。
        self._configs: dict[tuple[str, str, str], ConfigModel] = {}

    def config_for(self, body_key: str, task_key: str) -> ConfigModel:
        """取(本体, 任务)在当前所选配置文件下的配置模型。

        优先缓存,否则从**本体**配置 YAML 载入,套上任务的 agent 默认与默认指令
        (本体配置缺失则用默认指令起步)。
        """
        cache_key = self._cache_key(body_key, task_key)
        if cache_key in self._configs:
            return self._configs[cache_key]
        body = registry.get_body(body_key)
        task = registry.get_task(task_key)
        config_path = Path(self.current_config_file) if self.current_config_file else body.config_path()
        try:
            model = ConfigModel.from_yaml_text(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.debug("load config %s failed, using default prompt: %s", config_path, exc)
            model = ConfigModel.from_dict({"env": {"cfg": {"prompt": task.default_query}}})
        # 任务级默认(如 pick_banana 的 fast/技能/步数):配置未显式设置时填入。
        for name, val in task.agent_defaults.items():
            if model.get(f"agent.{name}") is None:
                model.set(f"agent.{name}", val)
        # 默认开启轨迹记录,让「历史」页开箱即用。
        if model.get("agent.enable_tracing") is None:
            model.set("agent.enable_tracing", True)
        # 默认用快速模式(fastagent):真机运行更快、可重复。
        if model.get("agent.exec_mode") is None:
            model.set("agent.exec_mode", "fastagent")
        # 任务指令:本体配置不含 prompt(与任务无关),用任务默认指令预填「配置 → 任务指令」框
        # (用户可改;不改就用它)。
        if not model.get("env.cfg.prompt"):
            model.set("env.cfg.prompt", task.default_query)
        self._configs[cache_key] = model
        return model

    def set_config(self, body_key: str, task_key: str, model: ConfigModel) -> None:
        self._configs[self._cache_key(body_key, task_key)] = model

    def _cache_key(self, body_key: str, task_key: str) -> tuple[str, str, str]:
        return (body_key, task_key, self.current_config_file or "")

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
        return self.host.is_busy() or (self.engine is not None and self.engine.is_running())

    @property
    def engine(self):
        current = self.host.current_engine
        if current is not self._engine_template:
            self._engine = current.subscribe() if current is not None else None
            self._engine_template = current
        return self._engine

    @engine.setter
    def engine(self, engine):
        self.host.current_engine = engine
        self._engine_template = engine
        self._engine = engine

    def remember_start_pose(self, body_key: str, joints: list[float], *, config_source=None) -> None:
        """记下某次运行开跑时的关节角(引擎的 ``start_pose`` 事件调用)。"""
        self._start_pose = {
            "body": body_key,
            "config_file": self._config_source(body_key, config_source or self.current_config_file),
            "joints": [float(v) for v in joints],
        }

    def start_pose_joints(self) -> list[float] | None:
        """当前 (本体, 配置文件) 下可用的开跑姿态;换了本体/配置即返回 None。"""
        snapshot = self._start_pose
        if snapshot is None or self.current_body is None:
            return None
        if snapshot["body"] != self.current_body:
            return None
        if snapshot["config_file"] != self._config_source(self.current_body, self.current_config_file):
            return None
        return list(snapshot["joints"])

    @staticmethod
    def _config_source(body_key, path):
        source = Path(path) if path else registry.get_body(body_key).config_path()
        return str(source.expanduser().resolve())

    def prime_detector_models(self, body_key: str, task_key: str) -> list[str]:
        """真机运行前把已下好的本地视觉模型目录写进检测器配置项,返回仍缺失的模型名。

        找到本地快照目录后写入检测器配置的 ``gdino_model_id`` / ``sam2_model_id``
        (指向本地目录即可离线加载,绕过「联网下载 / 已缓存却仍在线校验」的卡顿),而非设进程级
        环境变量。新配置只在未指定模型时接收环境路径并写入配置;旧配置保留环境变量优先级。
        显式非默认模型不探测缓存;任务不含本地检测器、或「禁用视觉服务」开关打开时直接跳过。
        """
        config = self.config_for(body_key, task_key)
        if config.get("gui.disable_vision"):
            return []
        detector = config.local_detector()
        if detector is None:
            return []  # 该任务不使用视觉检测器
        canonical = isinstance(config.data.get("detector"), dict)
        needed = [
            (
                "gdino_model_id",
                "GroundingDINO",
                "GDINO_MODEL_ID",
                local_models.GDINO_REPO,
                local_models.looks_like_gdino_dir,
            ),
        ]
        if detector.get("use_sam2", True):
            needed.append(
                (
                    "sam2_model_id",
                    "SAM2",
                    "SAM2_MODEL_ID",
                    local_models.SAM2_REPO,
                    local_models.looks_like_sam2_dir,
                )
            )
        missing: list[str] = []
        for field, name, env_var, repo_id, validator in needed:
            current = detector.get(field)
            env_model = os.environ.get(env_var, "").strip()
            if env_model and not canonical:
                continue  # Legacy parsing applies the environment override itself.
            if env_model and field not in detector:
                # Canonical parsing is environment-independent; persist the
                # GUI's default choice so export and sidecar startup agree.
                detector[field] = env_model
                continue
            if current and current != repo_id:
                continue  # 配置里已显式指定非默认模型(本地路径/换了模型),尊重
            found = local_models.detect_local_model(repo_id, validator)
            if found is not None:
                detector[field] = str(found)  # 写进配置项,而非进程级环境变量
            else:
                missing.append(name)
        return missing
