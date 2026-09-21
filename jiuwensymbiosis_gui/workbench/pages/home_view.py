# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""主页(NiceGUI 版):本体 + 配置文件选择 + 任务列表(一行一个)+ 操作按钮。

点任务卡片即「选中」它(高亮 + 顶部显示「当前任务」);「运行」「配置」按钮作用于当前
选中的任务。开局自动选中第一个任务,消除「没有当前任务」的死角。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from nicegui import ui

from jiuwensymbiosis_gui.workbench import registry
from jiuwensymbiosis_gui.workbench.app_state import AppState

__all__ = ["HomeView"]

# 「配置文件」下拉里代表「本体默认配置」的取值(其余取值是配置文件绝对路径)。
_DEFAULT_CONFIG = ""
_DEFAULT_LABEL = "原始配置模板"


class HomeView:
    """主页视图。``on_run`` / ``on_config`` 作用于当前选中任务。"""

    def __init__(self, state: AppState, *, on_run: Callable[[str], None], on_config: Callable[[str], None]) -> None:
        self._state = state
        self._on_run = on_run
        self._on_config = on_config
        self._cards: dict[str, Any] = {}
        self._selected: str | None = None
        self._build()

    def _build(self) -> None:
        bodies = registry.list_bodies()
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("本体:")
            self._body = ui.select(
                {b.key: b.display_name for b in bodies},
                value=self._state.current_body or (bodies[0].key if bodies else None),
                on_change=lambda _e: self._on_body_change(),
            ).props("outlined dense")
            ui.label("配置文件:").classes("ml-4")  # 与本体下拉拉开一点距离
            self._config_file = ui.select(
                {_DEFAULT_CONFIG: _DEFAULT_LABEL},
                value=_DEFAULT_CONFIG,
                on_change=lambda _e: self._on_config_file_change(),
            ).props("outlined dense")
        self._current = ui.label("").classes("text-blue-600 font-bold")
        with ui.scroll_area().classes("w-full grow border rounded"):
            self._list = ui.column().classes("w-full gap-2 p-2")
        with ui.row().classes("gap-2"):
            self._run_btn = ui.button("▶ 运行", on_click=self._run_current).props("color=primary")
            self._cfg_btn = ui.button("⚙ 配置", on_click=self._config_current)
        initial_config = self._state.current_config_file
        self._refresh_config_files()
        if initial_config:
            options = dict(self._config_file.options)
            options[initial_config] = Path(initial_config).name
            self._config_file.set_options(options, value=initial_config)
            self._state.current_config_file = initial_config
        self._refresh_cards()

    def _on_body_change(self) -> None:
        """换本体:配置文件候选随本体重建(旧选择对新本体无意义),再刷新任务列表。"""
        self._refresh_config_files()
        self._refresh_cards()

    def reload_configs(self) -> None:
        """配置目录里新增/更新了可选配置后重建下拉(保留当前选择)。"""
        self._refresh_config_files(keep_selection=True)

    def _refresh_config_files(self, *, keep_selection: bool = False) -> None:
        """按当前本体重建配置文件下拉;``keep_selection`` 为假(换本体)时重置为原始配置模板。"""
        body_key = self._body.value
        options = {_DEFAULT_CONFIG: _DEFAULT_LABEL}
        if body_key is not None:
            options.update({str(path): path.stem for path in registry.alternate_configs(body_key)})
        current = self._config_file.value
        if keep_selection and current and Path(current).is_file():
            options.setdefault(current, Path(current).name)
        value = current if keep_selection and current in options else _DEFAULT_CONFIG
        self._config_file.set_options(options, value=value)
        self._state.current_config_file = value or None

    def _on_config_file_change(self) -> None:
        self._state.current_config_file = self._config_file.value or None

    def selected_task(self) -> str | None:
        return self._selected

    def reload(self) -> None:
        """按注册表重建任务列表(如「另存为新任务」后刷新主页)。"""
        self._refresh_cards()

    def _run_current(self) -> None:
        if self._selected is not None:
            self._on_run(self._selected)

    def _config_current(self) -> None:
        if self._selected is not None:
            self._on_config(self._selected)

    def _select(self, task_key: str) -> None:
        self._selected = task_key
        self._state.current_task = task_key
        self._current.set_text(f"当前任务:{registry.get_task(task_key).display_name}")
        self._highlight()

    def _highlight(self) -> None:
        for key, card in self._cards.items():
            if key == self._selected:
                card.classes(add="ring-2 ring-blue-500 bg-blue-50")
            else:
                card.classes(remove="ring-2 ring-blue-500 bg-blue-50")

    def _refresh_cards(self) -> None:
        body_key = self._body.value
        # 选中的本体是运行/配置的依据(配置属本体):在此单点同步到共享状态。
        self._state.current_body = body_key
        self._list.clear()
        self._cards = {}
        tasks = registry.tasks_for_body(body_key)
        with self._list:
            for task in tasks:
                card = ui.card().classes("w-full cursor-pointer")
                with card:
                    ui.label(task.display_name).classes("font-bold")
                    ui.label(task.description).classes("text-gray-600 text-sm")
                card.on("click", lambda _e, k=task.key: self._select(k))
                self._cards[task.key] = card

        keys = [t.key for t in tasks]
        if tasks and (self._selected is None or self._selected not in keys):
            self._select(tasks[0].key)
        elif tasks:
            self._highlight()
        else:
            self._selected = None
            self._current.set_text("当前任务:该本体暂无任务")
