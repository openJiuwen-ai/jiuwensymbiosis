# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""主页(NiceGUI 版):本体选择 + 模拟模式开关 + 任务列表(一行一个)+ 操作按钮。

点任务卡片即「选中」它(高亮 + 顶部显示「当前任务」);「运行」「配置」按钮作用于当前
选中的任务。开局自动选中第一个任务,消除「没有当前任务」的死角。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from jiuwensymbiosis.gui import registry
from jiuwensymbiosis.gui.app_state import AppState

__all__ = ["HomeView"]


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
                value=bodies[0].key if bodies else None,
                on_change=lambda _e: self._refresh_cards(),
            ).props("outlined dense")
            self._caps = ui.label("").classes("text-green-700 text-sm")
            ui.space()
            self._mock = ui.switch("🧪 模拟模式(不连接硬件)", value=False, on_change=lambda e: self._set_mock(e.value))
        self._current = ui.label("").classes("text-blue-600 font-bold")
        ui.label("点任务选择它;再用下方的「运行」「配置」操作当前选中的任务。").classes("text-gray-500 text-sm")
        with ui.scroll_area().classes("w-full grow border rounded"):
            self._list = ui.column().classes("w-full gap-2 p-2")
        with ui.row().classes("gap-2"):
            self._run_btn = ui.button("▶ 运行", on_click=self._run_current).props("color=primary")
            self._cfg_btn = ui.button("⚙ 配置", on_click=self._config_current)
        self._set_mock(False)
        self._refresh_cards()

    def is_mock(self) -> bool:
        return bool(self._mock.value)

    def selected_task(self) -> str | None:
        return self._selected

    def reload(self) -> None:
        """按注册表重建任务列表(如「另存为新任务」后刷新主页)。"""
        self._refresh_cards()

    def _set_mock(self, value: bool) -> None:
        self._state.mock = bool(value)

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
        body = registry.get_body(body_key)
        self._caps.set_text("能力:" + " · ".join(body.capability_badges))
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
