# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""主界面装配(NiceGUI 版):顶部导航 + 五个页面 + 编排。

取代 Qt 版的 ``QMainWindow`` + ``QStackedWidget`` + 菜单栏。持有跨页共享状态
(``AppState``:当前任务、配置缓存、工作区、正在运行的引擎),把各页动作接到运行链路。
同一时刻只允许一个运行(检测 sidecar 端口/日志是进程级单例)。
"""

from __future__ import annotations

from nicegui import ui

from jiuwensymbiosis.gui import ABOUT_TEXT, APP_NAME, registry
from jiuwensymbiosis.gui.app_state import AppState
from jiuwensymbiosis.gui.pages.config_view import ConfigView
from jiuwensymbiosis.gui.pages.history_view import HistoryView
from jiuwensymbiosis.gui.pages.home_view import HomeView
from jiuwensymbiosis.gui.pages.run_view import RunView
from jiuwensymbiosis.gui.pages.settings_view import SettingsView
from jiuwensymbiosis.gui.run_engine import RunEngine

__all__ = ["build_layout", "Layout"]

_HISTORY = "历史"


class Layout:
    """一个客户端连接的整页布局。"""

    def __init__(self, state: AppState) -> None:
        self._state = state
        self._build()

    def _build(self) -> None:
        with ui.header().classes("items-center justify-between"):
            ui.label(APP_NAME).classes("text-lg font-bold")
            ui.button("关于", on_click=lambda: ui.notify(ABOUT_TEXT, multi_line=True, close_button="好")).props(
                "flat color=white"
            )

        with ui.tabs().classes("w-full") as self._tabs:
            self._home_tab = ui.tab("主页")
            self._config_tab = ui.tab("配置")
            self._run_tab = ui.tab("运行")
            self._history_tab = ui.tab(_HISTORY)
            self._settings_tab = ui.tab("设置")

        with ui.tab_panels(self._tabs, value=self._home_tab, on_change=self._on_nav).classes("w-full grow"):
            with ui.tab_panel(self._home_tab):
                self._home = HomeView(self._state, on_run=self._start_run, on_config=self._open_config)
            with ui.tab_panel(self._config_tab):
                self._config = ConfigView(on_run=self._run_current_config, on_back=lambda: self._goto(self._home_tab))
            with ui.tab_panel(self._run_tab):
                self._run = RunView(on_stop=self._stop_run, on_fix=self._state.apply_fix)
            with ui.tab_panel(self._history_tab):
                self._history = HistoryView(self._state.workspace)
            with ui.tab_panel(self._settings_tab):
                self._settings = SettingsView(self._state.workspace, on_workspace_change=self._set_workspace)

    # ------------------------------------------------------------------ 导航
    def _goto(self, tab: object) -> None:
        self._tabs.set_value(tab)

    def _on_nav(self, e: object) -> None:
        if getattr(e, "value", None) == _HISTORY:
            self._history.set_workspace(self._state.workspace)

    # ------------------------------------------------------------------ 配置 / 运行
    def _open_config(self, task_key: str) -> None:
        self._state.current_task = task_key
        task = registry.get_task(task_key)
        self._config.load(task.display_name, self._state.config_for_task(task_key), mock=self._state.mock)
        self._goto(self._config_tab)

    def _run_current_config(self) -> None:
        if self._state.current_task is None:
            ui.notify("请先在主页点选一个任务(点一下任务卡片即可)。", type="warning")
            self._goto(self._home_tab)
            return
        self._start_run(self._state.current_task)

    def _start_run(self, task_key: str) -> None:
        if self._state.is_busy():
            ui.notify("已有任务在运行,请等待其结束或先停止。", type="warning")
            return
        self._state.current_task = task_key
        task = registry.get_task(task_key)
        model = self._state.config_for_task(task_key)
        engine = RunEngine(task, model.data, mock=self._state.mock, workspace=self._state.workspace)
        self._state.engine = engine
        self._goto(self._run_tab)
        self._run.attach(engine)

    def _stop_run(self) -> None:
        if self._state.engine is not None:
            self._state.engine.request_stop()

    def _set_workspace(self, workspace: str) -> None:
        self._state.workspace = workspace
        self._history.set_workspace(workspace)


def build_layout(state: AppState) -> Layout:
    """在当前客户端页面里构建整页布局。"""
    return Layout(state)
