# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""主界面装配(NiceGUI 版):顶部导航 + 五个页面 + 编排。

取代 Qt 版的 ``QMainWindow`` + ``QStackedWidget`` + 菜单栏。持有跨页共享状态
(``AppState``:当前任务、配置缓存、工作区、正在运行的引擎),把各页动作接到运行链路。
同一时刻只允许一个运行(检测 sidecar 端口/日志是进程级单例)。
"""

from __future__ import annotations

from nicegui import app, ui

from jiuwensymbiosis.gui import ABOUT_TEXT, APP_NAME, registry
from jiuwensymbiosis.gui.app_state import AppState
from jiuwensymbiosis.gui.pages.config_view import ConfigView
from jiuwensymbiosis.gui.pages.history_view import HistoryView
from jiuwensymbiosis.gui.pages.home_view import HomeView
from jiuwensymbiosis.gui.pages.run_view import RunView
from jiuwensymbiosis.gui.pages.settings_view import SettingsView
from jiuwensymbiosis.gui.pages.tools_view import ToolsView
from jiuwensymbiosis.gui.run_engine import RunEngine

__all__ = ["build_layout", "Layout"]

_HISTORY = "历史"
_CONFIG = "配置"
_TOOLS = "工具"


class Layout:
    """一个客户端连接的整页布局。"""

    def __init__(self, state: AppState) -> None:
        self._state = state
        self._build()

    def _build(self) -> None:
        about = self._build_about_dialog()
        self._quit_dialog = self._build_quit_dialog()
        self._restart_dialog = self._build_restart_dialog()
        self._bye_dialog = self._build_bye_dialog()
        self._restarting_dialog = self._build_restarting_dialog()
        with ui.header().classes("items-center justify-between"):
            ui.label(APP_NAME).classes("text-lg font-bold")
            with ui.row().classes("items-center gap-1"):
                ui.button("关于", on_click=about.open).props("flat color=white")
                ui.button("重启", on_click=self._confirm_restart).props("flat color=white")
                ui.button("退出", on_click=self._confirm_quit).props("flat color=white")

        with ui.tabs().classes("w-full") as self._tabs:
            self._home_tab = ui.tab("主页")
            self._config_tab = ui.tab("配置")
            self._run_tab = ui.tab("运行")
            self._tools_tab = ui.tab(_TOOLS)
            self._history_tab = ui.tab(_HISTORY)
            self._settings_tab = ui.tab("设置")

        with ui.tab_panels(self._tabs, value=self._home_tab, on_change=self._on_nav).classes("w-full grow"):
            with ui.tab_panel(self._home_tab):
                self._home = HomeView(self._state, on_run=self._start_run, on_config=self._open_config)
            with ui.tab_panel(self._config_tab):
                self._config = ConfigView(on_run=self._run_current_config, on_back=lambda: self._goto(self._home_tab))
            with ui.tab_panel(self._run_tab):
                self._run = RunView(on_stop=self._stop_run, on_fix=self._state.apply_fix, on_rerun=self._rerun)
            with ui.tab_panel(self._tools_tab):
                self._tools = ToolsView(self._state)
            with ui.tab_panel(self._history_tab):
                self._history = HistoryView(self._state.workspace)
            with ui.tab_panel(self._settings_tab):
                self._settings = SettingsView(self._state.workspace, on_workspace_change=self._set_workspace)

    def _build_quit_dialog(self) -> ui.dialog:
        """确认后关停整个应用(NiceGUI 服务器随之退出;重开请再点桌面图标/启动脚本)。"""
        with ui.dialog() as dialog, ui.card().classes("gap-3"):
            ui.label("退出 Jiuwen Symbiosis？").classes("text-base font-bold")
            ui.label("将关闭本应用（后台服务一并停止）。重新打开请再次点击桌面图标。").classes("text-sm")
            with ui.row().classes("self-end gap-2"):
                ui.button("取消", on_click=dialog.close).props("flat")
                ui.button("退出", on_click=self._do_quit).props("color=negative")
        return dialog

    def _build_restart_dialog(self) -> ui.dialog:
        """确认后重启整个应用:关停当前服务器并拉起一个新的(硬件/检测服务一并重连)。"""
        with ui.dialog() as dialog, ui.card().classes("gap-3"):
            ui.label("重启 Jiuwen Symbiosis？").classes("text-base font-bold")
            ui.label("将关停当前应用并重新启动(硬件/检测服务一并重连)。浏览器会自动打开新页面。").classes("text-sm")
            with ui.row().classes("self-end gap-2"):
                ui.button("取消", on_click=dialog.close).props("flat")
                ui.button("重启", on_click=self._do_restart).props("color=primary")
        return dialog

    def _confirm_quit(self) -> None:
        """点「退出」:运行中先拦一下(避免中途杀掉真机任务),否则弹确认框。"""
        if self._state.is_busy():
            ui.notify("有任务正在运行，请先到「运行」页点「■ 停止」再退出。", type="warning")
            return
        self._quit_dialog.open()

    def _confirm_restart(self) -> None:
        """点「重启」:运行中先拦一下(避免中途杀掉真机任务),否则弹确认框。"""
        if self._state.is_busy():
            ui.notify("有任务正在运行，请先到「运行」页点「■ 停止」再重启。", type="warning")
            return
        self._restart_dialog.open()

    def _do_restart(self) -> None:
        """确认重启:拉起接替进程(它等本进程让出端口后自己起服务器),亮「正在重启」再延时关停本进程。"""
        from jiuwensymbiosis.gui.app import spawn_replacement

        # 释放相机/CAN,免得接替进程重连硬件时被占用。
        self._tools.stop_preview()
        self._restart_dialog.close()
        self._restarting_dialog.open()
        spawn_replacement()
        ui.timer(0.8, app.shutdown, once=True)

    def _do_quit(self) -> None:
        """确认退出:先关确认框、亮「已关闭」、尝试关标签页,延时后再停服务器。

        ``app.shutdown()`` 会立刻断开与浏览器的连接,之后任何 UI 更新都送不到;所以先把
        「已关闭」提示 + ``window.close()`` 发出去,再用 ``ui.timer`` 延时停服务器。标签页能
        否真关取决于浏览器(手动打开的标签多数会拦脚本关闭,拦了就靠「已关闭」提示)。
        """
        from jiuwensymbiosis.gui.app import clear_instance_marker

        # 先撤「健康实例」标记:关停期间(app.shutdown 前的过渡期)端口仍在监听,新启动的进程据此
        # 判定旧实例在退、自己接手重开,而不是把浏览器指到这个马上要死的服务器上。
        clear_instance_marker()
        self._quit_dialog.close()
        self._bye_dialog.open()
        ui.run_javascript("window.close()")
        ui.timer(0.6, app.shutdown, once=True)

    # ------------------------------------------------------------------ 导航
    def _goto(self, tab: object) -> None:
        self._tabs.set_value(tab)

    def _on_nav(self, e: object) -> None:
        val = getattr(e, "value", None)
        if val != _TOOLS:
            # 离开工具页即请求停掉相机预览,释放 RealSense/CAN,免得正常运行时相机被占用。
            self._tools.stop_preview(wait=False)
        if val == _HISTORY:
            self._history.set_workspace(self._state.workspace)
        elif val == _CONFIG:
            # 切标签进配置页也按当前选中任务 + 当前模拟开关重建(与点卡片进入行为一致):
            # 主页改了选中任务或模拟↔真机后,配置页据此更新,因仿真置灰的控件恢复可点。
            self._sync_config_view()
        elif val == _TOOLS:
            # 进工具页按当前模拟开关/选中任务/配置重算前置校验(主页改动后据此更新引导)。
            self._tools.refresh()

    # ------------------------------------------------------------------ 配置 / 运行
    def _open_config(self, task_key: str) -> None:
        self._state.current_task = task_key
        self._sync_config_view()
        self._goto(self._config_tab)

    def _sync_config_view(self) -> None:
        """按当前选中任务 + 当前模拟开关重建配置表单。无选中任务则不动。"""
        task_key = self._state.current_task
        if task_key is None:
            return
        task = registry.get_task(task_key)
        self._config.load(task.display_name, self._state.config_for_task(task_key), mock=self._state.mock)

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
        # 开始正常运行前,确保工具页的相机预览已停止并释放硬件(阻塞等待,否则相机被占用)。
        self._tools.stop_preview()
        self._state.current_task = task_key
        # 真机运行前先把已下好的本地视觉模型喂给检测器(避免它去 huggingface.co 联网下载
        # 933MB 卡住);找不到就直接展示「错误诊断」引导用户定位/换镜像,而非空跑到超时。
        if not self._state.mock:
            missing = self._state.prime_detector_models(task_key)
            if missing:
                self._goto(self._run_tab)
                self._run.show_model_help(missing)
                return
        task = registry.get_task(task_key)
        model = self._state.config_for_task(task_key)
        engine = RunEngine(task, model.data, mock=self._state.mock, workspace=self._state.workspace)
        self._state.engine = engine
        self._goto(self._run_tab)
        self._run.attach(engine)

    def _stop_run(self) -> None:
        if self._state.engine is not None:
            self._state.engine.request_stop()

    def _rerun(self) -> None:
        """用刚跑完那次的同一配置重跑(克隆引擎,不受运行后改动的配置/模拟开关影响)。"""
        engine = self._state.engine
        if engine is None or self._state.is_busy():
            return
        fresh = engine.clone()
        self._state.engine = fresh
        self._goto(self._run_tab)
        self._run.attach(fresh)

    def _set_workspace(self, workspace: str) -> None:
        self._state.workspace = workspace
        self._history.set_workspace(workspace)

    # ------------------------------------------------------------------ 弹窗构建
    @staticmethod
    def _build_about_dialog() -> ui.dialog:
        """居中矩形「关于」弹窗(替代底部滑出的通知条)。"""
        with ui.dialog() as dialog, ui.card().classes("max-w-md gap-3"):
            ui.label(APP_NAME).classes("text-lg font-bold")
            ui.label(ABOUT_TEXT).classes("text-sm leading-relaxed whitespace-pre-wrap")
            ui.button("了解", on_click=dialog.close).props("flat").classes("self-end")
        return dialog

    @staticmethod
    def _build_bye_dialog() -> ui.dialog:
        """退出后的「已关闭」提示:shutdown 会立即断连,先亮这句,避免页面看起来像卡死。"""
        with ui.dialog().props("persistent") as dialog, ui.card().classes("items-center gap-2"):
            ui.label("Jiuwen Symbiosis 已关闭").classes("text-lg font-bold")
            ui.label("可以关闭此标签页了。").classes("text-sm text-gray-600")
        return dialog

    @staticmethod
    def _build_restarting_dialog() -> ui.dialog:
        """重启中提示:shutdown 会立即断连,先亮这句,新页面稍候由接替进程自动打开。"""
        with ui.dialog().props("persistent") as dialog, ui.card().classes("items-center gap-2"):
            ui.label("正在重启 Jiuwen Symbiosis…").classes("text-lg font-bold")
            ui.label("新页面稍候自动打开,可关闭此标签页。").classes("text-sm text-gray-600")
        return dialog


def build_layout(state: AppState) -> Layout:
    """在当前客户端页面里构建整页布局。"""
    return Layout(state)
