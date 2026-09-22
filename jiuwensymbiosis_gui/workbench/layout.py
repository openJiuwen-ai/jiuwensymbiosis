# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""主界面装配(NiceGUI 版):顶部导航 + 五个页面 + 编排。

取代 Qt 版的 ``QMainWindow`` + ``QStackedWidget`` + 菜单栏。持有跨页共享状态
(``AppState``:当前任务、配置缓存、工作区、正在运行的引擎),把各页动作接到运行链路。
同一时刻只允许一个运行(检测 sidecar 端口/日志是进程级单例)。

页面区(标签面板)整体接受拖入的 YAML:文件在浏览器侧读成文本送回,是本体配置就弹框问
「只应用」还是「同时存为可选配置」。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import app, ui

from jiuwensymbiosis_gui.workbench import ABOUT_TEXT, APP_NAME, registry
from jiuwensymbiosis_gui.workbench.app_state import AppState
from jiuwensymbiosis_gui.workbench.pages.config_view import ConfigView
from jiuwensymbiosis_gui.workbench.pages.history_view import HistoryView
from jiuwensymbiosis_gui.workbench.pages.home_view import HomeView
from jiuwensymbiosis_gui.workbench.pages.run_view import RunView
from jiuwensymbiosis_gui.workbench.pages.settings_view import SettingsView
from jiuwensymbiosis_gui.workbench.pages.tools_view import ToolsView
from jiuwensymbiosis_gui.workbench.run_engine import RunEngine

__all__ = ["build_layout", "Layout"]

_HISTORY = "历史"
_CONFIG = "配置"
_TOOLS = "工具"

_APPLY_ONLY = "apply"
_APPLY_AND_SAVE = "save"

# 拖放:只收第一个 .yaml/.yml 文件,浏览器侧读成文本后回传(超过 2MB 的当作误拖忽略)。
_DROP_JS = (
    "async (e) => { const f = e.dataTransfer?.files?.[0];"
    " if (!f || !/\\.ya?ml$/i.test(f.name) || f.size > 2000000) return;"
    " emit({name: f.name, text: await f.text()}); }"
)


class Layout:
    """一个客户端连接的整页布局。"""

    def __init__(self, state: AppState) -> None:
        self._state = state
        self._build()

    def _build(self) -> None:
        about = self._build_about_dialog()
        self._quit_dialog = self._build_quit_dialog()
        self._restart_dialog = self._build_restart_dialog()
        self._return_dialog = self._build_return_dialog()
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

        with ui.tab_panels(self._tabs, value=self._home_tab, on_change=self._on_nav).classes("w-full grow") as panels:
            with ui.tab_panel(self._home_tab):
                self._home = HomeView(self._state, on_run=self._start_run, on_config=self._open_config)
            with ui.tab_panel(self._config_tab):
                self._config = ConfigView(
                    on_run=self._run_current_config,
                    on_back=lambda: self._goto(self._home_tab),
                    on_config_saved=self._home.reload_configs,
                )
            with ui.tab_panel(self._run_tab):
                self._run = RunView(
                    on_stop=self._stop_run,
                    on_fix=self._state.apply_fix,
                    on_rerun=self._rerun,
                    on_start_pose=self._remember_start_pose,
                    on_return_to_start=self._confirm_return_to_start,
                )
            with ui.tab_panel(self._tools_tab):
                self._tools = ToolsView(self._state, on_open_config=self._open_config_field)
            with ui.tab_panel(self._history_tab):
                self._history = HistoryView(self._state.workspace)
            with ui.tab_panel(self._settings_tab):
                self._settings = SettingsView(self._state.workspace, on_workspace_change=self._set_workspace)

        self._dropped: tuple[str, str] | None = None
        self._drop_dialog, self._drop_name, self._drop_choice = self._build_drop_dialog()
        panels.on("dragover.prevent", js_handler="() => {}")  # 不 preventDefault 浏览器就不允许放下
        panels.on("drop.prevent", self._on_yaml_dropped, js_handler=_DROP_JS)
        ui.timer(0.2, self._sync_shared_run)

    def _sync_shared_run(self) -> None:
        engine = self._state.engine
        if engine is not None and engine is not self._run.engine:
            self._run.attach(engine)

    # ------------------------------------------------------------------ 拖入配置
    def _build_drop_dialog(self) -> tuple[ui.dialog, Any, Any]:
        with ui.dialog() as dialog, ui.card().classes("w-[34rem]"):
            name = ui.label("").classes("text-lg font-bold")
            choice = ui.radio({_APPLY_ONLY: "只应用", _APPLY_AND_SAVE: "应用并存储"}, value=_APPLY_ONLY)
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("取消", on_click=dialog.close).props("flat")
                ui.button("确认", on_click=self._confirm_drop).props("color=primary")
        return dialog, name, choice

    def _on_yaml_dropped(self, e: Any) -> None:
        """浏览器侧读好的 YAML 文本到达:是本体配置才弹框,否则一句提示。"""
        payload = e.args[0] if isinstance(e.args, list) and e.args else e.args
        if not isinstance(payload, dict):
            return
        name, text = str(payload.get("name", "")), str(payload.get("text", ""))
        if not registry.is_body_config_text(text):
            ui.notify(f"{name} 不是可用的本体配置(需包含 env.cfg.low_level)。", type="negative")
            return
        self._dropped = (name, text)
        self._drop_name.set_text(name)
        self._drop_choice.set_value(_APPLY_ONLY)
        self._drop_dialog.open()

    def _confirm_drop(self) -> None:
        """应用拖入的配置(可选同时存进本体配置目录)。"""
        if self._dropped is None:
            return
        name, text = self._dropped
        model = self._state.current_config()
        body_key = self._state.current_body
        if model is None or body_key is None:
            ui.notify("请先在主页选择一个本体与任务。", type="warning")
            return
        model.replace_from_yaml(text)
        self._drop_dialog.close()
        self._dropped = None
        self._sync_config_view()
        if self._drop_choice.value != _APPLY_AND_SAVE:
            ui.notify(f"已应用 {name}", type="positive", timeout=2000)
            return
        try:
            path = registry.save_body_config(body_key, name, text)
        except (ValueError, OSError) as exc:
            ui.notify(f"已应用 {name}，但保存失败:{exc}", type="negative")
            return
        self._home.reload_configs()
        ui.notify(f"已应用并保存:{path}", type="positive", timeout=2500)

    def _build_quit_dialog(self) -> ui.dialog:
        """确认后关停整个应用(NiceGUI 服务器随之退出;重开请再点桌面图标/启动脚本)。"""
        return self._confirm_dialog(
            title="退出 Jiuwen Symbiosis？",
            body="将请求停止当前任务和工具，确认硬件收尾后关闭应用。清理未完成时会保留页面和占用。",
            confirm_label="退出",
            confirm_props="color=negative",
            on_confirm=self._do_quit,
        )

    def _build_restart_dialog(self) -> ui.dialog:
        """确认后重启整个应用:关停当前服务器并拉起一个新的(硬件/检测服务一并重连)。"""
        return self._confirm_dialog(
            title="重启 Jiuwen Symbiosis？",
            body="将请求停止当前任务和工具，确认硬件收尾后重新启动。清理未完成时不会启动新实例。",
            confirm_label="重启",
            confirm_props="color=primary",
            on_confirm=self._do_restart,
        )

    def _build_return_dialog(self) -> ui.dialog:
        """确认后机械臂自主运动回到本次运行开跑时的姿态。"""
        return self._confirm_dialog(
            title="回到起始位？",
            body="机械臂会自动运动回到本次运行开始时的姿态，请先离开工作区。",
            confirm_label="回到起始位",
            confirm_props="color=negative",
            on_confirm=self._do_return_to_start,
        )

    def _confirm_quit(self) -> None:
        """应用关闭负责停止所有客户端的操作,包括刷新前启动的维护工具。"""
        self._quit_dialog.open()

    def _confirm_restart(self) -> None:
        """确认只请求关闭;真正重启仍由硬件清理结果决定。"""
        self._restart_dialog.open()

    def _do_restart(self) -> None:
        """确认重启:拉起接替进程(它等本进程让出端口后自己起服务器),亮「正在重启」再延时关停本进程。

        接替进程不开浏览器:本页面的 socket 断开后会一直重连,连上新服务器时握手失败(clientId
        对不上),NiceGUI 前端据此自行 reload——于是同一个标签页被复用,重启多少次都不多开页面。
        """
        from jiuwensymbiosis_gui.workbench.app import spawn_replacement

        # 释放相机/CAN,免得接替进程重连硬件时被占用。
        if not self._state.host.close()["closed"]:
            ui.notify("硬件仍在收尾，暂时不能重启。请等待或核对占用记录后重试。", type="warning")
            return
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
        from jiuwensymbiosis_gui.workbench.app import clear_instance_marker

        if not self._state.host.close()["closed"]:
            ui.notify("硬件仍在收尾，暂时不能退出。请等待或核对占用记录后重试。", type="warning")
            return

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
            self._tools.release_hardware(wait=False)
        if val == _HISTORY:
            self._history.set_workspace(self._state.workspace)
        elif val == _CONFIG:
            # 切标签进配置页也按当前选中任务重建(与点卡片进入行为一致):主页改了选中任务后,
            # 配置页据此更新。
            self._sync_config_view()
        elif val == _TOOLS:
            # 进工具页按当前选中任务/配置重算前置校验(主页改动后据此更新引导)。
            self._tools.refresh()

    # ------------------------------------------------------------------ 配置 / 运行
    def _open_config(self, task_key: str) -> None:
        self._state.current_task = task_key
        self._sync_config_view()
        self._goto(self._config_tab)

    def _open_config_field(self, path: str) -> None:
        """跳到「配置」页并停在某个字段上(工具页「这项没填」类提示的落点)。

        必须自己先 ``_sync_config_view()``:``_on_nav`` 认的是浏览器传回的标签**名**,而
        ``_goto`` 传的是 Tab 对象,那几条分支都不会命中,表单不会被重建。定位放在切标签
        之后,这样 ``_on_nav`` 日后改成认得 Tab 对象了也不会把落点冲掉。
        """
        if self._state.current_body is None or self._state.current_task is None:
            ui.notify("请先在主页选择一个本体与任务,再去改配置。", type="warning")
            self._goto(self._home_tab)
            return
        self._sync_config_view()
        self._goto(self._config_tab)
        if not self._config.reveal_field(path):
            ui.notify(f"「配置」页的表单里没有 {path},请在「原始 YAML」里改。", type="warning")

    def _sync_config_view(self) -> None:
        """按当前选中本体+任务重建配置表单。无选中本体/任务则不动。"""
        task_key = self._state.current_task
        body_key = self._state.current_body
        if task_key is None or body_key is None:
            return
        task = registry.get_task(task_key)
        self._config.load(
            task.display_name,
            self._state.config_for(body_key, task_key),
            body_key=body_key,
        )

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
        body_key = self._state.current_body
        if body_key is None:
            ui.notify("请先在主页选择一个本体。", type="warning")
            self._goto(self._home_tab)
            return
        # 开始正常运行前,确保工具页已放开相机与机械臂(阻塞等待)。放不开就不开跑:标定的
        # 自动采集阶段中途停不下来,硬上会变成两边同时占相机、同时对机械臂下指令。
        if not self._tools.release_hardware():
            ui.notify("「工具」页还在占用相机与机械臂，等它结束后再运行。", type="negative")
            return
        self._state.current_task = task_key
        # 真机运行前先把已下好的本地视觉模型喂给检测器(避免它去 huggingface.co 联网下载
        # 933MB 卡住);找不到就直接展示「错误诊断」引导用户定位/换镜像,而非空跑到超时。
        missing = self._state.prime_detector_models(body_key, task_key)
        if missing:
            self._goto(self._run_tab)
            self._run.show_model_help(missing)
            return
        task = registry.get_task(task_key)
        model = self._state.config_for(body_key, task_key)
        engine = RunEngine(
            task,
            model.data,
            workspace=self._state.workspace,
            body_key=body_key,
            config_source=self._state.current_config_file,
            host=self._state.host,
        )
        self._state.engine = engine
        self._goto(self._run_tab)
        self._run.attach(engine)

    def _stop_run(self) -> None:
        if self._state.engine is not None:
            self._state.engine.request_stop()

    def _rerun(self) -> None:
        """重跑同一本体、同一任务,带上配置页此刻的配置。

        本体/任务取自引擎(界面此后可能已切走),配置现取,所以「改完配置点重新执行」跑的就是
        改后的那份——与配置页的「用当前配置运行」一致。
        """
        engine = self._state.engine
        if engine is None or self._state.is_busy():
            return
        model = self._state.config_for(engine.body_key, engine.task_key)
        source = self._state.current_config_file or registry.get_body(engine.body_key).config_path()
        fresh = engine.rerun_with(model.data, config_source=source)
        self._state.engine = fresh
        self._goto(self._run_tab)
        self._run.attach(fresh)

    # ------------------------------------------------------------------ 回到起始位
    def _remember_start_pose(self, joints: list[float]) -> None:
        """起始姿态归属于原任务配置,不随查看页面的选择改变。"""
        engine = self._state.engine
        if engine is not None and joints:
            self._state.remember_start_pose(engine.body_key, joints, config_source=engine.config_source)

    def _confirm_return_to_start(self) -> None:
        """点「回到起始位」:先把拦不住的情况说清楚,能走再弹确认框。"""
        if self._state.is_busy():
            ui.notify("有任务正在运行，请先停止再回位。", type="warning")
            return
        if self._state.start_pose_joints() is None:
            ui.notify("换过本体或配置文件后，上次的起始姿态就不适用了；重新跑一次即可。", type="warning")
            return
        self._return_dialog.open()

    def _do_return_to_start(self) -> None:
        """确认回位:先要回硬件,再借运行引擎跑一次纯关节运动。"""
        self._return_dialog.close()
        joints = self._state.start_pose_joints()
        engine = self._state.engine
        if joints is None or engine is None:
            return
        # 与开跑同一道门:回位同样要独占机械臂。
        if not self._tools.release_hardware():
            ui.notify("「工具」页还在占用机械臂，等它结束后再回位。", type="negative")
            return
        # 沿用刚跑完那个引擎实例,而不是 clone():运行页的事件轮询盯的就是它,换一个
        # 实例回位进度就送不到界面上了。回位线程不看运行留下的停止标志与取消 token。
        self._goto(self._run_tab)
        engine.start_return_to(joints)

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
        """退出后的「已关闭」提示。"""
        return Layout._notice_dialog("Jiuwen Symbiosis 已关闭", "可以关闭此标签页了。")

    @staticmethod
    def _build_restarting_dialog() -> ui.dialog:
        """重启中提示;接替进程起来后本页面自行重连刷新(勿关标签页)。"""
        return Layout._notice_dialog("正在重启 Jiuwen Symbiosis…", "本页面会在服务就绪后自动刷新,请勿关闭。")

    @staticmethod
    def _confirm_dialog(
        *,
        title: str,
        body: str,
        confirm_label: str,
        confirm_props: str,
        on_confirm: Callable[[], None],
    ) -> ui.dialog:
        """确认框:标题 + 说明 + 「取消 / <确认>」两个按钮。"""
        with ui.dialog() as dialog, ui.card().classes("gap-3"):
            ui.label(title).classes("text-base font-bold")
            ui.label(body).classes("text-sm")
            with ui.row().classes("self-end gap-2"):
                ui.button("取消", on_click=dialog.close).props("flat")
                ui.button(confirm_label, on_click=on_confirm).props(confirm_props)
        return dialog

    @staticmethod
    def _notice_dialog(title: str, body: str) -> ui.dialog:
        """persistent 提示框:shutdown 会立即断连,先亮标题+一句说明,避免页面看起来像卡死。"""
        with ui.dialog().props("persistent") as dialog, ui.card().classes("items-center gap-2"):
            ui.label(title).classes("text-lg font-bold")
            ui.label(body).classes("text-sm text-gray-600")
        return dialog


def build_layout(state: AppState) -> Layout:
    """在当前客户端页面里构建整页布局。"""
    return Layout(state)
