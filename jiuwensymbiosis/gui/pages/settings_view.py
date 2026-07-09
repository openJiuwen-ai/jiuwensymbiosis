# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""设置页(NiceGUI 版):运行记录保存位置 + 界面语言。

模型端点 / api-key 在各任务的「配置 → 模型」分组里编辑;这里只放两个全局项。浏览器
模式没有原生目录选择框,工作区路径直接文本输入(本机 localhost 单用户,粘贴即可)。
"""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

__all__ = ["SettingsView"]


class SettingsView:
    """全局设置页。工作区变更经 ``on_workspace_change`` 回传。"""

    def __init__(self, workspace: str, *, on_workspace_change: Callable[[str], None]) -> None:
        with ui.column().classes("w-full gap-3 p-2"):
            ui.label("运行记录保存位置:").classes("font-bold")
            ui.input(value=workspace, on_change=lambda e: on_workspace_change(e.value)).classes("w-full").props(
                "outlined dense"
            )
            ui.label("任务的执行记录(可在「历史」页回看)保存在此目录,一般无需改动。").classes("text-gray-500 text-sm")
            ui.separator()
            with ui.row().classes("items-center gap-2"):
                ui.label("界面语言:")
                ui.label("简体中文")
