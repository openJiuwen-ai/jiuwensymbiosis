# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""设置页(NiceGUI 版):程序信息 + 运行记录保存位置 + 界面语言。

模型端点 / api-key 在各任务的「配置 → 模型」分组里编辑;这里只放全局项。浏览器
模式没有原生目录选择框,工作区路径直接文本输入(本机 localhost 单用户,粘贴即可)。
「程序信息」显示本窗口实际导入的源码目录与解释器,便于确认「跑的是哪份副本」。
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from nicegui import ui

import jiuwensymbiosis

__all__ = ["SettingsView"]


def _package_dir() -> str:
    """本进程实际导入的 jiuwensymbiosis 源码目录(排查「跑的是哪份副本」用)。"""
    return str(Path(jiuwensymbiosis.__file__).resolve().parent)


class SettingsView:
    """全局设置页。工作区变更经 ``on_workspace_change`` 回传。"""

    def __init__(self, workspace: str, *, on_workspace_change: Callable[[str], None]) -> None:
        with ui.column().classes("w-full gap-3 p-2"):
            ui.label("当前运行的代码目录:").classes("font-bold")
            ui.input(value=_package_dir()).classes("w-full").props("outlined dense readonly")
            ui.separator()
            ui.label("Python 解释器:").classes("font-bold")
            ui.input(value=sys.executable).classes("w-full").props("outlined dense readonly")
            ui.separator()
            ui.label("运行记录保存位置:").classes("font-bold")
            ui.input(value=workspace, on_change=lambda e: on_workspace_change(e.value)).classes("w-full").props(
                "outlined dense"
            )
            ui.label("一般无需改动。").classes("text-gray-500 text-sm")
