# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""启动预检:确认图形界面依赖(NiceGUI)就位。

浏览器模式的图形界面只需一个 Python 依赖 ``nicegui``,不需要任何系统级显示库(不像
Qt 的 xcb 平台插件)。NiceGUI 缺失是可捕获的 ``ImportError``,但它在导入那一刻才抛;
放到预检里提前拦下,才能给出清晰中文指引而非裸 traceback。

范围仅限图形界面:命令行 / 无头运行 / 作为库导入都不需要 NiceGUI。
"""

from __future__ import annotations

import importlib.util

__all__ = ["nicegui_installed", "preflight_message"]


def nicegui_installed() -> bool:
    """NiceGUI 是否可用。用 ``find_spec`` 只探测、不导入。"""
    return importlib.util.find_spec("nicegui") is not None


def preflight_message() -> str | None:
    """缺 NiceGUI 时返回中文安装指引;就绪时返回 ``None``。"""
    if nicegui_installed():
        return None
    return (
        "无法启动图形界面:缺少 NiceGUI(图形界面必需)。\n"
        "(注意:仅图形界面需要;命令行 / 无头运行 / 作为库导入都不需要。)\n"
        "\n请在已激活的 conda 环境中安装图形界面依赖:\n"
        '    pip install -e ".[gui]"\n'
        "\n安装后重新运行本程序即可。"
    )
