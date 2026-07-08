# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""NiceGUI 应用启动装配(浏览器模式)。

``run()`` 假定调用方(``__main__.main``)已先调用 ``clear_proxy_env()`` 完成代理卫生,
再导入本模块——因为构建页面会间接导入 openjiuwen。绝不使用 ``native=True``(那会经
pywebview 拉系统 WebKitGTK,属 LGPL);只以本机 ``127.0.0.1`` 浏览器模式运行。
"""

from __future__ import annotations

from jiuwensymbiosis.gui import APP_NAME
from jiuwensymbiosis.utils.logging import configure_logging

__all__ = ["run"]

# 默认监听端口。仅绑定回环地址,不对外暴露。
_DEFAULT_PORT = 8770


def run(*, host: str = "127.0.0.1", port: int = _DEFAULT_PORT, show: bool = True) -> int:
    """构建页面并进入 NiceGUI/uvicorn 事件循环(阻塞至退出),返回进程退出码。"""
    configure_logging("INFO")

    from nicegui import ui

    from jiuwensymbiosis.gui.app_state import AppState
    from jiuwensymbiosis.gui.layout import build_layout

    @ui.page("/")
    def index() -> None:
        # 每个客户端连接一份独立状态,避免多标签共享同一运行引擎时争抢事件队列。
        build_layout(AppState())

    ui.run(
        host=host,
        port=port,
        title=APP_NAME,
        show=show,
        reload=False,
        native=False,
        show_welcome_message=False,
    )
    return 0
