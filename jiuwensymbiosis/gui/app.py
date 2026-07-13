# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""NiceGUI 应用启动装配(浏览器模式)。

``run()`` 假定调用方(``__main__.main``)已先调用 ``clear_proxy_env()`` 完成代理卫生,
再导入本模块——因为构建页面会间接导入 openjiuwen。绝不使用 ``native=True``(那会经
pywebview 拉系统 WebKitGTK,属 LGPL);只以本机 ``127.0.0.1`` 浏览器模式运行。
"""

from __future__ import annotations

from pathlib import Path

from jiuwensymbiosis.gui import APP_NAME
from jiuwensymbiosis.utils.logging import configure_logging

__all__ = ["run"]

# 默认监听端口。仅绑定回环地址,不对外暴露。
_DEFAULT_PORT = 8770

# 浏览器标签图标:openJiuwen 官方品牌标(矢量)。缺失时回落到 NiceGUI 默认,不报错。
_FAVICON = Path(__file__).parent / "assets" / "favicon.svg"


def _server_already_running(host: str, port: int) -> bool:
    """该 host:port 是否已有实例在监听(用户又点了一次图标/启动脚本)。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        return probe.connect_ex((host, port)) == 0


def run(*, host: str = "127.0.0.1", port: int = _DEFAULT_PORT, show: bool = True) -> int:
    """构建页面并进入 NiceGUI/uvicorn 事件循环(阻塞至退出),返回进程退出码。"""
    configure_logging("INFO")

    # 已有实例在跑(用户又点了一次图标)?别再起第二个(会撞端口而崩),直接把浏览器
    # 指到已在跑的那个页面。仅在需要弹浏览器时(show)这么做;headless/测试照常起。
    if show and _server_already_running(host, port):
        import webbrowser

        webbrowser.open(f"http://{host}:{port}")
        return 0

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
        favicon=str(_FAVICON) if _FAVICON.exists() else None,
        show=show,
        reload=False,
        native=False,
        show_welcome_message=False,
    )
    return 0
