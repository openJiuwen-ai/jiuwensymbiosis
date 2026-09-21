# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""NiceGUI 应用启动装配(浏览器模式)。

``run()`` 假定调用方(``__main__.main``)已先调用 ``clear_proxy_env()`` 完成代理卫生,
再导入本模块——因为构建页面会间接导入 openjiuwen。绝不使用 ``native=True``(那会经
pywebview 拉系统 WebKitGTK,属 LGPL);只以本机 ``127.0.0.1`` 浏览器模式运行。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path

from jiuwensymbiosis.utils.logging import configure_logging, get_logger
from jiuwensymbiosis_gui.plugin import LaunchOptions
from jiuwensymbiosis_gui.workbench import APP_NAME

__all__ = ["run", "clear_instance_marker", "spawn_replacement", "set_replay_workspace", "replay_url"]

logger = get_logger(__name__)

# 默认监听端口。仅绑定回环地址,不对外暴露。
_DEFAULT_PORT = 8770

# 端口被旧实例占着时等它让出多久:能开浏览器的启动路径超时就指过去,不必久等;重启的接替
# 进程(不开浏览器)没有这条退路,等不到就彻底没界面,故多等一会儿。
_SHOW_PORT_WAIT_S = 1.5
_TAKEOVER_PORT_WAIT_S = 15.0

# 浏览器标签图标:openJiuwen 官方品牌标(矢量)。缺失时回落到 NiceGUI 默认,不报错。
_FAVICON = Path(__file__).parent / "assets" / "favicon.svg"


def _server_already_running(host: str, port: int) -> bool:
    """该 host:port 是否已有实例在监听(用户又点了一次图标/启动脚本)。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        return probe.connect_ex((host, port)) == 0


def _wait_for_port_release(host: str, port: int, *, timeout: float) -> bool:
    """轮询等占用 host:port 的实例让出端口。让出返回 True;到 timeout 仍被占返回 False。"""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _server_already_running(host, port):
            return True
        time.sleep(0.1)
    return False


# 本进程持有的「健康实例」标记路径:起服务器时落、退出/关停时删。新启动的进程据此分辨占着
# 端口的是健康实例(有标记→秒开浏览器)还是正在关停的旧实例(无标记→等它让出后自己重开)。
_INSTANCE_MARKER: Path | None = None
_LAUNCH_OPTIONS = LaunchOptions()
_INSTANCE_TOKEN = uuid.uuid4().hex


def _marker_path(port: int) -> Path:
    return Path(tempfile.gettempdir()) / f"jiuwensymbiosis-gui-{port}.lock"


def _mark_instance_healthy(port: int) -> None:
    """落下本实例的健康标记(内容为 pid,仅供排查;判定只看文件是否存在)。"""
    global _INSTANCE_MARKER
    path = _marker_path(port)
    path.write_text(
        json.dumps({"pid": os.getpid(), "token": _INSTANCE_TOKEN, "launch": _LAUNCH_OPTIONS.identity()}),
        encoding="utf-8",
    )
    _INSTANCE_MARKER = path


def clear_instance_marker() -> None:
    """删除本进程落下的健康标记(退出/关停时调用,幂等)。"""
    global _INSTANCE_MARKER
    if _INSTANCE_MARKER is not None:
        try:
            record = json.loads(_INSTANCE_MARKER.read_text(encoding="utf-8"))
            if isinstance(record, dict) and record.get("token") == _INSTANCE_TOKEN:
                _INSTANCE_MARKER.unlink(missing_ok=True)
        except (OSError, ValueError):
            logger.debug("Instance marker already missing or invalid")
        _INSTANCE_MARKER = None


def _healthy_instance_marked(port: int) -> bool:
    """Verify the listening app against its marker and requested launch identity."""
    import urllib.request

    try:
        marker = json.loads(_marker_path(port).read_text(encoding="utf-8"))
        if not isinstance(marker, dict) or marker.get("launch") != _LAUNCH_OPTIONS.identity():
            return False
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        host = _LAUNCH_OPTIONS.host
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        url = f"http://{host}:{port}/_jiuwen/instance"
        with opener.open(url, timeout=0.5) as response:
            identity = json.loads(response.read(8192))
        return bool(identity == {"token": marker.get("token"), "launch": marker.get("launch")})
    except (OSError, ValueError, TypeError):
        return False


def spawn_replacement() -> None:
    """为「重启」拉起一个接替本进程的新 GUI 进程;调用方随后关停本进程。

    先撤健康标记:新进程会看到端口仍被占但无标记,据此判定「旧实例在退」,等端口让出后
    自己起服务器(而非只把浏览器指过来)——与手动重开走同一套单实例逻辑。带 ``--no-browser``
    起:接替进程不再开新标签页,原页面在新服务器起来后由 NiceGUI 的重连逻辑自行刷新,重启
    多少次都只有一个标签页。分离式启动(``start_new_session``),使新进程不随本进程退出而被
    带走。命令固定、无用户输入、不走 shell。
    """
    import subprocess
    import sys

    clear_instance_marker()
    subprocess.Popen(
        [sys.executable, "-m", "jiuwensymbiosis_gui", *_LAUNCH_OPTIONS.arguments(restart=True)], start_new_session=True
    )


# 历史回放:经已在跑的本机 HTTP 服务打开,而非 file://。后者会被部分浏览器(如 snap/flatpak
# 沙箱下的 Firefox)以「无法访问本地文件」拒绝。记录当前服务地址与历史页所用工作区,供
# replay_url() 构造链接、/replay 路由据此定位轨迹文件。
_active_host = "127.0.0.1"
_active_port = _DEFAULT_PORT
_active_workspace: str | None = None

# 回放请求里的轨迹名只允许安全文件名(无路径分隔符):配合下方的 traces 目录归属校验防目录穿越。
_SAFE_STEM = re.compile(r"^[\w.-]+$")


def set_replay_workspace(workspace: str) -> None:
    """记录历史页当前所用工作区,供 /replay 路由据此解析轨迹文件。"""
    global _active_workspace
    _active_workspace = workspace


def replay_url(stem: str) -> str:
    """构造某条轨迹的回放地址(经本机 HTTP 服务)。

    用 ``urlunsplit`` 按部件拼(scheme 作数据),调用点无 ``http://`` 明文串。
    """
    from urllib.parse import quote, urlunsplit

    return urlunsplit(("http", f"{_active_host}:{_active_port}", f"/replay/{quote(stem)}", "", ""))


def _resolve_trace(stem: str, workspace: str | None) -> Path | None:
    """把回放请求的 stem 安全解析成 ``<workspace>/traces/<stem>.json``;越界/不存在返回 None。

    只接受安全文件名(无路径分隔符 / ``..``),并确认解析后仍在 traces 目录内,防目录穿越。
    """
    if not workspace or not _SAFE_STEM.match(stem):
        return None
    traces_dir = (Path(workspace) / "traces").resolve()
    path = (traces_dir / f"{stem}.json").resolve()
    if path.parent != traces_dir or not path.is_file():
        return None
    return path


def _port_takeover(host: str, port: int, *, show: bool) -> str:
    """端口已被占用时怎么办:``start`` 自己起 / ``open`` 指向在跑的实例 / ``abort`` 放弃。

    有健康标记=另一实例在正常跑(关了标签页又点图标)→ 指过去秒开;没标记=刚点「退出/重启」
    正在关停的旧实例(已先撤标记)→ 等它让出端口后自己接手。重启的接替进程(``show=False``)
    没有「指过去」这条退路,等不到就只能放弃。
    """
    if not _server_already_running(host, port):
        return "start"
    if show and _healthy_instance_marked(port):
        return "open"
    timeout = _SHOW_PORT_WAIT_S if show else _TAKEOVER_PORT_WAIT_S
    if _wait_for_port_release(host, port, timeout=timeout):
        return "start"
    return "abort"


def run(
    *, host: str = "127.0.0.1", port: int = _DEFAULT_PORT, show: bool = True, options: LaunchOptions | None = None
) -> int:
    """构建页面并进入 NiceGUI/uvicorn 事件循环(阻塞至退出),返回进程退出码。

    ``show=False``(重启的接替进程 / headless)不开浏览器:原标签页会在本服务器起来后自动
    重连刷新,不必再开一个。
    """
    global _LAUNCH_OPTIONS
    _LAUNCH_OPTIONS = options or LaunchOptions(host=host, port=port, open_browser=show)
    host, port, show = _LAUNCH_OPTIONS.host, _LAUNCH_OPTIONS.port, _LAUNCH_OPTIONS.open_browser
    configure_logging("INFO")

    action = _port_takeover(host, port, show=show)
    if action == "open":
        from nicegui.helpers import schedule_browser

        # schedule_browser 在守护线程里开浏览器;join 等它开完再返回,否则进程先退会把线程杀掉。
        thread, _ = schedule_browser("http", host, port)
        thread.join(3.0)
        return 0
    if action == "abort":
        from jiuwensymbiosis_gui.startup import _show_error_dialog

        message = f"端口 {port} 一直被占用，图形界面未能启动。\n请确认旧实例已经退出，再手动重开图形界面。"
        logger.error(message)
        # --no-browser is also used by a desktop restart's replacement process.
        # That process has no terminal and cannot update the old browser page.
        _show_error_dialog("JiuwenSymbiosis 启动失败", message)
        return 1

    global _active_host, _active_port
    _active_host, _active_port = host, port

    from nicegui import app as _ng_app
    from nicegui import ui

    from jiuwensymbiosis_gui.workbench.app_host import AppHost
    from jiuwensymbiosis_gui.workbench.app_state import AppState
    from jiuwensymbiosis_gui.workbench.layout import build_layout

    application = AppHost()

    def close_application():
        report = application.close()
        if not report["closed"]:
            logger.error("Application shutdown left unconfirmed hardware cleanup: %s", report)
        return report

    _ng_app.on_shutdown(close_application)

    @ui.page("/")
    def index() -> None:
        # 页面只持有选择和事件游标；任务和工具由同一个应用宿主持有。
        state = AppState(
            workspace=str(_LAUNCH_OPTIONS.workspace) if _LAUNCH_OPTIONS.workspace else None, host=application
        )
        if _LAUNCH_OPTIONS.config_path is not None:
            import yaml

            from jiuwensymbiosis_gui.workbench.registry import list_bodies

            raw = yaml.safe_load(_LAUNCH_OPTIONS.config_path.read_text(encoding="utf-8")) or {}
            adapter = str(raw.get("adapter") or "piper")
            body = next((body for body in list_bodies() if body.adapter == adapter), None)
            if body is None:
                raise ValueError(f"工作台未注册配置对应的本体: {adapter}")
            state.current_body = body.key
            state.current_config_file = str(_LAUNCH_OPTIONS.config_path)
        build_layout(state)

    @_ng_app.get("/_jiuwen/instance")
    def _instance() -> dict:
        return {"token": _INSTANCE_TOKEN, "launch": _LAUNCH_OPTIONS.identity()}

    @_ng_app.get("/replay/{stem}")
    def _replay(stem: str):
        # 历史页回放:按需渲染自包含 HTML 经本机 HTTP 服务返回(绕开 file:// 沙箱限制)。
        from fastapi.responses import HTMLResponse

        path = _resolve_trace(stem, _active_workspace)
        if path is None:
            return HTMLResponse("轨迹不存在或不可访问。", status_code=404)
        import json

        from jiuwensymbiosis.agent.trace_html import render_trace_html

        try:
            html = render_trace_html(json.loads(path.read_text(encoding="utf-8")), trace_path=path)
        except (OSError, ValueError) as exc:
            return HTMLResponse(f"回放渲染失败:{exc}", status_code=500)
        return HTMLResponse(html)

    _mark_instance_healthy(port)
    try:
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
    except KeyboardInterrupt:
        pass
    finally:
        cleanup = close_application()
        clear_instance_marker()
    return 0 if cleanup["closed"] else 1
