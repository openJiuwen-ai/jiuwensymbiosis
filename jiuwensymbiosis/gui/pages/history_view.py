# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""历史页(NiceGUI 版):列出已记录的执行轨迹,复用内置 HTML 回放在浏览器打开。

轨迹 JSON 由 ``TraceRail`` 写入 ``<workspace>/traces/``。这里扫描该目录、展示每次运行
的摘要,并用 ``render_trace_html`` 生成自包含 HTML(内联相机帧)供浏览器查看。
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from nicegui import ui

from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["HistoryView"]


class HistoryView:
    """执行轨迹列表 + 浏览器回放。"""

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace
        self._selected: Path | None = None
        with ui.row().classes("w-full no-wrap gap-4"):
            with ui.column().classes("w-1/2 gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("历史运行(最新在前):").classes("text-sm text-gray-600")
                    ui.button("↻ 刷新", on_click=self.refresh).props("flat dense")
                    ui.button("🌐 在浏览器打开回放", on_click=self._open).props("flat dense")
                with ui.scroll_area().classes("w-full h-96 border rounded"):
                    self._list = ui.column().classes("w-full gap-1")
            with ui.column().classes("w-1/2 gap-1"):
                ui.label("摘要:").classes("text-sm text-gray-600")
                self._summary = ui.label("选择一条轨迹查看摘要。").classes("whitespace-pre-wrap")
        self.refresh()

    def set_workspace(self, workspace: str) -> None:
        """设置工作区并刷新列表。"""
        self._workspace = workspace
        self.refresh()

    def refresh(self) -> None:
        """扫描 ``<workspace>/traces/*.json`` 重建列表(按修改时间倒序)。"""
        self._list.clear()
        traces_dir = Path(self._workspace) / "traces"
        if not traces_dir.is_dir():
            return
        files = sorted(traces_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        with self._list:
            for path in files:
                ui.label(path.stem).classes("cursor-pointer hover:bg-gray-100 px-2 py-1 rounded text-sm").on(
                    "click", lambda _e, p=path: self._show(p)
                )

    def _show(self, path: Path) -> None:
        self._selected = path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._summary.set_text(f"读取失败:{exc}")
            return
        entries = data.get("entries", []) or []
        ok = sum(1 for e in entries if e.get("success", True))
        self._summary.set_text(
            "\n".join(
                [
                    f"任务指令:{data.get('query', '—')}",
                    f"本体:{data.get('robot_name', '—')}",
                    f"开始时间:{data.get('started_at', '—')}",
                    f"步骤数:{len(entries)}(成功 {ok} / 失败 {len(entries) - ok})",
                ]
            )
        )

    def _open(self) -> None:
        if self._selected is None:
            ui.notify("请先选择一条轨迹。", type="warning")
            return
        # 经已在跑的本机 HTTP 服务打开(而非 file://:后者会被部分浏览器沙箱拒绝)。渲染在
        # /replay 路由里按需完成;这里只告诉路由该用哪个工作区,再让浏览器指过去。
        from jiuwensymbiosis.gui import app as gui_app

        gui_app.set_replay_workspace(self._workspace)
        webbrowser.open(gui_app.replay_url(self._selected.stem))
