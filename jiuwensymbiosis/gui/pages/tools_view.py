# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The 「工具」 page (NiceGUI): master-detail — tool list on the left, the selected tool's workspace on the right.

Holds task-agnostic debug / calibration tools. A new tool = one entry in ``_TOOLS`` + a method that
builds its workspace.

The only tool so far, 「感知测试」, shows the live camera; clicking anywhere prints that point's
base-frame (x, y, z). Real hardware only — ``PerceptionEngine`` runs the preview + reprojection on a
background thread, and this view updates only by draining its event queue via ``drain()`` (the same
``ui.timer`` polling as the run page).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from nicegui import ui

from jiuwensymbiosis.gui import registry
from jiuwensymbiosis.gui.app_state import AppState
from jiuwensymbiosis.gui.perception_engine import PerceptionEngine
from jiuwensymbiosis.gui.run_engine import resolve_real_session_config

__all__ = ["ToolsView"]

# Tool list: (key, icon, name). Add a tool here and build its workspace in _build.
_TOOLS: list[tuple[str, str, str]] = [
    ("perception", "🎯", "感知测试"),
]


def _dig(data: Any, *keys: str) -> Any:
    """Read a value from a nested dict by keys; return None if any level is missing / not a dict."""
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


class ToolsView:
    """The 「工具」 page view. Currently only 「感知测试」; the list + workspace layout is ready for more tools."""

    def __init__(self, state: AppState) -> None:
        """Build the tool list + workspace and attach the timer that drains perception-engine events."""
        self._state = state
        self._engine: PerceptionEngine | None = None
        self._last_xyz: tuple[float, float, float] | None = None
        self._errored = False
        self._selected_tool = _TOOLS[0][0]
        self._tool_rows: dict[str, Any] = {}

        self._dispatch: dict[str, Callable[[Any], None]] = {
            "preview_started": self._on_preview_started,
            "frame": self._on_frame,
            "point_result": self._on_point_result,
            "error": self._on_error,
            "preview_stopped": self._on_preview_stopped,
        }

        # 「感知测试」 workspace UI handles: actually built in _build_perception_panel; declared here first.
        self._context: Any = None
        self._start_btn: Any = None
        self._stop_btn: Any = None
        self._banner: Any = None
        self._camera: Any = None
        self._pixel_lbl: Any = None
        self._depth_lbl: Any = None
        self._xyz_lbl: Any = None
        self._zc_lbl: Any = None
        self._copy_btn: Any = None
        self._status: Any = None

        self._build()
        ui.timer(0.1, self._drain)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        with ui.row().classes("w-full no-wrap gap-4"):
            with ui.column().classes("w-56 shrink-0 gap-1"):
                ui.label("工具").classes("text-sm text-gray-500")
                for key, icon, label in _TOOLS:
                    row = ui.row().classes(
                        "w-full items-center cursor-pointer rounded px-2 py-2 hover:bg-gray-100 no-wrap gap-2"
                    )
                    with row:
                        ui.label(icon)
                        ui.label(label).classes("text-sm")
                    row.on("click", lambda _e, k=key: self._select_tool(k))
                    self._tool_rows[key] = row
                ui.label("更多工具即将到来\n(标定、检测测试…)").classes(
                    "text-gray-400 text-xs whitespace-pre-line mt-2"
                )
            with ui.column().classes("grow gap-2"):
                self._build_perception_panel()
        self._highlight_tool()
        self.refresh()

    def _select_tool(self, key: str) -> None:
        # Only one tool for now; the selection + highlight state is kept so more tools can switch by key later.
        self._selected_tool = key
        self._highlight_tool()

    def _highlight_tool(self) -> None:
        for key, row in self._tool_rows.items():
            if key == self._selected_tool:
                row.classes(add="bg-blue-50 ring-1 ring-blue-400")
            else:
                row.classes(remove="bg-blue-50 ring-1 ring-blue-400")

    # ------------------------------------------------------------ 感知测试 workspace
    def _build_perception_panel(self) -> None:
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("感知测试").classes("text-lg font-bold")
            self._context = ui.label("").classes("text-sm text-gray-500 grow")
            self._start_btn = ui.button("▶ 开始预览", on_click=self._start).props("color=primary")
            self._stop_btn = ui.button("■ 停止", on_click=self._stop).props("color=negative")
            self._stop_btn.disable()

        ui.label(
            "点画面任意位置 → 显示该点在基座坐标系下的 (x, y, z)。青色十字随光标移动=瞄准处;点击后红色圈=已测量的点。"
        ).classes("text-sm text-gray-600")

        self._banner = (
            ui.label().classes("w-full").style("background:#fff3cd; color:#7a5b00; padding:6px; border-radius:4px;")
        )
        self._banner.set_visibility(False)

        with ui.row().classes("w-full no-wrap gap-4"):
            with ui.column().classes("w-2/3 gap-1"):
                # Don't override object-fit: interactive_image's click→pixel math (.js: image_x =
                # offsetX*naturalW/clientW) assumes the image fills the element box, so object-fit:contain
                # letterboxing would map clicks to the wrong pixel. size=(640,480) gives a fixed 4:3 box
                # (= piper camera resolution); cursor:crosshair aligns the pointer with the crosshair
                # (the arrow's hotspot is its tip, otherwise it looks offset).
                self._camera = (
                    ui.interactive_image(
                        size=(640, 480),
                        cross="#22d3ee",
                        on_mouse=self._on_image_mouse,
                        events=["mousedown"],
                    )
                    .classes("w-full rounded")
                    .style("background:#111; max-width:760px; cursor:crosshair;")
                )
            with ui.column().classes("w-1/3 gap-2"):
                with ui.card().classes("w-full gap-1"):
                    ui.label("点选读数").classes("font-bold")
                    self._pixel_lbl = ui.label("像素:—").classes("text-sm")
                    self._depth_lbl = ui.label("深度:—").classes("text-sm")
                    self._xyz_lbl = ui.label("基座 (x, y, z):—").classes("font-mono text-sm")
                    self._zc_lbl = ui.label("").classes("font-mono text-xs text-amber-700")
                    self._zc_lbl.set_visibility(False)
                    self._copy_btn = ui.button("复制坐标", on_click=self._copy).props("flat dense")
                    self._copy_btn.disable()
                self._status = ui.label("").classes("text-sm text-gray-500")

    # ------------------------------------------------------------------ lifecycle
    def refresh(self) -> None:
        """Recompute preconditions (mock switch / selected task / config); call on page entry or state change."""
        self._update_context()
        if self.is_previewing():
            return
        reason = self._precondition_block()
        if reason:
            self._banner.set_text("⚠️ " + reason)
            self._banner.set_visibility(True)
            self._start_btn.disable()
        else:
            self._banner.set_visibility(False)
            self._start_btn.enable()

    def is_previewing(self) -> bool:
        return self._engine is not None and self._engine.is_running()

    def _precondition_block(self) -> str | None:
        """Return the first (most relevant) blocking reason as Chinese guidance; None when all pass."""
        st = self._state
        if st.mock:
            return "感知测试需要真机与深度相机,请在主页关闭「🧪 模拟模式」。"
        task_key = st.current_task
        if task_key is None:
            return "请先在主页选择一个任务(决定用哪个本体与配置)。"
        low_level = _dig(st.config_for_task(task_key).data, "env", "cfg", "low_level") or {}
        # camera_serial / calib_path are common adapter config keys (piper puts them under env.cfg.low_level).
        if not low_level.get("camera_serial") and not os.environ.get("CAMERA_SERIAL"):
            return "请先在「配置」页填写相机序列号(camera_serial),否则没有实时画面。"
        if not low_level.get("calib_path"):
            return "需要手眼标定文件(calib_path)才能把像素换算成基座坐标。"
        return None

    def _update_context(self) -> None:
        task_key = self._state.current_task
        if task_key is None:
            self._context.set_text("")
            return
        task = registry.get_task(task_key)
        mode = "模拟" if self._state.mock else "真机"
        self._context.set_text(f"本体:{task.body_key}   配置:{task.display_name}   模式:{mode}")

    # ------------------------------------------------------------------ interaction
    def _start(self) -> None:
        reason = self._precondition_block()
        if reason:
            ui.notify(reason, type="warning")
            return
        task_key = self._state.current_task
        if task_key is None:
            return
        task = registry.get_task(task_key)
        body = registry.get_body(task.body_key)
        cfg_data = resolve_real_session_config(self._state.config_for_task(task_key).data, task.config_path().parent)
        z_correction = float(_dig(cfg_data, "env", "cfg", "low_level", "z_correction_mm") or 0.0)
        self._engine = PerceptionEngine(lambda: body.build_real_session(cfg_data), z_correction_mm=z_correction)
        self._errored = False
        self._reset_readout()
        self._banner.set_visibility(False)
        self._start_btn.disable()
        self._stop_btn.enable()
        self._status.set_text("正在连接相机…")
        self._engine.start()

    def _stop(self) -> None:
        if self._engine is not None:
            self._engine.stop()
        self._stop_btn.disable()
        self._status.set_text("正在停止…")

    def stop_preview(self, *, wait: bool = True) -> None:
        """Stop the preview and release the camera/CAN (leaving the 「工具」 page, or before a normal run; idempotent).

        A normal run reopens the same camera; ``wait=True`` joins the worker thread (which calls
        env.disconnect() to free the RealSense + CAN before exiting), so release finishes before the run
        connects — avoiding a busy camera. Leaving the page uses ``wait=False``: only set the stop flag,
        don't block the event loop.
        """
        engine = self._engine
        if engine is None or not engine.is_running():
            return
        engine.stop()
        if wait:
            engine.join(timeout=3.0)
        self._stop_btn.disable()
        if self._precondition_block() is None:
            self._start_btn.enable()

    def _on_image_mouse(self, e: Any) -> None:
        """Click on the image: hand the pixel coords (image's native pixel frame) to the engine to reproject."""
        engine = self._engine
        if engine is None or not engine.is_running():
            return
        engine.request_point(e.image_x, e.image_y)

    def _copy(self) -> None:
        if self._last_xyz is None:
            return
        x, y, z = self._last_xyz
        ui.notify(f"坐标 (mm):{x:.1f}, {y:.1f}, {z:.1f}", type="positive")

    # ------------------------------------------------------------------ event handling
    def _drain(self) -> None:
        engine = self._engine
        if engine is None:
            return
        for tag, payload in engine.drain():
            handler = self._dispatch.get(tag)
            if handler is not None:
                handler(payload)

    def _on_preview_started(self, _payload: Any) -> None:
        self._status.set_text("预览中,点击画面任意位置取点。")

    def _on_frame(self, uri: str) -> None:
        self._camera.set_source(uri)

    def _on_point_result(self, r: dict) -> None:
        if not r.get("ok"):
            self._status.set_text("取点失败:" + str(r.get("reason", "")))
            return
        u, v = r["u"], r["v"]
        self._pixel_lbl.set_text(f"像素:({u}, {v})")
        self._depth_lbl.set_text(f"深度:{r['depth_m']:.3f} m")
        self._xyz_lbl.set_text(f"基座 (x, y, z):({r['x']:.1f}, {r['y']:.1f}, {r['z']:.1f}) mm")
        self._last_xyz = (r["x"], r["y"], r["z"])
        if "z_corrected" in r:
            self._zc_lbl.set_text(
                f"抓取校正后 Z:{r['z_corrected']:.1f} mm (z_correction_mm={r['z_correction_mm']:+.0f})"
            )
            self._zc_lbl.set_visibility(True)
        else:
            self._zc_lbl.set_visibility(False)
        self._copy_btn.enable()
        self._draw_marker(u, v)
        self._status.set_text("预览中,点击画面任意位置取点。")

    def _on_error(self, payload: dict) -> None:
        self._errored = True
        reason = str(payload.get("reason", ""))
        self._banner.set_text("⚠️ " + reason)
        self._banner.set_visibility(True)
        self._status.set_text("已停止:" + reason)

    def _on_preview_stopped(self, _payload: Any) -> None:
        self._stop_btn.disable()
        if self._precondition_block() is None:
            self._start_btn.enable()
        if not self._errored:
            self._status.set_text("已停止。")

    # ------------------------------------------------------------------ internal
    def _reset_readout(self) -> None:
        self._last_xyz = None
        self._pixel_lbl.set_text("像素:—")
        self._depth_lbl.set_text("深度:—")
        self._xyz_lbl.set_text("基座 (x, y, z):—")
        self._zc_lbl.set_visibility(False)
        self._copy_btn.disable()
        self._camera.set_content("")
        self._status.set_text("")

    def _draw_marker(self, u: float, v: float) -> None:
        """Draw a red ring + center dot at the measured point (distinct from the cyan aim cross; image-pixel coords)."""
        self._camera.set_content(
            f'<circle cx="{u}" cy="{v}" r="8" fill="none" stroke="#ff3b30" stroke-width="2"/>'
            f'<circle cx="{u}" cy="{v}" r="2" fill="#ff3b30"/>'
        )
