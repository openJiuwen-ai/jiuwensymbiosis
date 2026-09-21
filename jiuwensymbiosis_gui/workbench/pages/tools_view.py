# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The 「工具」 page (NiceGUI): master-detail — tool list on the left, the selected tool's workspace on the right.

Holds task-agnostic debug / calibration tools. A new tool = one entry in ``_TOOLS`` + a method that
builds its workspace into its own container.

「感知测试」 shows the live camera; clicking anywhere prints that point's base-frame (x, y, z).
「手眼标定」 is the four-step calibration wizard (see ``calibration_view``). Both are real-hardware
only and both run their work on a background thread, so this view updates only by draining event
queues via ``drain()`` (the same ``ui.timer`` polling as the run page).

Only one tool may hold the camera/bus at a time: switching tools stops the one being left.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nicegui import ui

from jiuwensymbiosis_gui.workbench import registry
from jiuwensymbiosis_gui.workbench.app_state import AppState
from jiuwensymbiosis_gui.workbench.maintenance import MaintenanceOwner
from jiuwensymbiosis_gui.workbench.pages.calibration_view import CalibrationView
from jiuwensymbiosis_gui.workbench.pages.hardware_view import HardwareView
from jiuwensymbiosis_gui.workbench.perception_engine import PerceptionEngine

__all__ = ["ToolsView"]

# Tool list: (key, name). Add a tool here and build its workspace in _build.
_TOOLS: list[tuple[str, str]] = [
    ("perception", "感知测试"),
    ("calibration", "手眼标定"),
    ("hardware", "硬件控制"),
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

    def __init__(self, state: AppState, *, on_open_config: Callable[[str], None] = lambda _path: None) -> None:
        """Build the tool list + workspace and attach the timer that drains perception-engine events.

        ``on_open_config`` jumps to the 「配置」 page and lands on one dotted config path.
        """
        self._state = state
        self._on_open_config = on_open_config
        self._engine: PerceptionEngine | None = None
        self._last_xyz: tuple[float, float, float] | None = None
        self._errored = False
        self._selected_tool = _TOOLS[0][0]
        self._tool_rows: dict[str, Any] = {}
        self._panels: dict[str, Any] = {}
        self._calibration: CalibrationView | None = None
        self._hardware: HardwareView | None = None

        self._dispatch: dict[str, Callable[[Any], None]] = {
            "preview_started": self._on_preview_started,
            "frame": self._on_frame,
            "point_result": self._on_point_result,
            "error": self._on_error,
            "preview_stopped": self._on_preview_stopped,
            "gap": self._on_gap,
        }

        # 「感知测试」 workspace UI handles: actually built in _build_perception_panel; declared here first.
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
                for key, label in _TOOLS:
                    row = ui.row().classes(
                        "w-full items-center cursor-pointer rounded px-2 py-2 hover:bg-gray-100 no-wrap gap-2"
                    )
                    with row:
                        ui.label(label).classes("text-sm")
                    row.on("click", lambda _e, k=key: self._select_tool(k))
                    self._tool_rows[key] = row
            with ui.column().classes("grow gap-2"):
                self._panels["perception"] = ui.column().classes("w-full gap-2")
                with self._panels["perception"]:
                    self._build_perception_panel()
                self._panels["calibration"] = ui.column().classes("w-full gap-2")
                with self._panels["calibration"]:
                    self._calibration = CalibrationView(self._state, on_open_config=self._on_open_config)
                self._panels["hardware"] = ui.column().classes("w-full gap-2")
                with self._panels["hardware"]:
                    self._hardware = HardwareView(self._state)
        self._highlight_tool()
        self._show_selected()
        self.refresh()

    def _select_tool(self, key: str) -> None:
        """Switch tools, stopping whatever the tool being left was holding.

        The camera and the arm bus are single-owner: leaving 「感知测试」 with the preview still
        running would keep the RealSense open, and the calibration tool would then fail to connect.
        """
        if key == self._selected_tool:
            return
        if self._selected_tool == "perception":
            self.stop_preview(wait=True)
        elif self._selected_tool == "calibration" and self._calibration is not None:
            self._calibration.stop(wait=True)
        elif self._selected_tool == "hardware" and self._hardware is not None:
            self._hardware.stop(wait=True)
        self._selected_tool = key
        self._highlight_tool()
        self._show_selected()
        self.refresh()

    def _show_selected(self) -> None:
        for key, panel in self._panels.items():
            panel.set_visibility(key == self._selected_tool)

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
            ui.space()
            self._start_btn = ui.button("▶ 开始预览", on_click=self._start).props("color=primary")
            self._stop_btn = ui.button("■ 停止", on_click=self._stop).props("color=negative")
            self._stop_btn.disable()

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
        """Recompute the selected tool's preconditions; call on page entry or state change."""
        if self._selected_tool == "calibration":
            if self._calibration is not None:
                self._calibration.refresh()
            return
        if self._selected_tool == "hardware":
            if self._hardware is not None:
                self._hardware.refresh()
            return
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
        task_key = st.current_task
        body_key = st.current_body
        if task_key is None or body_key is None:
            return "请先在主页选择一个本体与任务(决定用哪个本体与配置)。"
        config = st.config_for(body_key, task_key)
        if config.get("gui.disable_vision"):
            return "已在「配置」页勾选「禁用视觉服务」;感知测试需要相机,请先取消该开关。"
        low_level = _dig(config.data, "env", "cfg", "low_level") or {}
        # camera_serial / calib_path are common adapter config keys (piper puts them under env.cfg.low_level).
        if not low_level.get("camera_serial") and not os.environ.get("CAMERA_SERIAL"):
            return "请先在「配置」页填写相机序列号(camera_serial),否则没有实时画面。"
        if not low_level.get("calib_path"):
            return "需要手眼标定文件(calib_path)才能把像素换算成基座坐标。"
        return None

    # ------------------------------------------------------------------ interaction
    def _start(self) -> None:
        reason = self._precondition_block()
        if reason:
            ui.notify(reason, type="warning")
            return
        task_key = self._state.current_task
        body_key = self._state.current_body
        if task_key is None or body_key is None:
            return
        body = registry.get_body(body_key)
        config_source = Path(self._state.current_config_file) if self._state.current_config_file else body.config_path()
        cfg_data = self._state.config_for(body_key, task_key).data
        z_correction = float(_dig(cfg_data, "env", "cfg", "low_level", "z_correction_mm") or 0.0)
        admission = MaintenanceOwner(
            self._state.host.runtime(self._state.workspace), config_source, self._state.workspace, adapter=body.adapter
        )
        self._engine = PerceptionEngine(
            lambda: body.build_real_session(cfg_data, include_sidecars=False),
            z_correction_mm=z_correction,
            admission=admission,
            config_snapshot=cfg_data,
        )
        self._errored = False
        self._reset_readout()
        self._banner.set_visibility(False)
        self._start_btn.disable()
        self._stop_btn.enable()
        self._status.set_text("正在连接相机…")
        self._state.host.start_tool(self._engine)

    def _stop(self) -> None:
        if self._engine is not None:
            self._engine.stop()
        self._stop_btn.disable()
        self._status.set_text("正在停止…")

    def release_hardware(self, *, wait: bool = True) -> bool:
        """Stop every tool on this page that can hold the camera / arm bus (idempotent).

        All three tools open the arm connection (and the first two the RealSense), so leaving the
        page, restarting, or starting a normal run must release whichever one is active — not just
        the perception preview.

        Returns whether they actually let go. Two phases cannot be interrupted on request:
        calibration's automatic capture, and hand guiding with the arm parked outside its soft
        limits (restoring torque there would fail and leave the arm limp). A caller that is about
        to drive the hardware itself must check this rather than assume the request was honoured.
        With ``wait=False`` nothing has had time to stop, so the answer is normally ``False``.
        """
        self.stop_preview(wait=wait)
        released = self._calibration is None or self._calibration.stop(wait=wait)
        released = (self._hardware is None or self._hardware.stop(wait=wait)) and released
        return released and not self.is_previewing()

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

    def _on_gap(self, payload: dict) -> None:
        dropped = int(payload.get("dropped", 0))
        self._status.set_text(f"界面事件流有缺口，已丢弃 {dropped} 条旧事件。")

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
