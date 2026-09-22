# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""「硬件控制」工具面板(NiceGUI):松开力矩,用手把机械臂摆到想要的位置。

一条直线的四步:连接 → 读警示并确认 → 用手摆位(实时看关节角)→ 恢复力矩并断开。

界面只做展示与编排,力矩逻辑全在驱动的 ``HandGuidingDriver``,事件靠 ``drain()``
轮询消费,与「感知测试」「手眼标定」一致。

两处与常规工具不同的地方,都是因为这里的后果是机械臂会失去支撑:

* **确认门**。支不支持松力矩取决于驱动实现了哪个端口,连上才知道,所以警示文案
  等引擎的 ``mode`` 事件到了再填 —— 该事件严格早于力矩发生变化。
* **越限预警**。恢复力矩要把实测关节角写回目标,越出软限位会被驱动拒绝并让机械臂
  停在失力矩状态。所以关节表实时标红越限项,「恢复力矩」也会被引擎挡回来。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nicegui import ui

from jiuwensymbiosis.utils.logging import get_logger
from jiuwensymbiosis_gui.workbench import registry
from jiuwensymbiosis_gui.workbench.app_state import AppState
from jiuwensymbiosis_gui.workbench.hardware_engine import HardwareEngine, HardwareSetup
from jiuwensymbiosis_gui.workbench.maintenance import MaintenanceOwner

logger = get_logger(__name__)

__all__ = ["HardwareView"]

# 松力矩方式的指引与危险级别。键是引擎 ``mode`` 事件的取值 —— 判据是驱动实现了哪个
# 端口(``HandGuidingDriver``),不是机型名,所以新本体实现与否都会自动落到正确一档。
_MODE_TEXT = {
    "hand_guiding": (
        "⚠️ 确认后机械臂会立刻失去力矩并下坠，夹爪也会松开、夹着的东西会掉 —— 请先用手托住，再确认。",
        True,
    ),
    "unsupported": (
        "本机型的驱动没有实现松力矩接口，这个工具帮不上忙。请用厂商的示教器或拖拽模式摆位。",
        False,
    ),
}

# 危险动作的键盘快捷键。两只手都在托机械臂时去够鼠标既别扭又不安全,空格能盲按;
# 要求连按两下是因为这两下的后果都关乎机械臂是否有支撑,不能被一次误触触发。
_DOUBLE_TAP_S = 0.8
_KEY_HINT_IDLE = "也可以连按两下空格确认——手托着机械臂时不必去够鼠标。"
_KEY_HINT_ARMED = "再按一下空格确认。"


def _blur_active_element() -> None:
    """把焦点从刚点过的按钮上挪开。

    焦点粘在按钮上时 ``ui.keyboard`` 会忽略按键(``ignore`` 默认含 button),而空格
    又会被浏览器当成"再点一次这个按钮"——鼠标点过一次之后快捷键就全乱了。
    """
    try:
        ui.run_javascript("document.activeElement?.blur()")
    except Exception as exc:
        logger.debug("失焦请求未发出(没有已连接的客户端): %s", exc)


class HardwareView:
    """「硬件控制」面板。一个实例服务一次完整的摆位会话,可重复开始。"""

    def __init__(self, state: AppState) -> None:
        """建面板并挂上事件轮询与键盘监听。"""
        self._state = state
        self._engine: HardwareEngine | None = None
        self._space_at = 0.0
        self._joint_rows: dict[str, dict[str, Any]] = {}

        self._dispatch: dict[str, Callable[[Any], None]] = {
            "mode": self._on_mode,
            "state": self._on_state,
            "blocked": self._on_blocked,
            "phase": self._on_phase,
            "log": self._on_log,
            "done": self._on_done,
            "error": self._on_error,
            "gap": self._on_gap,
        }

        self._build()
        ui.keyboard(on_key=self._on_key, repeating=False)
        ui.timer(0.1, self._drain)

    # ================================================================== 布局
    def _build(self) -> None:
        with ui.column().classes("w-full gap-3"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label("硬件控制").classes("text-lg font-bold")
                ui.space()
                self._body_label = ui.label("").classes("text-sm text-gray-500")

            ui.label("松开力矩，用手把机械臂摆到想要的位置，再恢复力矩固定住。").classes("text-sm text-gray-600")

            self._blocker = (
                ui.label().classes("w-full").style("background:#fff3cd; color:#7a5b00; padding:6px; border-radius:4px;")
            )
            self._blocker.set_visibility(False)

            self._btn_connect = ui.button("连接机械臂", on_click=self._connect).props("color=primary")

            self._hint = ui.label().classes("w-full")
            self._hint.set_visibility(False)
            self._btn_confirm = ui.button("确认，松开力矩", on_click=self._confirm).props("color=negative")
            self._key_hint = ui.label(_KEY_HINT_IDLE).classes("text-xs text-gray-500")
            self._show_gate(False)

            self._readout = ui.card().classes("w-full gap-1")
            with self._readout:
                ui.label("关节角").classes("font-bold")
                self._joint_box = ui.column().classes("gap-0")
                self._pose_label = ui.label("").classes("font-mono text-xs text-gray-500")
            self._readout.set_visibility(False)

            self._btn_restore = ui.button("恢复力矩并断开", on_click=self._restore).props("color=positive")
            self._btn_restore.set_visibility(False)

            self._status = ui.label("").classes("text-sm text-gray-500")
            self._log_box = ui.log(max_lines=200).classes("w-full h-32 text-xs")

    def _show_gate(self, visible: bool) -> None:
        self._btn_confirm.set_visibility(visible)
        self._key_hint.set_visibility(visible)
        if not visible:
            self._space_at = 0.0
            self._key_hint.set_text(_KEY_HINT_IDLE)

    # ================================================================== 生命周期
    def refresh(self) -> None:
        """重算前置校验;进工具页或主页改动后调用。"""
        body_key = self._state.current_body
        self._body_label.set_text("" if body_key is None else registry.get_body(body_key).display_name)
        if self.is_active():
            return
        reason = self._precondition_block()
        self._blocker.set_text("" if reason is None else "⚠️ " + reason)
        self._blocker.set_visibility(reason is not None)
        if reason is None:
            self._btn_connect.enable()
        else:
            self._btn_connect.disable()

    def is_active(self) -> bool:
        """引擎是否还占着机械臂。"""
        return self._engine is not None and self._engine.is_running()

    def stop(self, *, wait: bool = False) -> bool:
        """请求收工并释放硬件(离开页面 / 开跑前调用;幂等),返回是否真的放开了。

        松力矩期间机械臂正靠操作者的手撑着,而恢复力矩要求当前姿态在软限位内 ——
        越限时引擎会拒绝恢复。所以这里停不下来是正常的一种结果,调用方必须看返回值,
        不能假定调用完硬件就归还了。
        """
        engine = self._engine
        if engine is None or not engine.is_running():
            return True
        engine.stop()
        if wait:
            engine.join(timeout=5.0)
        return not engine.is_running()

    def _precondition_block(self) -> str | None:
        """返回第一条阻塞原因(中文引导);全部通过返回 None。"""
        if self._state.current_body is None or self._state.current_task is None:
            return "请先在主页选择一个本体与任务(决定用哪个本体与配置)。"
        if self._state.is_busy():
            return "有任务正在运行，请先到「运行」页停止，再操作硬件。"
        return None

    # ================================================================== 动作
    def _connect(self) -> None:
        reason = self._precondition_block()
        if reason:
            ui.notify(reason, type="warning")
            return
        body_key = self._state.current_body
        task_key = self._state.current_task
        if body_key is None or task_key is None:
            return
        body = registry.get_body(body_key)
        config_source = Path(self._state.current_config_file) if self._state.current_config_file else body.config_path()
        config_data = self._state.config_for(body_key, task_key).data
        admission = MaintenanceOwner(
            self._state.host.runtime(self._state.workspace), config_source, self._state.workspace, adapter=body.adapter
        )
        self._engine = HardwareEngine(
            HardwareSetup(build_session=body.build_real_session, config_data=config_data), admission=admission
        )
        self._reset_readout()
        self._btn_connect.disable()
        self._blocker.set_visibility(False)
        self._status.set_text("正在连接…")
        self._state.host.start_tool(self._engine)

    def _confirm(self) -> None:
        """用户已读过警示:放行工作线程去松力矩。"""
        if self._engine is None:
            return
        _blur_active_element()
        self._show_gate(False)
        self._engine.confirm_release()

    def _restore(self) -> None:
        if self._engine is None:
            return
        _blur_active_element()
        self._status.set_text("正在同步目标并恢复力矩…")
        self._engine.request_restore()

    def _on_key(self, e: Any) -> None:
        """连按两下空格 = 点当前那个危险按钮(确认松力矩 / 恢复力矩)。"""
        if not e.action.keydown or e.action.repeat or not e.key.space:
            return
        if self._engine is None or not self._engine.is_running():
            return
        if self._btn_confirm.visible:
            action = self._confirm
        elif self._btn_restore.visible:
            action = self._restore
        else:
            return
        now = time.monotonic()
        if now - self._space_at > _DOUBLE_TAP_S:
            self._space_at = now
            self._key_hint.set_text(_KEY_HINT_ARMED)
            return
        action()

    # ================================================================== 事件
    def _drain(self) -> None:
        engine = self._engine
        if engine is None:
            return
        for tag, payload in engine.drain():
            handler = self._dispatch.get(tag)
            if handler is not None:
                handler(payload)

    def _on_mode(self, payload: dict) -> None:
        """已连接、力矩尚未变化:按驱动**实际**支持的方式给出指引。"""
        mode = str(payload.get("mode") or "")
        text, dangerous = _MODE_TEXT.get(mode, ("已连接。确认工作区安全后继续。", False))
        self._hint.set_text(text)
        self._hint.style(
            "background:#fee2e2; color:#991b1b; padding:8px; border-radius:4px; font-weight:600;"
            if dangerous
            else "background:#eff6ff; color:#1e40af; padding:8px; border-radius:4px;"
        )
        self._hint.set_visibility(True)
        self._show_gate(dangerous)
        self._status.set_text("" if dangerous else "已断开。")

    def _on_state(self, payload: dict) -> None:
        self._render_joints(payload)
        pose = payload.get("pose")
        if isinstance(pose, dict):
            self._pose_label.set_text(
                "位姿 (x, y, z) mm:" + f"({pose.get('x', 0.0):.1f}, {pose.get('y', 0.0):.1f}, {pose.get('z', 0.0):.1f})"
            )

    def _on_blocked(self, payload: dict) -> None:
        names = "、".join(f"{v['name']} {v['direction']}" for v in payload.get("violations", []))
        self._status.set_text(f"力矩没有恢复:{names} 超出软限位。请把它搬回范围内再点一次。")
        self._status.style("color:#b91c1c;")
        ui.notify(f"这些关节超出软限位，先搬回来:{names}", type="negative")

    def _on_phase(self, payload: dict) -> None:
        if payload.get("phase") == "guiding":
            self._readout.set_visibility(True)
            self._btn_restore.set_visibility(True)
        self._status.set_text(str(payload.get("msg", "")))
        self._status.style("color:inherit;")

    def _on_log(self, payload: dict) -> None:
        self._log_box.push(f"{payload.get('level', '')} {payload.get('msg', '')}")

    def _on_done(self, _payload: dict) -> None:
        self._btn_restore.set_visibility(False)
        self._readout.set_visibility(False)
        self._hint.set_visibility(False)
        self._status.set_text("力矩已恢复，已断开。机械臂停在你摆的位置。")
        self._status.style("color:#15803d;")
        self.refresh()

    def _on_error(self, payload: dict) -> None:
        reason = str(payload.get("reason", ""))
        if payload.get("fatal"):
            self._hint.set_text(
                f"⚠️ 力矩恢复失败，机械臂可能仍处于失力矩状态 —— 请立刻用手托住，并检查电机总线。原因:{reason}"
            )
            self._hint.style("background:#fee2e2; color:#991b1b; padding:8px; border-radius:4px; font-weight:600;")
            self._hint.set_visibility(True)
        else:
            self._blocker.set_text("⚠️ " + reason)
            self._blocker.set_visibility(True)
        self._show_gate(False)
        self._btn_restore.set_visibility(False)
        self._status.set_text("已停止:" + reason)
        self._status.style("color:#b91c1c;")
        self._btn_connect.enable()

    def _on_gap(self, payload: dict) -> None:
        dropped = int(payload.get("dropped", 0))
        self._status.set_text(f"界面事件流有缺口，已丢弃 {dropped} 条旧事件。")

    # ================================================================== 读数渲染
    def _render_joints(self, payload: dict) -> None:
        """首帧按限位建行,之后只改文字 —— 5Hz 重建整张表会让页面一直闪。"""
        limits = payload.get("limits") or []
        if not self._joint_rows and limits:
            self._build_joint_rows(limits)
        joints = payload.get("joints") or []
        violating = {v["name"] for v in payload.get("violations", [])}
        for index, entry in enumerate(limits):
            row = self._joint_rows.get(entry["name"])
            if row is None or index >= len(joints):
                continue
            row["value"].set_text(f"{joints[index]:8.2f}")
            bad = entry["name"] in violating
            row["value"].style("color:#b91c1c; font-weight:700;" if bad else "color:inherit; font-weight:400;")
            row["note"].set_text("超出软限位" if bad else "")

    def _build_joint_rows(self, limits: list[dict]) -> None:
        with self._joint_box:
            for entry in limits:
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.label(entry["name"]).classes("text-sm w-32")
                    value = ui.label("—").classes("font-mono text-sm w-20 text-right")
                    ui.label(f"[{entry['low']:.1f}, {entry['high']:.1f}]").classes("font-mono text-xs text-gray-500")
                    note = ui.label("").classes("text-xs").style("color:#b91c1c;")
                self._joint_rows[entry["name"]] = {"value": value, "note": note}

    def _reset_readout(self) -> None:
        self._joint_box.clear()
        self._joint_rows.clear()
        self._pose_label.set_text("")
        self._readout.set_visibility(False)
        self._btn_restore.set_visibility(False)
        self._hint.set_visibility(False)
        self._show_gate(False)
