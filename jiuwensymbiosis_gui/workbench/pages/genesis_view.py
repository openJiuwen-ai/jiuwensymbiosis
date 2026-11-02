"""Workbench mount for Genesis video, measured state and permission-gated chat."""

import asyncio
import json
import uuid
from collections.abc import Callable
from hashlib import sha256
from importlib.resources import files
from typing import Any

import markdown2
from nicegui import ui

from jiuwensymbiosis.agent.config import ModelSpec
from jiuwensymbiosis.simulators.genesis.chat import ChatError, GenesisChat, state_snapshot

__all__ = ["GenesisView"]


class GenesisView:
    """Keep each browser's chat local; the application owns the shared renderer."""

    def __init__(
        self,
        *,
        status: Callable[[], dict[str, Any]],
        model: Callable[[], ModelSpec],
        open_models: Callable[[], None],
        preview=None,
    ) -> None:
        self._status = status
        self._model = model
        self._chat = GenesisChat(status, preview=preview)
        self._request: asyncio.Task | None = None
        self._frame_id = "genesis-" + uuid.uuid4().hex
        self._focused = False
        self._chat_collapsed = False
        self._image_cards: list = []
        css = files("jiuwensymbiosis.simulators.genesis.web").joinpath("assets/workbench.css").read_bytes()
        version = sha256(css).hexdigest()[:16]
        ui.add_head_html(f'<link rel="stylesheet" href="/genesis/assets/workbench.css?v={version}">')
        with ui.element("section").classes("genesis-workspace") as self._workspace:
            with ui.row().classes("genesis-heading"):
                with ui.row().classes("items-center gap-3"):
                    ui.icon("view_in_ar").classes("genesis-mark")
                    with ui.column().classes("gap-0"):
                        ui.label("仿真工作台").classes("genesis-title")
                        ui.label("GENESIS / NYX").classes("genesis-eyebrow")
                with ui.row().classes("items-center gap-2"):
                    self._backend_badge = ui.badge("正在检查后台…").props("outline").classes("genesis-badge")
                    self._robot_stop = (
                        ui.button("停止机器人", icon="stop_circle", on_click=self._stop_robot)
                        .props("flat dense")
                        .classes("genesis-stop")
                    )
                    self._robot_stop.set_visibility(False)
                    ui.button("状态检查", icon="monitor_heart", on_click=self._query_status).props("flat dense")
                    self._collapse_button = (
                        ui.button(icon="chevron_right", on_click=self._toggle_chat)
                        .props("flat dense aria-label=收起或展开对话 aria-expanded=true")
                        .tooltip("收起或展开对话")
                    )
                    self._focus_button = ui.button("专注画面", icon="width_wide", on_click=self._toggle_focus).props(
                        "flat dense"
                    )
            with (
                ui.splitter(reverse=True, value=340, limits=(280, 540))
                .props("unit=px")
                .classes("genesis-stage") as splitter
            ):
                with splitter.before:
                    ui.element("iframe").props(
                        f'id={self._frame_id} src=/genesis/viewer title="Genesis Nyx 实时画面" allow="autoplay; fullscreen"'
                    ).classes("genesis-player")
                with splitter.after:
                    self._build_chat(open_models)
        with ui.dialog() as self._state_dialog, ui.card().classes("genesis-state-dialog"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("仿真状态与查询结果").classes("text-lg font-semibold")
                ui.button(icon="close", on_click=self._state_dialog.close).props(
                    "flat round dense aria-label=关闭状态检查"
                )
            with ui.row().classes("gap-2"):
                ui.button("查询状态", icon="refresh", on_click=self._query_status).props("flat dense")
                ui.button("查询物体位置", icon="view_in_ar", on_click=self._query_objects).props("flat dense")
            ui.label("查询无需模型；位置单位米、世界坐标、Z 向上。仅展示本次采样可用的状态。").classes(
                "text-xs text-gray-500"
            )
            self._query_label = ui.label("仅供人工诊断：此处状态快照不会发送给 Agent。").classes("text-sm")
            self._state_summary = ui.column().classes("genesis-state-summary")
            with ui.expansion("原始数据", icon="data_object").classes("w-full"):
                self._result = (
                    ui.code("{}", language="json").classes("w-full").style("max-height: 40vh; overflow: auto")
                )
        ui.timer(1, self._refresh_badge)
        ui.context.client.on_disconnect(self._cancel)
        with ui.dialog() as self._image_dialog, ui.card().classes("w-full max-w-4xl"):
            self._image_title = ui.label()
            self._image_full = ui.image().props("fit=contain").classes("w-full")
            ui.button("关闭图片", on_click=self._image_dialog.close).props("flat")

    def _build_chat(self, open_models: Callable[[], None]) -> None:
        with ui.element("aside").classes("genesis-chat"):
            with ui.column().classes("genesis-chat-header"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("forum", size="18px")
                        ui.label("Symbiosis-sim").classes("font-semibold")
                        self._mode_badge = ui.badge("只读").props("outline").classes("genesis-badge")
                    self._clear_button = ui.button(icon="delete_sweep", on_click=self._clear).props(
                        "flat round dense size=sm aria-label=清空对话"
                    )
                    self._clear_button.tooltip("清空对话")
                with ui.row().classes("w-full items-center justify-between gap-1"):
                    self._model_label = ui.label("使用统一模型配置").classes("genesis-model-label")
                    ui.button(icon="tune", on_click=open_models).props(
                        "flat round dense size=sm aria-label=模型设置"
                    ).tooltip("模型设置")
                ui.label("纯视觉 · 仅头部相机 + 动作反馈").classes("genesis-muted")
                self._action_switch = ui.switch("允许 Agent 控制仿真", on_change=self._change_mode).props("dense")
                self._action_switch.set_enabled(self._chat.preview is not None)
                self._mode_help = ui.label("已关闭：仅用头部相机看图，不执行运动。").classes("genesis-muted")
            with ui.scroll_area().classes("genesis-messages") as self._scroll:
                self._messages = ui.column().classes("w-full gap-3 p-3")
            self._welcome()
            with ui.column().classes("genesis-composer"):
                self._chat_status = ui.label("纯视觉 · 仅头部相机图像和动作反馈；不提供场景真值").classes(
                    "genesis-muted"
                )
                self._input = (
                    ui.textarea(placeholder="描述头部画面，或输入导航目标…")
                    .props("outlined rows=2 maxlength=8000 aria-label=仿真对话输入")
                    .classes("w-full")
                )
                self._input.on("keydown.enter.exact.prevent", self._send)
                with ui.row().classes("w-full items-center justify-between gap-1"):
                    ui.label("Enter 发送 · Shift + Enter 换行").classes("genesis-muted")
                    self._send_button = ui.button("发送", icon="arrow_upward", on_click=self._send).props(
                        "unelevated dense"
                    )
                    self._cancel_button = ui.button("停止回复", on_click=self._cancel).props("outline dense")
                    self._cancel_button.disable()
                    self._cancel_button.set_visibility(False)

    def _welcome(self) -> None:
        with self._messages:
            with ui.column().classes("genesis-welcome"):
                ui.icon("chat_bubble_outline", size="24px").classes("genesis-welcome-icon")
                ui.label("从头部相机开始").classes("font-medium")
                ui.label("让 Agent 用头部相机观察；开启控制后可发送导航目标。").classes("genesis-muted")
                for query in ("用头部相机看看前方有什么？", "头部画面中哪里可以通行？", "用头部相机找一找门或沙发。"):
                    ui.button(query, on_click=lambda query=query: self._input.set_value(query)).props(
                        "flat no-caps dense"
                    ).classes("genesis-suggestion")

    def _change_mode(self, event) -> None:
        try:
            self._chat.actions_enabled = bool(event.value)
        except ChatError as exc:
            self._action_switch.set_value(self._chat.actions_enabled)
            self._chat_status.set_text(str(exc))
            return
        if not event.value and self._request is not None:
            self._cancel()
        self._mode_badge.set_text("可控制" if event.value else "只读")
        self._cancel_button.set_text("停止任务" if event.value else "停止回复")
        self._input.props(
            'placeholder="输入控制指令，例如：向左移动 20 厘米…"'
            if event.value
            else 'placeholder="用头部相机观察前方…"'
        )
        self._refresh_badge()

    def _toggle_chat(self) -> None:
        """Hide the panel without unmounting chat or the video session."""
        self._chat_collapsed = not self._chat_collapsed
        self._workspace.classes(
            add="genesis-chat-collapsed" if self._chat_collapsed else "",
            remove="" if self._chat_collapsed else "genesis-chat-collapsed",
        )
        self._collapse_button.props(f"aria-expanded={str(not self._chat_collapsed).lower()}")
        self._collapse_button.props(f"icon={'chevron_left' if self._chat_collapsed else 'chevron_right'}")

    def _toggle_focus(self) -> None:
        """Change only presentation; keep the iframe and its WebRTC session mounted."""
        self._focused = not self._focused
        self._workspace.classes(
            add="genesis-focus" if self._focused else "", remove="" if self._focused else "genesis-focus"
        )
        self._focus_button.set_text("退出专注" if self._focused else "专注画面")
        self._focus_button.props(f"aria-pressed={str(self._focused).lower()}")

    def _refresh_badge(self) -> None:
        if self._request is None:
            self._set_pending(self._chat.busy)
        status = self._read_status()
        snapshot = state_snapshot(status, actions_enabled=self._chat.actions_enabled)
        if not self._chat.actions_enabled:
            help_text = "已关闭：仅用头部相机看图，不执行运动。"
            if self._chat.action_running:
                help_text += "当前任务尚未退出。"
        elif not snapshot["fresh"]:
            help_text = "已开启：等待新鲜的仿真状态；请检查后台连接与运行状态。"
        elif not snapshot["robot_actions_available"]:
            help_text = "已开启：请选择 MOZ1，并切换为“MOZ1 底盘物理”模式。"
        else:
            help_text = "已开启：可控制前后、左右横移和旋转；关闭将取消当前任务。"
        self._mode_help.set_text(help_text)
        self._robot_stop.set_visibility(
            (status.get("robot") or {}).get("control_mode") == "wheel_velocity_physics" or self._chat.motion_running
        )
        self._robot_stop.set_enabled(self._chat.preview is not None)
        state = snapshot["state"]
        label = {
            "ready": "后台就绪",
            "starting": "后台启动中",
            "reconfiguring": "后台切换中",
            "stopped": "后台未启动",
            "failed": "后台异常",
            "unavailable": "服务不可达 · 状态未知",
        }.get(state, "后台不可用")
        if state == "ready" and not snapshot["fresh"]:
            label = "状态采样已过期"
        self._backend_badge.set_text(label)
        try:
            self._model_label.set_text(f"模型：{self._model().model_name}")
        except (RuntimeError, ValueError):
            self._model_label.set_text("模型设置尚未就绪，请打开“模型”页。")

    def _read_status(self) -> dict[str, Any]:
        try:
            return self._status()
        except (RuntimeError, OSError):
            return {"state": "unavailable"}

    async def _stop_robot(self) -> None:
        """Cancel Agent ownership and request zero wheel speed; do not claim a measured stop."""
        status = self._read_status()
        mobile = (status.get("robot") or {}).get("control_mode") == "wheel_velocity_physics"
        if self._chat.preview is None or not (mobile or self._chat.motion_running):
            self._chat_status.set_text("当前没有可停止的机器人控制；请检查仿真状态。")
            return
        self._cancel()
        try:
            await asyncio.to_thread(
                self._chat.preview.base_command,
                vx=0,
                vy=0,
                wz=0,
                scene_instance_id=status.get("scene", {}).get("instance_id", ""),
                config_revision=status.get("config_revision", -1),
            )
        except (RuntimeError, ValueError, OSError) as exc:
            self._chat_status.set_text(f"停止请求未确认：{exc}")
        else:
            self._chat_status.set_text("已请求停止机器人，等待底盘制动与物理读回。")

    def _show_snapshot(self, snapshot: dict, *, objects_only: bool = False) -> None:
        fresh = "新鲜" if snapshot["fresh"] else "不可用或过期，未提供当前物体坐标"
        self._query_label.set_text(f"查询：{snapshot['queried_at']} · 发送/查询时状态{fresh}")
        self._state_summary.clear()
        settings = snapshot.get("settings", {})
        scene = snapshot.get("scene", {})
        mode = {
            "static_background": "静态视觉背景",
            "robot_preview": "机器人姿态检视",
            "robot_mobile": "机器人底盘物理",
        }.get(scene.get("mode"), "动态物理场景" if scene else "暂无可用场景状态")
        if scene.get("background_id"):
            mode = "室内场景与机器人"
        collision = scene.get("collision")
        rows = {
            "场景模式": mode,
            "场景碰撞": "已启用" if collision is True else "未启用" if collision is False else "未知",
            "渲染设置": (
                f"{settings['width']} × {settings['height']} · {settings.get('spp', '—')} 采样"
                if settings.get("width") and settings.get("height")
                else "暂无配置"
            ),
            "FPS 上限": str(settings.get("fps", "—")),
        }
        if snapshot["fresh"] and "object_positions" in snapshot:
            rows["动态物体读回"] = f"{len(snapshot['object_positions'])} 个；不包含全部静态家具"
        with self._state_summary:
            for title, value in rows.items():
                with ui.column().classes("genesis-state-item"):
                    ui.label(title).classes("genesis-muted")
                    ui.label(value)
        data = snapshot
        if objects_only:
            data = {
                k: v
                for k, v in snapshot.items()
                if k in {"queried_at", "sample_age_s", "fresh", "state", "coordinate_frame", "object_positions"}
            }
        self._result.set_content(json.dumps(data, ensure_ascii=False, indent=2))

    def _query_status(self) -> None:
        self._show_snapshot(state_snapshot(self._read_status(), actions_enabled=self._chat.actions_enabled))
        self._state_dialog.open()

    def _query_objects(self) -> None:
        self._show_snapshot(state_snapshot(self._read_status()), objects_only=True)
        self._state_dialog.open()

    def _append(self, text: str, *, sent: bool = False) -> None:
        with self._messages:
            if sent:
                ui.chat_message(text, name="你", sent=True).classes("w-full")
            else:
                with ui.chat_message(name="Symbiosis-sim").classes("w-full"):
                    rendered = str(
                        markdown2.markdown(text, safe_mode="escape", extras=["tables", "fenced-code-blocks"])
                    )
                    rendered = rendered.replace("<table>", '<div class="table-scroll"><table>').replace(
                        "</table>", "</table></div>"
                    )
                    ui.html(rendered).classes("genesis-reply w-full")
        self._scroll.scroll_to(percent=1)

    def _set_pending(self, pending: bool) -> None:
        pending = pending or self._chat.busy
        for control in (self._input, self._send_button, self._clear_button):
            control.set_enabled(not pending)
        self._action_switch.set_enabled(self._chat.preview is not None and (self._chat.actions_enabled or not pending))
        self._cancel_button.set_enabled(pending)
        self._cancel_button.set_visibility(pending)
        self._send_button.set_visibility(not pending)

    def _append_frame(self, frame: dict) -> None:
        caption = f"{frame['camera_name']} · 与本轮模型输入相同的头部图像"

        def expand() -> None:
            self._image_title.set_text(caption)
            self._image_full.set_source(frame["image_url"])
            self._image_dialog.open()

        with self._messages:
            with ui.chat_message(name="相机观测").classes("w-full") as card:
                ui.label(caption).classes("genesis-muted")
                ui.image(frame["image_url"]).props(f'alt="{frame["camera_name"]} 采样图像"').classes(
                    "w-full cursor-zoom-in"
                ).on("click", expand)
                ui.label("已加入本轮模型输入 · 点击放大").classes("genesis-muted")
        self._image_cards.append(card)
        if len(self._image_cards) > 12:
            self._image_cards.pop(0).delete()
        self._chat_status.set_text("已获取相机图像，模型正在理解…")
        self._scroll.scroll_to(percent=1)

    async def _send(self) -> None:
        query = (self._input.value or "").strip()
        if self._request is not None or not query:
            return
        try:
            spec = self._model()
            changed = self._chat.configure(spec)
        except (ChatError, RuntimeError, ValueError) as exc:
            self._chat_status.set_text(str(exc) if isinstance(exc, ChatError) else "模型设置尚未就绪，请检查“模型”页。")
            return
        if changed:
            self._messages.clear()
            self._image_cards.clear()
            self._image_full.set_source("")
        self._input.set_value("")
        self._append(query, sent=True)
        action_mode = self._chat.actions_enabled
        self._set_pending(True)
        self._chat_status.set_text("Agent 正在处理任务，可按需调用相机…")
        self._request = asyncio.create_task(self._chat.send(query, spec, on_frame=self._append_frame))
        try:
            reply = await self._request
            self._append(reply)
            self._chat_status.set_text(
                "任务返回 · 已记录动作与相机观测"
                if action_mode
                else "回复完成 · 本轮相机图片已在对话中展示"
                if (self._chat.last_execution or {}).get("frames")
                else "回复完成 · 本轮未获取新图像"
            )
        except asyncio.CancelledError:
            self._chat_status.set_text(
                "已请求取消动作，等待底盘制动与任务退出。" if action_mode else "回复已停止，仿真继续运行。"
            )
        except ChatError as exc:
            self._chat_status.set_text(str(exc))
        finally:
            if self._chat.last_snapshot is not None:
                self._show_snapshot(self._chat.last_snapshot)
            self._request = None
            self._set_pending(False)

    def _cancel(self) -> None:
        self._chat.cancel()
        if self._request is not None:
            self._request.cancel()

    def _clear(self) -> None:
        if self._request is not None:
            return
        self._chat.reset()
        self._messages.clear()
        self._image_cards.clear()
        self._image_dialog.close()
        self._image_full.set_source("")
        self._welcome()
        self._chat_status.set_text("对话已清空。")

    def deactivate(self) -> None:
        """Disconnect video while keeping the application's physics scene alive."""
        ui.run_javascript(
            f"document.getElementById('{self._frame_id}')?.contentWindow"
            "?.postMessage({type:'genesis-disconnect'}, location.origin)"
        )
