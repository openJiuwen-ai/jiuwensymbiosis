"""Head-camera-grounded conversation with explicit, revocable Agent control permission."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from functools import partial
from typing import Any
from urllib.parse import urlsplit

from jiuwensymbiosis.agent.config import ModelSpec, build_model
from jiuwensymbiosis.simulators.genesis.base_motion import MOZ1_SUPPORTS_LATERAL
from jiuwensymbiosis.simulators.genesis.visual_policy import OBSERVATION_POLICY, visual_history
from jiuwensymbiosis.utils.logging import get_logger

__all__ = ["GenesisChat", "ChatError", "state_snapshot"]

logger = get_logger(__name__)
_PROMPT = """你是 Symbiosis-sim 纯视觉仿真助手，用中文回答。
当前没有连接头部相机或动作工具，只能讨论用户提供的信息，不能执行动作或观察场景。
你没有收到仿真状态、场景名称、坐标、物体位置或图像。需要观察时请说明头部相机尚未连接。
不猜测当前环境，不把历史文字当作当前观测。"""


class ChatError(Exception):
    """A user-facing error containing no provider response or credentials."""


def state_snapshot(status: dict[str, Any], *, actions_enabled: bool = False) -> dict[str, Any]:
    """Operator-only diagnostic snapshot. Never include this in model input."""
    measured = status.get("measured_at")
    age = max(0.0, time.monotonic() - measured) if isinstance(measured, (int, float)) else None
    fresh = status.get("state") == "ready" and age is not None and age <= 2
    snapshot = {
        "source": "Genesis structured state (not vision)",
        "queried_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sample_age_s": round(age, 3) if age is not None else None,
        "fresh": fresh,
        "state": status.get("state", "unavailable"),
        "connection": status.get("connection", "closed"),
        "settings": {
            k: v for k, v in status.get("settings", {}).items() if k in {"width", "height", "fps", "spp", "dt"}
        },
        "robot_actions_available": False,
    }
    if fresh:
        if status.get("base_observation"):
            snapshot["base_observation"] = status["base_observation"]
            snapshot["scene_contacts"] = status.get("scene_contacts")
            snapshot["robot_actions_available"] = actions_enabled
            snapshot["supports_lateral"] = MOZ1_SUPPORTS_LATERAL
        snapshot.update(
            {k: status[k] for k in ("steps", "sim_time", "cycle", "streaming", "object_positions") if k in status}
        )
        scene_keys = {
            "id",
            "background_id",
            "robot_id",
            "spawn_ground_m",
            "spawn_support_verified",
            "collision_meshes",
            "collision_method",
            "mode",
            "collision",
            "reset_interval_steps",
            "visible_meshes",
            "visible_triangles",
            "bounds_m",
            "preview_exposure_ev",
            "lighting",
        }
        snapshot["scene"] = {k: v for k, v in status.get("scene", {}).items() if k in scene_keys}
        snapshot["camera"] = {
            k: v for k, v in status.get("camera", {}).items() if k in {"pos", "lookat", "up", "revision"}
        }
        snapshot["coordinate_frame"] = "world; metres; Z-up"
        if status.get("robot"):
            robot_keys = {
                "name",
                "source_format",
                "links",
                "dofs",
                "joint_names",
                "joint_positions",
                "fixed_base",
                "control_mode",
                "collision",
                "planning_connected",
            }
            snapshot["robot"] = {k: v for k, v in status["robot"].items() if k in robot_keys}
    return deepcopy(snapshot)


class GenesisChat:
    """Read-only by default; explicit action mode reuses the shared robot Agent."""

    def __init__(
        self,
        status: Callable[[], dict[str, Any]],
        *,
        model_factory: Callable | None = None,
        timeout: float = 90,
        preview=None,
    ) -> None:
        self._status = status
        self._model_factory = model_factory or partial(build_model, use_shared_http_client=False)
        self._timeout = timeout
        self._spec: ModelSpec | None = None
        self._history: list[dict[str, str]] = []
        self._pending: asyncio.Task | None = None
        self._generation = 0
        self.preview = preview
        self._actions_enabled = False
        self._action_task: asyncio.Task | None = None
        self._action_cancel: threading.Event | None = None
        self._task_has_motion = False
        self.last_execution: dict | None = None
        self.last_snapshot: dict[str, Any] | None = None
        self.last_vision: dict[str, Any] | None = None

    @property
    def busy(self) -> bool:
        return self._pending is not None or self.action_running

    @property
    def action_running(self) -> bool:
        return self._action_task is not None and not self._action_task.done()

    @property
    def motion_running(self) -> bool:
        return self.action_running and self._task_has_motion

    @property
    def actions_enabled(self) -> bool:
        return self._actions_enabled

    @actions_enabled.setter
    def actions_enabled(self, enabled: bool) -> None:
        """Revocation also reaches the worker thread, including a pending model call."""
        enabled = bool(enabled)
        if enabled == self._actions_enabled:
            return
        if enabled:
            if self.preview is None:
                raise ChatError("当前页面没有连接仿真控制接口。")
            if self.busy:
                raise ChatError("请等待当前回复或任务退出后再开启控制。")
        self._actions_enabled = enabled
        if not enabled:
            self.cancel()
        # Do not carry earlier read-only refusals across a permission change.
        self._history.clear()

    def configure(self, spec: ModelSpec) -> bool:
        """Validate the applied profile and report whether conversation history changed."""
        if self.busy:
            raise ChatError("正在生成回复，请先等待或停止。")
        url = urlsplit(spec.api_base)
        valid_address = url.scheme in {"http", "https"} and bool(url.hostname)
        if not valid_address or url.username or url.password:
            raise ChatError("请在“模型”页配置有效的 HTTP/HTTPS API 基础地址。")
        if not spec.model_name.strip() or not spec.provider.strip():
            raise ChatError("请先在“模型”页选择并应用模型。")
        if spec.provider == "OpenAI" and not spec.api_key.strip():
            raise ChatError("请在“模型”页填写 API Key；本地免鉴权服务可填 EMPTY。")
        changed = spec != self._spec
        if changed:
            self.reset()
            self._spec = deepcopy(spec)
        return changed

    def cancel(self) -> None:
        if self._action_cancel is not None:
            self._action_cancel.set()
        if self._pending is not None:
            self._generation += 1
            self._pending.cancel()

    def reset(self) -> None:
        self.cancel()
        self._generation += 1
        self._history.clear()
        self._spec = None
        self.last_snapshot = None
        self.last_execution = None
        self.last_vision = None

    async def recognize(self, spec: ModelSpec) -> dict[str, Any]:
        """One explicit image-only request, even when motion permission is enabled.

        Do not include state, scene labels or chat history: this observation must
        come from pixels, and cannot dispatch robot actions.
        """
        from jiuwensymbiosis.simulators.genesis.vision import PROMPT, QUERY, image_url, save_result

        self.configure(spec)
        if self.preview is None:
            raise ChatError("当前页面没有连接 Genesis 相机。")
        generation = self._generation
        self._pending = asyncio.current_task()
        self.last_vision = None
        try:
            try:
                frame = await asyncio.to_thread(self.preview.capture_image)
            except (RuntimeError, OSError):
                raise ChatError("相机采样失败：请确认后台就绪、视角稳定，再重新识别。") from None
            if generation != self._generation:
                raise asyncio.CancelledError
            messages = [
                {"role": "system", "content": PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": QUERY},
                        {"type": "image_url", "image_url": {"url": image_url(frame["jpeg"])}},
                    ],
                },
            ]
            started = time.monotonic()
            async with asyncio.timeout(self._timeout):
                reply = await self._model_factory(deepcopy(spec)).invoke(messages, timeout=self._timeout)
            if generation != self._generation:
                raise asyncio.CancelledError
            if getattr(reply, "tool_calls", None):
                raise ChatError("识别请求返回了工具调用；未执行动作，请重试。")
            text = reply.content
            if not isinstance(text, str) or not text.strip():
                raise ChatError("模型未返回识别文本，请检查图片输入支持。")
            if spec.api_key and spec.api_key != "EMPTY":
                text = text.replace(spec.api_key, "[已隐藏]")
            self.last_vision = save_result(
                frame, text[:16000], model=spec.model_name, duration_s=time.monotonic() - started
            )
            return self.last_vision
        except TimeoutError:
            raise ChatError("图像识别超时，请稍后重新识别。") from None
        except (ChatError, asyncio.CancelledError):
            raise
        except Exception as exc:
            logger.warning("Genesis vision failed (%s)", type(exc).__name__)
            raise ChatError("图像识别失败，请检查模型服务及图片输入支持。") from None
        finally:
            self._pending = None

    async def send(self, query: str, spec: ModelSpec, *, on_frame: Callable | None = None) -> str:
        query = query.strip()
        if not query or len(query) > 8000:
            raise ChatError("请输入 1–8000 字的消息。")
        self.configure(spec)
        generation = self._generation
        self._pending = asyncio.current_task()
        self.last_execution = None
        try:
            if self.preview is not None:
                text = await self._send_agent(query, spec, on_frame=on_frame)
                if generation != self._generation:
                    raise asyncio.CancelledError
                return self._remember(query, text, spec)
            self.last_snapshot = state_snapshot(self._status(), actions_enabled=self.actions_enabled)
            messages = [
                {"role": "system", "content": _PROMPT},
                *visual_history(self._history),
                {
                    "role": "user",
                    "content": query,
                },
            ]
            async with asyncio.timeout(self._timeout):
                reply = await self._model_factory(deepcopy(spec)).invoke(messages, timeout=self._timeout)
            if generation != self._generation:
                raise asyncio.CancelledError
            if getattr(reply, "tool_calls", None):
                raise ChatError("模型请求了未开放的工具，本轮未执行任何动作。")
            text = reply.content
            if not isinstance(text, str) or not text.strip():
                raise ChatError("模型未返回文本，请检查模型设置后重试。")
            return self._remember(query, text, spec)
        except TimeoutError:
            raise ChatError("模型响应超时，请检查服务后重试。") from None
        except (ChatError, asyncio.CancelledError):
            raise
        except Exception as exc:
            logger.warning("Genesis chat failed (%s)", type(exc).__name__)
            raise ChatError("模型请求失败，请检查模型配置；看图需要模型同时支持图片输入和工具调用。") from None
        finally:
            self._pending = None

    def _remember(self, query: str, text: str, spec: ModelSpec) -> str:
        if spec.api_key and spec.api_key != "EMPTY":
            text = text.replace(spec.api_key, "[已隐藏]")
        text = text[:16000]
        self._history.extend(
            {"role": role, "content": content, "observation_policy": OBSERVATION_POLICY}
            for role, content in (("user", query), ("assistant", text))
        )
        while len(self._history) > 12 or sum(len(m["content"]) for m in self._history) > 48000:
            del self._history[:2]
        return text

    async def _send_agent(self, query: str, spec: ModelSpec, *, on_frame: Callable | None = None) -> str:
        if self.preview is None:
            raise ChatError("当前页面没有连接机器人控制接口。")
        snapshot = state_snapshot(self._status(), actions_enabled=self.actions_enabled)
        self.last_snapshot = snapshot
        if self.actions_enabled and not snapshot["fresh"]:
            raise ChatError("仿真后台未就绪或状态已过期，请启动后台并等待就绪。")
        if self.actions_enabled and not snapshot["robot_actions_available"]:
            raise ChatError("请先选择 MOZ1，并将“机器人模式”切换为“MOZ1 底盘物理”；姿态检视和背景场景不能移动。")
        from jiuwensymbiosis.simulators.genesis.agent import run_agent_query

        self._action_cancel = threading.Event()
        self._task_has_motion = self.actions_enabled
        cancelled = self._action_cancel
        generation = self._generation
        loop = asyncio.get_running_loop()

        def deliver(frame: dict) -> None:
            if on_frame is not None and not cancelled.is_set() and generation == self._generation:
                on_frame(frame)

        self._action_task = asyncio.create_task(
            asyncio.to_thread(
                run_agent_query,
                self.preview,
                deepcopy(spec),
                query,
                cancelled=cancelled,
                actions_enabled=self.actions_enabled,
                history=deepcopy(self._history),
                on_frame=lambda frame: loop.call_soon_threadsafe(deliver, frame),
            )
        )
        try:
            async with asyncio.timeout(240):
                execution: dict = await asyncio.shield(self._action_task)
                self.last_execution = execution
            self.last_snapshot = state_snapshot(self._status(), actions_enabled=self.actions_enabled)
            return str(execution["reply"]) or "本轮未返回回复，请检查模型的图片输入和工具调用支持。"
        except BaseException:
            self._action_cancel.set()
            # Keep the task referenced: busy remains true until the session releases
            # control. Cancelling an asyncio waiter cannot stop its worker thread.
            self._action_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
            raise
