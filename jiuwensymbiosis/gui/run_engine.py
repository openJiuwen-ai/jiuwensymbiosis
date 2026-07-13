# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""RunEngine —— 在后台线程里跑一次任务,把运行时事件放进线程安全队列供界面拉取。

整段 ``connect → run_robot_task → disconnect`` 在单个后台 ``threading.Thread`` 内完成
(``run_robot_task`` 内部 ``asyncio.run`` 每次新建事件循环,worker 线程无既有循环,安全)。
``UIBridgeRail`` 与日志 handler 的回调都在该 worker 线程触发,只往 ``queue.Queue`` 里塞
``(event, payload)`` 元组——绝不直接碰界面控件。界面侧用 ``ui.timer`` 周期 ``drain()`` 取事件
并更新 NiceGUI 元素,跨线程只经这一个队列。

本模块**无 Qt / 无 nicegui 依赖**,可独立单测。事件标签:
``run_started`` / ``step_started`` / ``step_finished`` / ``frame`` / ``narration`` /
``safety_event`` / ``log`` / ``run_finished``。``frame`` 的载荷是编码好的 data URI 字符串。

同一时刻只应有一个运行(日志/检测 sidecar 端口是进程级单例),由界面负责串行化。
"""

from __future__ import annotations

import copy
import logging
import queue
import uuid
from collections import deque
from pathlib import Path
from threading import Thread
from typing import Any

from jiuwensymbiosis.agent import ModelSpec, RobotAgentConfig, run_robot_task
from jiuwensymbiosis.gui import imaging
from jiuwensymbiosis.gui.bridge import UIBridgeRail
from jiuwensymbiosis.gui.config_model import ConfigModel
from jiuwensymbiosis.gui.mock_sessions import build_scripted_mock_model
from jiuwensymbiosis.gui.registry import TaskDef, get_body
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["RunEngine", "QueueLogHandler", "default_workspace"]


def default_workspace() -> str:
    """GUI 默认工作区(轨迹/会话落盘处)。"""
    return str(Path.home() / ".jiuwensymbiosis" / "gui_workspace")


class QueueLogHandler(logging.Handler):
    """把 ``jiuwensymbiosis`` 的日志记录塞进事件队列,并留一段尾缓冲供失败诊断。

    直接持有队列(而非引擎)以免跨对象访问受保护成员。
    """

    def __init__(self, events: queue.Queue, level: int = logging.INFO) -> None:
        """绑定事件队列;自带日志尾环形缓冲(默认 400 行)。"""
        super().__init__(level)
        self._events = events
        self._buffer: deque[str] = deque(maxlen=400)

    def emit(self, record: logging.LogRecord) -> None:
        """把一条日志转成 dict 入队并留档到缓冲(日志绝不因界面而抛异常)。"""
        try:
            msg = record.getMessage()
            self._buffer.append(f"{record.levelname} {record.name}: {msg}")
            self._events.put(("log", {"level": record.levelname, "name": record.name, "msg": msg}))
        except Exception:  # 日志 handler 内不能再走日志系统(会递归),交给 logging 内建的错误处理
            self.handleError(record)

    def log_tail(self) -> str:
        """返回最近若干条日志(拼成文本),供 ``diagnose`` 精确判断失败原因。"""
        return "\n".join(self._buffer)


class RunEngine:
    """一次任务运行的后台线程 + 事件队列 + ``UIBridgeRail`` 的 emitter。"""

    def __init__(
        self,
        task: TaskDef,
        config_data: dict[str, Any],
        *,
        mock: bool,
        workspace: str | None = None,
    ) -> None:
        """记录本次运行的任务、配置数据、模拟开关与工作区。"""
        self._task = task
        self._config = ConfigModel.from_dict(config_data)
        self._mock = mock
        self._workspace = workspace or default_workspace()
        self._events: queue.Queue = queue.Queue()
        self._thread: Thread | None = None
        self._stop = False

    # -------------------------------------------------- UIBridgeRail emitter 接口
    def step_started(self, info: dict) -> None:
        self._events.put(("step_started", info))

    def step_finished(self, info: dict) -> None:
        self._events.put(("step_finished", info))

    def frame(self, rgb: Any) -> None:
        try:
            uri = imaging.to_data_uri(rgb)
        except Exception as exc:  # 坏帧不应中断运行
            logger.debug("frame encode failed: %s", exc)
            return
        self._events.put(("frame", uri))

    def narration(self, text: str) -> None:
        self._events.put(("narration", text))

    def safety_event(self, info: dict) -> None:
        self._events.put(("safety_event", info))

    # ------------------------------------------------------------------ 控制
    def start(self) -> None:
        """启动后台线程(幂等:已在运行则忽略)。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = Thread(target=self._run, name="jiuwen-gui-run", daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        """请求停止(下一步开始前生效)。"""
        self._stop = True

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def drain(self) -> list[tuple[str, Any]]:
        """非阻塞取出当前队列里的全部事件,供界面 ``ui.timer`` 周期消费。"""
        events: list[tuple[str, Any]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return events

    # ------------------------------------------------------------------ 运行
    def _run(self) -> None:
        """后台线程主体:构建会话/配置、连接、运行、断开,并把结果入队。"""
        log = get_logger("jiuwensymbiosis")
        handler = QueueLogHandler(self._events)
        log.addHandler(handler)
        try:
            session, agent_cfg, query = self._build()
            self._events.put(
                (
                    "run_started",
                    {"task": self._task.display_name, "body": self._task.body_key, "mock": self._mock, "query": query},
                )
            )
            conv_id = f"gui-{uuid.uuid4().hex[:8]}"
            with session:
                self._emit_initial_frame(session)
                # 连接已完成,接下来 run_robot_task 先做 fast 的唯一云端大模型调用(编译动作序列),
                # 通常要等十几~几十秒。给个明确提示:这段是在等云侧模型响应,不是本地卡死。
                # 第一条执行指令的叙述到达后会自动覆盖。mock 无云端,不提示。
                if not self._mock:
                    self.narration("等待云侧服务响应中…")
                result = run_robot_task(session, query, agent_cfg, conversation_id=conv_id)
            self._events.put(
                (
                    "run_finished",
                    {"ok": True, "result": result, "conversation_id": conv_id, "workspace": self._workspace},
                )
            )
        except Exception as exc:  # 运行失败需回传界面而非崩溃
            logger.exception("GUI 任务运行失败")
            self._events.put(
                (
                    "run_finished",
                    {"ok": False, "error": str(exc), "error_type": type(exc).__name__, "log_tail": handler.log_tail()},
                )
            )
        finally:
            log.removeHandler(handler)

    # ------------------------------------------------------------------ 内部
    def _build(self) -> tuple[Any, RobotAgentConfig, str]:
        """把界面选择组装成 (session, agent_cfg, query)。"""
        body = get_body(self._task.body_key)
        data = self._config.data

        agent_cfg = RobotAgentConfig.from_dict(data.get("agent"))
        agent_cfg.model_spec = ModelSpec(**(data.get("model") or {}))
        agent_cfg.workspace = self._workspace

        if self._mock:
            session = body.build_mock_session()
            # 模拟:只把 LLM 换成脚本化离线模型,其余(mode / 技能 / 安全 / 拍照 /
            # 追踪)全按配置页显示的值运行,使模拟忠实于配置。
            agent_cfg.model = build_scripted_mock_model(
                list(self._task.mock_script), final_text=self._task.mock_final_text
            )
            # fast 需真实 LLM 编译动作序列,模拟(脚本化模型)用不了 → 强制逐步。
            agent_cfg.exec_mode = "agent"
        else:
            session = body.build_real_session(self._real_session_config())

        bridge = UIBridgeRail(self, session, should_stop=lambda: self._stop)
        agent_cfg.extra_rails = list(agent_cfg.extra_rails or []) + [bridge]

        query = self._config.get("env.cfg.prompt") or self._task.default_query
        return session, agent_cfg, str(query)

    def _real_session_config(self) -> dict[str, Any]:
        """真机会话所用配置 = 界面编辑过的完整配置(深拷贝,避免改到界面在用的那份)。

        另把相对 ``calib_path`` 解析成绝对路径:真机会话经 ``from_dict`` 构建,没有文件
        上下文,不像 ``from_yaml`` 会相对 yaml 所在目录解析,否则标定文件按运行目录找不到
        会导致连接失败。
        """
        data = copy.deepcopy(self._config.data)
        env = data.get("env")
        cfg = env.get("cfg") if isinstance(env, dict) else None
        low_level = cfg.get("low_level") if isinstance(cfg, dict) else None
        if isinstance(low_level, dict):
            calib = low_level.get("calib_path")
            if isinstance(calib, str) and calib and not Path(calib).is_absolute():
                resolved = (self._task.config_path().parent / calib).resolve()
                if resolved.exists():
                    low_level["calib_path"] = str(resolved)
        return data

    def _emit_initial_frame(self, session: Any) -> None:
        """连接后先推一帧初始相机画面,让主视觉区不为空。"""
        try:
            rgb = session.env.get_observation().rgb
        except Exception as exc:  # 取帧失败不影响运行
            logger.debug("initial frame capture failed: %s", exc)
            return
        if rgb is not None:
            self.frame(rgb)
