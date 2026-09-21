# coding: utf-8
"""Workbench request adaptation and rendering of shared runtime facts."""

from __future__ import annotations

import copy
import logging
import queue
import traceback
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from jiuwensymbiosis.runtime import decode_frame
from jiuwensymbiosis.runtime import default_workspace as runtime_default_workspace
from jiuwensymbiosis.utils.logging import get_logger
from jiuwensymbiosis_gui.workbench import humanize, imaging
from jiuwensymbiosis_gui.workbench.app_host import AppHost
from jiuwensymbiosis_gui.workbench.config_model import ConfigModel
from jiuwensymbiosis_gui.workbench.registry import TaskDef, get_body

logger = get_logger(__name__)

__all__ = ["RunEngine", "QueueLogHandler", "default_workspace"]


def default_workspace() -> str:
    """GUI 默认工作区(轨迹/会话落盘处)。"""
    return str(runtime_default_workspace())


def strip_vision_services(config_data: dict[str, Any]) -> dict[str, Any]:
    """返回去掉全部视觉服务的配置副本(供「禁用视觉服务」开关用)。

    剥掉顶层 ``api_servers``(检测器 sidecar 不再 spawn)与 ``env.cfg.low_level.camera_serial``
    (env 不再声明 ``vision.*`` 能力、相机不打开)。本体无关:piper / so101 都是这套键。
    深拷贝,不改界面在用的那份配置。
    """
    data = copy.deepcopy(config_data)
    data.pop("api_servers", None)
    env = data.get("env")
    cfg = env.get("cfg") if isinstance(env, dict) else None
    low_level = cfg.get("low_level") if isinstance(cfg, dict) else None
    if isinstance(low_level, dict):
        low_level.pop("camera_serial", None)
    data.pop("camera_serial", None)  # 兼容极少数把 camera_serial 放平铺顶层的配置
    return data


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
            if record.exc_info:
                # 带上 traceback,否则 GUI 的 log_tail / 诊断只看得到消息、看不到堆栈,
                # 异常就会退化成 KeyError('object') 这种"意义不明"的裸信息。
                msg = f"{msg}\n{''.join(traceback.format_exception(*record.exc_info)).rstrip()}"
            self._buffer.append(f"{record.levelname} {record.name}: {msg}")
            self._events.put(("log", {"level": record.levelname, "name": record.name, "msg": msg}))
        except Exception:  # 日志 handler 内不能再走日志系统(会递归),交给 logging 内建的错误处理
            self.handleError(record)

    def log_tail(self) -> str:
        """返回最近若干条日志(拼成文本),供 ``diagnose`` 精确判断失败原因。"""
        return "\n".join(self._buffer)


class RunEngine:
    """Per-client workbench adapter; Runtime owns task execution and cleanup."""

    def __init__(
        self,
        task: TaskDef,
        config_data: dict[str, Any],
        *,
        workspace=None,
        body_key: str,
        config_source=None,
        host=None,
        runtime=None,
    ):
        self._task = task
        self._body_key = body_key
        self._config = ConfigModel.from_dict(copy.deepcopy(config_data))
        self._workspace = workspace or default_workspace()
        self._source = Path(config_source) if config_source else get_body(body_key).config_path()
        self._host = host or AppHost()
        self._runtime = runtime or self._host.runtime(self._workspace)
        self._job_id = None
        self._started = False
        self._start_failure = None
        self._binding = None
        self._cursor = 0
        self._request_id = uuid.uuid4().hex
        self._local_events: queue.Queue = queue.Queue()
        self._return_operation = None

    @property
    def body_key(self):
        return self._body_key

    @property
    def config_source(self):
        return self._source.expanduser().resolve()

    @property
    def task_key(self):
        return str(self._task.key)

    @property
    def job_id(self):
        return self._job_id

    def subscribe(self):
        """Another page gets its own cursor for the same accepted job."""
        subscriber: RunEngine = copy.copy(self)
        subscriber._cursor = 0
        subscriber._local_events = queue.Queue()
        if self._start_failure is not None:
            subscriber._local_events.put(("run_finished", self._start_failure))
        return subscriber

    def rerun_with(self, config_data: dict[str, Any], *, config_source: str | Path) -> RunEngine:
        """重跑时一起替换配置及其来源，保持相对路径的解析基准一致。"""
        return RunEngine(
            self._task,
            copy.deepcopy(config_data),
            workspace=self._workspace,
            body_key=self._body_key,
            config_source=config_source,
            host=self._host,
            runtime=self._runtime,
        )

    def start(self):
        if self._started:
            return
        self._started = True
        try:
            self._binding = self._runtime.prepare_binding(self._source, config_snapshot=self._real_session_config())
            query = self._config.get("env.cfg.prompt") or self._task.default_query
            job = self._runtime.submit_task(self._binding.binding_id, self._request_id, str(query))
            self._job_id = job["job_id"]
        except Exception as exc:
            self._start_failure = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self._local_events.put(("run_finished", self._start_failure))

    def request_stop(self):
        if self._job_id:
            self._runtime.cancel(self._job_id)

    def is_running(self):
        if self._return_operation and self._return_operation.is_running():
            return True
        return self._job_id is not None and not self._runtime.get_job(self._job_id).get("cleanup", {}).get(
            "released", False
        )

    def join(self, timeout=None):
        if self._job_id:
            self._runtime.wait_for_job(self._job_id, timeout)
        if self._return_operation:
            self._return_operation.join(timeout)

    def _real_session_config(self):
        data = self._config.data
        if self._config.get("gui.disable_vision"):
            data = strip_vision_services(data)
        data = copy.deepcopy(data)
        adapter = get_body(self._body_key).adapter
        if data.get("adapter", adapter) != adapter:
            raise ValueError("配置中的 adapter 与当前选择的本体不一致，请重新选择本体和配置。")
        data.setdefault("adapter", adapter)
        return data

    def start_return_to(self, joints):
        if self.is_running() or self._job_id is None:
            return
        pose = self._runtime.get_job(self._job_id).get("start_pose")
        if not pose or pose["binding_id"] != self._binding.binding_id or list(joints) != pose["joints"]:
            raise ValueError("return target must match the original job binding and observation")
        binding = self._binding.without_sidecars()
        self._return_operation = self._host.return_to(
            self._runtime, binding, joints, lambda kind, data: self._local_events.put((kind, data))
        )

    def drain(self):
        output = []
        while True:
            try:
                output.append(self._local_events.get_nowait())
            except queue.Empty:
                break
        if self._job_id is None:
            return output
        batch = self._runtime.read_events(self._job_id, self._cursor, limit=500)
        self._cursor = batch["next_seq"]
        snapshot = None
        if batch["gap"]:
            snapshot = self._runtime.get_job(self._job_id)
            output.append(
                ("log", {"level": "WARNING", "name": "runtime", "msg": "较早事件已超出保留窗口；当前任务状态已保留。"})
            )
            if snapshot.get("start_pose"):
                output.append(("start_pose", {**snapshot["start_pose"], "body": self._body_key}))
        for event in batch["events"]:
            kind, data = event["kind"], event["data"]
            if kind == "run_started":
                data = {**data, "task": self._task.display_name, "body": self._body_key}
                output.append(("narration", "等待云侧服务响应中…"))
            elif kind in {"step_started", "step_finished"}:
                data = {**data, "label": humanize.friendly_label(data.get("tool", ""), data.get("params", {}))}
                if kind == "step_started":
                    output.append(("narration", humanize.narration(data.get("tool", ""), data.get("params", {}))))
            elif kind in {"frame", "step_frame"}:
                reference = data if kind == "frame" else data["artifact"]
                try:
                    uri = imaging.to_data_uri(decode_frame(self._runtime.read_artifact(reference)))
                except (OSError, ValueError) as exc:
                    logger.debug("preview unavailable: %s", exc)
                    continue
                data = uri if kind == "frame" else {"index": data["index"], "uri": uri}
            elif kind == "start_pose":
                data = {**data, "body": self._body_key}
            elif kind == "blocked":
                output.append(("narration", "硬件收尾尚未确认；资源继续占用，请检查原始日志。"))
            elif (
                kind == "run_finished"
                and isinstance(data.get("result"), dict)
                and data["result"].get("result_type") == "stopped"
            ):
                data = {**data, "result": {**data["result"], "output": "用户已停止运行"}}
            output.append((kind, data))
        if snapshot is not None:
            if snapshot.get("cleanup", {}).get("released") and snapshot.get("result") is not None:
                output.append(("run_finished", snapshot["result"]))
            elif snapshot.get("phase") == "blocked":
                output.append(("narration", "硬件收尾尚未确认；资源继续占用，请检查原始日志。"))
        return output
