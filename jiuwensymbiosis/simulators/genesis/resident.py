"""A GUI lease on a per-user simulator service; detaching keeps the world alive."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from jiuwensymbiosis.simulators.genesis.preview import _worker_environment
from jiuwensymbiosis.simulators.genesis.resident_transport import (
    MAX_REQUEST,
    MAX_RESPONSE,
    PROTOCOL,
    receive_message,
    runtime_fingerprint,
    send_message,
)
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)


class ResidentPreview:
    """Attach explicitly; status never starts a service or replaces its selected world."""

    def __init__(self, *, state_dir: Path | None = None) -> None:
        self.state_dir = (state_dir or Path.home() / ".jiuwensymbiosis" / "genesis-preview").resolve()
        self.path = self.state_dir / "service.sock"
        self._client_id = uuid.uuid4().hex
        self._attached = False
        self._lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._state: dict[str, Any] = {"state": "stopped", "message": "Genesis 后台未连接"}
        self._fingerprint = runtime_fingerprint()

    def _rpc(self, method: str, *, timeout: float = 5, cancelled: threading.Event | None = None, **params: Any) -> Any:
        if cancelled is not None and cancelled.is_set():
            raise RuntimeError("任务已取消。")
        if method not in {"ping", "attach", "detach", "stop"} and not self._attached:
            raise RuntimeError("Genesis 客户端连接已失效，请重新连接。")
        request_id = uuid.uuid4().hex
        sent_cancel = False

        def poll() -> None:
            nonlocal sent_cancel
            if cancelled is not None and cancelled.is_set() and not sent_cancel:
                reply = self._object("cancel_request", request_id=request_id)
                sent_cancel = reply.get("cancelled") is True

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(3)
                connection.connect(str(self.path))
                send_message(
                    connection,
                    {"method": method, "client_id": self._client_id, "id": request_id, "params": params},
                    limit=MAX_REQUEST,
                )
                reply = receive_message(connection, limit=MAX_RESPONSE, timeout=timeout, poll=poll)
        except OSError as exc:
            raise RuntimeError("Genesis 后台连接中断，请重新连接；运行中的任务不会自动重试。") from exc
        if not reply.get("ok"):
            error = {"ValueError": ValueError, "TimeoutError": TimeoutError}.get(reply.get("error"), RuntimeError)
            raise error(reply.get("message", "Genesis 后台请求失败"))
        return reply["result"]

    def _object(self, method: str, **params: Any) -> dict:
        result = self._rpc(method, **params)
        if not isinstance(result, dict):
            raise RuntimeError("Genesis 后台返回了无效的对象响应。")
        return result

    def _ping(self) -> dict | None:
        try:
            return self._object("ping", timeout=1)
        except (RuntimeError, TimeoutError):
            return None

    def _ensure_service(self) -> None:
        if sys.platform != "linux":
            raise RuntimeError("Genesis 后台目前需要 Linux。")
        if len(os.fsencode(self.path)) >= 108:
            raise ValueError("Genesis state directory is too long for a Unix socket")
        import fcntl

        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.state_dir / "launch.lock").open("a") as launch:
            fcntl.flock(launch, fcntl.LOCK_EX)
            health = self._ping()
            if health is None:
                # An unresponsive live owner is not a stale socket. Never spawn a
                # duplicate GPU owner or terminate a process by a stored PID.
                with (self.state_dir / "service.lock").open("a") as ownership:
                    try:
                        fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise RuntimeError("Genesis 后台仍持有资源，但未响应。请检查后台日志。") from exc
                with (self.state_dir / "resident.log").open("ab") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "jiuwensymbiosis.simulators.genesis.resident_server",
                            "--state-dir",
                            str(self.state_dir),
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        cwd=self.state_dir,
                        env=_worker_environment(),
                        start_new_session=True,
                    )
                deadline = time.monotonic() + 15
                while health is None and time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Genesis 后台启动失败，请检查 resident.log。")
                    time.sleep(0.1)
                    health = self._ping()
                if health is None:
                    raise TimeoutError("Genesis 后台未及时响应，请检查 resident.log。")
            if health.get("protocol") != PROTOCOL or health.get("fingerprint") != self._fingerprint:
                raise RuntimeError("Genesis 后台版本与当前 GUI 不一致，请释放后台资源后重启。")

    def _attach(self, *, launch: bool = False) -> None:
        with self._lock:
            if self._attached:
                return
            if launch:
                self._ensure_service()
            elif self._ping() is None:
                raise RuntimeError("Genesis 后台未运行，请先启动。")
            self._client_id = uuid.uuid4().hex
            self._rpc("attach")
            self._attached = True
            self._heartbeat_stop = threading.Event()
            self._heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._heartbeat_thread.start()

    def _heartbeat(self) -> None:
        stop = self._heartbeat_stop
        while not stop.wait(1):
            try:
                self._rpc("heartbeat", timeout=2)
            except (RuntimeError, TimeoutError) as exc:
                with self._lock:
                    if self._heartbeat_stop is stop and not stop.is_set():
                        self._attached = False
                        self._state = {"state": "failed", "message": str(exc)}
                return  # Lost ownership is terminal for existing work.

    def status(self) -> dict:
        if not self._attached:
            return self._state.copy()
        try:
            self._state = self._object("status")
        except (RuntimeError, TimeoutError) as exc:
            self._state = {"state": "failed", "message": str(exc)}
        return self._state.copy()

    def start(self) -> dict:
        if self._attached and (self._heartbeat_thread is None or not self._heartbeat_thread.is_alive()):
            self._heartbeat_stop.set()
            self._attached = False
        try:
            self._attach(launch=True)
            return self._object("start", timeout=15)
        except (RuntimeError, TimeoutError, ValueError) as exc:
            self._state = {"state": "failed", "message": str(exc)}
            raise

    def close(self) -> dict:
        """Detach this GUI, cancelling its work without releasing the GPU world."""
        with self._lock:
            if not self._attached:
                return {"detached": True}
            result = self._object("detach", timeout=20)
            self._heartbeat_stop.set()
            self._attached = False
            if not result["detached"]:
                raise RuntimeError(result["error"])
            return result

    def stop(self) -> dict:
        """Explicit user operation: release the shared backend and all GPU resources."""
        self._attach()  # Also permits releasing an incompatible older service.
        result = self._object("stop", timeout=20)
        self._heartbeat_stop.set()
        self._attached = False
        self._state = result
        return result

    def configure(self, **options: Any) -> dict:
        self._attach(launch=True)
        return self._object("configure", **options)

    def reset_scene(self) -> dict:
        return self._object("reset_scene", timeout=20)

    def latest_frame(self) -> dict | None:
        if not self._attached:
            return None
        response = self._object("poll")
        self._state = response["status"]
        frame = response["frame"]
        if frame is not None and not isinstance(frame, dict):
            raise RuntimeError("Genesis 后台返回了无效的视频响应。")
        return frame

    def acknowledge_frame(self, sequence: int) -> None:
        self._rpc("acknowledge_frame", sequence=sequence)

    def request_keyframe(self) -> None:
        self._rpc("request_keyframe")

    def set_streaming(self, enabled: bool) -> None:
        if self._attached:
            self._rpc("set_streaming", enabled=enabled)

    def camera_command(self, action: str, **values: float) -> dict:
        return self._object("camera_command", action=action, **values)

    def select(
        self, *, action: str, u: float, v: float, scene_instance_id: str, config_revision: int, camera_revision: int
    ) -> dict:
        return self._object(
            "select",
            timeout=12,
            action=action,
            u=u,
            v=v,
            scene_instance_id=scene_instance_id,
            config_revision=config_revision,
            camera_revision=camera_revision,
        )

    def base_command(self, **values: Any) -> dict:
        return self._object("base_command", **values)

    def capture_image(
        self, camera_name: str = "observer", *, cancelled: threading.Event | None = None, **options: Any
    ) -> dict:
        return self._object("capture_image", camera_name=camera_name, timeout=15, cancelled=cancelled, **options)

    def query_scene(self, operation: str, params: dict, *, cancelled: threading.Event | None = None) -> dict:
        return self._object("query_scene", operation=operation, params=params, timeout=65, cancelled=cancelled)

    def acquire_motion(self) -> str:
        owner = self._rpc("acquire_motion")
        if not isinstance(owner, str):
            raise RuntimeError("Genesis 后台返回了无效的控制权响应。")
        return owner

    def release_motion(self, owner: str) -> None:
        self._rpc("release_motion", owner=owner, timeout=15)

    def cancel_motion(self, owner: str, request_id: str | None = None) -> None:
        self._rpc("cancel_motion", owner=owner, request_id=request_id)

    def motion_state(self, owner: str) -> dict:
        return self._object("motion_state", owner=owner)

    def navigate_relative(
        self, owner: str, dx: float, dy: float, dyaw: float, *, cancelled: threading.Event | None = None
    ) -> dict:
        return self._object("navigate_relative", owner=owner, dx=dx, dy=dy, dyaw=dyaw, timeout=110, cancelled=cancelled)

    def move_arm(
        self, owner: str, *, targets=None, poses=None, group=None, speed=1.0, effector=None, opened=None, cancelled=None
    ) -> dict:
        return self._object(
            "move_arm",
            owner=owner,
            targets=targets,
            poses=poses,
            group=group,
            speed=speed,
            effector=effector,
            opened=opened,
            timeout=190,
            cancelled=cancelled,
        )
