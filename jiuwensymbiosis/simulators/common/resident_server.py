"""Per-user background owner of the GPU worker, independent of every GUI process."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import signal
import socket
import socketserver
import struct
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jiuwensymbiosis.simulators.common.preview import SimulationPreview
from jiuwensymbiosis.simulators.common.resident_transport import (
    MAX_REQUEST,
    MAX_RESPONSE,
    PROTOCOL,
    receive_message,
    runtime_fingerprint,
    send_message,
)
from jiuwensymbiosis.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)
_CLIENT_ID = re.compile(r"[a-f0-9]{32}\Z")
_CANCELLABLE = {"navigate_relative", "policy_action", "capture_image", "query_scene"}
_METHODS = {
    "start",
    "configure",
    "reset_scene",
    "capture_image",
    "query_scene",
    "camera_command",
    "select",
    "base_command",
    "acquire_motion",
    "release_motion",
    "cancel_motion",
    "motion_state",
    "navigate_relative",
    "policy_action",
}


@dataclass
class ClientLease:
    last_seen: float
    owner: str | None = None
    closed: bool = False
    cleanup_error: str | None = None
    requests: dict[str, threading.Event] = field(default_factory=dict)


class ResidentHost:
    """Serialize client ownership; the existing preview still owns all worker commands."""

    def __init__(
        self, preview: SimulationPreview, *, clock: Callable[[], float] = time.monotonic, lease_s: float = 6.0
    ) -> None:
        self.preview = preview
        self.clock, self.lease_s = clock, lease_s
        self.fingerprint = runtime_fingerprint()
        self.lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self.clients: dict[str, ClientLease] = {}
        self.viewer: str | None = None
        self.closing = False
        self.shutdown_ready = False

    def status(self, client_id: str) -> dict:
        state = self.preview.status()
        with self.lock:
            return {
                **state,
                "resident": {
                    "pid": os.getpid(),
                    "protocol": PROTOCOL,
                    "fingerprint": self.fingerprint,
                    "clients": sum(not client.closed for client in self.clients.values()),
                    "viewer_owned": self.viewer == client_id,
                    "cleanup_errors": [c.cleanup_error for c in self.clients.values() if c.cleanup_error],
                },
            }

    def dispatch(self, message: dict) -> Any:
        if isinstance(message, dict) and message.get("method") in {"start", "configure", "stop"}:
            with self._lifecycle_lock:
                return self._dispatch(message)
        return self._dispatch(message)

    def _dispatch(self, message: dict) -> Any:
        if not isinstance(message, dict):
            raise ValueError("Invalid resident request")
        method, client_id = message.get("method"), message.get("client_id", "")
        request_id, params = message.get("id", ""), message.get("params", {})
        if not _CLIENT_ID.fullmatch(client_id) or not _CLIENT_ID.fullmatch(request_id):
            raise ValueError("Invalid resident request identity")
        if not isinstance(params, dict) or "cancelled" in params:
            raise ValueError("Invalid resident parameters")
        if method == "ping":
            return {"protocol": PROTOCOL, "pid": os.getpid(), "fingerprint": self.fingerprint}
        with self.lock:
            if self.closing:
                raise RuntimeError("仿真后台正在释放资源。")
            if method == "attach":
                if client_id in self.clients:
                    raise ValueError("Client identity already used")
                self.clients[client_id] = ClientLease(self.clock())
                return {"attached": True}
            client = self.clients.get(client_id)
            if client is None or client.closed:
                raise RuntimeError("仿真客户端连接已失效，请重新连接。")
            client.last_seen = self.clock()
            if method == "heartbeat":
                return {"alive": True}
            if method == "cancel_request":
                event = client.requests.get(params.get("request_id", ""))
                if event is not None:
                    event.set()
                return {"cancelled": event is not None}
        if method == "detach":
            return self.detach(client_id)
        if method == "status":
            return self.status(client_id)
        if method == "poll":
            with self.lock:
                frame = self.preview.latest_frame() if self.viewer == client_id else None
            return {"status": self.status(client_id), "frame": frame}
        if method == "stop":
            with self.lock:
                self.closing = True
                for lease in self.clients.values():
                    for event in lease.requests.values():
                        event.set()
            try:
                result = self.preview.stop()
                self.shutdown_ready = True
                return result
            except Exception:
                with self.lock:
                    self.closing = False
                raise
        with self.lock:
            if client.closed or self.closing:
                raise RuntimeError("仿真客户端连接已失效。")
            if method == "set_streaming":
                enabled = params.get("enabled")
                if not isinstance(enabled, bool):
                    raise ValueError("streaming must be boolean")
                if enabled:
                    if self.viewer not in (None, client_id):
                        raise RuntimeError("另一个 GUI 已连接视频，请先断开原连接。")
                    self.preview.set_streaming(True)
                    self.viewer = client_id
                elif self.viewer == client_id:
                    self.preview.set_streaming(False)
                    self.viewer = None
                return None
            if method in {"acknowledge_frame", "request_keyframe"}:
                if self.viewer == client_id:
                    getattr(self.preview, method)(**params)
                return None
            if method not in _METHODS:
                raise ValueError("Unknown resident operation")
            requires_viewer = method in {"camera_command", "select"} or (
                method == "base_command" and any(params.get(key, 0) != 0 for key in ("vx", "vy", "wz"))
            )
            if requires_viewer and self.viewer != client_id:
                raise RuntimeError("请连接当前 GUI 的视频后操作。")
            if method in {"release_motion", "cancel_motion", "motion_state", "navigate_relative", "policy_action"}:
                if client.owner is None or params.get("owner") != client.owner:
                    raise RuntimeError("该客户端没有此运动任务的控制权。")
            if method == "acquire_motion":
                client.owner = self.preview.acquire_motion()
                return client.owner
            if method in {"camera_command", "base_command", "cancel_motion", "motion_state"}:
                return getattr(self.preview, method)(**params)
            event = threading.Event()
            if request_id in client.requests:
                raise ValueError("Duplicate request identity")
            client.requests[request_id] = event
        try:
            if method in _CANCELLABLE:
                params = {**params, "cancelled": event}
            result = getattr(self.preview, method)(**params)
            if method == "release_motion":
                with self.lock:
                    if client.owner == params.get("owner"):
                        client.owner = None
            return result
        finally:
            with self.lock:
                client.requests.pop(request_id, None)

    def detach(self, client_id: str) -> dict:
        """Cancel this client's work; keep unknown stop ownership for explicit recovery."""
        with self.lock:
            client = self.clients.get(client_id)
            if client is None:
                return {"detached": True}
            client.closed = True
            for event in client.requests.values():
                event.set()
            if self.viewer == client_id:
                self.preview.set_streaming(False)
                self.viewer = None
            owner = client.owner
        try:
            if owner is not None:
                self.preview.release_motion(owner)
        except Exception as exc:
            logger.exception("Disconnected Genesis client could not confirm stopping")
            with self.lock:
                client.cleanup_error = str(exc)
            return {"detached": False, "error": str(exc)}
        with self.lock:
            client.owner = None
            self.clients.pop(client_id, None)
        return {"detached": True}

    def expire_clients(self) -> None:
        with self.lock:
            expired = [
                key
                for key, client in self.clients.items()
                if not client.closed and self.clock() - client.last_seen > self.lease_s
            ]
            # Revoke every expired client before waiting for any physical stop.
            for key in expired:
                self.clients[key].closed = True
                for event in self.clients[key].requests.values():
                    event.set()
        for key in expired:
            self.detach(key)


class ResidentServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: Path, host: ResidentHost) -> None:
        self.host = host
        super().__init__(str(path), ResidentHandler)
        path.chmod(0o600)


class ResidentHandler(socketserver.BaseRequestHandler):
    server: ResidentServer

    def handle(self) -> None:
        if sys.platform == "darwin":
            # Darwin has getpeereid(), not Linux SO_PEERCRED. Keep the same
            # same-user boundary; never substitute an unauthenticated socket.
            import ctypes

            uid_value, gid_value = ctypes.c_uint(), ctypes.c_uint()
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.getpeereid(self.request.fileno(), ctypes.byref(uid_value), ctypes.byref(gid_value)) != 0:
                return
            uid = uid_value.value
        else:
            _, uid, _ = struct.unpack("3i", self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if uid != os.getuid():
            return
        try:
            message = receive_message(self.request, limit=MAX_REQUEST, timeout=3, binary=False)
            result = self.server.host.dispatch(message)
            reply = {"ok": True, "result": result}
        except (ValueError, RuntimeError, TimeoutError) as exc:
            reply = {"ok": False, "error": type(exc).__name__, "message": str(exc)[:2000]}
        except Exception:
            logger.exception("Genesis resident request failed")
            reply = {"ok": False, "error": "RuntimeError", "message": "仿真后台请求失败，请查看后台日志。"}
        try:
            send_message(self.request, reply, limit=MAX_RESPONSE)
        except (OSError, ValueError):
            logger.warning("Genesis client disconnected before receiving its result")
        finally:
            if self.server.host.shutdown_ready:
                threading.Thread(target=self.server.shutdown, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.state_dir.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    configure_logging(level="INFO", log_dir=root)
    path = root / "service.sock"
    with (root / "service.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        path.unlink(missing_ok=True)
        host = ResidentHost(SimulationPreview(state_dir=root))
        server = ResidentServer(path, host)
        finished = threading.Event()

        def watch() -> None:
            while not finished.wait(1):
                host.expire_clients()

        def shutdown(signum, frame) -> None:
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        threading.Thread(target=watch, daemon=True).start()
        try:
            server.serve_forever(poll_interval=0.2)
        finally:
            finished.set()
            host.preview.stop()
            server.server_close()
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
