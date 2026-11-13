"""Bounded JSON/binary messages over a user-private Unix socket; never pickle."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import socket
import struct
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROTOCOL = 2
MAX_REQUEST = 64 * 1024
MAX_RESPONSE = 64 * 1024 * 1024


def runtime_fingerprint() -> str:
    """Detect an already running backend from a different installed implementation."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.parent.rglob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _encode(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__binary__": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if hasattr(value, "dtype"):
        import numpy as np

        if isinstance(value, np.ndarray) and value.dtype in (np.uint8, np.float32) and value.nbytes <= 32 * 1024**2:
            return {
                "__binary__": "array",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": base64.b64encode(value.tobytes()).decode("ascii"),
            }
        raise ValueError("Unsupported simulator array payload")
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__binary__" not in value:
        return {key: _decode(item) for key, item in value.items()}
    data = base64.b64decode(value["data"], validate=True)
    if value["__binary__"] == "bytes":
        return data
    if value["__binary__"] != "array" or value.get("dtype") not in {"uint8", "float32"}:
        raise ValueError("Invalid simulator binary response")
    shape = value["shape"]
    if not isinstance(shape, list) or not 1 <= len(shape) <= 3 or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid simulator array shape")
    size = math.prod(shape) * (1 if value["dtype"] == "uint8" else 4)
    if size != len(data) or size > 32 * 1024**2:
        raise ValueError("Simulator array size mismatch")
    import numpy as np

    return np.frombuffer(data, dtype=value["dtype"]).reshape(shape).copy()


def send_message(connection: socket.socket, value: Any, *, limit: int) -> None:
    payload = json.dumps(_encode(value), allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > limit:
        raise ValueError("Simulator message exceeds size limit")
    connection.settimeout(10)
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def receive_message(
    connection: socket.socket,
    *,
    limit: int,
    timeout: float,
    poll: Callable[[], None] | None = None,
    binary: bool = True,
) -> Any:
    deadline = time.monotonic() + timeout
    connection.settimeout(min(timeout, 0.2))

    def read(count: int) -> bytes:
        parts = bytearray()
        while len(parts) < count:
            if time.monotonic() >= deadline:
                raise TimeoutError("Simulator backend response timed out")
            if poll:
                poll()
            try:
                part = connection.recv(count - len(parts))
            except TimeoutError:
                continue
            if not part:
                raise ConnectionError("Simulator backend connection closed")
            parts.extend(part)
        return bytes(parts)

    size = struct.unpack("!I", read(4))[0]
    if size > limit:
        raise ValueError("Simulator message exceeds size limit")
    value = json.loads(read(size))
    return _decode(value) if binary else value
