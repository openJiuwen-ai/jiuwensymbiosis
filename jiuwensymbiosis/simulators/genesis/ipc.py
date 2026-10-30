"""Linux mmap mailbox: bounded latest frame/control, crash-safe OS file locks.

This module is also imported by the standalone worker without package initialization.
It must remain standard-library-only and must never unpickle peer input.
"""

from __future__ import annotations

import contextlib
import json
import mmap
import struct
import threading
from pathlib import Path

_CONTROL = 8
_STATUS = 4104
_FRAME = 20488
_PIXELS = 24584


class Mailbox:
    """One control writer and one producer of bounded, acknowledged H264 packets."""

    def __init__(self, path: Path, *, frame_bytes: int = 0) -> None:
        self._file = path.open("r+b")
        if frame_bytes:
            self._file.truncate(_PIXELS + frame_bytes)
        self._data = mmap.mmap(self._file.fileno(), 0)
        self._thread_lock = threading.RLock()

    @contextlib.contextmanager
    def _locked(self):
        import fcntl

        with self._thread_lock:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)

    def _read(self, offset: int) -> dict:
        size = struct.unpack_from("<I", self._data, offset)[0]
        start = offset + 4
        end = start + size
        return json.loads(self._data[start:end]) if size else {}

    def _write(self, offset: int, limit: int, value: dict) -> None:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        if len(data) + 4 > limit:
            raise ValueError("Mailbox metadata exceeds its fixed budget")
        struct.pack_into("<I", self._data, offset, len(data))
        start = offset + 4
        end = start + len(data)
        self._data[start:end] = data

    def write_control(self, value: dict) -> None:
        with self._locked():
            self._write(_CONTROL, _STATUS - _CONTROL, value)

    def read_control(self) -> dict:
        with self._locked():
            return self._read(_CONTROL)

    def write_status(self, value: dict) -> None:
        with self._locked():
            self._write(_STATUS, _FRAME - _STATUS, value)

    def read_status(self) -> dict:
        with self._locked():
            return self._read(_STATUS)

    def write_frame(self, metadata: dict, payload: bytes) -> None:
        if not 0 < len(payload) <= len(self._data) - _PIXELS:
            raise ValueError("Encoded frame exceeds mailbox capacity")
        with self._locked():
            end = _PIXELS + len(payload)
            self._data[_PIXELS:end] = payload
            self._write(_FRAME, _PIXELS - _FRAME, {**metadata, "size": len(payload)})

    def read_frame(self, previous_sequence: int = -1) -> dict | None:
        with self._locked():
            metadata = self._read(_FRAME)
            if not metadata or metadata["sequence"] == previous_sequence:
                return None
            end = _PIXELS + metadata["size"]
            return {**metadata, "h264": self._data[_PIXELS:end]}

    def acknowledge(self, sequence: int) -> None:
        """Allow the producer to publish its next dependent P frame."""
        with self._locked():
            struct.pack_into("<I", self._data, 4, sequence)

    def acknowledged(self) -> int:
        with self._locked():
            return int(struct.unpack_from("<I", self._data, 4)[0])

    def request_stop(self) -> None:
        self._data[0] = 1

    def stopping(self) -> bool:
        return bool(self._data[0])

    def close(self) -> None:
        self._data.close()
        self._file.close()
