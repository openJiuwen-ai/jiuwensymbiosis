"""Bounded previews and opaque references; GUI encodes images for its transport."""

from __future__ import annotations

import io
import json
import re
import threading
import uuid
from pathlib import Path
from typing import cast

import numpy as np

_TOKEN = re.compile(r"[a-f0-9]{32}")
MAX_BYTES = 32 * 1024 * 1024


def _valid_slot(slot: object) -> bool:
    """A frame slot is ``latest`` or a 1..128 step index."""
    if not isinstance(slot, str):
        return False
    return slot == "latest" or (slot.isdigit() and 1 <= int(slot) <= 128)


def _valid_frame_array(array: np.ndarray) -> bool:
    """A storable frame is an RGB/RGBA array within the byte budget."""
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        return False
    return not array.dtype.hasobject and array.nbytes <= MAX_BYTES


class ArtifactStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def frame(self, job_id: str, rgb: np.ndarray, *, slot: str = "latest") -> dict | None:
        """Store one latest preview or at most 128 step previews per job."""
        if not _TOKEN.fullmatch(job_id):
            raise ValueError("invalid job ID")
        if not _valid_slot(slot):
            return None
        array = np.asarray(rgb)
        if not _valid_frame_array(array):
            return None
        directory = self.directory / job_id
        directory.mkdir(exist_ok=True)
        with self._lock:
            target = directory / f"{slot}.npy"
            temporary = directory / f"{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return {"job_id": job_id, "slot": slot, "kind": "numpy-frame"}

    def read(self, reference: dict) -> bytes:
        job_id, slot = reference.get("job_id", ""), reference.get("slot", "")
        if not isinstance(job_id, str) or not _TOKEN.fullmatch(job_id):
            raise ValueError("invalid artifact reference")
        if reference.get("kind") == "trace":
            if not isinstance(slot, str) or not _TOKEN.fullmatch(slot):
                raise ValueError("invalid trace reference")
            manifest = self.directory / job_id / f"{slot}.ref.json"
            if not manifest.resolve().is_relative_to(self.directory):
                raise ValueError("artifact reference escapes runtime directory")
            record = json.loads(manifest.read_text())
            target = Path(record["path"]).resolve()
            if str(target) != record["path"] or not target.is_relative_to(Path(record["root"])):
                raise ValueError("trace target changed after registration")
            with target.open("rb") as stream:
                content = stream.read(MAX_BYTES + 1)
            if len(content) > MAX_BYTES:
                raise ValueError("trace exceeds read limit")
            return content
        if not _valid_slot(slot):
            raise ValueError("invalid artifact slot")
        target = (self.directory / job_id / f"{slot}.npy").resolve()
        if not target.is_relative_to(self.directory):
            raise ValueError("artifact escapes the runtime directory")
        with self._lock, target.open("rb") as stream:
            content = stream.read(MAX_BYTES + 4097)
        if len(content) > MAX_BYTES + 4096:
            raise ValueError("artifact exceeds read limit")
        return content

    def register_trace(self, job_id: str, path: Path) -> dict:
        """Execution-side registration; the public read API accepts references only."""
        if not _TOKEN.fullmatch(job_id):
            raise ValueError("invalid job ID")
        target = path.resolve(strict=True)
        if target.suffix != ".json" or not target.name.startswith(f"gui-{job_id}_"):
            raise ValueError("trace does not belong to this job")
        slot = uuid.uuid4().hex
        directory = self.directory / job_id
        directory.mkdir(exist_ok=True)
        (directory / f"{slot}.ref.json").write_text(json.dumps({"path": str(target), "root": str(target.parent)}))
        return {"job_id": job_id, "slot": slot, "kind": "trace"}

    def latest(self, job_id: str) -> dict | None:
        if not _TOKEN.fullmatch(job_id):
            raise ValueError("invalid job ID")
        if (self.directory / job_id / "latest.npy").is_file():
            return {"job_id": job_id, "slot": "latest", "kind": "numpy-frame"}
        return None


def decode_frame(content: bytes) -> np.ndarray:
    """Decode the documented numpy-frame format without allowing pickle."""
    return cast(np.ndarray, np.load(io.BytesIO(content), allow_pickle=False))
