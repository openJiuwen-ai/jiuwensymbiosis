"""Immutable, metre-based evidence shared by simulation, replay and real cameras."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def frozen_array(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    # A bytes owner prevents a caller from re-enabling writes on the buffer.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def pack(**arrays: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, allow_pickle=False, **arrays)
    return stream.getvalue()


def unpack(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


@dataclass(frozen=True)
class Observation:
    """Rectified RGB-D at one frozen optical-camera pose; no object truth fields.

    ``timestamp_s`` is in the named clock; map poses use metres, optical axes are
    right/down/forward. A database accepts exactly one scope and calibration.
    """

    observation_id: str
    scene_epoch: str
    map_frame: str
    calibration_id: str
    timestamp_s: float
    clock_id: str
    pose_source: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    map_from_camera: np.ndarray

    def __post_init__(self) -> None:
        for name in ("observation_id", "scene_epoch", "map_frame", "calibration_id", "clock_id", "pose_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 1024:
                raise ValueError(f"Invalid {name}")
        if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("Invalid capture timestamp")
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            raise ValueError("RGB must be uint8 H x W x 3")
        if min(self.rgb.shape[:2]) < 2 or self.rgb.shape[0] * self.rgb.shape[1] > 16_000_000:
            raise ValueError("Image dimensions outside observation budget")
        for name, dtype in (
            ("rgb", np.uint8),
            ("depth_m", np.float32),
            ("intrinsics", np.float64),
            ("map_from_camera", np.float64),
        ):
            object.__setattr__(self, name, frozen_array(getattr(self, name), dtype))
        d, k, t = self.depth_m, self.intrinsics, self.map_from_camera
        if d.shape != self.rgb.shape[:2] or np.isinf(d).any() or np.any(d[np.isfinite(d)] <= 0):
            raise ValueError("Depth must match RGB, with positive metres or NaN for missing samples")
        if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError("Invalid pinhole intrinsics")
        if not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
            raise ValueError("Only rectified zero-skew pinhole images are supported")
        if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1]):
            raise ValueError("Invalid map_from_camera transform")
        if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(t[:3, :3]), 1, atol=1e-5
        ):
            raise ValueError("Camera transform must be a proper rigid transform")

    def metadata(self) -> dict:
        fields = (
            "observation_id",
            "scene_epoch",
            "map_frame",
            "calibration_id",
            "timestamp_s",
            "clock_id",
            "pose_source",
        )
        return {field: getattr(self, field) for field in fields}

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in ("rgb", "depth_m", "intrinsics", "map_from_camera")}


@dataclass(frozen=True)
class Detection:
    local_id: str
    label: str
    score: float
    mask: np.ndarray
    kind: str = "object"
    class_id: int | None = None
    embedding: np.ndarray | None = None

    def __post_init__(self) -> None:
        valid_score = bool(np.isfinite(self.score)) and 0 <= self.score <= 1
        if not self.local_id or not self.label or not valid_score:
            raise ValueError("Invalid detection identity, label or score")
        if self.kind not in {"object", "surface", "scene"}:
            raise ValueError("Invalid detection kind")
        mask = np.asarray(self.mask)
        if mask.ndim != 2 or not np.isin(mask, [0, 1]).all() or not mask.any():
            raise ValueError("Expected a nonempty binary instance mask")
        object.__setattr__(self, "mask", frozen_array(mask, np.bool_))
        if self.embedding is not None:
            vector = np.asarray(self.embedding, dtype=np.float32)
            valid_shape = vector.ndim == 1 and 3 <= vector.size <= 4096
            if not valid_shape or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
                raise ValueError("Appearance embedding must be a finite unit vector")
            object.__setattr__(self, "embedding", frozen_array(vector, np.float32))

    def metadata(self) -> dict:
        result = {
            "local_id": self.local_id,
            "label": self.label,
            "score": self.score,
            "kind": self.kind,
            "class_id": self.class_id,
        }
        if self.embedding is not None:
            result["embedding"] = self.embedding.tolist()
        return result


@dataclass(frozen=True)
class Detections:
    """Successful inference only. Model exceptions must never produce this record."""

    instances: tuple[Detection, ...]
    provenance_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instances", tuple(self.instances))
        ids = [d.local_id for d in self.instances]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate local detection IDs")
        provenance = json.loads(self.provenance_json)
        for field in ("model_id", "weights_sha256", "vocabulary_sha256", "inference"):
            if not provenance.get(field):
                raise ValueError(f"Missing detector provenance: {field}")
        appearance = provenance.get("appearance")
        dimensions = {len(d.embedding) for d in self.instances if d.embedding is not None}
        if appearance is not None:
            if not isinstance(appearance, dict) or not appearance.get("space_id") or not appearance.get("dimension"):
                raise ValueError("Require versioned appearance provenance")
            if any(d.embedding is None for d in self.instances) or dimensions - {appearance["dimension"]}:
                raise ValueError("Appearance space must cover every detection with the declared dimension")
        elif dimensions:
            raise ValueError("Embeddings require appearance provenance")
        object.__setattr__(self, "provenance_json", canonical(provenance))


def evidence_fingerprint(observation: Observation, detections: Detections) -> str:
    digest = hashlib.sha256(canonical(observation.metadata()).encode())
    digest.update(detections.provenance_json.encode())
    arrays = list(observation.arrays().values())
    for detection in detections.instances:
        digest.update(canonical(detection.metadata()).encode())
        arrays.append(detection.mask)
    for array in arrays:
        digest.update(canonical([array.dtype.str, list(array.shape)]).encode())
        # Canonicalize NaN bit patterns so serialization doesn't change identity.
        digest.update(np.nan_to_num(array, nan=-1).tobytes())
    return digest.hexdigest()


def discovery_fingerprint(observation: Observation, detections: Detections) -> str:
    """Bind annotations to original sensing/segmentation, independently of descriptors."""
    from dataclasses import replace

    provenance = json.loads(detections.provenance_json)
    provenance.pop("appearance", None)
    raw = Detections(tuple(replace(d, embedding=None) for d in detections.instances), canonical(provenance))
    return evidence_fingerprint(observation, raw)


def observation_from_arrays(metadata: dict, arrays: dict[str, np.ndarray]) -> Observation:
    return Observation(
        **metadata,
        rgb=arrays["rgb"],
        depth_m=arrays["depth_m"],
        intrinsics=arrays["intrinsics"],
        map_from_camera=arrays["map_from_camera"],
    )
