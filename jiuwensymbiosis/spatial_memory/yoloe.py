"""Optional local YOLOE-PF discovery. Importing core memory never loads this model."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
from typing import Any

import numpy as np

from jiuwensymbiosis.spatial_memory.models import Detection, Detections, canonical


def label_kind(label: str) -> str:
    """Conservative lexical routing, versioned as policy rather than model truth."""
    words = label.casefold().replace("_", " ").split()
    if label.casefold() in {"darkness", "light", "sunlight", "shadow", "home interior", "interior", "indoors"} or (
        words and words[-1].endswith("room")
    ):
        return "scene"
    if words and words[-1] in {"floor", "wall", "ceiling", "flooring"}:
        return "surface"
    return "object"


class YoloEPromptFree:
    """Explicit local model owner; no automatic download or fallback inference.

    The caller chooses device and a verified checkpoint digest. This adapter is
    intentionally independent of the target-conditioned GDINO/SAM2 interface.
    """

    def __init__(
        self, weights: str | Path, sha256: str, *, device: str = "0", confidence: float = 0.25, max_det: int = 100
    ):
        path = Path(weights).expanduser().resolve(strict=True)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != sha256:
            raise ValueError("YOLOE checkpoint digest mismatch")
        if not 0 <= confidence <= 1 or type(max_det) is not int or not 1 <= max_det <= 1000:
            raise ValueError("Invalid inference thresholds")
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        from ultralytics import YOLOE

        self.model: Any = YOLOE(str(path))
        if self.model.task != "segment" or len(self.model.names) != 4585 or not path.name.endswith("-pf.pt"):
            raise ValueError("Expected a 4585-name prompt-free segmentation checkpoint")
        self.arguments: dict[str, Any] = {
            "device": device,
            "imgsz": 640,
            "conf": confidence,
            "iou": 0.7,
            "max_det": max_det,
            "retina_masks": True,
            "quantize": 32,
            "nms": True,
            "agnostic_nms": True,
            "batch": 1,
            "verbose": False,
            "save": False,
        }
        self.provenance = {
            "model_id": path.name,
            "weights_sha256": actual,
            "vocabulary_sha256": hashlib.sha256(canonical(self.model.names).encode()).hexdigest(),
            "ultralytics_version": importlib.metadata.version("ultralytics"),
            "inference": {"mode": "prompt_free", "input_color": "PIL_RGB", **self.arguments},
            "kind_policy": "lexical-object-surface-scene-v3",
            "truncation_status": "unknown",
            "inventory_complete": False,
        }

    def detect(self, rgb: np.ndarray) -> Detections:
        from PIL import Image

        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Expected uint8 RGB")
        result = self.model.predict(Image.fromarray(rgb), **self.arguments)[0]
        boxes = result.boxes
        if boxes is None:
            raise RuntimeError("YOLOE returned no boxes container")
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        if len(scores):
            if result.masks is None:
                raise RuntimeError("YOLOE detections have no segmentation masks")
            masks = result.masks.data.cpu().numpy()
            if masks.shape != (len(scores), *rgb.shape[:2]):
                raise RuntimeError("YOLOE masks are not in original image coordinates")
        else:
            masks = np.empty((0, *rgb.shape[:2]), dtype=bool)
        detections = tuple(
            Detection(
                str(i),
                str(result.names[int(cls)]),
                float(score),
                mask,
                label_kind(str(result.names[int(cls)])),
                int(cls),
            )
            for i, (cls, score, mask) in enumerate(zip(classes, scores, masks, strict=True))
        )
        return Detections(
            detections,
            canonical({**self.provenance, "output_limit_reached": len(detections) >= self.arguments["max_det"]}),
        )
