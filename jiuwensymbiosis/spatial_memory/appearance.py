"""Optional, local-only object descriptors; model runtimes never enter the memory core."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from jiuwensymbiosis.spatial_memory.models import Detections, canonical


class AppearanceEncoder(Protocol):
    """Attach versioned descriptors while preserving raw segmentation evidence."""

    def enrich(self, rgb: np.ndarray, detections: Detections) -> Detections: ...


def model_manifest(directory: Path) -> dict:
    """Hash local configuration and model bytes, independent of their absolute path."""
    names = ["config.json", "preprocessor_config.json"]
    weights = sorted(directory.glob("*.safetensors")) or sorted(directory.glob("pytorch_model*.bin"))
    if not weights or any(not (directory / name).is_file() for name in names):
        raise ValueError("Require local model weights, config.json and preprocessor_config.json")
    entries = {}
    for file in [*(directory / name for name in names), *weights]:
        with file.open("rb") as stream:
            entries[file.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"files": entries, "sha256": hashlib.sha256(canonical(entries).encode()).hexdigest()}


def object_crops(rgb: np.ndarray, masks: list[np.ndarray], *, size: int = 224) -> tuple[list, list[np.ndarray]]:
    """Use identical masked, square, padded crops for all learned descriptors.

    Letterboxing preserves the full object; the image processor must not crop it
    again. Soft masks retain tiny-object contributions when patch pooling.
    """
    from PIL import Image

    images, weights = [], []
    for mask in masks:
        if mask.shape != rgb.shape[:2] or not mask.any():
            raise ValueError("Nonempty object mask must align with RGB")
        ys, xs = np.nonzero(mask)
        padding = max(1, int(np.ceil(0.1 * max(np.ptp(xs) + 1, np.ptp(ys) + 1))))
        x0, x1 = max(0, int(xs.min()) - padding), min(rgb.shape[1], int(xs.max()) + padding + 1)
        y0, y1 = max(0, int(ys.min()) - padding), min(rgb.shape[0], int(ys.max()) + padding + 1)
        selected = mask[y0:y1, x0:x1]
        pixels = np.where(selected[..., None], rgb[y0:y1, x0:x1], 127).astype(np.uint8)
        height, width = selected.shape
        side = max(height, width)
        square = np.full((side, side, 3), 127, dtype=np.uint8)
        square_mask = np.zeros((side, side), dtype=np.float32)
        y, x = (side - height) // 2, (side - width) // 2
        square[y : y + height, x : x + width] = pixels
        square_mask[y : y + height, x : x + width] = selected
        images.append(Image.fromarray(square).resize((size, size), Image.Resampling.BICUBIC))
        weights.append(np.asarray(Image.fromarray(square_mask).resize((size, size), Image.Resampling.BILINEAR)))
    return images, weights


class TransformersAppearance:
    """CLIP image projection or DINOv3 CLS / mask-pooled patch descriptors.

    Explicit construction loads local, previously downloaded weights. The owner
    chooses a device and releases this object when its processing session ends.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        kind: str,
        device: str = "cuda:0",
        batch_size: int = 16,
        expected_sha256: str | None = None,
    ) -> None:
        if kind not in {"clip", "dinov3-cls", "dinov3-patch"} or not 1 <= batch_size <= 128:
            raise ValueError("Unknown appearance kind or invalid batch size")
        import torch
        import transformers
        from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModelWithProjection

        self.directory = Path(directory).resolve()
        manifest = model_manifest(self.directory)
        if expected_sha256 is not None and manifest["sha256"] != expected_sha256:
            raise ValueError("Appearance model digest mismatch")
        config = json.loads((self.directory / "config.json").read_text())
        if kind == "clip" and config.get("model_type") != "clip":
            raise ValueError("CLIP requires a CLIP model")
        if kind.startswith("dinov3") and config.get("model_type") != "dinov3_vit":
            raise ValueError("DINOv3 requires an official-compatible DINOv3 ViT model, not DINOv2")
        self.kind, self.batch_size = kind, batch_size
        self.device = torch.device(f"cuda:{device}" if device.isdigit() else device)
        self.model: Any
        self.processor: Any
        if kind == "clip":
            self.processor = CLIPImageProcessor.from_pretrained(self.directory, local_files_only=True)
            self.model = CLIPVisionModelWithProjection.from_pretrained(self.directory, local_files_only=True)
            dimension = self.model.config.projection_dim
        else:
            self.processor = AutoImageProcessor.from_pretrained(self.directory, local_files_only=True)
            self.model = AutoModel.from_pretrained(self.directory, local_files_only=True, trust_remote_code=False)
            dimension = self.model.config.hidden_size
        self.model.to(self.device).eval()
        self.size = 224
        self.provenance = {
            "kind": kind,
            "dimension": int(dimension),
            "model": manifest,
            "preprocess": {"size": self.size, "background": 127, "crop": "masked-square-10pct-pad-v1"},
            "transformers_version": transformers.__version__,
            "precision": "float32",
        }
        self.provenance["space_id"] = hashlib.sha256(canonical(self.provenance).encode()).hexdigest()
        self.timings: list[dict] = []

    def enrich(self, rgb: np.ndarray, detections: Detections) -> Detections:
        import torch
        import torch.nn.functional as functional

        if any(d.embedding is not None for d in detections.instances):
            raise ValueError("Detections already have appearance features; choose the raw frozen masks")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        images, masks = object_crops(rgb, [d.mask for d in detections.instances], size=self.size)
        vectors: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                inputs = self.processor(
                    images=images[start : start + self.batch_size],
                    return_tensors="pt",
                    do_resize=False,
                    do_center_crop=False,
                ).to(self.device)
                output = self.model(**inputs)
                if self.kind == "clip":
                    encoded = output.image_embeds
                elif self.kind == "dinov3-cls":
                    encoded = output.last_hidden_state[:, 0]
                else:
                    tokens = output.last_hidden_state[:, 1 + self.model.config.num_register_tokens :]
                    side = self.size // self.model.config.patch_size
                    if tokens.shape[1] != side * side:
                        raise ValueError("DINOv3 patch grid does not match transformed image")
                    pixels = torch.as_tensor(np.stack(masks[start : start + self.batch_size]), device=self.device)
                    weights = functional.interpolate(pixels[:, None], (side, side), mode="area").flatten(1)
                    encoded = (tokens * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-8)
                vectors.extend(functional.normalize(encoded.float(), dim=-1).cpu().numpy())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.timings.append({"instances": len(images), "elapsed_s": time.perf_counter() - started})
        provenance = json.loads(detections.provenance_json)
        provenance["appearance"] = self.provenance
        return Detections(
            tuple(
                replace(detection, embedding=vector)
                for detection, vector in zip(detections.instances, vectors, strict=True)
            ),
            canonical(provenance),
        )
