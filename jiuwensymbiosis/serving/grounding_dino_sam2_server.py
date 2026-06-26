# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""GroundingDINO (+ SAM2) open-vocabulary detection server.

A license-clean open-vocabulary detection server. The client
(``detector_client``) and everything downstream (``detect_and_centroid`` →
projection → grasp_z/place_z → agent) consume this ``/segment`` contract:

  POST /segment  {image_base64, text_prompt}
    -> {results: [ {mask_base64, shape, box, score, label}, ... ]}

Why this backend:
  * GroundingDINO (IDEA-Research, Apache-2.0) does open-vocabulary text→box
    detection (license-clean concept/text prompting).
  * SAM2 (Meta, Apache-2.0) turns each box into a high-quality mask, so the
    mask centroid stays accurate for irregular / non-box objects.
  * GroundingDINO is loaded via HF ``transformers`` (AutoModelForZeroShot-
    ObjectDetection), which avoids compiling GroundingDINO's custom CUDA op
    (painful on CUDA 12.8).

Accuracy choices (see PiperConfig / the api_servers YAML):
  * detector default = ``IDEA-Research/grounding-dino-base`` (Swin-B, not Tiny).
  * segmenter default = ``facebook/sam2.1-hiera-large`` (best masks).
  * box/text thresholds are configurable; the downstream still picks the
    highest-score detection, so set them for precision (don't grab the wrong
    object) and tune on the real scene.

Run directly::

    python -m jiuwensymbiosis.serving.grounding_dino_sam2_server \
        --host 127.0.0.1 --port 8114 \
        --gdino-model-id IDEA-Research/grounding-dino-base \
        --sam2-model-id facebook/sam2.1-hiera-large

``--no-sam2`` runs GroundingDINO only (box rectangle as the mask, box center as
the centroid) — lightest/fastest; fine for box-like objects.

First run downloads the model weights from HuggingFace (can take minutes); the
sidecar's ``startup_timeout_s`` must allow for it.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import io
import logging
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jiuwensymbiosis.serving.gdino_sam2")

app = FastAPI(title="jiuwensymbiosis GroundingDINO+SAM2 server")

# --- global model state (populated in main()) -------------------------------
_GDINO_PROCESSOR: Any | None = None
_GDINO_MODEL: Any | None = None
_SAM2_MODEL: Any | None = None  # transformers Sam2Model, or None when --no-sam2
_SAM2_PROCESSOR: Any | None = None  # transformers Sam2Processor
_DEVICE: str = "cuda"
_BOX_THR: float = 0.35
_TEXT_THR: float = 0.25
_USE_SAM2: bool = True

# Serialize GPU access — concurrent inference can OOM on small cards.
_GPU_SEMAPHORE = asyncio.Semaphore(1)

# Cap detections returned per call (GroundingDINO rarely emits many, but the
# downstream only ever uses the top few). Override via env JIUWEN_VIS_TOPK.
_SEGMENT_TOPK = int(os.environ.get("JIUWEN_VIS_TOPK", "32"))


async def _run_on_gpu(fn, *args, **kwargs):
    """Run a blocking function on a thread pool while serializing GPU access."""
    async with _GPU_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# --- helpers ----------------------------------------------------------------
def _to_numpy(tensor: Any) -> np.ndarray:
    """Convert a torch tensor or array-like to a plain numpy array.

    torch tensors MUST be moved to host first: a CUDA tensor has ``.numpy()``
    but calling it raises ("can't convert cuda:0 device type tensor to numpy").
    So route any torch tensor through ``detach().cpu()`` BEFORE the generic
    ``.numpy()`` fallback (which is only for non-torch array-likes).
    """
    if isinstance(tensor, torch.Tensor):
        t = tensor.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


def _decode_image(b64: str) -> Image.Image:
    """Decode a base64-encoded image string into an RGB PIL Image."""
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}") from exc


def _encode_mask(mask: np.ndarray) -> str:
    """Encode a boolean numpy mask as a base64 uint8 byte string."""
    return base64.b64encode(mask.astype(np.uint8).tobytes()).decode("utf-8")


class SegmentRequest(BaseModel):
    image_base64: str
    text_prompt: str


class MaskData(BaseModel):
    shape: list[int]
    box: list[float]
    score: float
    label: str
    mask_base64: str


class SegmentResponse(BaseModel):
    results: list[MaskData]


# --- inference --------------------------------------------------------------
def _normalize_prompt(text_prompt: str) -> str:
    """GroundingDINO expects a lowercase caption ending with a period."""
    text = text_prompt.lower().strip()
    if not text.endswith("."):
        text = text + " ."
    return text


def _gdino_detect(pil_image: Image.Image, text: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (boxes_xyxy_pixels[N,4], scores[N]) for the caption ``text``."""
    # _GDINO_PROCESSOR / _GDINO_MODEL are module-level import-guard objects
    # (Any | None); this function runs only after server startup has loaded
    # them, so they are non-None here.
    inputs = _GDINO_PROCESSOR(images=pil_image, text=text, return_tensors="pt").to(_DEVICE)  # type: ignore[misc]
    with torch.no_grad():
        outputs = _GDINO_MODEL(**inputs)  # type: ignore[misc]
    target_sizes = [(pil_image.height, pil_image.width)]  # (h, w)
    # transformers renamed ``box_threshold`` -> ``threshold`` across versions;
    # try the modern kwarg first, fall back to the legacy one.
    try:
        res = _GDINO_PROCESSOR.post_process_grounded_object_detection(  # type: ignore[union-attr]
            outputs,
            inputs["input_ids"],
            threshold=_BOX_THR,
            text_threshold=_TEXT_THR,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        res = _GDINO_PROCESSOR.post_process_grounded_object_detection(  # type: ignore[union-attr]
            outputs,
            inputs["input_ids"],
            box_threshold=_BOX_THR,
            text_threshold=_TEXT_THR,
            target_sizes=target_sizes,
        )[0]
    boxes = _to_numpy(res["boxes"]).reshape(-1, 4)
    scores = _to_numpy(res["scores"]).reshape(-1)
    return boxes, scores


def _box_to_mask(box_xyxy: np.ndarray, h: int, w: int) -> np.ndarray:
    """GDINO-only fallback: a filled rectangle. Its centroid == the box center."""
    x1, y1, x2, y2 = box_xyxy
    m = np.zeros((h, w), dtype=bool)
    xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
    xi2, yi2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if xi2 > xi1 and yi2 > yi1:
        m[yi1:yi2, xi1:xi2] = True
    return m


def _sam2_masks(pil_image: Image.Image, boxes: np.ndarray) -> list[np.ndarray]:
    """One high-quality boolean mask per GroundingDINO box, via transformers SAM2.

    ``boxes`` is (N, 4) xyxy in pixels. Returns N boolean HxW masks (original
    image size).
    """
    # _SAM2_PROCESSOR / _SAM2_MODEL are module-level import-guard objects
    # (Any | None); this function runs only after server startup has loaded
    # them (and only when --no-sam2 is not set). Per-line type: ignore rather
    # than assert — assert is reserved for tests (`python -O` strips it).
    input_boxes = [[[float(x) for x in b] for b in boxes]]  # (batch=1, N, 4)
    inputs = _SAM2_PROCESSOR(images=pil_image, input_boxes=input_boxes, return_tensors="pt").to(_DEVICE)  # type: ignore[misc]
    ctx = torch.autocast(_DEVICE, dtype=torch.bfloat16) if "cuda" in _DEVICE else torch.autocast("cpu")
    with torch.inference_mode(), ctx:
        outputs = _SAM2_MODEL(**inputs)  # type: ignore[misc]
    # Upsample low-res logits back to the original image size.
    masks = _SAM2_PROCESSOR.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0]  # type: ignore[union-attr]
    masks = _to_numpy(masks)  # (N, M, H, W) — M masks per box
    iou = getattr(outputs, "iou_scores", None)
    iou = _to_numpy(iou) if iou is not None else None

    out: list[np.ndarray] = []
    n = boxes.shape[0]
    for i in range(n):
        m = masks[i]
        if m.ndim == 3:  # (M, H, W): keep the highest-IoU mask
            if iou is not None:
                scores_i = iou[0, i] if iou.ndim == 3 else iou.reshape(n, -1)[i]
                m = m[int(np.argmax(scores_i))]
            else:
                m = m[0]
        out.append(np.asarray(m) > 0.0)
    return out


def _do_segment(pil_image: Image.Image, text_prompt: str) -> SegmentResponse:
    """Run GroundingDINO detection and optional SAM2 segmentation on one image."""
    t0 = time.perf_counter()
    text = _normalize_prompt(text_prompt)
    boxes, scores = _gdino_detect(pil_image, text)
    t_det = time.perf_counter() - t0
    if boxes.shape[0] == 0:
        logger.info("[gdino] /segment prompt=%r det=%.0fms kept=0", text_prompt, t_det * 1000.0)
        return SegmentResponse(results=[])

    order = np.argsort(scores)[::-1][:_SEGMENT_TOPK]
    boxes, scores = boxes[order], scores[order]

    h, w = pil_image.height, pil_image.width
    t1 = time.perf_counter()
    if _USE_SAM2 and _SAM2_MODEL is not None:
        masks = _sam2_masks(pil_image, boxes)
    else:
        masks = [_box_to_mask(b, h, w) for b in boxes]
    t_seg = time.perf_counter() - t1

    items: list[MaskData] = []
    for b, s, m in zip(boxes, scores, masks, strict=True):
        if not m.any():
            continue
        items.append(
            MaskData(
                mask_base64=_encode_mask(m),
                shape=list(m.shape),
                box=[float(x) for x in b],
                score=float(s),
                label=text_prompt,
            )
        )
    logger.info(
        "[gdino] /segment prompt=%r det=%.0fms seg=%.0fms kept=%d/%d sam2=%s",
        text_prompt,
        t_det * 1000.0,
        t_seg * 1000.0,
        len(items),
        len(scores),
        bool(_USE_SAM2 and _SAM2_MODEL is not None),
    )
    return SegmentResponse(results=items)


# --- routes -----------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    """Return server health status including model readiness and backend."""
    return {
        "status": "ok" if _GDINO_MODEL is not None else "loading",
        "device": _DEVICE,
        "backend": "gdino_sam2" if (_USE_SAM2 and _SAM2_MODEL is not None) else "gdino",
    }


@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    """Run open-vocabulary detection and segmentation on the provided image."""
    if _GDINO_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    pil = _decode_image(req.image_base64)
    try:
        return await _run_on_gpu(_do_segment, pil, req.text_prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


# --- entry point ------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, load models, and start the uvicorn server."""
    parser = argparse.ArgumentParser(description="jiuwensymbiosis GroundingDINO(+SAM2) server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8114)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gdino-model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam2-model-id", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--no-sam2",
        dest="use_sam2",
        action="store_false",
        help="GroundingDINO only; use the box rectangle as the mask (faster).",
    )
    parser.set_defaults(use_sam2=True)
    args = parser.parse_args(argv)

    global _GDINO_PROCESSOR, _GDINO_MODEL, _SAM2_MODEL, _SAM2_PROCESSOR, _DEVICE
    global _BOX_THR, _TEXT_THR, _USE_SAM2
    _DEVICE = args.device
    _BOX_THR = args.box_threshold
    _TEXT_THR = args.text_threshold
    _USE_SAM2 = args.use_sam2

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Both models load via HF transformers (Apache-2.0, no custom CUDA op build).
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )

    logger.info("Loading GroundingDINO: %s", args.gdino_model_id)
    _GDINO_PROCESSOR = AutoProcessor.from_pretrained(args.gdino_model_id)
    _GDINO_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_model_id).to(_DEVICE)
    _GDINO_MODEL.eval()

    if _USE_SAM2:
        logger.info("Loading SAM2: %s", args.sam2_model_id)
        _SAM2_PROCESSOR = Sam2Processor.from_pretrained(args.sam2_model_id)
        _SAM2_MODEL = Sam2Model.from_pretrained(args.sam2_model_id).to(_DEVICE)
        _SAM2_MODEL.eval()
    else:
        logger.info("SAM2 disabled (--no-sam2): box rectangle used as mask.")

    logger.info(
        "GroundingDINO%s ready on %s; serving at http://%s:%d (box_thr=%.2f text_thr=%.2f)",
        "+SAM2" if _USE_SAM2 else "",
        _DEVICE,
        args.host,
        args.port,
        _BOX_THR,
        _TEXT_THR,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
