# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Detector HTTP client — talks to the open-vocabulary detection server
(``jiuwensymbiosis.serving.grounding_dino_sam2_server``).

A thin text-prompt segmentation client. Only the text-prompt path is
included; the server returns one mask/box/score per detection.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Callable, Optional

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)


def _encode_image(image: np.ndarray | Image.Image) -> str:
    """Encode an image (ndarray or PIL) to a base64 JPEG string."""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image, mode="RGB")
    else:
        pil_image = image.convert("RGB") if image.mode != "RGB" else image

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _decode_mask(mask_b64: str, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(mask_b64), dtype=np.uint8).reshape(shape)


def _post_with_retries(
    url: str,
    payload: dict,
    *,
    timeout_s: float = 30.0,
    max_attempts: int = 3,
    backoff_s: float = 1.0,
) -> dict:
    """POST JSON with simple retry/backoff. Raises on the final failure."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts:
                logger.warning(
                    "detector POST %s attempt %d/%d failed: %s; retrying in %.1fs",
                    url, attempt, max_attempts, exc, backoff_s,
                )
                time.sleep(backoff_s * attempt)
    raise RuntimeError(f"detector POST {url} failed after {max_attempts} attempts: {last_exc}")


def init_detector(
    service_url: str = "http://127.0.0.1:8114",
    *,
    timeout_s: float = 30.0,
) -> Callable[..., list[dict[str, Any]]]:
    """Return a callable ``segment_fn(image, text_prompt) -> [results]``.

    Each result is::

        {"mask": np.ndarray[bool], "box": [x1,y1,x2,y2], "score": float, "label": str}

    The callable is stateless; it makes one HTTP POST per call to
    ``{service_url}/segment``. If the server is unreachable, returns ``[]``
    and logs a warning rather than raising — the caller (xxxApi) treats
    "no detection" as a recoverable outcome.
    """
    def segment_fn(
        image: np.ndarray | Image.Image,
        text_prompt: str,
    ) -> list[dict[str, Any]]:
        payload = {
            "image_base64": _encode_image(image),
            "text_prompt": text_prompt,
        }
        try:
            data = _post_with_retries(
                f"{service_url}/segment", payload, timeout_s=timeout_s
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "detector service at %s unreachable: %s", service_url, exc,
            )
            return []

        results_data = data.get("results") or []
        if not results_data:
            logger.info(
                "detector returned no results for prompt: %r", text_prompt
            )
            return []

        results: list[dict[str, Any]] = []
        for item in results_data:
            mask_shape = tuple(item["shape"])
            mask = _decode_mask(item["mask_base64"], mask_shape).astype(bool)
            results.append(
                {
                    "mask": mask,
                    "box": item["box"],
                    "score": item["score"],
                    "label": item["label"],
                }
            )
        return results

    return segment_fn
