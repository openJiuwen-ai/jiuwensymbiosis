# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""VLM-based visual completion judge for the continuous watch loop.

Instead of the geometric check (is the pick object's detected xy near the place
target's xy?), this asks a **vision-language model**, from the live wrist-camera
image, whether the pick object is already placed on / in the target — a real
"visual understanding" judgment. On an unreachable or unclear VLM it falls back to
the caller-supplied geometric judge.

The endpoint is any OpenAI-compatible vision chat API (e.g. Qwen3-VL on SiliconFlow,
or a local vLLM serving a VLM). NOTE the demos' default text model (DeepSeek-V3.2)
is NOT a VLM — configure a VLM model to use this.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_POSITIVE = ("yes", "是", "已")
_NEGATIVE = ("no", "否", "没", "未")


def _encode_jpeg_b64(image: Any) -> str | None:
    """RGB HxWx3 numpy (``api.get_image()``) → base64 JPEG string, or None."""
    try:
        import cv2
        import numpy as np
    except ImportError:  # optional deps; degrade to geometric fallback
        return None
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    # api.get_image() is RGB; cv2 expects BGR for a correctly-coloured JPEG.
    bgr = cv2.cvtColor(arr.astype("uint8"), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _ask_vlm_yesno(
    image: Any,
    question: str,
    *,
    api_base: str,
    api_key: str,
    model_name: str,
    timeout_s: float = 20.0,
) -> bool | None:
    """Ask the VLM ``question`` about ``image``; return True/False, or None if the
    call fails or the answer is unclear."""
    import httpx

    b64 = _encode_jpeg_b64(image)
    if b64 is None:
        return None
    url = api_base.rstrip("/").removesuffix("/chat/completions") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 - any failure → None → caller falls back
        logger.warning("[vlm_judge] request failed: %s", exc)
        return None
    t = (text or "").strip().lower()
    if t.startswith("y") or any(kw in t for kw in _POSITIVE):
        return True
    if t.startswith("n") or any(kw in t for kw in _NEGATIVE):
        return False
    logger.info("[vlm_judge] unclear answer %r", text)
    return None


def make_vlm_completion_judge(
    *,
    api_base: str,
    api_key: str,
    model_name: str,
    timeout_s: float = 20.0,
    question_template: str | None = None,
    fallback: Callable[[Any, Any], bool] | None = None,
) -> Callable[[Any, Any], bool]:
    """Build an ``is_task_complete(api, config) -> bool`` that asks a VLM, from the
    live camera image (``api.get_image()``), whether the pick object is already on
    the place target.

    Args:
      api_base / api_key / model_name: OpenAI-compatible VLM endpoint.
      question_template: overrides the prompt; gets ``.format(chip=..., slot=...)``.
      fallback: used when the VLM is unreachable / unclear (e.g.
        ``geometric_completion_judge``). If None, an uncertain VLM is treated as
        "not complete" (the arm will then act only if the pick object is detected).
    """
    tmpl = question_template or (
        "这是机器人腕部相机看到的画面。「{chip}」是否已经放到「{slot}」上面 / 里面了？只回答 yes 或 no。"
    )

    def _judge(api: Any, config: Any) -> bool:
        """Ask the VLM whether the pick object is already on the place target."""
        img = None
        try:
            img = api.get_image()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[vlm_judge] get_image failed: %s", exc)
        if img is not None:
            ans = _ask_vlm_yesno(
                img,
                tmpl.format(chip=config.chip_object_name, slot=config.slot_object_name),
                api_base=api_base,
                api_key=api_key,
                model_name=model_name,
                timeout_s=timeout_s,
            )
            if ans is not None:
                return ans
        # VLM unavailable / unclear → geometric fallback (or conservative not-complete).
        if fallback is not None:
            return bool(fallback(api, config))
        return False

    return _judge
