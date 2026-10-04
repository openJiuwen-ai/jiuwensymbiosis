# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Session-owned HTTP detection client; imports no model or serving dependencies."""

from __future__ import annotations

import base64
import io
import math
import time
import uuid
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from PIL import Image

from jiuwensymbiosis.errors import InferenceServiceError
from jiuwensymbiosis.perception.config import DetectorConfig
from jiuwensymbiosis.utils.service_http import HttpEndpointConfig, HttpServiceClient
from jiuwensymbiosis.utils.validation import require_unit_interval

MAX_PIXELS = 16_777_216
MAX_DETECTIONS = 32


def _encode_image(image: np.ndarray | Image.Image) -> str:
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("detector image must be an HxWx3 RGB array")
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_mask_png(mask: np.ndarray) -> dict[str, str]:
    """Encode a binary mask without loss using the v1 mask schema."""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(buf, format="PNG")
    return {"mime_type": "image/png", "data_base64": base64.b64encode(buf.getvalue()).decode("ascii")}


def decode_mask_png(data: Any, *, width: int, height: int) -> np.ndarray:
    if not isinstance(data, dict) or data.get("mime_type") != "image/png":
        raise ValueError("mask must be an image/png object")
    encoded = data.get("data_base64")
    if not isinstance(encoded, str) or len(encoded) > MAX_PIXELS * 4:
        raise ValueError("invalid encoded mask size")
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
        if image.format != "PNG" or image.size != (width, height) or image.mode not in {"1", "L"}:
            raise ValueError("mask dimensions or mode do not match the image")
        mask = np.asarray(image)
        if not np.isin(mask, [0, 1, 255]).all():
            raise ValueError("mask is not binary")
        return mask.astype(bool)


def _validate_detection(item: Any, width: int, height: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("detection must be an object")
    box, score, label = item.get("box"), item.get("score"), item.get("label")
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in box)
    ):
        raise ValueError("invalid detection box")
    if not (0 <= box[0] <= box[2] <= width and 0 <= box[1] <= box[3] <= height):
        raise ValueError("detection box is outside the image")
    require_unit_interval(score, message="invalid detection score")
    if not isinstance(label, str) or len(label) > 4096:
        raise ValueError("invalid detection label")
    mask = decode_mask_png(item.get("mask"), width=width, height=height)
    return {"mask": mask, "box": box, "score": float(cast("int | float", score)), "label": label}


class DetectorClient:
    """Callable detector plus explicitly owned transport and readiness probe."""

    def __init__(self, config: DetectorConfig, *, request_scoped: bool = False) -> None:
        self.config = config
        endpoint = config.endpoint
        if config.mode == "local":
            endpoint = HttpEndpointConfig(url=cast(str, config.url))
        self.endpoint = endpoint
        self._http = (
            HttpServiceClient(endpoint, service="vision") if endpoint is not None and not request_scoped else None
        )
        self._cancel_token: Any = None

    def _request_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if self._http is not None:
            return self._http.request_json(*args, **kwargs)
        if self.endpoint is None:
            raise InferenceServiceError("Detection is disabled", service="vision")
        # Directly constructed APIs have no session resource registration. Keep
        # their legacy per-request lifetime; managed clients still reuse a pool.
        with HttpServiceClient(self.endpoint, service="vision", cancel_token=self._cancel_token) as client:
            return client.request_json(*args, **kwargs)

    def bind_cancel_token(self, token: Any) -> None:
        if self._http is not None:
            self._http.bind_cancel_token(token)
        self._cancel_token = token

    def open(self, *, lazy: bool = False) -> DetectorClient:
        """Open or reopen a session, optionally deferring request setup."""
        try:
            if self._http is not None:
                self._http.open(lazy=lazy)
        except BaseException:
            self.close()
            raise
        return self

    def cleanup_report(self) -> Any:
        from jiuwensymbiosis.agent.lifecycle import CleanupReport

        return CleanupReport(connected=self._http is not None and not self._http.closed)

    def close(self, timeout_s: float = 5.0) -> None:
        if self._http is not None:
            self._http.close(timeout_s=timeout_s)

    def __enter__(self) -> DetectorClient:
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def readiness(self) -> dict[str, Any]:
        if self.endpoint is None:
            raise InferenceServiceError("Detection is disabled", service="vision")
        request_id = uuid.uuid4().hex
        data = self._request_json("GET", "/v1/health", request_id=request_id)
        result = self._result(data, request_id)
        if result.get("service") != "vision" or result.get("status") != "ready":
            raise InferenceServiceError("Vision service is not ready", service="vision", request_id=request_id)
        return result

    @staticmethod
    def _result(data: dict, request_id: str) -> dict:
        if (
            data.get("schema_version") != 1
            or data.get("request_id") != request_id
            or not isinstance(data.get("result"), dict)
        ):
            raise InferenceServiceError(
                "Invalid vision response envelope",
                code="inference_protocol_error",
                service="vision",
                request_id=request_id,
            )
        return cast(dict[str, Any], data["result"])

    def __call__(
        self,
        image: np.ndarray | Image.Image,
        text_prompt: str,
        *,
        frame_id: str | None = None,
        captured_monotonic_s: float | None = None,
    ) -> list[dict[str, Any]]:
        if self.endpoint is None:
            raise InferenceServiceError("Detection is disabled; configure detector.mode", service="vision")
        request_id = uuid.uuid4().hex
        frame_id = frame_id or uuid.uuid4().hex
        started = time.monotonic() if captured_monotonic_s is None else captured_monotonic_s
        if not isinstance(text_prompt, str) or not text_prompt.strip() or len(text_prompt) > 4096:
            raise ValueError("text_prompt must contain 1..4096 characters")
        width, height = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
        if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
            raise ValueError("image exceeds pixel limit")
        self.check_frame_age(started, request_id)
        encoded = _encode_image(image)
        payload = {
            "schema_version": 1,
            "request_id": request_id,
            "frame_id": frame_id,
            "image": {"mime_type": "image/jpeg", "data_base64": encoded, "width": width, "height": height},
            "text_prompt": text_prompt,
            "top_k": MAX_DETECTIONS,
        }
        remaining_age = self.check_frame_age(started, request_id)
        request_timeout = self.endpoint.request_timeout_s
        data = self._request_json(
            "POST",
            "/v1/segment",
            payload,
            request_id=request_id,
            timeout_s=min(remaining_age, request_timeout) if remaining_age is not None else request_timeout,
        )
        self.check_frame_age(started, request_id)
        try:
            result = self._result(data, request_id)
            if result.get("frame_id") != frame_id or result.get("width") != width or result.get("height") != height:
                raise ValueError("vision response does not match the input frame")
            items = result["detections"]
            if not isinstance(items, list) or len(items) > MAX_DETECTIONS:
                raise ValueError("invalid detection count")
            results = [_validate_detection(item, width, height) for item in items]
        except (ValueError, KeyError, TypeError, OverflowError, OSError, Image.DecompressionBombError) as exc:
            raise InferenceServiceError(
                "Invalid vision result", code="inference_protocol_error", service="vision", request_id=request_id
            ) from exc
        self.check_frame_age(started, request_id)
        return results

    def check_frame_age(self, captured: float, request_id: str) -> float | None:
        """Check cancellation and return the remaining optional capture-age budget."""
        if self._cancel_token is not None:
            self._cancel_token.raise_if_set()
        if self.config.max_frame_age_s is None:
            return None
        age = time.monotonic() - captured
        if not math.isfinite(age) or age < 0 or age >= self.config.max_frame_age_s:
            raise InferenceServiceError(
                "Vision frame expired", code="inference_result_stale", service="vision", request_id=request_id
            )
        return self.config.max_frame_age_s - age


def check_detection_completion(
    seg_fn: Any, captured_monotonic_s: float, *, invalidate: Callable[[], None] | None = None
) -> None:
    """Guard publication after local geometry; an abandoned result cannot retain a cache."""
    if isinstance(seg_fn, DetectorClient):
        try:
            seg_fn.check_frame_age(captured_monotonic_s, "")
        except Exception:
            if invalidate is not None:
                invalidate()
            raise


def create_detector_client(config: DetectorConfig) -> DetectorClient:
    """Construct a lazy per-session client; never probes or starts a model."""
    return DetectorClient(config)


def segment_image(
    seg_fn: Any,
    image: np.ndarray,
    text_prompt: str,
    *,
    captured_monotonic_s: float | None = None,
    frame_id: str | None = None,
) -> list[dict]:
    """Forward capture identity to managed clients, preserving injected callable compatibility."""
    if isinstance(seg_fn, DetectorClient):
        return seg_fn(image, text_prompt, captured_monotonic_s=captured_monotonic_s, frame_id=frame_id)
    return cast(list[dict], seg_fn(image, text_prompt=text_prompt))


def init_detector(
    service_url: str = "http://127.0.0.1:8114",
    *,
    timeout_s: float = 30.0,
    request_scoped: bool = False,
    max_frame_age_s: float | None = None,
) -> DetectorClient:
    """Compatibility factory for the current /v1 API with an optional frame-age limit.

    Unmanaged APIs close each request's transport via ``request_scoped=True``.
    """
    return DetectorClient(
        DetectorConfig(
            mode="remote",
            endpoint=HttpEndpointConfig(url=service_url, request_timeout_s=timeout_s),
            max_frame_age_s=max_frame_age_s,
        ),
        request_scoped=request_scoped,
    )
