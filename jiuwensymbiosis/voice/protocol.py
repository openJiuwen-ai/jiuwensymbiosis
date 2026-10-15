# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded, model-independent WAV and response contracts shared by speech peers."""

from __future__ import annotations

import base64
import binascii
import io
import math
import wave
from typing import Any, cast

import numpy as np

from jiuwensymbiosis.errors import InferenceServiceError

__all__ = ["encode_audio", "decode_audio", "response_result"]


def encode_audio(audio: np.ndarray, sample_rate: int, *, max_duration_s: float = 120.0) -> dict[str, str]:
    """Encode mono PCM16 to WAV without importing a model or audio driver."""
    if audio.ndim != 1 or audio.dtype != np.int16 or not audio.size:
        raise ValueError("Audio must be non-empty mono int16 PCM")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 96000:
        raise ValueError("Unsupported audio sample rate")
    if audio.size / sample_rate > max_duration_s:
        raise ValueError("Audio exceeds the configured duration limit")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.astype("<i2", copy=False).tobytes())
    return {"mime_type": "audio/wav", "data_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}


def decode_audio(
    payload: Any, *, max_duration_s: float = 120.0, max_bytes: int = 16 * 1024 * 1024
) -> tuple[np.ndarray, int]:
    """Validate size/header before allocating decoded mono PCM16 samples."""
    if not isinstance(payload, dict) or payload.get("mime_type") != "audio/wav":
        raise ValueError("Audio must declare mime_type audio/wav")
    encoded = payload.get("data_base64")
    if not isinstance(encoded, str) or len(encoded) > ((max_bytes + 2) // 3) * 4:
        raise ValueError("Invalid or oversized audio data")
    try:
        data = base64.b64decode(encoded, validate=True)
        if len(data) > max_bytes:
            raise ValueError("Audio exceeds the byte limit")
        with wave.open(io.BytesIO(data), "rb") as wav:
            rate, frames = wav.getframerate(), wav.getnframes()
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                raise ValueError("Audio must be mono, uncompressed PCM16")
            if not 8000 <= rate <= 96000 or frames <= 0 or frames / rate > max_duration_s:
                raise ValueError("Invalid audio sample rate or duration")
            raw = wav.readframes(frames)
            if len(raw) != frames * 2:
                raise ValueError("Truncated WAV samples")
        return np.frombuffer(raw, dtype="<i2").astype(np.int16, copy=True), rate
    except (binascii.Error, wave.Error, EOFError) as exc:
        raise ValueError("Invalid PCM16 WAV data") from exc


def response_result(envelope: Any, request_id: str, *, service: str) -> dict[str, Any]:
    """Reject wrong response identity instead of attributing it to this utterance."""
    if not isinstance(envelope, dict):
        raise InferenceServiceError(
            "Invalid speech response envelope", code="inference_protocol_error", service=service
        )
    if (
        type(envelope.get("schema_version")) is not int
        or envelope.get("schema_version") != 1
        or envelope.get("request_id") != request_id
    ):
        raise InferenceServiceError(
            "Invalid speech response envelope", code="inference_protocol_error", service=service
        )
    if "error" in envelope or not isinstance(envelope.get("result"), dict):
        raise InferenceServiceError(
            "Invalid speech response envelope", code="inference_protocol_error", service=service
        )
    return cast(dict[str, Any], envelope["result"])


def validate_audio_metadata(result: dict[str, Any], audio: np.ndarray, sample_rate: int) -> None:
    """Check redundant response metadata against the actual WAV header."""
    duration = result.get("duration_s")
    if (
        result.get("sample_rate") != sample_rate
        or type(result.get("channels")) is not int
        or result.get("channels") != 1
    ):
        raise ValueError("Audio response metadata disagrees with the WAV header")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration):
        raise ValueError("Audio response metadata disagrees with the WAV header")
    if abs(duration - audio.size / sample_rate) > 1 / sample_rate:
        raise ValueError("Audio response metadata disagrees with the WAV header")
