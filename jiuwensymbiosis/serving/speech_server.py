# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Optional, independently deployed HTTP ASR/TTS service.

Start with ``python -m jiuwensymbiosis.serving.speech_server --help``.
Only selected model backends are imported; no microphone or speaker is used.
"""

from __future__ import annotations

import argparse
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from jiuwensymbiosis.serving.http_support import InferenceAdmission, configure_service
from jiuwensymbiosis.utils.logging import get_logger
from jiuwensymbiosis.utils.validation import require_positive_number
from jiuwensymbiosis.voice.protocol import decode_audio, encode_audio

logger = get_logger(__name__)


class _AudioPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mime_type: str
    data_base64: str


class _ASRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(min_length=1, max_length=128)
    utterance_id: str = Field(min_length=1, max_length=128)
    audio: _AudioPayload


class _TTSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1)
    voice: str = Field(default="default", min_length=1, max_length=128)


def create_app(
    *,
    asr=None,
    synthesizer=None,
    max_request_bytes: int = 16_777_216,
    max_response_bytes: int = 16_777_216,
    max_audio_duration_s: float = 30.0,
    max_tts_duration_s: float = 120.0,
    max_text_length: int = 2000,
    max_queue: int = 8,
    queue_timeout_s: float = 30.0,
    owns_backends: bool = False,
) -> FastAPI:
    """Build an ASR-only, TTS-only, or combined service, with injectable models."""
    if asr is None and synthesizer is None:
        raise ValueError("Enable at least one speech backend")
    for name, value in (
        ("max_request_bytes", max_request_bytes),
        ("max_response_bytes", max_response_bytes),
        ("max_text_length", max_text_length),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for duration_name, duration_value in (
        ("max_audio_duration_s", max_audio_duration_s),
        ("max_tts_duration_s", max_tts_duration_s),
        ("queue_timeout_s", queue_timeout_s),
    ):
        require_positive_number(duration_value, message=f"{duration_name} must be finite and positive")
    if isinstance(max_queue, bool) or not isinstance(max_queue, int) or max_queue < 0:
        raise ValueError("max_queue must be a non-negative integer")
    admission = InferenceAdmission(max_queue=max_queue, queue_timeout_s=queue_timeout_s)
    ready = False

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal ready
        attempted = []
        try:
            for backend in (asr, synthesizer):
                if backend is None:
                    continue
                # load() can allocate model resources before it reports failure.
                attempted.append(backend)
                load = getattr(backend, "load", None)
                if load is not None:
                    await admission.run(load)
            ready = True
            yield
        finally:
            ready = False
            if owns_backends:
                if not await admission.wait_for_idle(timeout_s=30.0):
                    raise RuntimeError("Speech model work is still pending; model handles were retained")
                cleanup_errors = []
                for backend in reversed(attempted):
                    close = getattr(backend, "close", None)
                    if close is not None:
                        try:
                            close()
                        except Exception as exc:
                            cleanup_errors.append(exc)
                if cleanup_errors:
                    raise ExceptionGroup("Speech backend cleanup is incomplete", cleanup_errors)

    app = FastAPI(title="JiuwenSymbiosis speech inference", lifespan=lifespan)
    configure_service(app, max_request_bytes=max_request_bytes)

    def success(request_id: str, result: dict[str, Any]) -> dict[str, Any]:
        envelope = {"schema_version": 1, "request_id": request_id, "result": result}
        if len(json.dumps(envelope, ensure_ascii=False).encode("utf-8")) > max_response_bytes:
            raise HTTPException(503, detail={"message": "Speech response exceeds the configured byte limit"})
        return envelope

    def require_ready(backend) -> None:
        if not ready or backend is None:
            raise HTTPException(503, detail={"code": "inference_unavailable", "message": "Speech backend is not ready"})

    @app.get("/v1/health")
    async def health(request: Request):
        return success(
            request.headers.get("x-request-id") or uuid.uuid4().hex,
            {
                "service": "speech",
                "ready": ready,
                "capabilities": [name for name, backend in (("asr", asr), ("tts", synthesizer)) if backend is not None],
                "formats": ["audio/wav; pcm_s16le; mono"],
                "voices": list(getattr(synthesizer, "voices", ())),
                "limits": {
                    "max_request_bytes": max_request_bytes,
                    "max_response_bytes": max_response_bytes,
                    "max_audio_duration_s": max_audio_duration_s,
                    "max_tts_duration_s": max_tts_duration_s,
                    "max_text_length": max_text_length,
                    "max_queue": max_queue,
                    "queue_timeout_s": queue_timeout_s,
                },
                "models": {"asr": getattr(asr, "model_name", None), "tts": getattr(synthesizer, "model_version", None)},
            },
        )

    @app.post("/v1/transcribe")
    async def transcribe(body: _ASRRequest, request: Request):
        request.state.request_id = body.request_id
        require_ready(asr)
        try:
            audio, rate = decode_audio(
                body.audio.model_dump(), max_duration_s=max_audio_duration_s, max_bytes=max_request_bytes
            )
            if rate != getattr(asr, "sample_rate", 16000):
                raise ValueError("The ASR model requires a different sample rate")
        except ValueError as exc:
            raise HTTPException(422, detail={"message": "Unsupported or invalid ASR audio"}) from exc
        try:
            text = await admission.run(asr.transcribe, audio, request=request)
            if text is not None and (not isinstance(text, str) or len(text) > 10000):
                raise ValueError("Invalid model transcript")
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("ASR inference failed (%s)", type(exc).__name__)
            raise HTTPException(503, detail={"message": "ASR inference failed"}) from exc
        return success(
            body.request_id,
            {"utterance_id": body.utterance_id, "text": text.strip() or None if isinstance(text, str) else None},
        )

    @app.post("/v1/synthesize")
    async def synthesize(body: _TTSRequest, request: Request):
        request.state.request_id = body.request_id
        require_ready(synthesizer)
        if (
            not body.text.strip()
            or len(body.text) > max_text_length
            or body.voice not in getattr(synthesizer, "voices", ("default",))
        ):
            raise HTTPException(422, detail={"message": "Invalid TTS text or unsupported voice"})
        try:
            audio, rate = await admission.run(synthesizer.synthesize, body.text, body.voice, request=request)
            encoded = encode_audio(audio, rate, max_duration_s=max_tts_duration_s)
            if len(encoded["data_base64"]) + 1024 > max_response_bytes:
                raise ValueError("Synthesized audio exceeds the response limit")
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("TTS inference failed (%s)", type(exc).__name__)
            raise HTTPException(503, detail={"message": "TTS inference failed"}) from exc
        return success(
            body.request_id, {"audio": encoded, "sample_rate": rate, "channels": 1, "duration_s": len(audio) / rate}
        )

    return app


def main(argv: list[str] | None = None) -> None:
    """Load explicitly selected, locally prepared models and serve HTTP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8115)
    parser.add_argument("--asr", choices=("disabled", "funasr"), default="disabled")
    parser.add_argument("--tts", choices=("disabled", "chattts"), default="disabled")
    parser.add_argument("--asr-model-path")
    parser.add_argument("--tts-model-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-request-bytes", type=int, default=16_777_216)
    parser.add_argument("--max-response-bytes", type=int, default=16_777_216)
    parser.add_argument("--max-audio-duration-s", type=float, default=30.0)
    parser.add_argument("--max-tts-duration-s", type=float, default=120.0)
    parser.add_argument("--max-text-length", type=int, default=2000)
    parser.add_argument("--max-queue", type=int, default=8)
    parser.add_argument("--queue-timeout-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    asr = synthesizer = None
    if args.asr == "funasr":
        if not args.asr_model_path or not Path(args.asr_model_path).expanduser().is_dir():
            parser.error("--asr funasr requires a prepared --asr-model-path directory")
        from jiuwensymbiosis.voice.asr import FunASRBackend

        asr = FunASRBackend(
            str(Path(args.asr_model_path).expanduser().resolve()), device=args.device, sample_rate=args.sample_rate
        )
    if args.tts == "chattts":
        if not args.tts_model_path:
            parser.error("--tts chattts requires --tts-model-path")
        from jiuwensymbiosis.serving.speech_models import ChatTTSSynthesizer

        synthesizer = ChatTTSSynthesizer(args.tts_model_path, device=args.device)
    if asr is None and synthesizer is None:
        parser.error("Enable --asr funasr or --tts chattts")
    app = create_app(
        asr=asr,
        synthesizer=synthesizer,
        owns_backends=True,
        max_request_bytes=args.max_request_bytes,
        max_response_bytes=args.max_response_bytes,
        max_audio_duration_s=args.max_audio_duration_s,
        max_tts_duration_s=args.max_tts_duration_s,
        max_text_length=args.max_text_length,
        max_queue=args.max_queue,
        queue_timeout_s=args.queue_timeout_s,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
