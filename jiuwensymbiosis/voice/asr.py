# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Speech-to-text — the ASR contract plus a FunASR backend and a mock.

``FunASRBackend`` ports ``n2_voice``'s ``get_asr_model`` + ``transcribe_audio``
(FunASR ``paraformer-zh``, optional VAD/punctuation). ``funasr`` and
``soundfile`` are imported lazily so the module imports without them.
``FixedASRBackend`` returns scripted transcripts for hardware-free tests.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from jiuwensymbiosis.errors import InferenceServiceError
from jiuwensymbiosis.utils.service_http import HttpEndpointConfig, HttpServiceClient
from jiuwensymbiosis.voice.protocol import encode_audio, response_result

if TYPE_CHECKING:
    from jiuwensymbiosis.voice.config import VoiceConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ASRBackend",
    "FunASRBackend",
    "FixedASRBackend",
    "RemoteASRBackend",
    "build_asr_backend",
]


@runtime_checkable
class ASRBackend(Protocol):
    """Transcribe one PCM utterance to text."""

    def transcribe(self, audio: np.ndarray) -> str | None:
        """Return the transcript, or ``None`` if nothing was recognised."""
        ...


class FunASRBackend:
    """FunASR ``paraformer-zh`` backend (lazy-loaded on first transcribe)."""

    def __init__(
        self,
        model_name: str = "paraformer-zh",
        device: str = "cuda:0",
        enable_vad: bool = False,
        enable_punc: bool = False,
        sample_rate: int = 16000,
    ):
        self.model_name = model_name
        self.device = device
        self.enable_vad = enable_vad
        self.enable_punc = enable_punc
        self.sample_rate = sample_rate
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from funasr import AutoModel

            kwargs: dict[str, object] = {
                "model": self.model_name,
                "disable_update": True,
                "device": self.device,
            }
            if self.enable_vad:
                kwargs["vad_model"] = "fsmn-vad"
            if self.enable_punc:
                kwargs["punc_model"] = "ct-punc"
            logger.info("[voice] 加载 ASR 模型 model=%s device=%s", self.model_name, self.device)
            self._model = AutoModel(**kwargs)
            logger.info("[voice] ASR 模型加载完成")
        return self._model

    def load(self) -> None:
        """Explicit eager loading for a reference service's startup readiness."""
        self._ensure_model()

    def close(self) -> None:
        """Release the model reference after synchronous inference has finished."""
        self._model = None

    def transcribe(self, audio: np.ndarray) -> str | None:
        if audio is None or len(audio) == 0:
            return None
        import os
        import tempfile

        import soundfile as sf

        model = self._ensure_model()
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(wav_path, audio, self.sample_rate, format="WAV", subtype="PCM_16")
            result = model.generate(input=wav_path)
            if result and result[0].get("text"):
                return str(result[0]["text"]).strip()
            return None
        finally:
            os.remove(wav_path)


class FixedASRBackend:
    """Mock backend that returns scripted transcripts, one per call."""

    def __init__(self, texts: Iterable[str]):
        self._texts = list(texts)
        self._idx = 0

    def transcribe(self, audio: np.ndarray) -> str | None:  # noqa: ARG002 — mock ignores audio
        if self._idx >= len(self._texts):
            return None
        text = self._texts[self._idx]
        self._idx += 1
        return text


class RemoteASRBackend:
    """Send one bounded WAV utterance to a configured HTTP inference service."""

    def __init__(
        self, endpoint: HttpEndpointConfig, *, sample_rate: int = 16000, max_audio_duration_s: float = 30.0, client=None
    ):
        self.sample_rate = sample_rate
        self.max_audio_duration_s = max_audio_duration_s
        self._client = client if client is not None else HttpServiceClient(endpoint, service="asr")
        self._owns_client = client is None

    def transcribe(self, audio: np.ndarray) -> str | None:
        return self.transcribe_utterance(audio, self.sample_rate, uuid.uuid4().hex)

    def transcribe_utterance(self, audio: np.ndarray, sample_rate: int, utterance_id: str) -> str | None:
        """The source sample rate belongs to this capture, not to the model config."""
        request_id = uuid.uuid4().hex
        payload = {
            "request_id": request_id,
            "utterance_id": utterance_id,
            "audio": encode_audio(audio, sample_rate, max_duration_s=self.max_audio_duration_s),
        }
        envelope = self._client.request_json("POST", "/v1/transcribe", payload, request_id=request_id)
        result = response_result(envelope, request_id, service="asr")
        text = result.get("text")
        if result.get("utterance_id") != utterance_id or "text" not in result:
            raise InferenceServiceError("Invalid ASR result", code="inference_protocol_error", service="asr")
        if text is not None and (not isinstance(text, str) or len(text) > 10000):
            raise InferenceServiceError("Invalid ASR result", code="inference_protocol_error", service="asr")
        return text.strip() or None if isinstance(text, str) else None

    def bind_cancel_token(self, token) -> None:
        self._client.bind_cancel_token(token)

    def close(self, timeout_s: float = 5.0) -> None:
        """Release owned HTTP resources within the caller's remaining budget."""
        if self._owns_client:
            self._client.close(timeout_s=timeout_s)


def build_asr_backend(config: VoiceConfig) -> ASRBackend:
    """Construct the ASR backend named by ``config.asr_backend``."""
    backend = config.asr_backend.lower()
    if backend == "funasr":
        return FunASRBackend(
            model_name=config.asr_model,
            device=config.asr_device,
            enable_vad=config.asr_enable_vad,
            enable_punc=config.asr_enable_punc,
            sample_rate=config.sample_rate,
        )
    if backend == "fixed":
        return FixedASRBackend([])
    if backend == "remote":
        if config.asr_endpoint is None:
            raise ValueError("voice.asr.endpoint is required for remote ASR")
        return RemoteASRBackend(
            config.asr_endpoint, sample_rate=config.sample_rate, max_audio_duration_s=config.max_audio_duration_s
        )
    if backend == "disabled":
        raise ValueError("Voice ASR is disabled; configure voice.asr.backend as remote or funasr, or use --voice-text")
    raise ValueError(f"未知 asr_backend: {config.asr_backend!r}")
