# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Optional local speech or HTTP synthesis followed by Agent-side playback."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

from jiuwensymbiosis.errors import InferenceServiceError
from jiuwensymbiosis.utils.logging import get_logger
from jiuwensymbiosis.utils.service_http import HttpEndpointConfig, HttpServiceClient
from jiuwensymbiosis.voice.audio import AudioPlayer, SoundDevicePlayer
from jiuwensymbiosis.voice.protocol import decode_audio, response_result, validate_audio_metadata

if TYPE_CHECKING:
    from jiuwensymbiosis.voice.config import VoiceConfig

logger = get_logger(__name__)
__all__ = ["TTSBackend", "NullTTS", "ChatTTSBackend", "RemoteTTSBackend", "build_tts_backend"]


class _SpeakerOptions(TypedDict):
    async_play: bool
    queue_size: int
    timeout_s: float


@runtime_checkable
class TTSBackend(Protocol):
    """Ordered feedback; preload never plays audio."""

    # fmt: off
    def speak(self, text: str) -> None:
        ...

    def preload(self, text: str) -> None:
        ...

    def wait(self) -> None:
        ...
    # fmt: on


class NullTTS:
    """No-model, no-audio default."""

    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        if text:
            self.spoken.append(text)
            logger.info("[voice][tts:null] %s", text)

    @staticmethod
    def preload(text: str) -> None:
        return None

    @staticmethod
    def wait() -> None:
        return None


class _OrderedSpeaker:
    """One bounded FIFO and one worker; queued feedback cannot grow threads."""

    def __init__(self, *, async_play: bool = True, queue_size: int = 8, timeout_s: float = 5.0):
        self.async_play = async_play
        self._queue_size = queue_size
        self._shutdown_timeout_s = timeout_s
        self._condition = threading.Condition()
        self._queue: deque[str] = deque()
        self._worker: threading.Thread | None = None
        self._active = False
        self._closed = False
        self._cancel_token = None
        self._unregister_cancel: Callable[[], None] | None = None
        self.last_error: Exception | None = None

    def bind_cancel_token(self, token) -> None:
        if self._unregister_cancel is not None:
            self._unregister_cancel()
        self._cancel_token = token
        self._unregister_cancel = token.on_cancel(self._cancel_playback) if token is not None else None

    def _cancel_playback(self) -> None:
        """Wake listeners immediately; active work remains registered until it exits."""
        with self._condition:
            self._closed = True
            self._queue.clear()
            self._condition.notify_all()

    def speak(self, text: str) -> None:
        if not text:
            return
        if self._cancel_token:
            self._cancel_token.raise_if_set()
        if not self.async_play:
            with self._condition:
                if self._closed:
                    raise RuntimeError("TTS backend is closed")
                if self._active:
                    raise InferenceServiceError("TTS playback is busy", code="inference_busy", service="tts")
                self._active = True
            finish = self._cancel_token.register_work("voice.tts.playback") if self._cancel_token else lambda: None
            try:
                if self._cancel_token:
                    self._cancel_token.raise_if_set()
                self._say(text)
            finally:
                finish()
                with self._condition:
                    self._active = False
                    self._condition.notify_all()
            return
        with self._condition:
            if self._closed:
                raise RuntimeError("TTS backend is closed")
            if len(self._queue) >= self._queue_size:
                raise InferenceServiceError("TTS playback queue is full", code="inference_busy", service="tts")
            self._queue.append(text)
            if self._worker is None:
                self._worker = threading.Thread(target=self._consume, daemon=True)
                try:
                    self._worker.start()
                except BaseException:
                    # A worker that never started cannot own its queued work.
                    self._worker = None
                    self._queue.pop()
                    raise
            self._condition.notify_all()

    def _consume(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._queue)
                if not self._queue:
                    return
                text = self._queue.popleft()
                self._active = True
            finish = self._cancel_token.register_work("voice.tts.playback") if self._cancel_token else lambda: None
            try:
                if self._cancel_token:
                    self._cancel_token.raise_if_set()
                self._say(text)
            except Exception as exc:
                self.last_error = exc
                logger.warning("[voice] TTS synthesis/playback failed (%s)", type(exc).__name__)
            finally:
                finish()
                with self._condition:
                    self._active = False
                    self._condition.notify_all()

    def _say(self, text: str) -> None:
        raise NotImplementedError

    @staticmethod
    def preload(text: str) -> None:
        """Explicit no-op: no unbounded synthesis cache is kept."""
        return None

    def wait(self, timeout_s: float | None = None) -> None:
        """Wait for normal playback without consuming the separate shutdown budget."""
        self._wait_for_idle(timeout_s, cancellable=True)

    def _wait_for_idle(self, timeout_s: float | None, *, cancellable: bool) -> None:
        with self._condition:
            finished = self._condition.wait_for(
                lambda: (
                    (not self._queue and not self._active)
                    or bool(cancellable and self._cancel_token and self._cancel_token.is_set())
                ),
                timeout=timeout_s,
            )
            if cancellable and self._cancel_token:
                self._cancel_token.raise_if_set()
            if not finished:
                raise TimeoutError("TTS work remains pending after the drain timeout")

    def close(self, *, discard: bool = False, timeout_s: float | None = None) -> None:
        """Drain normally; cancellation drops queued feedback but retains active work."""
        deadline = time.monotonic() + (self._shutdown_timeout_s if timeout_s is None else timeout_s)
        with self._condition:
            self._closed = True
            if discard:
                self._queue.clear()
            self._condition.notify_all()
        # Cancellation may abandon a listener, but never its cleanup evidence.
        self._wait_for_idle(max(0.0, deadline - time.monotonic()), cancellable=False)
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._worker.is_alive():
                raise TimeoutError("TTS worker is still active")
        if self._unregister_cancel is not None:
            self._unregister_cancel()
            self._unregister_cancel = None


class ChatTTSBackend(_OrderedSpeaker):
    """Explicit compatibility backend for an external local ``tts.py`` speaker.

    This adapter invokes safe_speak_text/speak_text, so it is never used as a
    remote synthesizer. Its external playback may be non-cancellable.
    """

    def __init__(self, module_path: str | Path, async_play: bool = True, **kwargs):
        super().__init__(async_play=async_play, **kwargs)
        self.module_path = Path(module_path).expanduser()
        self._speaker = None
        self._loaded = False

    def _ensure_speaker(self):
        if self._loaded:
            return self._speaker
        self._loaded = True
        if not self.module_path.exists():
            logger.warning("[voice] 未找到配置的 ChatTTS 模块")
            return None
        try:
            spec = spec_from_file_location("jws_voice_chattts", self.module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Unable to load the configured ChatTTS module")
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            self._speaker = getattr(module, "safe_speak_text", None) or getattr(module, "speak_text", None)
            if self._speaker is None:
                raise RuntimeError("ChatTTS module does not expose a speaker")
        except Exception as exc:
            logger.warning("[voice] ChatTTS module loading failed (%s)", type(exc).__name__)
        return self._speaker

    def _say(self, text: str) -> None:
        speaker = self._ensure_speaker()
        if speaker:
            speaker(text)


class RemoteTTSBackend(_OrderedSpeaker):
    """Synthesize remotely, validate WAV, then play at its real rate locally."""

    def __init__(
        self,
        endpoint: HttpEndpointConfig,
        *,
        voice: str = "default",
        player: AudioPlayer | None = None,
        playback_device=None,
        max_text_length: int = 2000,
        max_audio_duration_s: float = 120.0,
        client=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.voice = voice
        self.max_text_length = max_text_length
        self.max_audio_duration_s = max_audio_duration_s
        self._client = client if client is not None else HttpServiceClient(endpoint, service="tts")
        self._owns_client = client is None
        self._player = player if player is not None else SoundDevicePlayer(device=playback_device)
        self._owns_player = player is None
        self._max_audio_bytes = endpoint.max_response_bytes

    def bind_cancel_token(self, token) -> None:
        super().bind_cancel_token(token)
        self._client.bind_cancel_token(token)

    def _cancel_playback(self) -> None:
        super()._cancel_playback()
        if self._owns_player:
            self._player.close()

    def speak(self, text: str) -> None:
        if not isinstance(text, str) or len(text) > self.max_text_length:
            raise ValueError("TTS text exceeds the configured limit")
        super().speak(text)

    def _say(self, text: str) -> None:
        request_id = uuid.uuid4().hex
        envelope = self._client.request_json(
            "POST",
            "/v1/synthesize",
            {"request_id": request_id, "text": text, "voice": self.voice},
            request_id=request_id,
        )
        result = response_result(envelope, request_id, service="tts")
        try:
            audio, rate = decode_audio(
                result.get("audio"), max_duration_s=self.max_audio_duration_s, max_bytes=self._max_audio_bytes
            )
            validate_audio_metadata(result, audio, rate)
        except ValueError as exc:
            raise InferenceServiceError(
                "Invalid synthesized audio", code="inference_protocol_error", service="tts"
            ) from exc
        if self._cancel_token:
            self._cancel_token.raise_if_set()
        self._player.play(audio, rate)

    def close(self, *, discard: bool = False, timeout_s: float | None = None) -> None:
        deadline = time.monotonic() + (self._shutdown_timeout_s if timeout_s is None else timeout_s)

        def close_client() -> None:
            if self._owns_client and not self._client.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("TTS HTTP cleanup budget exhausted")
                self._client.close(timeout_s=remaining)

        try:
            if discard:
                # Reject new speech and drop queued items even if HTTP cleanup
                # times out. Retain the active worker and client for a retry.
                self._cancel_playback()
                close_client()
            super().close(discard=discard, timeout_s=max(0.0, deadline - time.monotonic()))
        finally:
            try:
                if not discard:
                    close_client()
            finally:
                if self._owns_player:
                    self._player.close()


def build_tts_backend(config: VoiceConfig) -> TTSBackend:
    """Construct only the explicitly selected model or remote backend."""
    backend = config.tts_backend.lower()
    if backend in ("null", "none", "off"):
        return NullTTS()
    kwargs: _SpeakerOptions = {
        "async_play": config.tts_async,
        "queue_size": config.tts_queue_size,
        "timeout_s": config.shutdown_timeout_s,
    }
    if backend == "chattts":
        if not config.tts_module_path:
            raise ValueError("voice.tts.module_path is required for the local ChatTTS backend")
        return ChatTTSBackend(config.tts_module_path, **kwargs)
    if backend == "remote":
        if config.tts_endpoint is None:
            raise ValueError("voice.tts.endpoint is required for remote TTS")
        return RemoteTTSBackend(
            config.tts_endpoint,
            voice=config.tts_voice,
            playback_device=config.playback_device,
            max_text_length=config.tts_max_text_length,
            max_audio_duration_s=config.tts_max_audio_duration_s,
            **kwargs,
        )
    raise ValueError(f"Unknown tts_backend: {config.tts_backend!r}")
