# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Voice-layer configuration.

Collapses ``n2_voice``'s ~60 scattered ``N2_*`` environment variables into one
declarative dataclass with a schema. Neutral defaults are chosen so the voice
layer is robot-agnostic and can run without a GPU (``tts_backend="null"``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from typing import Any

from jiuwensymbiosis.utils.service_http import HttpEndpointConfig
from jiuwensymbiosis.utils.validation import require_positive_number
from jiuwensymbiosis.voice.wake import WAKE_WORD, WAKE_WORD_VARIANTS

__all__ = ["VoiceConfig"]


@dataclass
class VoiceConfig:
    """Declarative configuration for the voice front-end.

    Attributes:
        wake_word: Primary wake word.
        wake_variants: Near-homophone spellings accepted as the wake word.
        wake_enabled: If ``False``, every transcript is treated as a command
            (no wake gating) — useful for push-to-talk or text fallback.
        asr_backend: ASR implementation id (remote, funasr, fixed, disabled).
        asr_model: FunASR model name.
        asr_device: Torch device for ASR (``"cuda:0"`` | ``"cpu"``).
        asr_enable_vad: Enable FunASR's built-in VAD model.
        asr_enable_punc: Enable FunASR's punctuation model.
        audio_backend: Capture backend (``"pulse"`` | ``"sounddevice"`` | ``"file"``).
        sample_rate: Capture sample rate (Hz).
        chunk: Samples per frame (480 = 30ms @ 16kHz, valid for WebRTC VAD).
        silence_frames: Trailing silent frames that end an utterance (30ms each);
            larger tolerates longer mid-phrase pauses without splitting.
        min_frames: Minimum frames before silence can end an utterance.
        max_frames: Capture cap (0 selects the max_audio_duration_s bound).
        timeout_frames: Frames of leading silence before giving up.
        energy_min: Energy threshold for the fallback VAD (when webrtcvad absent).
        vad_aggressiveness: webrtcvad sensitivity 0 (loosest) – 3 (strictest).
        preroll_frames: Frames kept before speech onset and prepended to the
            utterance so the first syllable is not clipped (0 disables).
        pulse_source: PulseAudio source device (empty = system default).
        tts_backend: TTS implementation id (remote, chattts, null).
        tts_module_path: Filesystem path to the ChatTTS ``tts.py`` module.
        tts_async: Play TTS on a background thread.
        ack_text: Spoken immediately on receiving a command (task may be slow).
        speak_ack: Whether to speak ``ack_text`` before dispatching.
    """

    # --- wake ---
    wake_word: str = WAKE_WORD
    wake_variants: tuple[str, ...] = field(default_factory=lambda: tuple(WAKE_WORD_VARIANTS))
    wake_enabled: bool = True

    # --- ASR ---
    asr_backend: str = "disabled"
    asr_model: str = "paraformer-zh"
    asr_device: str = "cuda:0"
    asr_enable_vad: bool = False
    asr_enable_punc: bool = False
    asr_endpoint: HttpEndpointConfig | None = None
    max_audio_duration_s: float = 30.0
    max_command_age_s: float = 20.0
    wake_command_timeout_s: float = 10.0

    # --- audio capture ---
    audio_backend: str = "pulse"
    sample_rate: int = 16000
    chunk: int = 480
    silence_frames: int = 25  # 拖尾静音 750ms 才断句；短停顿不再把一句切两段
    min_frames: int = 8
    max_frames: int = 0
    timeout_frames: int = 200
    energy_min: int = 400  # 帧能量(mean-abs)超此值即判为语音；安静的笔记本 DMIC 调低
    vad_aggressiveness: int = 2  # webrtcvad 灵敏度 0(最松)~3(最严)；DMIC 小声用 1~2
    preroll_frames: int = 8  # 起始前 240ms 一并带上，补回被吞的首字
    pulse_source: str = ""

    # --- TTS ---
    tts_backend: str = "null"
    tts_module_path: str | None = None
    tts_async: bool = True
    tts_endpoint: HttpEndpointConfig | None = None
    tts_voice: str = "default"
    playback_backend: str = "sounddevice"
    playback_device: str | int | None = None
    tts_queue_size: int = 8
    tts_max_text_length: int = 2000
    tts_max_audio_duration_s: float = 120.0
    shutdown_timeout_s: float = 5.0

    # --- behaviour ---
    ack_text: str = "收到指令，开始执行"
    speak_ack: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> VoiceConfig:
        """Normalize strict nested YAML, accepting explicit legacy backend fields.

        Flat Python attributes remain available during the configuration migration.
        Merely supplying a model/device never opts into loading a local model.
        """
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("voice must be a mapping")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - {"asr", "tts"}
        if unknown:
            raise ValueError(f"Unknown voice configuration fields: {sorted(unknown)}")
        kwargs = dict(data)
        for group, names, backend_fields in (
            (
                "asr",
                {
                    "backend": "asr_backend",
                    "model": "asr_model",
                    "device": "asr_device",
                    "enable_vad": "asr_enable_vad",
                    "enable_punc": "asr_enable_punc",
                    "endpoint": "asr_endpoint",
                    "max_audio_duration_s": "max_audio_duration_s",
                    "max_command_age_s": "max_command_age_s",
                },
                {
                    "funasr": {"model", "device", "enable_vad", "enable_punc"},
                    "remote": {"endpoint"},
                    "disabled": set(),
                    "fixed": set(),
                },
            ),
            (
                "tts",
                {
                    "backend": "tts_backend",
                    "module_path": "tts_module_path",
                    "async_play": "tts_async",
                    "endpoint": "tts_endpoint",
                    "voice": "tts_voice",
                    "playback_backend": "playback_backend",
                    "playback_device": "playback_device",
                    "queue_size": "tts_queue_size",
                    "max_text_length": "tts_max_text_length",
                    "max_audio_duration_s": "tts_max_audio_duration_s",
                },
                {
                    "chattts": {"module_path"},
                    "remote": {"endpoint", "voice", "playback_backend", "playback_device"},
                    "null": set(),
                },
            ),
        ):
            legacy = set(data) & set(names.values())
            if group not in data:
                if legacy:
                    warnings.warn(
                        f"Flat voice {group} fields are deprecated; use voice.{group}", DeprecationWarning, stacklevel=2
                    )
                if data.get(f"{group}_backend") == "remote":
                    forbidden = {names[k] for k in set().union(*backend_fields.values()) - backend_fields["remote"]}
                    if forbidden & legacy:
                        raise ValueError(f"Remote voice.{group} cannot configure local model parameters")
                continue
            if legacy:
                raise ValueError(f"voice.{group} conflicts with legacy fields: {sorted(legacy)}")
            section = kwargs.pop(group)
            if not isinstance(section, dict):
                raise ValueError(f"voice.{group} must be a mapping")
            extra = set(section) - set(names)
            if extra:
                raise ValueError(f"Unknown voice.{group} fields: {sorted(extra)}")
            backend = section.get("backend", "disabled" if group == "asr" else "null")
            if backend not in backend_fields:
                raise ValueError(f"Unknown voice.{group}.backend: {backend!r}")
            exclusive = set().union(*backend_fields.values())
            invalid = (set(section) & exclusive) - backend_fields[backend]
            if invalid:
                raise ValueError(f"voice.{group} backend {backend!r} does not accept {sorted(invalid)}")
            for key, value in section.items():
                kwargs[names[key]] = HttpEndpointConfig.from_dict(value) if key == "endpoint" else value
        if "wake_variants" in kwargs and kwargs["wake_variants"] is not None:
            kwargs["wake_variants"] = tuple(kwargs["wake_variants"])
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self) -> None:
        """Reject invalid bounds and missing explicitly-selected service settings."""
        for key in (
            "max_audio_duration_s",
            "max_command_age_s",
            "wake_command_timeout_s",
            "tts_max_audio_duration_s",
            "shutdown_timeout_s",
        ):
            require_positive_number(getattr(self, key), message=f"voice.{key} must be finite and positive")
        for key in (
            "sample_rate",
            "chunk",
            "silence_frames",
            "min_frames",
            "timeout_frames",
            "tts_queue_size",
            "tts_max_text_length",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"voice.{key} must be a positive integer")
        for key in ("max_frames", "preroll_frames", "energy_min"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"voice.{key} must be a non-negative integer")
        if self.asr_backend not in {"remote", "funasr", "fixed", "disabled"}:
            raise ValueError(f"Unknown asr_backend: {self.asr_backend!r}")
        if self.tts_backend not in {"remote", "chattts", "null", "none", "off"}:
            raise ValueError(f"Unknown tts_backend: {self.tts_backend!r}")
        for group in ("asr", "tts"):
            if getattr(self, f"{group}_backend") == "remote":
                endpoint = getattr(self, f"{group}_endpoint")
                if not isinstance(endpoint, HttpEndpointConfig):
                    raise ValueError(f"voice.{group}.endpoint is required")
        if self.tts_backend == "chattts" and not self.tts_module_path:
            raise ValueError("voice.tts.module_path is required for the local ChatTTS backend")
        if self.playback_backend != "sounddevice":
            raise ValueError("voice.tts.playback_backend must be sounddevice")
