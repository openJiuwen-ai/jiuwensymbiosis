# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Voice-layer configuration.

Collapses ``n2_voice``'s ~60 scattered ``N2_*`` environment variables into one
declarative dataclass with a schema. Neutral defaults are chosen so the voice
layer is robot-agnostic and can run without a GPU (``tts_backend="null"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

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
        asr_backend: ASR implementation id (``"funasr"`` | ``"fixed"``).
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
        max_frames: Hard cap on utterance length (0 = unlimited).
        timeout_frames: Frames of leading silence before giving up.
        energy_min: Energy threshold for the fallback VAD (when webrtcvad absent).
        vad_aggressiveness: webrtcvad sensitivity 0 (loosest) – 3 (strictest).
        preroll_frames: Frames kept before speech onset and prepended to the
            utterance so the first syllable is not clipped (0 disables).
        pulse_source: PulseAudio source device (empty = system default).
        tts_backend: TTS implementation id (``"null"`` | ``"chattts"``).
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
    asr_backend: str = "funasr"
    asr_model: str = "paraformer-zh"
    asr_device: str = "cuda:0"
    asr_enable_vad: bool = False
    asr_enable_punc: bool = False

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

    # --- behaviour ---
    ack_text: str = "收到指令，开始执行"
    speak_ack: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> VoiceConfig:
        """Build from a (YAML-loaded) mapping, ignoring unknown keys.

        ``wake_variants`` is coerced to a tuple so the dataclass stays hashable
        and immutable-by-convention.
        """
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "wake_variants" in kwargs and kwargs["wake_variants"] is not None:
            kwargs["wake_variants"] = tuple(kwargs["wake_variants"])
        return cls(**kwargs)
