# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Robot-agnostic voice front-end for jiuwensymbiosis.

Pipeline: microphone → wake word → ASR → ``on_command`` callback → TTS feedback.
The callback hands transcribed text to :func:`jiuwensymbiosis.run_robot_task`, so
the existing DeepAgent stays the decision-maker; this layer only adds the voice
I/O. See ``docs/voice-control-integration-design.md``.

Heavy/optional deps (funasr, sounddevice, soundfile, webrtcvad, ChatTTS) are
imported lazily by the real backends; importing this package pulls in none of
them. Install with ``pip install -e ".[voice]"``.
"""

from __future__ import annotations

from jiuwensymbiosis.voice.asr import (
    ASRBackend,
    FixedASRBackend,
    FunASRBackend,
    build_asr_backend,
)
from jiuwensymbiosis.voice.audio import (
    AudioSource,
    FileAudioSource,
    PulseAudioSource,
    RecordTuning,
    SoundDeviceSource,
    build_audio_source,
)
from jiuwensymbiosis.voice.config import VoiceConfig
from jiuwensymbiosis.voice.loop import OnCommand, VoiceLoop, result_to_speech
from jiuwensymbiosis.voice.tts import (
    ChatTTSBackend,
    NullTTS,
    TTSBackend,
    build_tts_backend,
)
from jiuwensymbiosis.voice.wake import (
    WAKE_WORD,
    WAKE_WORD_VARIANTS,
    normalize_wake_text,
    split_after_last_wake,
)

__all__ = [
    # config + orchestration
    "VoiceConfig",
    "VoiceLoop",
    "OnCommand",
    "result_to_speech",
    # wake
    "WAKE_WORD",
    "WAKE_WORD_VARIANTS",
    "normalize_wake_text",
    "split_after_last_wake",
    # audio
    "AudioSource",
    "RecordTuning",
    "FileAudioSource",
    "PulseAudioSource",
    "SoundDeviceSource",
    "build_audio_source",
    # asr
    "ASRBackend",
    "FunASRBackend",
    "FixedASRBackend",
    "build_asr_backend",
    # tts
    "TTSBackend",
    "NullTTS",
    "ChatTTSBackend",
    "build_tts_backend",
]
