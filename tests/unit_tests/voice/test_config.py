# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.voice.config."""

from __future__ import annotations

from jiuwensymbiosis.voice.config import VoiceConfig
from jiuwensymbiosis.voice.wake import WAKE_WORD


class TestVoiceConfig:
    def test_defaults(self):
        cfg = VoiceConfig()
        assert cfg.wake_word == WAKE_WORD
        assert cfg.wake_enabled is True
        assert cfg.tts_backend == "null"  # GPU-less default
        assert cfg.asr_backend == "funasr"
        assert isinstance(cfg.wake_variants, tuple)
        assert cfg.wake_word in cfg.wake_variants

    def test_from_dict_none(self):
        assert VoiceConfig.from_dict(None) == VoiceConfig()

    def test_from_dict_ignores_unknown_keys(self):
        cfg = VoiceConfig.from_dict({"wake_word": "小幺小幺", "totally_unknown": 1})
        assert cfg.wake_word == "小幺小幺"

    def test_from_dict_coerces_variants_to_tuple(self):
        cfg = VoiceConfig.from_dict({"wake_variants": ["a", "b"]})
        assert cfg.wake_variants == ("a", "b")

    def test_from_dict_roundtrip_fields(self):
        cfg = VoiceConfig.from_dict({"asr_device": "cpu", "tts_backend": "chattts", "speak_ack": False})
        assert cfg.asr_device == "cpu"
        assert cfg.tts_backend == "chattts"
        assert cfg.speak_ack is False
