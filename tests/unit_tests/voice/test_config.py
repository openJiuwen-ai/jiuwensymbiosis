# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.voice.config."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.voice.config import VoiceConfig
from jiuwensymbiosis.voice.wake import WAKE_WORD


class TestVoiceConfig:
    def test_defaults(self):
        cfg = VoiceConfig()
        assert cfg.wake_word == WAKE_WORD
        assert cfg.wake_enabled is True
        assert cfg.tts_backend == "null"  # GPU-less default
        assert cfg.asr_backend == "disabled"
        assert isinstance(cfg.wake_variants, tuple)
        assert cfg.wake_word in cfg.wake_variants

    def test_from_dict_none(self):
        assert VoiceConfig.from_dict(None) == VoiceConfig()

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="totally_unknown"):
            VoiceConfig.from_dict({"wake_word": "小幺小幺", "totally_unknown": 1})

    def test_from_dict_coerces_variants_to_tuple(self):
        cfg = VoiceConfig.from_dict({"wake_variants": ["a", "b"]})
        assert cfg.wake_variants == ("a", "b")

    def test_from_dict_roundtrip_fields(self):
        with pytest.warns(DeprecationWarning):
            cfg = VoiceConfig.from_dict(
                {"asr_device": "cpu", "tts_backend": "chattts", "tts_module_path": "/tmp/tts.py", "speak_ack": False}
            )
        assert cfg.asr_device == "cpu"
        assert cfg.tts_backend == "chattts"
        assert cfg.speak_ack is False

    @pytest.mark.parametrize("group", ["asr", "tts"])
    def test_nested_remote_accepts_url_only(self, group):
        cfg = VoiceConfig.from_dict(
            {
                group: {
                    "backend": "remote",
                    "endpoint": {"url": "https://speech.example/prefix"},
                },
            }
        )
        assert getattr(cfg, f"{group}_endpoint").url == "https://speech.example/prefix"
        assert getattr(cfg, f"{group}_backend") == "remote"

    @pytest.mark.parametrize(
        "data",
        [
            {"asr": {"backend": "remote"}},
            {"asr": {"backend": "remote", "device": "cuda:0", "endpoint": {"url": "http://speech"}}},
            {"asr": {"backend": "funasr"}, "asr_device": "cpu"},
            {"tts": {"backend": "remote", "module_path": "/tmp/tts.py", "endpoint": {"url": "http://speech"}}},
            {"wake_command_timeout_s": float("inf")},
            {"asr": {"max_audio_duration_s": 0}},
            {"tts": {"backend": None}},
            [],
        ],
    )
    def test_invalid_and_conflicting_config_is_rejected(self, data):
        with pytest.raises(ValueError):
            VoiceConfig.from_dict(data)

    def test_legacy_model_parameters_do_not_enable_implicit_funasr(self):
        with pytest.warns(DeprecationWarning):
            cfg = VoiceConfig.from_dict({"asr_model": "paraformer-zh", "asr_device": "cpu"})
        assert cfg.asr_backend == "disabled"
