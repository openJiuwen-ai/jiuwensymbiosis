# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.voice.tts."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from jiuwensymbiosis.voice.config import VoiceConfig
from jiuwensymbiosis.voice.tts import NullTTS, build_tts_backend


class TestNullTTS:
    def test_records_spoken_text(self):
        tts = NullTTS()
        tts.speak("你好")
        tts.speak("世界")
        assert tts.spoken == ["你好", "世界"]

    def test_empty_text_ignored(self):
        tts = NullTTS()
        tts.speak("")
        assert tts.spoken == []

    def test_preload_and_wait_noop(self):
        tts = NullTTS()
        assert tts.preload("x") is None
        assert tts.wait() is None


class TestBuildTTSBackend:
    def test_null_backend(self):
        assert isinstance(build_tts_backend(VoiceConfig(tts_backend="null")), NullTTS)

    def test_chattts_without_path_falls_back_to_null(self):
        # Missing tts_module_path must degrade gracefully, not crash.
        tts = build_tts_backend(VoiceConfig(tts_backend="chattts", tts_module_path=None))
        assert isinstance(tts, NullTTS)

    def test_chattts_with_path(self):
        from jiuwensymbiosis.voice.tts import ChatTTSBackend

        tts = build_tts_backend(VoiceConfig(tts_backend="chattts", tts_module_path="/tmp/tts.py"))
        assert isinstance(tts, ChatTTSBackend)

    def test_chattts_missing_module_speak_is_safe(self):
        from jiuwensymbiosis.voice.tts import ChatTTSBackend

        # Non-existent module path: _ensure_speaker warns and speak() no-ops.
        tts = ChatTTSBackend("/nonexistent/path/tts.py", async_play=False)
        tts.speak("无声也不报错")  # must not raise

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="tts_backend"):
            build_tts_backend(VoiceConfig(tts_backend="espeak"))


class TestChatTTSBackend:
    def test_wait_blocks_before_async_worker_acquires_playback_lock(self, monkeypatch):
        from jiuwensymbiosis.voice import tts as tts_module

        gate = threading.Event()
        played = threading.Event()
        wait_done = threading.Event()

        class GatedThread(threading.Thread):
            def __init__(self, *, target, daemon):
                super().__init__(target=lambda: (gate.wait(), target()), daemon=daemon)

        backend = tts_module.ChatTTSBackend("/unused", async_play=True)
        backend._loaded = True
        backend._speaker = lambda text: played.set()
        monkeypatch.setattr(
            tts_module,
            "threading",
            SimpleNamespace(Thread=GatedThread, current_thread=threading.current_thread),
        )

        backend.speak("最后一句")

        def wait_for_tts():
            backend.wait()
            wait_done.set()

        waiter = threading.Thread(target=wait_for_tts)
        waiter.start()
        try:
            assert not wait_done.wait(0.1)
            gate.set()
            assert wait_done.wait(1.0)
            assert played.is_set()
        finally:
            gate.set()
            waiter.join(timeout=1.0)
