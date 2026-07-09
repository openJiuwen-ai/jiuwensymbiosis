# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.voice.loop — VoiceLoop wiring and result_to_speech.

Fully mocked (FileAudioSource + FixedASRBackend + NullTTS): no microphone, GPU,
funasr, or robot. This is the M1 acceptance test for the voice layer.
"""

from __future__ import annotations

import numpy as np

from jiuwensymbiosis.voice.asr import FixedASRBackend
from jiuwensymbiosis.voice.audio import AudioSource, FileAudioSource
from jiuwensymbiosis.voice.config import VoiceConfig
from jiuwensymbiosis.voice.loop import VoiceLoop, result_to_speech
from jiuwensymbiosis.voice.tts import NullTTS


def _seg() -> np.ndarray:
    return np.ones(480, dtype=np.int16)


def _make_loop(config: VoiceConfig, transcripts: list[str], on_command):
    return VoiceLoop(
        config,
        on_command,
        audio=FileAudioSource([_seg()] * len(transcripts)),
        asr=FixedASRBackend(transcripts),
        tts=NullTTS(),
    )


class TestRunOnce:
    def test_strips_wake_word(self):
        loop = _make_loop(VoiceConfig(), ["九问九问把黑盒子拿起来"], on_command=lambda t: "ok")
        assert loop.run_once() == "把黑盒子拿起来"

    def test_no_wake_word_returns_none(self):
        loop = _make_loop(VoiceConfig(), ["今天天气不错"], on_command=lambda t: "ok")
        assert loop.run_once() is None

    def test_wake_only_returns_none(self):
        loop = _make_loop(VoiceConfig(), ["九问九问"], on_command=lambda t: "ok")
        assert loop.run_once() is None

    def test_two_phase_wake_then_command(self):
        # A wake-only segment arms the loop; the next segment (no wake word) is
        # taken as the command. Mirrors a natural pause after "九问九问".
        loop = _make_loop(
            VoiceConfig(),
            ["九问九问", "把黑盒子放到白盒子上面"],
            on_command=lambda t: "ok",
        )
        assert loop.run_once() is None  # wake only → armed
        assert loop.run_once() == "把黑盒子放到白盒子上面"  # next segment = command
        assert loop.run_once() is None  # disarmed again (no more input)

    def test_wake_disabled_passes_full_text(self):
        loop = _make_loop(VoiceConfig(wake_enabled=False), ["直接执行的指令"], on_command=lambda t: "ok")
        assert loop.run_once() == "直接执行的指令"

    def test_no_audio_returns_none(self):
        loop = VoiceLoop(
            VoiceConfig(),
            on_command=lambda t: "ok",
            audio=FileAudioSource([]),  # empty → EOF
            asr=FixedASRBackend([]),
            tts=NullTTS(),
        )
        assert loop.run_once() is None


class _InterruptAfter:
    """Audio source that yields one segment, then raises KeyboardInterrupt."""

    sample_rate = 16000

    def __init__(self):
        self._yielded = False

    def record_segment(self):
        if self._yielded:
            raise KeyboardInterrupt
        self._yielded = True
        return _seg()


class TestRunForever:
    def test_dispatch_speaks_ack_then_reply(self):
        captured: list[str] = []
        tts = NullTTS()
        loop = VoiceLoop(
            VoiceConfig(ack_text="收到", speak_ack=True),
            on_command=lambda t: (captured.append(t), "已完成")[1],
            audio=_InterruptAfter(),
            asr=FixedASRBackend(["九问九问向前走"]),
            tts=tts,
        )
        loop.run_forever()  # processes one command, then KeyboardInterrupt exits

        assert captured == ["向前走"]
        assert tts.spoken == ["收到", "已完成"]

    def test_callback_exception_does_not_crash_loop(self):
        tts = NullTTS()

        def boom(_text: str) -> str:
            raise RuntimeError("boom")

        loop = VoiceLoop(
            VoiceConfig(speak_ack=False),
            on_command=boom,
            audio=_InterruptAfter(),
            asr=FixedASRBackend(["九问九问跳舞"]),
            tts=tts,
        )
        loop.run_forever()  # must not propagate the RuntimeError
        assert "抱歉，执行出错了" in tts.spoken


class TestInjectedBackendsWin:
    def test_isinstance_audiosource_protocol(self):
        assert isinstance(FileAudioSource([]), AudioSource)


class TestResultToSpeech:
    def test_none(self):
        assert result_to_speech(None) == "好的，已完成"

    def test_plain_string(self):
        assert result_to_speech("抓取成功") == "抓取成功"

    def test_fast_path_ok(self):
        assert result_to_speech({"ok": True, "sequence": "..."}) == "好的，已完成"

    def test_fast_path_failure_includes_reason(self):
        out = result_to_speech({"ok": False, "reason": "compile_failed"})
        assert "compile_failed" in out

    def test_reply_key(self):
        assert result_to_speech({"reply": "你好呀"}) == "你好呀"

    def test_object_with_content(self):
        class _R:
            content = "完成了任务"

        assert result_to_speech(_R()) == "完成了任务"
