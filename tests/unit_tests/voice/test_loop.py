# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.voice.loop — VoiceLoop wiring and result_to_speech.

Fully mocked (FileAudioSource + FixedASRBackend + NullTTS): no microphone, GPU,
funasr, or robot. This is the M1 acceptance test for the voice layer.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

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


def test_disabled_asr_fails_before_capture():
    class Audio:
        sample_rate = 16000

        def record_segment(self):
            pytest.fail("Capture must not begin without an explicit ASR backend")

    loop = VoiceLoop(VoiceConfig(), lambda t: "ok", audio=Audio())
    with pytest.raises(ValueError, match="disabled"):
        loop.run_once()


def test_asr_failure_clears_two_phase_wake_and_never_dispatches():
    class ASR:
        def __init__(self):
            self.transcripts = iter(["九问九问", RuntimeError("model unreachable"), "拿起杯子", "九问九问拿起杯子"])

        def transcribe(self, audio):
            text = next(self.transcripts)
            if isinstance(text, Exception):
                raise text
            return text

    called = []
    loop = VoiceLoop(VoiceConfig(), called.append, audio=FileAudioSource([_seg()] * 4), asr=ASR(), tts=NullTTS())
    assert loop.run_once() is None  # Wake-only utterance.
    assert loop.run_once() is None  # ASR failure must reset the wake gate.
    assert loop.run_once() is None
    assert called == []
    command = loop.run_once()
    assert command == "拿起杯子"
    loop.handle_command(command)
    assert called == ["拿起杯子"]


def test_stale_asr_result_and_expired_wake_are_discarded(monkeypatch):
    from jiuwensymbiosis.voice import loop as module

    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    loop = _make_loop(VoiceConfig(wake_command_timeout_s=1), ["九问九问", "拿起杯子"], lambda t: "ok")
    assert loop.run_once() is None
    now[0] = 2
    assert loop.run_once() is None

    class LateASR:
        def transcribe(self, audio):
            now[0] += 30
            return "九问九问拿起杯子"

    loop = VoiceLoop(
        VoiceConfig(max_command_age_s=1),
        lambda t: pytest.fail("stale dispatch"),
        audio=FileAudioSource([_seg()]),
        asr=LateASR(),
        tts=NullTTS(),
    )
    assert loop.run_once() is None


def test_feedback_failure_does_not_block_or_repeat_command():
    class BrokenTTS:
        def speak(self, text):
            raise RuntimeError("speaker unavailable")

        def wait(self):
            pass

    called = []
    loop = VoiceLoop(
        VoiceConfig(),
        lambda t: (called.append(t), "完成")[1],
        audio=FileAudioSource([_seg()]),
        asr=FixedASRBackend(["九问九问拿起杯子"]),
        tts=BrokenTTS(),
    )
    command = loop.run_once()
    loop.handle_command(command)
    loop.handle_command(command)
    assert called == ["拿起杯子"]


def test_half_duplex_waits_before_capture():
    events = []

    class TTS(NullTTS):
        def wait(self):
            events.append("wait")

    class Audio:
        sample_rate = 16000

        def record_segment(self):
            events.append("capture")
            return _seg()

    loop = VoiceLoop(VoiceConfig(), lambda t: "ok", asr=FixedASRBackend(["九问九问拿起杯子"]), audio=Audio(), tts=TTS())
    assert loop.run_once() == "拿起杯子"
    assert events == ["wait", "capture"]


def test_context_closes_only_owned_backends(monkeypatch):
    from jiuwensymbiosis.voice import loop as module

    class ASR(FixedASRBackend):
        def __init__(self):
            super().__init__(["九问九问拿起杯子"])
            self.closed = False

        def close(self):
            self.closed = True

    owned, injected = ASR(), ASR()
    monkeypatch.setattr(module, "build_asr_backend", lambda cfg: owned)
    with VoiceLoop(VoiceConfig(), lambda t: "ok", audio=FileAudioSource([]), tts=NullTTS()):
        pass
    with VoiceLoop(VoiceConfig(), lambda t: "ok", asr=injected, audio=FileAudioSource([]), tts=NullTTS()):
        pass
    assert owned.closed
    assert not injected.closed


def test_remote_asr_cleanup_uses_budget_remaining_after_tts(monkeypatch):
    from jiuwensymbiosis.utils.service_http import HttpEndpointConfig
    from jiuwensymbiosis.voice import loop as module
    from jiuwensymbiosis.voice.asr import RemoteASRBackend

    now, budgets = [100.0], []
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    asr = RemoteASRBackend(HttpEndpointConfig(url="http://speech"))
    close_client = asr._client.close

    def close_http(timeout_s):
        budgets.append(timeout_s)
        close_client(timeout_s=timeout_s)

    class SlowTTS(NullTTS):
        def close(self, *, discard, timeout_s):
            assert timeout_s == 5
            now[0] += 2

    monkeypatch.setattr(asr._client, "close", close_http)
    monkeypatch.setattr(module, "build_asr_backend", lambda cfg: asr)
    monkeypatch.setattr(module, "build_tts_backend", lambda cfg: SlowTTS())
    with VoiceLoop(VoiceConfig(shutdown_timeout_s=5), lambda t: "ok", audio=FileAudioSource([])) as loop:
        loop.speak("完成")
    assert budgets == [3]
    assert asr._client.closed


def test_cancelled_late_transcript_never_dispatches():
    from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled

    token = CancelToken()

    class LateASR:
        def transcribe(self, audio):
            token.set()
            return "九问九问拿起杯子"

    commands = []
    loop = VoiceLoop(
        VoiceConfig(),
        commands.append,
        asr=LateASR(),
        tts=NullTTS(),
        audio=FileAudioSource([_seg()]),
        cancel_token=token,
    )
    with pytest.raises(RunCancelled):
        loop.run_once()
    assert not commands
    assert loop.close(cancelled=True).released


def test_slow_synchronous_ack_cannot_dispatch_an_expired_command(monkeypatch):
    from jiuwensymbiosis.voice import loop as module

    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])

    class SlowTTS(NullTTS):
        def speak(self, text):
            now[0] += 5

    commands = []
    loop = VoiceLoop(
        VoiceConfig(max_command_age_s=1),
        commands.append,
        tts=SlowTTS(),
        audio=FileAudioSource([_seg()]),
        asr=FixedASRBackend(["九问九问拿起杯子"]),
    )
    command = loop.run_once()
    loop.handle_command(command)
    assert not commands


def test_unfinished_playback_timeout_does_not_restart_capture():
    class PendingTTS(NullTTS):
        def wait(self):
            raise TimeoutError("still playing")

    class Audio:
        sample_rate = 16000

        def record_segment(self):
            pytest.fail("Never record while playback remains pending")

    loop = VoiceLoop(VoiceConfig(), lambda t: "ok", tts=PendingTTS(), audio=Audio(), asr=FixedASRBackend([]))
    with pytest.raises(TimeoutError):
        loop.run_once()


def test_cancellation_during_injected_playback_wait_never_starts_capture():
    from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled

    token = CancelToken()

    class TTS(NullTTS):
        def wait(self):
            token.set()

    class Audio:
        sample_rate = 16000

        def record_segment(self):
            pytest.fail("Cancelled playback must not open the next capture")

    loop = VoiceLoop(
        VoiceConfig(), lambda text: "", tts=TTS(), audio=Audio(), asr=FixedASRBackend([]), cancel_token=token
    )
    with pytest.raises(RunCancelled):
        loop.run_once()
