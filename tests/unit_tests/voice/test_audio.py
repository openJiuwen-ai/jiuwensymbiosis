# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the audio segmenter and mock source — no microphone required.

The recorder's VAD is injected with a deterministic energy gate so the test is
independent of whether ``webrtcvad`` is installed.
"""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.voice.audio import (
    FileAudioSource,
    RecordTuning,
    SoundDevicePlayer,
    _SegmentRecorder,
    build_audio_source,
)
from jiuwensymbiosis.voice.config import VoiceConfig

CHUNK = 10


class _EnergyVad:
    """Deterministic stand-in for _VadGate: loud frame == speech."""

    def __init__(self, threshold: float = 100.0):
        self.threshold = threshold

    def is_speech(self, data: np.ndarray) -> bool:
        return float(np.abs(data.astype(np.float64)).mean()) > self.threshold


def _silent() -> np.ndarray:
    return np.zeros(CHUNK, dtype=np.int16)


def _loud() -> np.ndarray:
    return np.full(CHUNK, 1000, dtype=np.int16)


class TestSegmentRecorder:
    def test_records_utterance_until_trailing_silence(self):
        # preroll_frames=0 isolates segmentation from the pre-roll behaviour
        # (covered separately in test_preroll_prepends_pre_onset_frames).
        tuning = RecordTuning(chunk=CHUNK, silence_frames=2, min_frames=1, timeout_frames=50, preroll_frames=0)
        rec = _SegmentRecorder(tuning, _EnergyVad())

        assert rec.feed(_silent()) is False
        assert rec.feed(_loud()) is False
        assert rec.feed(_loud()) is False
        # two trailing silent frames end it (silence_frames=2, frames>min_frames)
        assert rec.feed(_silent()) is False
        assert rec.feed(_silent()) is True
        assert rec.reason == "检测到静音"

        out = rec.result()
        assert out is not None
        # recorded frames: 2 loud + 2 silent = 4 frames of CHUNK samples
        assert len(out) == 4 * CHUNK

    def test_preroll_prepends_pre_onset_frames(self):
        # The two silent frames just before speech onset are kept and prepended,
        # so the quiet leading edge of the first syllable isn't clipped.
        tuning = RecordTuning(chunk=CHUNK, silence_frames=2, min_frames=1, timeout_frames=50, preroll_frames=2)
        rec = _SegmentRecorder(tuning, _EnergyVad())
        rec.feed(_silent())  # pre-onset silence → preroll ring
        rec.feed(_silent())  # pre-onset silence → preroll ring (holds last 2)
        rec.feed(_loud())  # onset: buffer seeded with 2 preroll + this frame
        rec.feed(_loud())
        rec.feed(_silent())
        assert rec.feed(_silent()) is True
        out = rec.result()
        assert out is not None
        # 2 preroll + 2 loud + 2 trailing silent = 6 frames
        assert len(out) == 6 * CHUNK

    def test_preroll_ring_is_bounded(self):
        # Three leading silent frames with preroll_frames=1 keep only the last.
        tuning = RecordTuning(chunk=CHUNK, silence_frames=2, min_frames=1, timeout_frames=50, preroll_frames=1)
        rec = _SegmentRecorder(tuning, _EnergyVad())
        rec.feed(_silent())
        rec.feed(_silent())
        rec.feed(_silent())
        rec.feed(_loud())  # buffer = 1 preroll + this loud
        rec.feed(_loud())
        rec.feed(_silent())
        assert rec.feed(_silent()) is True
        out = rec.result()
        assert out is not None
        # 1 preroll + 2 loud + 2 trailing silent = 5 frames
        assert len(out) == 5 * CHUNK

    def test_timeout_with_no_speech_returns_none(self):
        tuning = RecordTuning(chunk=CHUNK, timeout_frames=3)
        rec = _SegmentRecorder(tuning, _EnergyVad())
        done = False
        for _ in range(4):
            done = rec.feed(_silent())
        assert done is True
        assert rec.reason == "无语音超时"
        assert rec.result() is None

    def test_max_frames_cap(self):
        tuning = RecordTuning(chunk=CHUNK, silence_frames=99, min_frames=1, max_frames=3)
        rec = _SegmentRecorder(tuning, _EnergyVad())
        rec.feed(_loud())
        rec.feed(_loud())
        assert rec.feed(_loud()) is True  # frame 3 hits the cap
        assert rec.reason == "达到最长录音"


class TestFileAudioSource:
    def test_replays_segments_then_none(self):
        seg = np.ones(CHUNK, dtype=np.int16)
        src = FileAudioSource([seg])
        assert np.array_equal(src.record_segment(), seg)
        assert src.record_segment() is None


class TestBuildAudioSource:
    def test_file_backend(self):
        from jiuwensymbiosis.voice.audio import FileAudioSource as FAS

        src = build_audio_source(VoiceConfig(audio_backend="file"))
        assert isinstance(src, FAS)

    def test_pulse_backend(self):
        from jiuwensymbiosis.voice.audio import PulseAudioSource

        src = build_audio_source(VoiceConfig(audio_backend="pulse"))
        assert isinstance(src, PulseAudioSource)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="audio_backend"):
            build_audio_source(VoiceConfig(audio_backend="nope"))


def test_player_close_cannot_abort_before_stream_start(monkeypatch):
    starting, allow_start, closing, aborted = (threading.Event() for _ in range(4))
    events, errors = [], []

    class Stream:
        def start(self):
            starting.set()
            assert allow_start.wait(2)
            events.append("start")

        def write(self, audio):
            assert aborted.wait(2)

        def stop(self):
            pass

        def abort(self):
            events.append("abort")
            aborted.set()

        def close(self):
            events.append("close")

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=lambda **kwargs: Stream()))
    player = SoundDevicePlayer()

    def play():
        try:
            player.play(_loud(), 16000)
        except BaseException as exc:
            errors.append(exc)

    def close():
        closing.set()
        player.close()

    playback = threading.Thread(target=play)
    closer = threading.Thread(target=close)
    playback.start()
    try:
        assert starting.wait(1)
        closer.start()
        assert closing.wait(1)
        assert not aborted.wait(0.1)
    finally:
        allow_start.set()
        if closer.ident is not None:
            closer.join(3)
        playback.join(3)
    assert not playback.is_alive() and not closer.is_alive()
    assert not errors
    assert events == ["start", "abort", "close"]
    assert player._stream is None


def test_player_closes_stream_when_start_fails(monkeypatch):
    closed = []

    class Stream:
        def start(self):
            raise RuntimeError("device failed to start")

        def close(self):
            closed.append(True)

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=lambda **kwargs: Stream()))
    player = SoundDevicePlayer()
    with pytest.raises(RuntimeError, match="device failed to start"):
        player.play(_loud(), 16000)
    assert closed == [True]
    assert player._stream is None
