# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Audio capture — segment an utterance from a microphone (or a mock source).

``AudioSource`` is the contract the voice loop consumes. Real backends
(PulseAudio ``parec`` / ``sounddevice``) port ``n2_voice``'s
``record_speech_segment_*`` plus its frame-level state machine. ``FileAudioSource``
is the hardware-free mock used by tests and ``--audio-file`` smoke runs.

Heavy/optional dependencies (``sounddevice``, ``soundfile``, ``webrtcvad``,
the ``parec`` binary) are imported lazily inside the methods that need them, so
``import jiuwensymbiosis.voice`` never requires them.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from jiuwensymbiosis.voice.config import VoiceConfig

logger = logging.getLogger(__name__)

__all__ = [
    "AudioSource",
    "RecordTuning",
    "FileAudioSource",
    "PulseAudioSource",
    "SoundDeviceSource",
    "build_audio_source",
]


@runtime_checkable
class AudioSource(Protocol):
    """A source that yields one captured utterance per call."""

    sample_rate: int

    def record_segment(self) -> np.ndarray | None:
        """Capture and return one utterance as int16 PCM, or ``None`` if none."""
        ...


@dataclass
class RecordTuning:
    """Frame-level segmentation parameters (subset of :class:`VoiceConfig`)."""

    sample_rate: int = 16000
    chunk: int = 480
    silence_frames: int = 25
    min_frames: int = 8
    max_frames: int = 0
    timeout_frames: int = 200
    energy_min: int = 400
    vad_aggressiveness: int = 2
    preroll_frames: int = 8

    @classmethod
    def from_config(cls, config: VoiceConfig) -> RecordTuning:
        return cls(
            sample_rate=config.sample_rate,
            chunk=config.chunk,
            silence_frames=config.silence_frames,
            min_frames=config.min_frames,
            max_frames=config.max_frames,
            timeout_frames=config.timeout_frames,
            energy_min=config.energy_min,
            vad_aggressiveness=config.vad_aggressiveness,
            preroll_frames=config.preroll_frames,
        )


class _VadGate:
    """Voice-activity gate: triggers on an energy threshold OR WebRTC VAD.

    Hybrid on purpose. WebRTC VAD alone (especially at high aggressiveness) misses
    soft speech from quiet mics (e.g. laptop DMICs), which shows up as an endless
    "无语音超时" loop. The energy threshold catches those; WebRTC VAD (when present)
    catches speech that sits below the energy threshold. Either one is enough.
    """

    def __init__(self, sample_rate: int, energy_min: int, aggressiveness: int = 2):
        self.sample_rate = sample_rate
        self.energy_min = energy_min
        self._vad = None
        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(aggressiveness)
        except Exception:  # noqa: BLE001 — optional dep; energy threshold still works
            self._vad = None

    def is_speech(self, data: np.ndarray) -> bool:
        if float(np.abs(data.astype(np.float64)).mean()) > self.energy_min:
            return True
        if self._vad is not None:
            try:
                return bool(self._vad.is_speech(data.astype(np.int16).tobytes(), self.sample_rate))
            except Exception:  # noqa: BLE001 — bad frame size etc.
                pass
        return False


class _SegmentRecorder:
    """Frame-driven utterance segmenter shared by the live capture backends.

    Feed it one frame at a time via :meth:`feed`; it returns ``True`` once the
    utterance is complete (trailing silence, max length, or leading-silence
    timeout). :meth:`result` then returns the concatenated PCM (or ``None``).
    """

    def __init__(self, tuning: RecordTuning, vad: _VadGate):
        self.tuning = tuning
        self.vad = vad
        self.recording = False
        self.done = False
        self.buffer: list[np.ndarray] = []
        self.silence = 0
        self.frames = 0
        self.timeout = 0
        self.reason = ""
        # Ring buffer of the most recent pre-onset frames. Prepended when speech
        # starts so the quiet leading edge of the first syllable is not clipped
        # (the VAD only fires once the sound is already above threshold, which
        # otherwise swallows the first character — "第一个字被吞").
        self._preroll: deque[np.ndarray] = deque(maxlen=max(0, tuning.preroll_frames))

    def feed(self, data: np.ndarray) -> bool:
        speech = self.vad.is_speech(data)
        if not self.recording:
            self.timeout += 1
            if self.timeout > self.tuning.timeout_frames:
                self.done = True
                self.reason = "无语音超时"
                return True
            if speech:
                self.recording = True
                # Seed with the pre-roll so the utterance onset survives.
                self.buffer = [*self._preroll, data]
                self.silence = 0
                self.frames = len(self.buffer)
            else:
                self._preroll.append(data)
        else:
            self.buffer.append(data)
            self.frames += 1
            self.silence = 0 if speech else self.silence + 1
            if self.silence >= self.tuning.silence_frames and self.frames > self.tuning.min_frames:
                self.done = True
                self.reason = "检测到静音"
                return True
            if self.tuning.max_frames > 0 and self.frames >= self.tuning.max_frames:
                self.done = True
                self.reason = "达到最长录音"
                return True
        return False

    def result(self) -> np.ndarray | None:
        if not self.buffer:
            return None
        return np.concatenate(self.buffer).astype(np.int16)


class FileAudioSource:
    """Mock source that replays pre-supplied PCM segments (tests / smoke runs).

    Each call to :meth:`record_segment` returns the next segment, then ``None``
    once exhausted. Segments may be ``np.ndarray`` int16 PCM or paths to WAV
    files (loaded lazily via ``soundfile``).
    """

    def __init__(self, segments: list[np.ndarray | str | Path], sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._segments = list(segments)
        self._idx = 0

    def record_segment(self) -> np.ndarray | None:
        if self._idx >= len(self._segments):
            return None
        seg = self._segments[self._idx]
        self._idx += 1
        if isinstance(seg, (str, Path)):
            import soundfile as sf

            data, _ = sf.read(str(seg), dtype="int16")
            return np.asarray(data)
        return np.asarray(seg)


class PulseAudioSource:
    """Live capture via PulseAudio's ``parec`` (ports ``record_speech_segment_pulse``)."""

    def __init__(self, tuning: RecordTuning, pulse_source: str = ""):
        self.tuning = tuning
        self.sample_rate = tuning.sample_rate
        self.pulse_source = pulse_source
        self._vad = _VadGate(tuning.sample_rate, tuning.energy_min, tuning.vad_aggressiveness)

    def record_segment(self) -> np.ndarray | None:
        import shutil
        import subprocess

        parec = shutil.which("parec")
        if not parec:
            raise RuntimeError("未找到 parec，无法使用 PulseAudio 录音后端（改用 audio_backend=sounddevice）")

        rec = _SegmentRecorder(self.tuning, self._vad)
        cmd = [
            parec,
            "--raw",
            "--format=s16le",
            f"--rate={self.sample_rate}",
            "--channels=1",
            f"--latency-msec={max(10, int(self.tuning.chunk * 1000 / self.sample_rate))}",
        ]
        if self.pulse_source:
            cmd.append(f"--device={self.pulse_source}")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = proc.stdout
        if stdout is None:  # stdout=PIPE guarantees a stream; narrow for the type checker
            raise RuntimeError("无法打开 pactl 输出流（stdout=None）")
        try:
            frame_bytes = self.tuning.chunk * 2
            while not rec.done:
                chunk = stdout.read(frame_bytes)
                if len(chunk) < frame_bytes:
                    break
                rec.feed(np.frombuffer(chunk, dtype=np.int16).copy())
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
        if rec.reason:
            logger.debug("[voice] 录音停止: %s", rec.reason)
        return rec.result()


class SoundDeviceSource:
    """Live capture via ``sounddevice`` (ports ``record_speech_segment_sounddevice``)."""

    def __init__(self, tuning: RecordTuning):
        self.tuning = tuning
        self.sample_rate = tuning.sample_rate
        self._vad = _VadGate(tuning.sample_rate, tuning.energy_min, tuning.vad_aggressiveness)

    def record_segment(self) -> np.ndarray | None:
        import sounddevice as sd

        rec = _SegmentRecorder(self.tuning, self._vad)
        done_event = threading.Event()

        def callback(indata, frames, time_info, status):  # noqa: ARG001 — sd signature
            if rec.done:
                return
            if rec.feed(indata[:, 0].copy()):
                done_event.set()

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.tuning.chunk,
            callback=callback,
        )
        stream.start()
        try:
            done_event.wait()
        finally:
            stream.stop()
            stream.close()
        if rec.reason:
            logger.debug("[voice] 录音停止: %s", rec.reason)
        return rec.result()


def build_audio_source(config: VoiceConfig) -> AudioSource:
    """Construct the audio source named by ``config.audio_backend``.

    Note: ``"file"`` has no segments here — inject a :class:`FileAudioSource`
    explicitly for mock/smoke use.
    """
    backend = config.audio_backend.lower()
    tuning = RecordTuning.from_config(config)
    if backend in ("pulse", "pulseaudio", "pa"):
        return PulseAudioSource(tuning, pulse_source=config.pulse_source)
    if backend in ("sounddevice", "sd"):
        return SoundDeviceSource(tuning)
    if backend == "file":
        return FileAudioSource([], sample_rate=config.sample_rate)
    raise ValueError(f"未知 audio_backend: {config.audio_backend!r}")
