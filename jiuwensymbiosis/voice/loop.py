# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The voice loop — orchestrates capture → ASR → wake gate → callback → TTS.

This is the robot-agnostic seam. :class:`VoiceLoop` knows nothing about Piper,
N2, sessions, or the agent: it hands transcribed text to an ``on_command``
callback and speaks whatever string the callback returns. Wiring that callback
to :func:`jiuwensymbiosis.run_robot_task` lives in the demo, not here — so the
same loop drives any robot adapter unchanged.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.voice.asr import ASRBackend, RemoteASRBackend, build_asr_backend
from jiuwensymbiosis.voice.audio import AudioSource, build_audio_source
from jiuwensymbiosis.voice.config import VoiceConfig
from jiuwensymbiosis.voice.tts import TTSBackend, build_tts_backend
from jiuwensymbiosis.voice.wake import split_after_last_wake

logger = logging.getLogger(__name__)

__all__ = ["VoiceLoop", "OnCommand", "result_to_speech"]

# Takes the transcribed command text, returns the text to speak back.
OnCommand = Callable[[str], str]


def result_to_speech(
    result: Any,
    *,
    ok_text: str = "好的，已完成",
    fail_text: str = "抱歉，没能完成",
) -> str:
    """Best-effort: reduce a ``run_robot_task`` return value to one spoken line.

    Handles the fast-path ``dict`` shape (``{"ok": ..., "reason": ...}``), plain
    strings, ``None``, and agent results exposing ``content`` / ``output``.
    Falls back to ``ok_text`` when no message can be extracted.
    """
    if result is None:
        return ok_text
    if isinstance(result, str):
        return result.strip() or ok_text
    if isinstance(result, dict):
        if "ok" in result:
            if result.get("ok"):
                return ok_text
            reason = str(result.get("reason") or "").strip()
            return f"{fail_text}，{reason}" if reason else fail_text
        for key in ("reply", "answer", "output", "result", "content", "message"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ok_text
    for attr in ("content", "output"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ok_text


class VoiceLoop:
    """Listen for wake-word-prefixed commands and dispatch them via a callback.

    Backends default from ``config`` but can be injected (for tests / custom
    pipelines). Injected backends suppress the lazy default construction, so a
    fully-mocked loop never imports ``funasr`` / ``sounddevice``.
    """

    def __init__(
        self,
        config: VoiceConfig,
        on_command: OnCommand,
        *,
        asr: ASRBackend | None = None,
        tts: TTSBackend | None = None,
        audio: AudioSource | None = None,
        cancel_token: CancelToken | None = None,
    ):
        self.config = config
        self.on_command = on_command
        self._asr = asr
        self._tts = tts
        self._audio = audio
        self._owned = {"asr": asr is None, "tts": tts is None, "audio": audio is None}
        self.cancel_token = cancel_token if cancel_token is not None else CancelToken()
        self._closed = False
        self._epoch = 0
        self._pending_command: tuple[str, str, float, int] | None = None
        self._dispatched_id: str | None = None
        self._armed_until = 0.0
        self._unregister_audio_cancel: Callable[[], None] | None = None
        # Effective wake variants: always include the configured wake word.
        self._variants = tuple(dict.fromkeys((config.wake_word, *config.wake_variants)))
        # Two-phase wake: set once the wake word is heard alone, so the *next*
        # segment is taken as the command without repeating the wake word. A
        # natural pause between "九问九问" and the instruction splits them into
        # separate segments; this stitches them back together.
        self._armed = False

    # --- lazily-built backends (injected ones win) ---
    @property
    def audio(self) -> AudioSource:
        if self._closed:
            raise RuntimeError("VoiceLoop is closed")
        if self._audio is None:
            self._audio = build_audio_source(self.config)
            close = getattr(self._audio, "close", None)
            if close is not None:
                self._unregister_audio_cancel = self.cancel_token.on_cancel(close)
        return self._audio

    @property
    def asr(self) -> ASRBackend:
        if self._closed:
            raise RuntimeError("VoiceLoop is closed")
        if self._asr is None:
            self._asr = build_asr_backend(self.config)
            bind = getattr(self._asr, "bind_cancel_token", None)
            if bind is not None:
                bind(self.cancel_token)
        return self._asr

    @property
    def tts(self) -> TTSBackend:
        if self._closed:
            raise RuntimeError("VoiceLoop is closed")
        if self._tts is None:
            self._tts = build_tts_backend(self.config)
            bind = getattr(self._tts, "bind_cancel_token", None)
            if bind is not None:
                bind(self.cancel_token)
        return self._tts

    def speak(self, text: str) -> None:
        if text:
            try:
                self.tts.speak(text)
            except RunCancelled:
                raise
            except Exception as exc:
                logger.warning("[voice] Feedback failed (%s)", type(exc).__name__)

    def wait(self) -> None:
        if self._tts is not None:
            self._tts.wait()

    def __enter__(self) -> VoiceLoop:
        self.config.validate()
        # Diagnose disabled/missing ASR before acquiring a microphone.
        _ = self.asr
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        report = self.close(cancelled=exc_type is not None)
        if not report.released:
            raise RuntimeError(f"Voice frontend cleanup is incomplete: {report}") from exc

    def stop(self) -> None:
        """Invalidate pending transcripts and request cancellation."""
        self._epoch += 1
        self._armed = False
        self._pending_command = None
        self.cancel_token.set()

    def close(self, *, cancelled: bool = False) -> CleanupReport:
        """Close only constructed, owned resources; preserve failed handles."""
        deadline = time.monotonic() + self.config.shutdown_timeout_s
        self._closed = True
        self._epoch += 1
        self._armed = False
        self._pending_command = None
        if cancelled:
            self.stop()
        errors = []
        for name in ("tts", "audio", "asr"):
            resource = getattr(self, f"_{name}")
            if resource is None or not self._owned[name]:
                continue
            close = getattr(resource, "close", None)
            if close is None:
                setattr(self, f"_{name}", None)
                continue
            try:
                if name == "tts":
                    close(discard=cancelled, timeout_s=max(0.0, deadline - time.monotonic()))
                else:
                    # Cancellation can return before HTTP or synchronous ASR
                    # finishes unwinding. Keep its resources until work is idle.
                    if not self.cancel_token.wait_for_idle(max(0.0, deadline - time.monotonic())):
                        raise TimeoutError("Voice work remains pending during cleanup")
                    if isinstance(resource, RemoteASRBackend):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("ASR cleanup budget exhausted")
                        close(timeout_s=remaining)
                    else:
                        close()
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}")
            else:
                setattr(self, f"_{name}", None)
                if name == "audio" and self._unregister_audio_cancel is not None:
                    self._unregister_audio_cancel()
                    self._unregister_audio_cancel = None
        return CleanupReport(pending_work=self.cancel_token.pending_work, errors=tuple(errors))

    def run_once(self) -> str | None:
        """Capture one utterance and return its command text.

        Returns the text after the wake word (``wake_enabled``), the full
        transcript (wake disabled or already armed by a prior wake-only
        utterance), or ``None`` when there was no speech / no wake word / an
        empty command.

        Two-phase wake: hearing the wake word alone arms the loop and returns
        ``None``; the next non-empty segment is then taken as the command with
        no wake word required.
        """
        if self._closed:
            raise RuntimeError("VoiceLoop is closed")
        self.cancel_token.raise_if_set()
        asr = self.asr
        # Half duplex: an unfinished acknowledgement/reply is never recorded.
        self.wait()
        self.cancel_token.raise_if_set()
        self._pending_command = None
        epoch = self._epoch
        utterance_id = uuid.uuid4().hex
        if time.monotonic() >= self._armed_until:
            self._armed = False
        finish_capture = self.cancel_token.register_work("voice.capture")
        try:
            audio = self.audio.record_segment()
        finally:
            finish_capture()
        self.cancel_token.raise_if_set()
        if audio is None or len(audio) == 0:
            return None
        captured_at = time.monotonic()
        sample_rate = self.audio.sample_rate
        if audio.ndim != 1 or len(audio) / sample_rate > self.config.max_audio_duration_s:
            self._armed = False
            logger.warning("[voice] Invalid or oversized capture was discarded")
            return None
        finish = self.cancel_token.register_work("voice.asr")
        try:
            self.cancel_token.raise_if_set()
            transcribe_utterance = getattr(asr, "transcribe_utterance", None)
            if transcribe_utterance is not None:
                text = transcribe_utterance(audio, sample_rate, utterance_id)
            else:
                if hasattr(asr, "sample_rate") and asr.sample_rate != sample_rate:
                    raise ValueError("Audio source and local ASR sample rates differ; resample explicitly")
                text = asr.transcribe(audio)
            self.cancel_token.raise_if_set()
        except RunCancelled:
            self._armed = False
            raise
        except Exception as exc:
            self._armed = False
            logger.warning("[voice] ASR failed; utterance discarded (%s)", type(exc).__name__)
            return None
        finally:
            finish()
        if self._closed or epoch != self._epoch or time.monotonic() - captured_at > self.config.max_command_age_s:
            self._armed = False
            return None
        if not text:
            return None
        if not self.config.wake_enabled:
            command = text.strip() or None
            if command:
                self._pending_command = (command, utterance_id, captured_at, epoch)
            return command
        remainder = split_after_last_wake(text, self._variants)
        if remainder is None:
            if self._armed and time.monotonic() < self._armed_until:
                # Already woken; this whole segment is the command.
                self._armed = False
                command = text.strip() or None
                if command:
                    self._pending_command = (command, utterance_id, captured_at, epoch)
                return command
            logger.info("[voice] 未检测到唤醒词，忽略：%s", text)
            return None
        remainder = remainder.strip()
        if not remainder:
            # Wake word with no trailing command → arm and wait for the command
            # in the next segment (no wake word needed).
            self._armed = True
            self._armed_until = time.monotonic() + self.config.wake_command_timeout_s
            logger.info("[voice] 已唤醒，请说指令…")
            return None
        # Wake word + inline command in one breath.
        self._armed = False
        self._pending_command = (remainder, utterance_id, captured_at, epoch)
        return remainder

    def handle_command(self, text: str) -> None:
        """Speak the ack, run ``on_command``, and speak its reply.

        Shared by :meth:`run_forever` and one-shot callers (e.g. the demo's
        ``--text`` / ``--audio-file`` modes). A callback exception is logged and
        spoken, never propagated.
        """
        self.cancel_token.raise_if_set()
        pending = self._pending_command
        if pending is not None:
            command, utterance_id, captured_at, epoch = pending
            if text != command or utterance_id == self._dispatched_id or epoch != self._epoch:
                return
            if time.monotonic() - captured_at > self.config.max_command_age_s:
                return
            self._dispatched_id = utterance_id
        if self._closed:
            return
        logger.info("[voice] 指令：%s", text)
        if self.config.speak_ack:
            self.speak(self.config.ack_text)
        try:
            self.cancel_token.raise_if_set()
            # Synchronous remote acknowledgement can consume the remaining
            # command age budget. Recheck immediately before the side effect.
            if self._closed:
                return
            if pending is not None and (
                pending[3] != self._epoch or time.monotonic() - pending[2] > self.config.max_command_age_s
            ):
                return
            reply = self.on_command(text)
        except RunCancelled:
            raise
        except Exception:  # noqa: BLE001 — one bad command must not kill the loop
            logger.exception("[voice] 指令执行异常")
            self.speak("抱歉，执行出错了")
            return
        if reply:
            self.speak(reply)

    def run_forever(self) -> None:
        """Continuously listen, dispatch commands, and speak feedback (Ctrl-C to stop)."""
        logger.info("[voice] 开始监听，唤醒词=%r（Ctrl-C 退出）", self.config.wake_word)
        try:
            while True:
                text = self.run_once()
                if not text:
                    continue
                self.handle_command(text)
        except (KeyboardInterrupt, RunCancelled):
            logger.info("[voice] 收到中断，退出监听")
            self.stop()
        finally:
            report = self.close(cancelled=self.cancel_token.is_set())
            if not report.released:
                raise RuntimeError(f"Voice frontend cleanup is incomplete: {report}")
