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
from collections.abc import Callable
from typing import Any

from jiuwensymbiosis.voice.asr import ASRBackend, build_asr_backend
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
    ):
        self.config = config
        self.on_command = on_command
        self._asr = asr
        self._tts = tts
        self._audio = audio
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
        if self._audio is None:
            self._audio = build_audio_source(self.config)
        return self._audio

    @property
    def asr(self) -> ASRBackend:
        if self._asr is None:
            self._asr = build_asr_backend(self.config)
        return self._asr

    @property
    def tts(self) -> TTSBackend:
        if self._tts is None:
            self._tts = build_tts_backend(self.config)
        return self._tts

    def speak(self, text: str) -> None:
        if text:
            self.tts.speak(text)

    def wait(self) -> None:
        self.tts.wait()

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
        audio = self.audio.record_segment()
        if audio is None or len(audio) == 0:
            return None
        text = self.asr.transcribe(audio)
        if not text:
            return None
        if not self.config.wake_enabled:
            return text.strip() or None
        remainder = split_after_last_wake(text, self._variants)
        if remainder is None:
            if self._armed:
                # Already woken; this whole segment is the command.
                self._armed = False
                return text.strip() or None
            logger.info("[voice] 未检测到唤醒词，忽略：%s", text)
            return None
        remainder = remainder.strip()
        if not remainder:
            # Wake word with no trailing command → arm and wait for the command
            # in the next segment (no wake word needed).
            self._armed = True
            logger.info("[voice] 已唤醒，请说指令…")
            return None
        # Wake word + inline command in one breath.
        self._armed = False
        return remainder

    def handle_command(self, text: str) -> None:
        """Speak the ack, run ``on_command``, and speak its reply.

        Shared by :meth:`run_forever` and one-shot callers (e.g. the demo's
        ``--text`` / ``--audio-file`` modes). A callback exception is logged and
        spoken, never propagated.
        """
        logger.info("[voice] 指令：%s", text)
        if self.config.speak_ack:
            self.speak(self.config.ack_text)
        try:
            reply = self.on_command(text)
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
        except KeyboardInterrupt:
            logger.info("[voice] 收到中断，退出监听")
        finally:
            self.wait()
