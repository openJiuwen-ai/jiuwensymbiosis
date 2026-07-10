# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Text-to-speech feedback — the TTS contract, a no-op, and a ChatTTS backend.

``NullTTS`` is the default: it logs instead of speaking, so the voice layer runs
on CI / GPU-less machines. ``ChatTTSBackend`` ports ``n2_voice``'s
``get_tts_speaker`` / ``speak_tts`` / ``wait_for_tts``: it dynamically loads the
ChatTTS ``tts.py`` module and calls its ``safe_speak_text`` entry point on a
background thread.
"""

from __future__ import annotations

import logging
import threading
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jiuwensymbiosis.voice.config import VoiceConfig

logger = logging.getLogger(__name__)

__all__ = [
    "TTSBackend",
    "NullTTS",
    "ChatTTSBackend",
    "build_tts_backend",
]


@runtime_checkable
class TTSBackend(Protocol):
    """Speak text aloud (or no-op)."""

    def speak(self, text: str) -> None:
        """Speak ``text`` aloud (or no-op)."""
        ...

    def preload(self, text: str) -> None:
        """Pre-synthesize ``text`` so the next speak is instant (or no-op)."""
        ...

    def wait(self) -> None:
        """Block until any in-flight speech finishes."""
        ...


class NullTTS:
    """No-audio TTS: records and logs spoken text. Default backend."""

    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        if not text:
            return
        self.spoken.append(text)
        logger.info("[voice][tts:null] %s", text)

    @staticmethod
    def preload(text: str) -> None:
        return None

    @staticmethod
    def wait() -> None:
        return None


class ChatTTSBackend:
    """Speak via the ChatTTS ``tts.py`` module (lazy-loaded on first speak).

    The module is expected to expose ``safe_speak_text(text)`` (or ``speak_text``).
    If the path is missing or the entry point is absent, speaking degrades to a
    warning (the voice loop keeps working without audio feedback).
    """

    def __init__(self, module_path: str | Path, async_play: bool = True):
        self.module_path = Path(module_path).expanduser()
        self.async_play = async_play
        self._speaker = None
        self._loaded = False
        self._lock = threading.Lock()
        self._threads_lock = threading.Lock()
        self._threads: set[threading.Thread] = set()

    def _ensure_speaker(self):
        if self._loaded:
            return self._speaker
        self._loaded = True
        if not self.module_path.exists():
            logger.warning("[voice] 未找到 ChatTTS 模块: %s", self.module_path)
            return None
        try:
            spec = spec_from_file_location("jws_voice_chattts", self.module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"无法加载 TTS 模块: {self.module_path}")
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            speaker = getattr(module, "safe_speak_text", None) or getattr(module, "speak_text", None)
            if speaker is None:
                logger.warning("[voice] %s 未提供 safe_speak_text/speak_text", self.module_path)
                return None
            self._speaker = speaker
            logger.info("[voice] ChatTTS 已启用: %s", self.module_path)
        except Exception as exc:  # noqa: BLE001 — TTS is best-effort, never fatal
            logger.warning("[voice] ChatTTS 加载失败: %s", exc)
            self._speaker = None
        return self._speaker

    def speak(self, text: str) -> None:
        speaker = self._ensure_speaker()
        if not speaker or not text:
            return

        def _run():
            try:
                with self._lock:
                    speaker(text)
            finally:
                with self._threads_lock:
                    self._threads.discard(threading.current_thread())

        if self.async_play:
            thread = threading.Thread(target=_run, daemon=True)
            with self._threads_lock:
                self._threads.add(thread)
            try:
                thread.start()
            except Exception:
                with self._threads_lock:
                    self._threads.discard(thread)
                raise
        else:
            _run()

    @staticmethod
    def preload(text: str) -> None:
        return None

    def wait(self) -> None:
        while True:
            with self._threads_lock:
                threads = tuple(self._threads)
            if not threads:
                return
            current = threading.current_thread()
            for thread in threads:
                if thread is not current:
                    thread.join()


def build_tts_backend(config: VoiceConfig) -> TTSBackend:
    """Construct the TTS backend named by ``config.tts_backend``."""
    backend = config.tts_backend.lower()
    if backend in ("null", "none", "off"):
        return NullTTS()
    if backend == "chattts":
        if not config.tts_module_path:
            logger.warning("[voice] tts_backend=chattts 但未设置 tts_module_path，回退 NullTTS")
            return NullTTS()
        return ChatTTSBackend(config.tts_module_path, async_play=config.tts_async)
    raise ValueError(f"未知 tts_backend: {config.tts_backend!r}")
