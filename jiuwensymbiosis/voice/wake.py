# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Wake-word detection — pure text logic, no audio/model dependency.

Ported from ``n2_voice``'s ``split_after_last_wake`` / ``normalize_wake_text``
(originally in ``voice_n2_agent_full_llm_wake_asr.py``). The default wake word
and its near-homophone variants come from ``voice_n2_agent_common.py``; they are
the single source of truth for the framework's voice layer and can be overridden
per :class:`~jiuwensymbiosis.voice.config.VoiceConfig`.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "WAKE_WORD",
    "WAKE_WORD_VARIANTS",
    "normalize_wake_text",
    "split_after_last_wake",
]

# Default wake word and its near-homophone / accent variants. ASR frequently
# mis-transcribes "九问九问"; matching any variant keeps wake detection robust.
WAKE_WORD = "九问九问"
WAKE_WORD_VARIANTS: tuple[str, ...] = (
    "九问九问",
    "九万九万",
    "九问九万",
    "九 问 九 万",
    "九问九温",
    "九温九温",
    "九纹九纹",
    "就问就问",
    "久闻久闻",
    "旧闻旧闻",
    "纠问纠问",
    "九问就问",
    "就问九问",
    "究问究问",
    "九闻九闻",
    "九问九闻",
    "九问九纹",
    "九闻九问",
    "九纹九问",
    "九温九问",
    "九万九问",
    "九万九温",
    "九万九纹",
    "久问久问",
    "旧问旧问",
    "就闻就闻",
    "就温就温",
    "就纹就纹",
)


def normalize_wake_text(text: str) -> str:
    """Strip whitespace and Chinese/ASCII punctuation so wake matching is stable."""
    return "".join(text.strip().replace("，", "").replace("。", "").replace(",", "").replace(".", "").split())


def split_after_last_wake(
    text: str,
    variants: Iterable[str] = WAKE_WORD_VARIANTS,
) -> str | None:
    """Return the command text following the *last* wake-word occurrence.

    Args:
        text: Raw ASR transcript (may contain the wake word and punctuation).
        variants: Wake-word spellings to match. Defaults to
            :data:`WAKE_WORD_VARIANTS`.

    Returns:
        The substring after the latest matched variant (stripped), or ``None``
        if no variant is present. An empty string means the wake word was heard
        but no command followed it.
    """
    clean = normalize_wake_text(text)
    best_idx = -1
    best_variant = ""
    for variant in variants:
        normalized = normalize_wake_text(variant)
        idx = clean.rfind(normalized)
        if idx > best_idx:
            best_idx = idx
            best_variant = normalized
    if best_idx < 0:
        return None
    return clean[best_idx + len(best_variant) :].strip()
