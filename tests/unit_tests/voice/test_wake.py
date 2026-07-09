# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.voice.wake — pure wake-word logic, no hardware."""

from __future__ import annotations

from jiuwensymbiosis.voice.wake import (
    WAKE_WORD,
    WAKE_WORD_VARIANTS,
    normalize_wake_text,
    split_after_last_wake,
)


class TestNormalize:
    def test_strips_punctuation_and_spaces(self):
        assert normalize_wake_text(" 九问九问，向前。走 ") == "九问九问向前走"

    def test_ascii_punctuation(self):
        assert normalize_wake_text("hello, world.") == "helloworld"


class TestSplitAfterLastWake:
    def test_primary_wake_word(self):
        assert split_after_last_wake("九问九问把黑盒子拿起来") == "把黑盒子拿起来"

    def test_homophone_variant(self):
        # "九万九万" is a near-homophone variant of the wake word.
        assert split_after_last_wake("九万九万向前走") == "向前走"

    def test_no_wake_word_returns_none(self):
        assert split_after_last_wake("今天天气不错") is None

    def test_wake_word_only_returns_empty(self):
        assert split_after_last_wake("九问九问") == ""

    def test_uses_last_occurrence(self):
        assert split_after_last_wake("九问九问停 九问九问向左转") == "向左转"

    def test_custom_variants(self):
        assert split_after_last_wake("小幺小幺关灯", variants=["小幺小幺"]) == "关灯"

    def test_default_constants_present(self):
        assert WAKE_WORD == "九问九问"
        assert WAKE_WORD in WAKE_WORD_VARIANTS
