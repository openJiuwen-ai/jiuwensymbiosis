# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Anti-lunge guard: _forward_step caps every fine-approach advance to approach_forward_step_m, so a
single detection frame (incl. a distant false positive with a wildly overshot front_x) can only move
one small step before the next re-detect corrects it — instead of ramming a nearer real object."""

from jiuwensymbiosis.adapters.cruzr.api import _forward_step
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig


def test_far_advance_capped_to_one_step():
    cfg = CruzrConfig()  # approach_forward_step_m = 0.25
    # a 2.1 m overshoot (the hardware false positive) is capped to a single 0.25 m step, not a lunge
    assert _forward_step(2.1, cfg) == 0.25


def test_small_remainder_passes_through():
    cfg = CruzrConfig()
    assert _forward_step(0.2, cfg) == 0.2  # already within one step → gentle landing


def test_nonpositive_forward_passes_through():
    cfg = CruzrConfig()
    assert _forward_step(0.0, cfg) == 0.0
    assert _forward_step(-0.1, cfg) == -0.1  # bearing-only / already in-band


def test_cap_follows_config():
    cfg = CruzrConfig()
    cfg.approach_forward_step_m = 0.4
    assert _forward_step(2.1, cfg) == 0.4
