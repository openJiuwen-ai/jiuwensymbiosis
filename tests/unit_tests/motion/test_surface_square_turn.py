# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``plan_surface_square_turn``: rotation that squares the base to a table's near edge.

A table top has no usable face normal (vertical), so we square to the ground footprint:
the near edge is the footprint side facing the robot; its outward normal is the principal
axis most aligned with table→robot; the base faces the opposite (into the near edge).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from jiuwensymbiosis.motion.base_goal import plan_surface_square_turn

_CFG = SimpleNamespace(place_square_min_aspect=1.2)


def test_already_square_needs_no_turn():
    # Table straight ahead, near edge along base-Y (long axis = Y, yaw=π/2) → already squared.
    turn, ok = plan_surface_square_turn([450.0, 0.0, 0.0], math.pi / 2, 800.0, 500.0, _CFG)
    assert ok and abs(turn) < 1e-6


def test_oblique_footprint_yields_square_turn():
    # Centred table whose footprint is rotated 0.3 rad → face into the near edge = rotate 0.3.
    turn, ok = plan_surface_square_turn([450.0, 0.0, 0.0], 0.3, 800.0, 500.0, _CFG)
    assert ok and math.isclose(turn, 0.3, abs_tol=1e-6)


def test_near_square_footprint_not_applicable():
    # long≈short → near-edge axis ambiguous → not applicable (caller keeps centre-facing).
    _turn, ok = plan_surface_square_turn([450.0, 0.0, 0.0], 0.3, 600.0, 580.0, _CFG)
    assert ok is False


def test_degenerate_footprint_not_applicable():
    _turn, ok = plan_surface_square_turn([450.0, 0.0, 0.0], 0.3, 0.0, 0.0, _CFG)
    assert ok is False


def test_zero_range_not_applicable():
    _turn, ok = plan_surface_square_turn([0.0, 0.0, 0.0], 0.3, 800.0, 500.0, _CFG)
    assert ok is False
