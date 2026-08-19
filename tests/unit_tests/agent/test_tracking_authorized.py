# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Following a MOVING target must actually be switched on for the bodies that can do it.

The machinery (``agent/fast/realtime``) is body-agnostic and the authorization rule reads
capabilities, so nothing about this is per-robot. But the agent gates on
``api.capabilities & env.capabilities``, and ``motion.servo`` is a *marker* capability that
no action advertises — so an api that forgets to claim it silently loses tracking while the
env still insists the hardware streams. That is exactly what had happened: both arms could
follow a moving target and neither was allowed to.
"""

from __future__ import annotations

import pytest

from jiuwensymbiosis.agent.fast.sequence import TRACK_DETECT, TRACK_GRASP
from jiuwensymbiosis.agent.run import _resolve_fast_special_ops
from jiuwensymbiosis.introspect import build_session


def _ops(config: str):
    session = build_session(config)
    caps = frozenset(session.api.capabilities) & frozenset(session.env.capabilities)
    return _resolve_fast_special_ops(caps, session.api, session.env), session


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        # Eye-in-hand: the camera rides the arm, so a detection is RELATIVE and tracking
        # corrects step by step.
        ("configs/piper/piper.yaml", TRACK_DETECT),
        # Eye-to-hand: the camera is fixed to the world, so a detection is already an
        # ABSOLUTE point and tracking can servo straight at it.
        ("configs/so101/so101.yaml", TRACK_GRASP),
    ],
)
def test_an_arm_that_can_stream_is_allowed_to_follow(config, expected):
    ops, _ = _ops(config)
    assert expected in ops


@pytest.mark.parametrize("config", ["configs/piper/piper.yaml", "configs/so101/so101.yaml"])
def test_the_api_claims_every_capability_its_hardware_declares(config):
    """No env-only leftovers: a capability the env states and the api omits is an ability
    switched off by bookkeeping, and it fails silently — nothing errors, the feature is
    just absent. Keeping the two sides equal is what makes that impossible."""
    _, session = _ops(config)
    env_only = frozenset(session.env.capabilities) - frozenset(session.api.capabilities)
    assert not env_only, f"{config}: env declares {sorted(env_only)} but the api never claims it"
