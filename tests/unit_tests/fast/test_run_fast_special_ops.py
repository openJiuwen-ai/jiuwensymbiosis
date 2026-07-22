# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for the fast-path capability gate (``_resolve_fast_special_ops``).

The gate decides which special ops a session is authorized to compile/run,
purely from ``env.capabilities`` + which servo sinks the api/env expose. It is
extracted from ``run_fast_task`` so it can be tested without stubbing the
LLM/agent machinery. These tests pin the three binding sink combinations
(only-API, only-env, neither) across the eye-in-hand and eye-to-hand branches.
"""

from __future__ import annotations

from types import SimpleNamespace

from jiuwensymbiosis.agent.run import _resolve_fast_special_ops


def _api(*, has_get_pose: bool = True, has_servo_to_tip: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        get_pose=(lambda: None) if has_get_pose else None,
        servo_to_tip=(lambda p: None) if has_servo_to_tip else None,
    )


def _env(
    *,
    caps: frozenset[str],
    has_servo_to_flange: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        capabilities=caps,
        servo_to_flange=(lambda p: None) if has_servo_to_flange else None,
    )


# ---------------------------------------------------------------- eye-in-hand
_EYE_IN_HAND_CAPS = frozenset({"motion.servo", "vision.detection"})


def test_eye_in_hand_only_api_sink_enables_track_detect():
    # api.servo_to_tip present, env.servo_to_flange absent -> binding OK.
    api = _api(has_servo_to_tip=True)
    env = _env(caps=_EYE_IN_HAND_CAPS, has_servo_to_flange=False)
    assert _resolve_fast_special_ops(env.capabilities, api, env) == frozenset({"track_detect"})


def test_eye_in_hand_only_env_sink_enables_track_detect():
    # env.servo_to_flange present, api.servo_to_tip absent -> binding OK via
    # the env fallback; the gate must NOT require both sinks.
    api = _api(has_servo_to_tip=False)
    env = _env(caps=_EYE_IN_HAND_CAPS, has_servo_to_flange=True)
    assert _resolve_fast_special_ops(env.capabilities, api, env) == frozenset({"track_detect"})


def test_eye_in_hand_no_sink_disables_tracking():
    # Neither sink available -> binding_available False -> no special op.
    api = _api(has_servo_to_tip=False)
    env = _env(caps=_EYE_IN_HAND_CAPS, has_servo_to_flange=False)
    assert _resolve_fast_special_ops(env.capabilities, api, env) == frozenset()


def test_eye_in_hand_missing_get_pose_disables_tracking():
    # get_pose is the hard requirement (read_pose); without it neither sink
    # combo rescues tracking.
    api = _api(has_get_pose=False, has_servo_to_tip=True)
    env = _env(caps=_EYE_IN_HAND_CAPS, has_servo_to_flange=True)
    assert _resolve_fast_special_ops(env.capabilities, api, env) == frozenset()


# -------------------------------------------------------------- eye-to-hand
_EYE_TO_HAND_CAPS = frozenset({"motion.servo", "vision.detection", "grasp.parallel", "vision.eye_to_hand"})


def test_eye_to_hand_only_api_sink_enables_track_grasp():
    api = _api(has_servo_to_tip=True)
    env = _env(caps=_EYE_TO_HAND_CAPS, has_servo_to_flange=False)
    assert _resolve_fast_special_ops(env.capabilities, api, env) == frozenset({"track_grasp"})


def test_eye_to_hand_only_env_sink_enables_track_grasp():
    api = _api(has_servo_to_tip=False)
    env = _env(caps=_EYE_TO_HAND_CAPS, has_servo_to_flange=True)
    assert _resolve_fast_special_ops(env.capabilities, api, env) == frozenset({"track_grasp"})


def test_eye_to_hand_no_sink_disables_tracking():
    api = _api(has_servo_to_tip=False)
    env = _env(caps=_EYE_TO_HAND_CAPS, has_servo_to_flange=False)
    assert _resolve_fast_special_ops(env.capabilities, api, env) == frozenset()


def test_eye_to_hand_missing_grasp_disables_track_grasp():
    # eye-to-hand requires grasp.* alongside vision + servo.
    caps = frozenset({"motion.servo", "vision.detection", "vision.eye_to_hand"})
    api = _api(has_servo_to_tip=True)
    env = _env(caps=caps, has_servo_to_flange=False)
    assert _resolve_fast_special_ops(caps, api, env) == frozenset()
