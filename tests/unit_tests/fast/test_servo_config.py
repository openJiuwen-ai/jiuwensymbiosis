# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""servo_config_from_session:按会话运动档推导实时伺服节奏(本体无关,duck-typed)。"""

from __future__ import annotations

from types import SimpleNamespace

from jiuwensymbiosis.agent.fast import ServoConfig, servo_config_from_session


def _session(*, trajectory_hz=None, max_cartesian_vel_mm_s=None, has_cfg=True):
    cfg = None
    if has_cfg:
        runtime = SimpleNamespace(max_cartesian_vel_mm_s=max_cartesian_vel_mm_s)
        cfg = SimpleNamespace(trajectory_hz=trajectory_hz, motion_runtime=runtime)
    return SimpleNamespace(env=SimpleNamespace(cfg=cfg))


def test_derives_control_hz_and_step_from_motion_profile():
    # SO-101 safe 档:10 Hz,30 mm/s → 3 mm/tick。
    servo = servo_config_from_session(_session(trajectory_hz=10.0, max_cartesian_vel_mm_s=30.0))
    assert isinstance(servo, ServoConfig)
    assert servo.control_hz == 10.0
    assert servo.max_lin_step_mm == 3.0


def test_returns_none_when_cfg_missing_motion_attrs():
    # piper 式会话:cfg 无 trajectory_hz/motion_runtime → None(调用方回落框架默认)。
    assert servo_config_from_session(_session(has_cfg=True, trajectory_hz=None)) is None
    assert servo_config_from_session(_session(has_cfg=False)) is None


def test_returns_none_on_non_positive_or_nonfinite():
    assert servo_config_from_session(_session(trajectory_hz=0.0, max_cartesian_vel_mm_s=30.0)) is None
    assert servo_config_from_session(_session(trajectory_hz=10.0, max_cartesian_vel_mm_s=-5.0)) is None


def test_returns_none_when_out_of_servo_band_instead_of_raising():
    # control_hz 超出 ServoConfig 允许的 [1, 200]:helper 吞掉 ValueError 返回 None,不炸运行。
    assert servo_config_from_session(_session(trajectory_hz=500.0, max_cartesian_vel_mm_s=30.0)) is None
