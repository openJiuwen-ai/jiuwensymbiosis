# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""perception_engine:后台线程 + 事件队列驱动相机预览与点选反投影(纯逻辑,无 nicegui)。

用 scene-backed 会话(``MockArmEnv(scene=MockScene)`` + ``SceneMockApi``,跑真实的
pixel→base 反投影)经 ``session_factory`` 注入,验证点选取点自洽;并用无深度的模拟会话
验证「无深度相机」的错误引导路径。
"""

from __future__ import annotations

import time

import pytest

from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.env.mock import MockArmEnv
from jiuwensymbiosis.gui.mock_sessions import build_mock_robot_session
from jiuwensymbiosis.gui.perception_engine import PerceptionEngine
from tests.mocks.mock_scene import MockObject, MockScene
from tests.mocks.scene_api import SceneMockApi


def _scene_session() -> tuple[RobotSession, MockArmEnv]:
    """一个 scene-backed 会话:相机看向桌面中心一个高 50mm 的盒子(顶面投影到画面中心)。"""
    scene = MockScene(
        objects=[MockObject(name="box", base_xy_mm=(0.0, 0.0), size_mm=(80.0, 80.0, 50.0), color=(200, 50, 50))]
    )
    env = MockArmEnv(scene=scene)
    api = SceneMockApi(env)
    return RobotSession(env=env, api=api, name="scene"), env


def _drain_until(engine: PerceptionEngine, want_tag: str, timeout: float = 5.0) -> list[tuple[str, object]]:
    """轮询 ``drain()`` 累积事件,直到出现 ``want_tag`` 或超时。"""
    deadline = time.monotonic() + timeout
    events: list[tuple[str, object]] = []
    while time.monotonic() < deadline:
        events.extend(engine.drain())
        if any(tag == want_tag for tag, _ in events):
            return events
        time.sleep(0.02)
    return events


@pytest.mark.unit
def test_locates_clicked_pixel_and_applies_z_correction():
    session, env = _scene_session()
    engine = PerceptionEngine(lambda: session, z_correction_mm=-57.0)
    engine.start()
    try:
        started = _drain_until(engine, "frame")
        assert any(tag == "preview_started" for tag, _ in started)
        assert any(tag == "frame" for tag, _ in started)

        # 画面中心 (320,240) 正对盒子顶面中心 → 反投影应回到基座 (0,0,50)。
        engine.request_point(320, 240)
        events = _drain_until(engine, "point_result")
        results = [p for tag, p in events if tag == "point_result"]
        assert results, "expected a point_result event"
        r = results[-1]
        assert r["ok"] is True
        assert abs(r["x"] - 0.0) < 2.0
        assert abs(r["y"] - 0.0) < 2.0
        assert abs(r["z"] - 50.0) < 2.0
        # z 校正为展示层叠加,不改反投影本身。
        assert abs(r["z_corrected"] - (50.0 - 57.0)) < 2.0
        assert r["z_correction_mm"] == -57.0
    finally:
        engine.stop()
        engine.join(timeout=5.0)

    assert not engine.is_running()
    assert env._connected is False  # 停止后应断开相机


@pytest.mark.unit
def test_out_of_range_click_reports_reason_without_crashing():
    session, _env = _scene_session()
    engine = PerceptionEngine(lambda: session)
    engine.start()
    try:
        _drain_until(engine, "frame")
        engine.request_point(9999, 9999)  # 越界
        events = _drain_until(engine, "point_result")
        results = [p for tag, p in events if tag == "point_result"]
        assert results and results[-1]["ok"] is False
        assert "范围" in str(results[-1]["reason"])
    finally:
        engine.stop()
        engine.join(timeout=5.0)


@pytest.mark.unit
def test_reports_error_when_no_depth():
    # 无 scene 的模拟会话:get_observation 有 rgb 但 depth 为 None。
    session = build_mock_robot_session()
    engine = PerceptionEngine(lambda: session)
    engine.start()
    try:
        events = _drain_until(engine, "error")
        errors = [p for tag, p in events if tag == "error"]
        assert errors, "expected an error event when depth is unavailable"
        assert "深度" in str(errors[0]["reason"])
    finally:
        engine.stop()
        engine.join(timeout=5.0)

    assert not engine.is_running()
