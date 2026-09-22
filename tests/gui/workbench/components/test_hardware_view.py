# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""hardware_view:确认门文案、关节读数渲染与越限提示。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jiuwensymbiosis_gui.workbench.app_state import AppState
from jiuwensymbiosis_gui.workbench.pages.hardware_view import _DOUBLE_TAP_S, HardwareView

_LIMITS = [
    {"name": "shoulder_pan", "low": -110.0, "high": 110.0},
    {"name": "elbow_flex", "low": -96.83, "high": 96.83},
]


@pytest.fixture(autouse=True)
def isolate_runtime_dir(tmp_path, monkeypatch):
    """Keep local resource locks and runtime state inside pytest's temporary directory."""
    monkeypatch.setenv("JIUWENSYMBIOSIS_RUNTIME_DIR", str(tmp_path / "runtime"))


@pytest.fixture
def view(tmp_path):
    return HardwareView(AppState(workspace=str(tmp_path)))


def _fake_engine(**extra):
    calls: list[str] = []
    defaults = {
        "drain": list,
        "is_running": lambda: True,
        "confirm_release": lambda: calls.append("confirm"),
        "request_restore": lambda: calls.append("restore"),
        "stop": lambda: calls.append("stop"),
        "join": lambda timeout=None: None,
    }
    return SimpleNamespace(**{**defaults, **extra}), calls


def _space(*, repeat: bool = False):
    return SimpleNamespace(
        action=SimpleNamespace(keydown=True, keyup=False, repeat=repeat),
        key=SimpleNamespace(name=" ", space=True, escape=False, enter=False),
    )


class TestModeGate:
    def test_supported_body_gets_the_falling_warning_and_the_gate(self, view):
        view._on_mode({"mode": "hand_guiding"})

        assert "失去力矩" in view._hint.text
        assert "夹爪也会松开" in view._hint.text, "全松与标定示教不同，必须说清楚"
        assert view._btn_confirm.visible

    def test_unsupported_body_is_not_told_torque_will_drop(self, view):
        view._on_mode({"mode": "unsupported"})

        assert "失去力矩" not in view._hint.text
        assert not view._btn_confirm.visible, "没有可松的力矩就不该给确认按钮"

    def test_unknown_mode_falls_back_to_the_harmless_wording(self, view):
        view._on_mode({"mode": "something-new"})

        assert not view._btn_confirm.visible


class TestDoubleTapSpace:
    def test_one_tap_only_arms_the_confirm(self, view):
        view._engine, calls = _fake_engine()
        view._on_mode({"mode": "hand_guiding"})

        view._on_key(_space())

        assert calls == []
        assert "再按一下" in view._key_hint.text

    def test_second_tap_confirms(self, view):
        view._engine, calls = _fake_engine()
        view._on_mode({"mode": "hand_guiding"})

        view._on_key(_space())
        view._on_key(_space())

        assert calls == ["confirm"]

    def test_a_slow_second_tap_does_not_count(self, view):
        view._engine, calls = _fake_engine()
        view._on_mode({"mode": "hand_guiding"})

        view._on_key(_space())
        view._space_at -= _DOUBLE_TAP_S * 2
        view._on_key(_space())

        assert calls == []

    def test_held_key_repeats_are_ignored(self, view):
        view._engine, calls = _fake_engine()
        view._on_mode({"mode": "hand_guiding"})

        view._on_key(_space())
        view._on_key(_space(repeat=True))

        assert calls == []

    def test_space_restores_torque_once_guiding(self, view):
        view._engine, calls = _fake_engine()
        view._on_phase({"phase": "guiding", "msg": "已松开力矩"})

        view._on_key(_space())
        view._on_key(_space())

        assert calls == ["restore"]

    def test_space_does_nothing_with_no_armed_action(self, view):
        view._engine, calls = _fake_engine()

        view._on_key(_space())
        view._on_key(_space())

        assert calls == []


class TestJointReadout:
    def test_rows_are_built_once_and_then_only_updated(self, view):
        view._on_state({"limits": _LIMITS, "joints": [1.0, 2.0], "violations": [], "pose": None})
        rows = dict(view._joint_rows)

        view._on_state({"limits": _LIMITS, "joints": [3.0, 4.0], "violations": [], "pose": None})

        assert view._joint_rows == rows, "5Hz 重建整张表会让页面一直闪"
        assert view._joint_rows["shoulder_pan"]["value"].text.strip() == "3.00"

    def test_violating_joint_is_flagged(self, view):
        view._on_state(
            {
                "limits": _LIMITS,
                "joints": [150.0, 0.0],
                "violations": [{"name": "shoulder_pan", "direction": "调小"}],
                "pose": None,
            }
        )

        bad = view._joint_rows["shoulder_pan"]
        good = view._joint_rows["elbow_flex"]
        assert bad["note"].text == "超出软限位"
        assert good["note"].text == ""

    def test_a_joint_coming_back_into_range_clears_the_flag(self, view):
        view._on_state(
            {
                "limits": _LIMITS,
                "joints": [150.0, 0.0],
                "violations": [{"name": "shoulder_pan", "direction": "调小"}],
                "pose": None,
            }
        )
        view._on_state({"limits": _LIMITS, "joints": [10.0, 0.0], "violations": [], "pose": None})

        assert view._joint_rows["shoulder_pan"]["note"].text == ""

    def test_missing_readings_leave_the_table_alone(self, view):
        view._on_state({"limits": _LIMITS, "joints": None, "violations": [], "pose": None})

        assert view._joint_rows["shoulder_pan"]["value"].text == "—"


class TestOutcomes:
    def test_blocked_names_the_joint_and_the_direction(self, view):
        view._on_blocked({"violations": [{"name": "elbow_flex", "direction": "调大"}]})

        assert "elbow_flex 调大" in view._status.text
        assert "力矩没有恢复" in view._status.text

    def test_fatal_error_tells_the_operator_to_catch_the_arm(self, view):
        view._on_error({"reason": "restore_all_torque failed", "fatal": True})

        assert "用手托住" in view._hint.text
        assert view._hint.visible
        assert not view._btn_restore.visible

    def test_ordinary_error_goes_to_the_banner_not_the_danger_box(self, view):
        view._on_error({"reason": "串口打不开", "fatal": False})

        assert "串口打不开" in view._blocker.text
        assert not view._hint.visible

    def test_done_reports_the_arm_stayed_where_it_was_put(self, view):
        view._on_phase({"phase": "guiding", "msg": "已松开力矩"})
        view._on_done({"phase": "guiding"})

        assert "力矩已恢复" in view._status.text
        assert not view._btn_restore.visible


class TestHardwareRelease:
    def test_idle_view_has_nothing_to_release(self, view):
        assert view.stop() is True

    def test_a_running_engine_that_will_not_let_go_reports_false(self, view):
        view._engine, _ = _fake_engine()

        assert view.stop(wait=True) is False, "越限时恢复被拒,硬件并没有归还"

    def test_a_stopped_engine_reports_true(self, view):
        running = [True]

        def stop() -> None:
            running[0] = False

        view._engine, _ = _fake_engine(is_running=lambda: running[0], stop=stop)

        assert view.stop(wait=True) is True
