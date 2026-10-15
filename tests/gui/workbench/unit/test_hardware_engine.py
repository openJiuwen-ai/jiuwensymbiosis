# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""hardware_engine:确认门、实时读数与恢复前的越限拦截(不碰硬件)。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from jiuwensymbiosis.agent.lifecycle import CleanupReport, HardwareCleanupError
from jiuwensymbiosis.env.protocol import HandGuidingRecoveryError
from jiuwensymbiosis.runtime.resources import ResourceBusyError
from jiuwensymbiosis_gui.workbench.hardware_engine import HardwareEngine, HardwareSetup, limit_violations

_LIMITS = {"shoulder_pan": (-110.0, 110.0), "elbow_flex": (-96.83, 96.83)}


class _Driver:
    """满足 ``HandGuidingDriver`` 的假驱动(真类:runtime_checkable 用 getattr_static)。"""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.fail_restore = False

    @contextmanager
    def hand_guiding(self, *, include_end_effector: bool = False):
        self.events.append(f"release:{include_end_effector}")
        try:
            yield
        finally:
            if self.fail_restore:
                raise HandGuidingRecoveryError("restore_all_torque failed after hand guiding.")
            self.events.append("restore")


class _Env:
    def __init__(self, *, joints, limits=_LIMITS, supported=True) -> None:
        self.joint_limits = dict(limits)
        self._joints = list(joints)
        self.low_level = _Driver() if supported else SimpleNamespace()

    def set_joints(self, joints) -> None:
        self._joints = list(joints)

    def get_observation(self):
        return SimpleNamespace(joints=list(self._joints), pose={"x": 1.0, "y": 2.0, "z": 3.0})

    def hand_guiding(self, *, include_end_effector: bool = False):
        return self.low_level.hand_guiding(include_end_effector=include_end_effector)


class _Session:
    def __init__(self, env: _Env, report: CleanupReport | None = None) -> None:
        self.env = env
        self.entered = False
        self.report = CleanupReport() if report is None else report

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.entered = False
        return False

    def cleanup_report(self) -> CleanupReport:
        return self.report if not self.report.released else CleanupReport(connected=self.entered)


class _Binding:
    def __init__(self, session: _Session) -> None:
        self.session = session

    def build_session(self):
        return self.session


class _Admission:
    def __init__(self, *, binding=None, error: Exception | None = None) -> None:
        self.binding = binding
        self.error = error
        self.calls: list[tuple[dict, str, object]] = []
        self.reports: list[CleanupReport] = []

    def acquire(self, config_snapshot, operation, *, stop=None):
        self.calls.append((config_snapshot, operation, stop))
        if self.error is not None:
            raise self.error
        return self.binding, object()

    def finish(self, _lease, report: CleanupReport) -> bool:
        self.reports.append(report)
        return report.released


def _engine(env: _Env, *, admission=None) -> tuple[HardwareEngine, dict]:
    seen: dict = {}

    def build(config_data, *, include_sidecars=True):
        seen["include_sidecars"] = include_sidecars
        seen["config"] = config_data
        return _Session(env)

    return HardwareEngine(HardwareSetup(build_session=build, config_data={"env": {}}), admission=admission), seen


def _wait_for(engine: HardwareEngine, tag: str, *, timeout: float = 3.0) -> dict:
    """轮询事件队列直到某个 tag 出现;超时即判定引擎卡住。"""
    deadline = time.monotonic() + timeout
    collected: list[tuple[str, object]] = []
    while time.monotonic() < deadline:
        collected.extend(engine.drain())
        for name, payload in collected:
            if name == tag:
                return payload
        time.sleep(0.01)
    raise AssertionError(f"等不到 {tag} 事件,已收到: {[n for n, _ in collected]}")


class TestLimitViolations:
    def test_reports_which_way_to_move(self):
        out = limit_violations([120.0, 0.0], _LIMITS)
        assert [(v["name"], v["direction"]) for v in out] == [("shoulder_pan", "调小")]

    def test_below_the_floor_asks_to_increase(self):
        assert limit_violations([0.0, -120.0], _LIMITS)[0]["direction"] == "调大"

    def test_in_range_is_clean(self):
        assert limit_violations([0.0, 0.0], _LIMITS) == []

    def test_mismatched_lengths_refuse_to_guess(self):
        assert limit_violations([0.0], _LIMITS) == []

    def test_missing_readings_refuse_to_guess(self):
        assert limit_violations(None, _LIMITS) == []
        assert limit_violations([0.0, 0.0], None) == []


class TestConfirmationGate:
    def test_torque_is_not_released_before_the_user_confirms(self):
        env = _Env(joints=[0.0, 0.0])
        engine, _ = _engine(env)
        engine.start()

        assert _wait_for(engine, "mode")["mode"] == "hand_guiding"
        time.sleep(0.15)
        assert env.low_level.events == [], "确认之前不得松力矩"

        engine.confirm_release()
        _wait_for(engine, "phase")
        assert env.low_level.events[0] == "release:True"

        engine.request_restore()
        engine.join(timeout=3.0)
        assert env.low_level.events == ["release:True", "restore"]

    def test_unsupported_body_never_reaches_hand_guiding(self):
        env = _Env(joints=[0.0, 0.0], supported=False)
        engine, _ = _engine(env)
        engine.start()

        assert _wait_for(engine, "mode")["mode"] == "unsupported"
        engine.join(timeout=3.0)
        assert not engine.is_running()
        assert not engine.is_guiding()

    def test_the_session_skips_detector_sidecars(self):
        env = _Env(joints=[0.0, 0.0], supported=False)
        engine, seen = _engine(env)
        engine.start()
        engine.join(timeout=3.0)

        assert seen["include_sidecars"] is False


class TestRestoreGate:
    def test_out_of_limit_pose_blocks_restore_and_keeps_torque_off(self):
        env = _Env(joints=[0.0, 0.0])
        engine, _ = _engine(env)
        engine.start()
        _wait_for(engine, "mode")
        engine.confirm_release()
        _wait_for(engine, "phase")

        env.set_joints([150.0, 0.0])
        engine.request_restore()

        blocked = _wait_for(engine, "blocked")
        assert blocked["violations"][0]["name"] == "shoulder_pan"
        assert engine.is_guiding(), "拦下之后必须留在手引导里"
        assert env.low_level.events == ["release:True"], "力矩不得恢复"

        env.set_joints([10.0, 0.0])
        engine.request_restore()
        engine.join(timeout=3.0)
        assert env.low_level.events == ["release:True", "restore"]

    def test_stop_is_blocked_by_limits_too(self):
        """停止请求带着越限姿态去恢复,只会换来一个仍然失力矩的异常。"""
        env = _Env(joints=[150.0, 0.0])
        engine, _ = _engine(env)
        engine.start()
        _wait_for(engine, "mode")
        engine.confirm_release()
        _wait_for(engine, "phase")

        engine.stop()

        _wait_for(engine, "blocked")
        assert env.low_level.events == ["release:True"]
        env.set_joints([0.0, 0.0])
        engine.request_restore()
        engine.join(timeout=3.0)

    def test_recovery_failure_is_fatal(self):
        env = _Env(joints=[0.0, 0.0])
        engine, _ = _engine(env)
        engine.start()
        _wait_for(engine, "mode")
        engine.confirm_release()
        _wait_for(engine, "phase")

        env.low_level.fail_restore = True
        engine.request_restore()

        error = _wait_for(engine, "error")
        assert error["fatal"] is True
        assert not engine.is_guiding()


class TestStateStream:
    def test_readings_carry_joints_limits_and_violations(self):
        env = _Env(joints=[120.0, 0.0])
        engine, _ = _engine(env)
        engine.start()
        _wait_for(engine, "mode")
        engine.confirm_release()

        state = _wait_for(engine, "state")
        assert state["joints"] == [120.0, 0.0]
        assert [entry["name"] for entry in state["limits"]] == list(_LIMITS)
        assert state["violations"][0]["name"] == "shoulder_pan"
        assert state["pose"]["x"] == 1.0

        env.set_joints([0.0, 0.0])
        engine.request_restore()
        engine.join(timeout=3.0)

    def test_observation_failure_does_not_end_guiding(self):
        env = _Env(joints=[0.0, 0.0])

        def boom():
            raise RuntimeError("serial hiccup")

        engine, _ = _engine(env)
        engine.start()
        _wait_for(engine, "mode")
        engine.confirm_release()
        _wait_for(engine, "phase")

        env.get_observation = boom
        time.sleep(0.3)
        assert engine.is_guiding()

        env.get_observation = lambda: SimpleNamespace(joints=[0.0, 0.0], pose=None)
        engine.request_restore()
        engine.join(timeout=3.0)
        assert env.low_level.events == ["release:True", "restore"]


@pytest.mark.parametrize("supported", [True, False])
def test_engine_start_is_idempotent(supported):
    env = _Env(joints=[0.0, 0.0], supported=supported)
    engine, _ = _engine(env)
    engine.start()
    engine.start()
    if supported:
        _wait_for(engine, "mode")
        engine.confirm_release()
        _wait_for(engine, "phase")
        engine.request_restore()
    engine.join(timeout=3.0)
    assert not engine.is_running()


def test_admission_busy_does_not_connect_or_start_a_session():
    env = _Env(joints=[0.0, 0.0])
    admission = _Admission(error=ResourceBusyError("task already owns the robot"))
    engine, seen = _engine(env, admission=admission)

    engine.start()
    engine.stop()
    engine.join(timeout=1.0)

    assert not engine.is_running()
    assert seen == {}
    assert admission.calls[0][1] == "hardware_guiding"
    events = engine.drain()
    assert any(tag == "error" and "task already owns" in payload["reason"] for tag, payload in events)


def test_thread_start_failure_finishes_the_reservation(monkeypatch):
    from jiuwensymbiosis_gui.workbench import hardware_engine as hardware_module

    class FailedThread:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("thread start failed")

        def is_alive(self) -> bool:
            return False

    env = _Env(joints=[0.0, 0.0], supported=False)
    session = _Session(env)
    admission = _Admission(binding=_Binding(session))
    monkeypatch.setattr(hardware_module, "Thread", FailedThread)
    engine, _seen = _engine(env, admission=admission)

    engine.start()
    engine.stop()
    engine.join(timeout=1.0)

    assert not engine.is_running()
    assert admission.reports == [CleanupReport()]
    assert any(tag == "error" for tag, _ in engine.drain())


def test_release_failure_is_reported_and_keeps_the_resource_blocked():
    env = _Env(joints=[0.0, 0.0], supported=False)
    session = _Session(env, CleanupReport(errors=("driver close failed",)))
    admission = _Admission(binding=_Binding(session))
    engine, _seen = _engine(env, admission=admission)

    engine.start()
    engine.join(timeout=3.0)
    events = engine.drain()

    assert admission.reports[0].errors == ("driver close failed",)
    assert any(tag == "error" and "driver close failed" in payload["reason"] for tag, payload in events)


def test_torque_recovery_failure_blocks_lease_even_if_session_disconnects_cleanly():
    env = _Env(joints=[0.0, 0.0])
    env.low_level.fail_restore = True
    session = _Session(env)
    admission = _Admission(binding=_Binding(session))
    engine, _seen = _engine(env, admission=admission)

    engine.start()
    assert _wait_for(engine, "mode")["mode"] == "hand_guiding"
    engine.confirm_release()
    _wait_for(engine, "phase")
    engine.request_restore()
    engine.join(timeout=3.0)

    errors = [payload for tag, payload in engine.drain() if tag == "error"]
    assert errors and errors[0]["fatal"] is True
    assert any("torque recovery state is unknown" in error for error in admission.reports[0].errors)
    assert admission.reports[0].connected is False
    assert not admission.reports[0].released


def test_binding_cleanup_error_without_a_session_keeps_the_lease_blocked():
    class BrokenBinding:
        def build_session(self):
            raise HardwareCleanupError("build hardware session", cleanup_errors=(RuntimeError("env rollback"),))

    env = _Env(joints=[0.0, 0.0], supported=False)
    admission = _Admission(binding=BrokenBinding())
    engine, _seen = _engine(env, admission=admission)

    engine.start()
    engine.join(timeout=2.0)

    assert admission.reports
    assert not admission.reports[0].released
    assert "hardware cleanup unconfirmed" in admission.reports[0].errors[-1]
    assert "env rollback" in admission.reports[0].errors[-1]
    assert engine._lease is not None
