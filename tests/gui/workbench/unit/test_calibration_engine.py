# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""calibration_engine:配置装配与结果摊平(不碰硬件)。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jiuwensymbiosis.agent.lifecycle import CleanupReport, HardwareCleanupError
from jiuwensymbiosis.calibration.domain.ports import ManualGuidanceRecoveryError
from jiuwensymbiosis.env.protocol import HandGuidingRecoveryError
from jiuwensymbiosis_gui.workbench.board_print import BoardParams
from jiuwensymbiosis_gui.workbench.calibration_engine import (
    CalibrationEngine,
    CalibrationSetup,
    TeachingAborted,
    _outcome_payload,
)

_PROFILE = {
    "output_relpath": "so101/calibration/so101_eye_to_hand.json",
    "profile": {"trajectory": {"space": "joint"}, "capture_gate": {"reach_rotation_deg": 4.0}},
    "env_overrides": {"joint_limits": {"shoulder_pan": [-180.0, 180.0]}, "z_min_safe_mm": -1000000.0},
}


@pytest.fixture
def engine(tmp_path):
    setup = CalibrationSetup(
        adapter_module="jiuwensymbiosis.adapters.so101",
        config_data={
            "env": {"cfg": {"low_level": {"port": "/dev/ttyACM0", "joint_limits": {"shoulder_pan": [-90, 90]}}}}
        },
        board=BoardParams(),
        out_path=tmp_path / "calib.json",
        waypoint_path=tmp_path / "wp.npz",
        calibration_profile=_PROFILE,
    )
    made = CalibrationEngine(setup)
    yield made
    made.close()


class TestSessionConfig:
    def test_overrides_widen_limits_without_touching_the_caller_dict(self, engine):
        """标定需要放宽软限位,但界面在用的那份配置不能被改。"""
        original = engine._setup.config_data
        merged = engine._session_config()

        low_level = merged["env"]["cfg"]["low_level"]
        assert low_level["joint_limits"] == {"shoulder_pan": [-180.0, 180.0]}
        assert low_level["z_min_safe_mm"] == -1000000.0
        assert low_level["port"] == "/dev/ttyACM0", "未被覆盖的硬件字段应原样保留"
        assert original["env"]["cfg"]["low_level"]["joint_limits"] == {"shoulder_pan": [-90, 90]}

    def test_missing_overrides_pass_config_through(self, tmp_path):
        setup = CalibrationSetup(
            adapter_module="jiuwensymbiosis.adapters.piper",
            config_data={"env": {"cfg": {"low_level": {"can_port": "can0"}}}},
            board=BoardParams(),
            out_path=tmp_path / "c.json",
            waypoint_path=tmp_path / "w.npz",
            calibration_profile={"profile": {"trajectory": {"space": "joint"}}},
        )
        made = CalibrationEngine(setup)
        try:
            assert made._session_config()["env"]["cfg"]["low_level"] == {"can_port": "can0"}
        finally:
            made.close()


class TestProfileYaml:
    def test_written_once_and_carries_the_calibration_block(self, engine):
        """两条装配路径必须读同一份档案,否则阈值可能不一致。"""
        first = engine._profile_yaml()
        assert first == engine._profile_yaml()

        data = yaml.safe_load(first.read_text(encoding="utf-8"))["calibration"]
        assert data["trajectory"]["space"] == "joint"
        assert data["capture_gate"]["reach_rotation_deg"] == 4.0
        assert data["adapter_module"] == "jiuwensymbiosis.adapters.so101"
        assert Path(data["output"]).name == "calib.json"

    def test_close_removes_the_temporary_file(self, engine):
        path = engine._profile_yaml()
        assert path.exists()
        engine.close()
        assert not path.exists()
        engine.close()  # 幂等


class TestOutcomePayload:
    def test_published_calibration(self, tmp_path):
        outcome = SimpleNamespace(
            candidate=False,
            artifact_path=tmp_path / "calib.json",
            decision=SimpleNamespace(accept=True, failed_checks=(), reasons=()),
            result=SimpleNamespace(n_stations=14, method="PARK", quality=None),
        )
        payload = _outcome_payload(outcome)
        assert payload["candidate"] is False
        assert payload["accept"] is True
        assert payload["n_stations"] == 14
        assert payload["method"] == "PARK"

    def test_candidate_carries_reasons_for_the_operator(self, tmp_path):
        outcome = SimpleNamespace(
            candidate=True,
            artifact_path=tmp_path / "calib.candidate.json",
            decision=SimpleNamespace(
                accept=False,
                failed_checks=("observability_flange_axes",),
                reasons=("rotation axes are degenerate",),
            ),
            result=SimpleNamespace(n_stations=9, method="PARK", quality=None),
        )
        payload = _outcome_payload(outcome)
        assert payload["candidate"] is True
        assert payload["failed_checks"] == ["observability_flange_axes"]
        assert payload["reasons"] == ["rotation axes are degenerate"]

    def test_no_solve_leaves_fields_empty_rather_than_guessing(self):
        payload = _outcome_payload(SimpleNamespace(candidate=True, artifact_path=None, decision=None, result=None))
        assert payload["artifact_path"] is None
        assert payload["accept"] is None
        assert payload["quality"] == {}
        assert payload["n_stations"] == 0

    @pytest.mark.parametrize("invariant", ["base_target", "flange_target"])
    def test_quality_carries_the_mount_specific_invariant_frame(self, tmp_path, invariant):
        """刚性不变量按 mount 不同;界面靠这个字段选标签,不能写死成 flange_target。"""
        quality = SimpleNamespace(
            reprojection=SimpleNamespace(summary=SimpleNamespace(mean=0.4, max=0.9)),
            axxb_residual=SimpleNamespace(
                rotation_deg=SimpleNamespace(mean=0.1),
                translation_mm=SimpleNamespace(mean=0.5),
            ),
            target_consistency=SimpleNamespace(
                invariant_frame=invariant,
                translation_residual_mm=SimpleNamespace(max=1.2, std=0.7),
                rotation_residual_deg=SimpleNamespace(max=0.3),
            ),
        )
        outcome = SimpleNamespace(
            candidate=False,
            artifact_path=tmp_path / "c.json",
            decision=SimpleNamespace(accept=True, failed_checks=(), reasons=()),
            result=SimpleNamespace(n_stations=12, method="PARK", quality=quality),
        )
        assert _outcome_payload(outcome)["quality"]["rigidity"]["invariant_frame"] == invariant


class _NoCameraDevice:
    """预览取帧永远失败的假设备:引擎只把它当一条提示,不影响确认门的判定。"""

    @staticmethod
    def capture_calibration_frame():
        raise RuntimeError("no camera in this test")


class TestTeachingGate:
    """力矩指引必须早于力矩变化 —— 确认门放行前不得进入 ``collect_waypoints``。"""

    def test_confirm_releases_the_gate(self, engine):
        engine.confirm_teaching()
        assert engine._await_start(_NoCameraDevice()) is True

    def test_stop_does_not_release_the_gate(self, engine):
        """停止不能被当成确认:它中止这一轮,力矩一次都不会松。"""
        engine.stop()
        with pytest.raises(TeachingAborted):
            engine._await_start(_NoCameraDevice())

    def test_recording_a_waypoint_does_not_release_the_gate(self, engine):
        """误触「记录当前姿态」不能被当成确认 —— 那会跳过危险提示直接松力矩。"""
        engine.record_waypoint()
        assert engine._await_start(_NoCameraDevice()) is False


class _HoldableDevice(_NoCameraDevice):
    """实现了 ``GuidanceHold`` 的假设备,记下力矩被托住 / 松开的顺序。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def hold_arm(self) -> None:
        self.calls.append("hold")

    def release_arm(self) -> None:
        self.calls.append("release")


class TestPauseResume:
    """暂停 / 继续改的是机械臂力矩,不是给 ``collect_waypoints`` 的答复。"""

    def test_pause_holds_the_arm_and_keeps_waiting(self, engine):
        device = _HoldableDevice()
        engine.pause_teaching()
        engine.resume_teaching()
        engine.record_waypoint()

        assert engine._await_command(device) == ""
        assert device.calls == ["hold", "release"], "暂停/继续必须真的动力矩,而不只是改界面"

    def test_pause_reports_the_new_state_only_after_the_arm_is_held(self, engine):
        device = _HoldableDevice()
        engine.pause_teaching()
        engine.finish_teaching()
        engine._await_command(device)

        assert [payload for tag, payload in engine.drain() if tag == "paused"] == [{"paused": True}]

    def test_unsupported_device_is_not_reported_as_paused(self, engine):
        """报了 paused 界面就会说"可以松手了" —— 力矩没恢复时那是句会伤人的假话。"""
        engine.pause_teaching()
        engine.finish_teaching()
        engine._await_command(_NoCameraDevice())

        assert not [tag for tag, _payload in engine.drain() if tag == "paused"]


class TestLimpArmIsFatal:
    def test_a_driver_reported_recovery_failure_reaches_the_fatal_branch(self, engine):
        """力矩没恢复是唯一需要人立刻用手接住的错误。报成普通错误就没人去接 —— 所以
        引擎捕获的类型必须和驱动真正抛出的那一个是同一个类,而不是同名的另一个。"""

        def torque_did_not_come_back() -> None:
            raise HandGuidingRecoveryError("restore_all_torque failed after hand guiding.")

        engine._guarded(torque_did_not_come_back)

        errors = [payload for tag, payload in engine.drain() if tag == "error"]
        assert errors and errors[0]["fatal"] is True


class TestAbort:
    def test_abort_raises_out_of_the_prompt_so_no_archive_is_written(self, engine):
        """中止就是"不要这一轮的数据":必须半途退出 ``collect_waypoints``,而不是走正常结束。"""
        engine.abort_teaching()
        with pytest.raises(TeachingAborted):
            engine._await_command(_NoCameraDevice())


class TestFailurePath:
    def test_connection_failure_reports_the_real_cause(self, tmp_path):
        """在 contextmanager 里 except 会让它走完却没 yield,调用方只收到一句
        ``generator didn't yield``,把串口打不开之类的真实原因整个盖掉。"""
        engine = CalibrationEngine(
            CalibrationSetup(
                adapter_module="jiuwensymbiosis.adapters.no_such_robot",
                config_data={},
                board=BoardParams(),
                out_path=tmp_path / "c.json",
                waypoint_path=tmp_path / "w.npz",
                calibration_profile={"profile": {"trajectory": {"space": "joint"}}},
            )
        )
        try:
            engine.start_teaching()
            engine.join(timeout=15)
            errors = [payload for tag, payload in engine.drain() if tag == "error"]
        finally:
            engine.close()

        assert errors, "连接失败必须报给界面,不能只留一个后台线程的 traceback"
        reason = errors[0]["reason"]
        assert "didn't yield" not in reason, reason
        assert "no_such_robot" in reason, reason
        assert errors[0]["fatal"] is False


class _MaintenanceAdmission:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.snapshots: list[tuple[dict, str, object]] = []
        self.reports: list[CleanupReport] = []

    def acquire(self, snapshot, operation, *, stop=None):
        self.snapshots.append((snapshot, operation, stop))
        if self.error is not None:
            raise self.error
        return object(), object()

    def finish(self, _lease, report: CleanupReport) -> bool:
        self.reports.append(report)
        return report.released


def test_calibration_admission_uses_profile_overrides_before_thread_start(engine):
    admission = _MaintenanceAdmission()
    engine._admission = admission
    engine._run_teaching = lambda: None

    engine.start_teaching()
    engine.join(timeout=2.0)

    assert len(admission.snapshots) == 1
    snapshot, operation, stop = admission.snapshots[0]
    assert operation == "calibration_teaching"
    assert snapshot["env"]["cfg"]["low_level"]["joint_limits"]["shoulder_pan"] == [-180.0, 180.0]
    assert callable(stop)
    assert admission.reports == [CleanupReport()]


def test_admission_failure_does_not_start_a_calibration_stage(engine):
    admission = _MaintenanceAdmission(RuntimeError("robot is already leased"))
    engine._admission = admission
    called: list[bool] = []
    engine._run_teaching = lambda: called.append(True)

    engine.start_teaching()
    engine.stop()
    engine.join(timeout=1.0)

    assert not engine.is_running()
    assert called == []
    assert [payload["reason"] for tag, payload in engine.drain() if tag == "error"] == [
        "RuntimeError: robot is already leased"
    ]


def test_thread_start_failure_releases_calibration_lease(monkeypatch, engine):
    from jiuwensymbiosis_gui.workbench import calibration_engine as calibration_module

    class FailedThread:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("thread start failed")

        def is_alive(self) -> bool:
            return False

    admission = _MaintenanceAdmission()
    engine._admission = admission
    monkeypatch.setattr(calibration_module, "Thread", FailedThread)

    engine.start_teaching()
    engine.stop()
    engine.join(timeout=1.0)

    assert not engine.is_running()
    assert admission.reports == [CleanupReport()]
    assert any("thread start failed" in payload["reason"] for tag, payload in engine.drain() if tag == "error")


def test_hardware_cleanup_error_without_session_keeps_calibration_lease(engine):
    admission = _MaintenanceAdmission()
    engine._admission = admission

    def fail_binding() -> None:
        raise HardwareCleanupError("build calibration session", cleanup_errors=(RuntimeError("rollback"),))

    engine._run_teaching = lambda: engine._guarded(fail_binding)
    engine.start_teaching()
    engine.join(timeout=2.0)

    assert admission.reports
    assert not admission.reports[0].released
    assert "hardware cleanup unconfirmed" in admission.reports[0].errors[-1]
    assert "rollback" in admission.reports[0].errors[-1]
    assert engine._lease is not None


def test_manual_guidance_recovery_error_is_persisted_in_maintenance_report(engine):
    admission = _MaintenanceAdmission()
    engine._admission = admission

    def fail_torque_restore() -> None:
        raise ManualGuidanceRecoveryError("torque restore unknown")

    engine._run_teaching = lambda: engine._guarded(fail_torque_restore)
    engine.start_teaching()
    engine.join(timeout=2.0)

    assert admission.reports
    assert not admission.reports[0].released
    assert any("manual guidance torque recovery state is unknown" in error for error in admission.reports[0].errors)
    assert engine._lease is not None
    errors = [payload for tag, payload in engine.drain() if tag == "error"]
    assert errors and errors[0]["fatal"] is True
