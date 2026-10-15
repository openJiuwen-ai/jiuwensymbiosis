# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for the explicit calibration use-case entry points."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.calibration.artifacts.archives import dump_waypoint_archive, load_waypoint_archive
from jiuwensymbiosis.calibration.domain.ports import JointState
from jiuwensymbiosis.calibration.workflows.collect import collect_waypoints
from jiuwensymbiosis.calibration.workflows.execute import execute_calibration
from jiuwensymbiosis.calibration.workflows.preflight import PreflightError
from jiuwensymbiosis.calibration.workflows.workflows import (
    CalibrationRunOptions,
    WorkflowDependencies,
)


class _Device:
    camera_mount = "eye_to_hand"

    def __init__(self) -> None:
        self.values = np.zeros(2)

    def get_flange_transform_mm(self):
        return np.eye(4)

    def get_joint_state(self):
        return JointState(
            values=self.values,
            unit="deg",
            order=("j1", "j2"),
            periodic=(False, False),
        )

    def move_joint_vector(self, values):
        self.values = np.asarray(values, dtype=np.float64)


def test_execute_calibration_is_independently_callable_in_dry_run(tmp_path):
    archive = tmp_path / "waypoints.npz"
    dump_waypoint_archive(
        archive,
        space="joint",
        joint_values=np.array([[0.0, 0.0], [5.0, 0.0]]),
        joint_unit="deg",
        joint_order=("j1", "j2"),
        joint_periodic=(False, False),
        adapter_module="unknown",
        mount="eye_to_hand",
        config=None,
    )
    outcome = execute_calibration(_Device(), archive, CalibrationRunOptions(dry_run=True))
    assert outcome.artifact_path is None
    assert outcome.candidate is False


def test_dry_run_validates_live_joint_metadata(tmp_path):
    archive = tmp_path / "waypoints.npz"
    dump_waypoint_archive(
        archive,
        space="joint",
        joint_values=np.array([[0.0, 0.0], [5.0, 0.0]]),
        joint_unit="deg",
        joint_order=("j1", "j2"),
        joint_periodic=(False, False),
        adapter_module="unknown",
        mount="eye_to_hand",
        config=None,
    )
    device = _Device()
    device.get_joint_state = lambda: JointState(  # type: ignore[method-assign]
        values=np.zeros(2),
        unit="rad",
        order=("j1", "j2"),
        periodic=(False, False),
    )

    with pytest.raises(PreflightError, match="joint unit"):
        execute_calibration(device, archive, CalibrationRunOptions(dry_run=True))


def test_dynamic_camera_preflight_runs_before_first_motion(tmp_path):
    archive = tmp_path / "waypoints.npz"
    dump_waypoint_archive(
        archive,
        space="joint",
        joint_values=np.array([[0.0, 0.0], [5.0, 0.0]]),
        joint_unit="deg",
        joint_order=("j1", "j2"),
        joint_periodic=(False, False),
        adapter_module="unknown",
        mount="eye_to_hand",
        config=None,
    )
    events: list[str] = []

    class _CameraFailureDevice(_Device):
        def move_joint_vector(self, values):
            events.append("move")
            super().move_joint_vector(values)

        def capture_calibration_frame(self):
            events.append("camera")
            raise RuntimeError("camera unavailable")

    with pytest.raises(PreflightError, match="camera preflight"):
        execute_calibration(_CameraFailureDevice(), archive, CalibrationRunOptions())
    assert events == ["camera"]


def test_below_solver_minimum_returns_candidate(tmp_path, monkeypatch):
    from jiuwensymbiosis.calibration.workflows import publication

    monkeypatch.setattr(
        publication,
        "solve_eye_to_hand",
        lambda *_args, **_kwargs: pytest.fail("solver must not run with fewer than three stations"),
    )
    outcome = publication.solve_and_publish(
        [object(), object()],
        np.eye(3),
        mount="eye_to_hand",
        out_path=tmp_path / "calibration.json",
        dependencies=WorkflowDependencies(options=CalibrationRunOptions()),
    )
    assert outcome.candidate is True
    assert outcome.result is None
    assert outcome.artifact_path is None


def test_reload_validation_uses_unique_private_path_and_preserves_sibling(tmp_path):
    from jiuwensymbiosis.calibration.domain.models import Station, ViewDetection
    from jiuwensymbiosis.calibration.workflows import publication

    out_path = tmp_path / "calibration.json"
    deterministic_path = out_path.with_suffix(".reload.json")
    deterministic_path.write_text("keep", encoding="utf-8")
    stations = [
        Station(tf_base_flange=np.eye(4), detection=ViewDetection(ok=True, tf_cam_target=np.eye(4))) for _ in range(3)
    ]
    result = SimpleNamespace(tf_base_cam=np.eye(4), method="PARK")
    decision = SimpleNamespace(accept=True, reasons=())
    seen: list = []

    def validator(path, *_args, **_kwargs):
        seen.append(path)

    outcome = publication.publish_with_reload(
        stations,
        np.eye(3),
        result,
        decision,
        mount="eye_to_hand",
        out_path=out_path,
        dependencies=WorkflowDependencies(options=CalibrationRunOptions(), reload_validator=validator),
    )
    assert outcome.candidate is False
    assert seen and seen[0] != deterministic_path
    assert seen[0].name.startswith(".calibration.reload-")
    assert not seen[0].exists()
    assert deterministic_path.read_text(encoding="utf-8") == "keep"


def test_eye_in_hand_publication_omits_object_anchor(tmp_path):
    import json
    from types import SimpleNamespace

    from jiuwensymbiosis.calibration.workflows import publication

    out_path = tmp_path / "eye-in-hand.json"
    result = SimpleNamespace(tf_flange_cam=np.eye(4), method="PARK")
    decision = SimpleNamespace(accept=True, reasons=())
    outcome = publication.publish_with_reload(
        [object(), object(), object()],
        np.eye(3),
        result,
        decision,
        mount="eye_in_hand",
        out_path=out_path,
        dependencies=WorkflowDependencies(
            options=CalibrationRunOptions(),
            reload_validator=lambda *_args, **_kwargs: None,
        ),
    )

    assert outcome.candidate is False
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["T_flange_cam"]["matrix_4x4"] == np.eye(4).tolist()
    assert "object" not in payload


def test_collect_waypoints_is_independently_callable(tmp_path):
    answers = iter(["", "", "q"])
    device = _Device()

    def prompt(_message):
        device.values = device.values + 1.0
        return next(answers)

    config = tmp_path / "config.yaml"
    config.write_text("calibration:\n  trajectory:\n    space: joint\n", encoding="utf-8")
    archive = collect_waypoints(
        device,
        tmp_path / "collected.npz",
        CalibrationRunOptions(config=config),
        prompt_fn=prompt,
    )
    loaded = load_waypoint_archive(archive)
    assert loaded["space"] == "joint"
    np.testing.assert_allclose(loaded["joint_values"], [[1.0, 1.0], [2.0, 2.0]])


def test_collect_waypoints_accepts_eye_in_hand(tmp_path):
    # collect is mount-neutral — eye-in-hand is accepted and archived.
    device = _Device()
    device.camera_mount = "eye_in_hand"
    answers = iter(["", "", "q"])
    config = tmp_path / "config.yaml"
    config.write_text("calibration:\n  trajectory:\n    space: joint\n", encoding="utf-8")

    archive = collect_waypoints(
        device,
        tmp_path / "collected.npz",
        CalibrationRunOptions(config=config),
        prompt_fn=lambda _message: next(answers),
    )
    loaded = load_waypoint_archive(archive)
    assert loaded["metadata"]["mount"] == "eye_in_hand"


def test_execute_calibration_accepts_eye_in_hand_dry_run(tmp_path):
    # execute is mount-neutral — an eye-in-hand archive dry-runs fine.
    archive = tmp_path / "waypoints.npz"
    dump_waypoint_archive(
        archive,
        space="joint",
        joint_values=np.array([[0.0, 0.0], [5.0, 0.0]]),
        joint_unit="deg",
        joint_order=("j1", "j2"),
        joint_periodic=(False, False),
        adapter_module="unknown",
        mount="eye_in_hand",
        config=None,
    )
    device = _Device()
    device.camera_mount = "eye_in_hand"

    outcome = execute_calibration(device, archive, CalibrationRunOptions(dry_run=True))
    assert outcome.artifact_path is None
    assert outcome.candidate is False


def test_publication_empty_stations_candidate_even_for_eye_in_hand(tmp_path):
    from jiuwensymbiosis.calibration.workflows.publication import solve_and_publish
    from jiuwensymbiosis.calibration.workflows.workflows import WorkflowDependencies

    # publication is mount-neutral; empty stations -> candidate for EIH too.
    outcome = solve_and_publish(
        [],
        np.eye(3),
        mount="eye_in_hand",
        out_path=tmp_path / "calibration.json",
        dependencies=WorkflowDependencies(options=CalibrationRunOptions()),
    )
    assert outcome.candidate is True
    assert outcome.artifact_path is None


def test_import_waypoints_accepts_eye_in_hand(tmp_path):
    from jiuwensymbiosis.calibration.workflows.collect import import_waypoints

    archive = tmp_path / "eye-in-hand.npz"
    dump_waypoint_archive(
        archive,
        space="joint",
        joint_values=np.array([[0.0, 0.0], [5.0, 0.0]]),
        joint_unit="deg",
        joint_order=("j1", "j2"),
        joint_periodic=(False, False),
        adapter_module="unknown",
        mount="eye_in_hand",
        config=None,
    )

    out = import_waypoints(archive, tmp_path / "normalised.npz")
    loaded = load_waypoint_archive(out)
    assert loaded["metadata"]["mount"] == "eye_in_hand"


@pytest.mark.parametrize("camera_mount", [None, "ceiling"])
def test_collect_waypoints_requires_explicit_valid_camera_mount(tmp_path, camera_mount):
    device = _Device()
    device.camera_mount = camera_mount
    config = tmp_path / "config.yaml"
    config.write_text("calibration:\n  trajectory:\n    space: joint\n", encoding="utf-8")

    with pytest.raises(PreflightError, match="camera_mount"):
        collect_waypoints(
            device,
            tmp_path / "collected.npz",
            CalibrationRunOptions(config=config),
            prompt_fn=lambda _message: pytest.fail("prompt must not run before mount validation"),
        )


# ===========================================================================
# Mount-neutral solve dispatch
# ===========================================================================
def test_solve_stations_dispatches_eye_in_hand_solver(monkeypatch):
    """solve_stations routes eye_in_hand to the EIH solver + EIH acceptance policy."""
    from jiuwensymbiosis.calibration.domain.models import EyeInHandResult
    from jiuwensymbiosis.calibration.workflows import publication

    stations = [object(), object(), object()]
    calls: dict[str, str] = {}

    def fake_eih(*_a, **_k):
        calls["solver"] = "eih"
        return EyeInHandResult(
            tf_flange_cam=np.eye(4),
            method="PARK",
            n_stations=3,
            intrinsics=np.eye(3),
            quality=SimpleNamespace(
                reprojection=SimpleNamespace(per_view_rms_px=(0.1, 0.1, 0.1), summary=SimpleNamespace(mean=0.1)),
                axxb_residual=SimpleNamespace(
                    rotation_deg=SimpleNamespace(mean=0.1),
                    translation_mm=SimpleNamespace(mean=1.0),
                ),
                target_consistency=SimpleNamespace(translation_residual_mm=SimpleNamespace(std=0.5)),
            ),
        )

    def fake_eih_policy(*_a, **_k):
        calls["policy"] = "eih"
        return SimpleNamespace(
            _decide_calls=0,
        )

    # Monkeypatch the solver and policy used by publication.solve_stations.
    class _FakePolicy:
        def decide(self, _quality):
            calls["policy"] = "eih"
            from jiuwensymbiosis.calibration.domain.models import CalibrationDecision

            return CalibrationDecision(accept=True, failed_checks=(), reasons=())

    monkeypatch.setattr(publication, "solve_eye_in_hand", fake_eih)
    monkeypatch.setattr(publication, "EyeInHandAcceptancePolicy", _FakePolicy)

    result, decision = publication.solve_stations(
        stations,
        np.eye(3),
        mount="eye_in_hand",
        dependencies=WorkflowDependencies(options=CalibrationRunOptions()),
    )
    assert calls["solver"] == "eih"
    assert calls["policy"] == "eih"
    assert decision.accept is True


def test_solve_stations_dispatches_eye_to_hand_solver(monkeypatch):
    """solve_stations routes eye_to_hand to the ETH solver + ETH acceptance policy."""
    from jiuwensymbiosis.calibration.domain.models import EyeToHandResult
    from jiuwensymbiosis.calibration.workflows import publication

    stations = [object(), object(), object()]
    calls: dict[str, str] = {}

    def fake_eth(*_a, **_k):
        calls["solver"] = "eth"
        return EyeToHandResult(
            tf_base_cam=np.eye(4),
            method="PARK",
            n_stations=3,
            intrinsics=np.eye(3),
            quality=SimpleNamespace(
                reprojection=SimpleNamespace(per_view_rms_px=(0.1, 0.1, 0.1), summary=SimpleNamespace(mean=0.1)),
                axxb_residual=SimpleNamespace(
                    rotation_deg=SimpleNamespace(mean=0.1),
                    translation_mm=SimpleNamespace(mean=1.0),
                ),
                target_consistency=SimpleNamespace(translation_residual_mm=SimpleNamespace(std=0.5)),
            ),
        )

    class _FakeEthPolicy:
        observability_thresholds = object()
        target_consistency_thresholds = object()

        def __init__(self, **kwargs):
            if "observability_thresholds" in kwargs:
                self.observability_thresholds = kwargs["observability_thresholds"]
            if "target_consistency_thresholds" in kwargs:
                self.target_consistency_thresholds = kwargs["target_consistency_thresholds"]

        def decide(self, _quality):
            calls["policy"] = "eth"
            from jiuwensymbiosis.calibration.domain.models import CalibrationDecision

            return CalibrationDecision(accept=True, failed_checks=(), reasons=())

    monkeypatch.setattr(publication, "solve_eye_to_hand", fake_eth)
    monkeypatch.setattr(publication, "EyeToHandAcceptancePolicy", _FakeEthPolicy)

    result, decision = publication.solve_stations(
        stations,
        np.eye(3),
        mount="eye_to_hand",
        dependencies=WorkflowDependencies(options=CalibrationRunOptions()),
    )
    assert calls["solver"] == "eth"
    assert calls["policy"] == "eth"
    assert decision.accept is True


class TestReprojectionAdvice:
    """Reprojection magnitude is reported with remediation hints, never rejected.

    The eye-to-hand acceptance policy deliberately gates observability + rigidity
    only; a large RMS can come from board/detector quality rather than a wrong
    hand-eye solution, so it must stay advisory.
    """

    @staticmethod
    def _report(values):
        from jiuwensymbiosis.calibration.domain.models import ReprojectionReport, VerifyStat

        array = np.asarray(values, dtype=np.float64)
        return ReprojectionReport(
            per_view_rms_px=tuple(float(v) for v in array),
            summary=VerifyStat(float(array.mean()), float(array.max()), float(array.std())),
        )

    def test_large_error_warns_with_remediation_hints(self, caplog):
        from jiuwensymbiosis.calibration.workflows.publication import _log_reprojection_advice

        with caplog.at_level("INFO", logger="jiuwensymbiosis.calibration.publication"):
            _log_reprojection_advice(self._report([100.0, 98.0, 102.0]), mount="eye_to_hand")

        text = caplog.text
        assert "100.00px" in text
        assert "board flatness" in text
        assert "intrinsics" in text
        assert any(record.levelname == "WARNING" for record in caplog.records)

    def test_good_error_reported_at_info(self, caplog):
        from jiuwensymbiosis.calibration.workflows.publication import _log_reprojection_advice

        with caplog.at_level("INFO", logger="jiuwensymbiosis.calibration.publication"):
            _log_reprojection_advice(self._report([0.3, 0.4, 0.2]), mount="eye_to_hand")

        assert "good" in caplog.text
        assert all(record.levelname != "WARNING" for record in caplog.records)

    def test_missing_values_warn_without_raising(self, caplog):
        from jiuwensymbiosis.calibration.domain.models import ReprojectionReport, VerifyStat
        from jiuwensymbiosis.calibration.workflows.publication import _log_reprojection_advice

        empty = ReprojectionReport(per_view_rms_px=(), summary=VerifyStat(0.0, 0.0, 0.0))
        with caplog.at_level("INFO", logger="jiuwensymbiosis.calibration.publication"):
            _log_reprojection_advice(empty, mount="eye_to_hand")

        assert "missing or non-finite" in caplog.text

    def test_large_error_still_accepted_by_eye_to_hand_policy(self):
        """The advisory is not a gate: reproj magnitude alone must not fail acceptance."""
        from jiuwensymbiosis.calibration.domain.quality import (
            EyeToHandAcceptancePolicy,
            ObservabilityReport,
        )

        decision = EyeToHandAcceptancePolicy().decide(
            SimpleNamespace(
                reprojection=self._report([100.0, 98.0, 102.0]),
                observability=ObservabilityReport(
                    n_valid_pairs=6,
                    n_relative_axes=3,
                    max_relative_rotation_deg=45.0,
                    max_relative_translation_mm=120.0,
                    max_axis_separation_deg=40.0,
                    n_relative_axes_camera=3,
                    max_relative_rotation_camera_deg=45.0,
                    max_relative_translation_camera_mm=120.0,
                    max_axis_separation_camera_deg=40.0,
                    n_duplicates=0,
                    svd_condition_min=1.0,
                ),
                target_consistency=SimpleNamespace(
                    translation_residual_mm=SimpleNamespace(std=0.5, max=1.0),
                    rotation_residual_deg=SimpleNamespace(max=0.5),
                ),
            )
        )
        assert decision.accept is True
        assert "reproj" not in decision.failed_checks
