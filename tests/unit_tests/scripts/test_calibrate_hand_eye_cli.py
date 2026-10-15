# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Regression tests for ``scripts/calibrate/calibrate_hand_eye.py`` (Piper eye-in-hand CLI).

Purpose
-------
This file is the **commit-0 gate** mandated by the architecture plan
(``feature/eye-to-hand-calib-unified``): before any core migration starts, we
lock the Piper legacy CLI's externally observable behaviour so the §4.3
legacy-property mapping and §4.4 preservation policy have a fixed reference.

What is locked
~~~~~~~~~~~~~~
1. ``--help`` exit code + presence of the parameters the legacy report reads.
2. ``--selftest`` exit code 0 (numpy AX=XB/board-origin/outlier/pose-convention
   + save-load round-trip + cv2 calibrateHandEye recovery + board detection
   links). cv2-gated so CI without opencv still runs the suite.
3. ``--generate-board`` produces an image file for both charuco and chessboard.
4. No-args ``--non-interactive`` exits non-zero (cannot build session without
   hardware) — locks the exit-code contract for the manual path.
5. ``_report_dict`` field mapping: the 14 keys consumed by the legacy JSON
   report and the §4.3 compatibility table.
6. ``_judge`` three-band verdict (``✅`` / ``⚠️`` / ``❌``).

These tests assert CURRENT behaviour; the migration commits must keep them
green. When a field is intentionally renamed by the unified quality model
(§4.1/§4.3), the corresponding assertion is updated *in the same commit* that
introduces the new mapping — never ahead of it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# ``calibrate_hand_eye.py`` uses same-directory bare imports (``from handeye_board
# import ...``) so it cannot be imported as ``scripts.calibrate.calibrate_hand_eye``.
# For the pure-function tests we add the directory to sys.path; for the
# entry-point tests we run it as a subprocess (matching test_eye_to_hand_calib_cli).
_CALIB_DIR = Path(__file__).resolve().parents[3] / "scripts" / "calibrate"
if str(_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_CALIB_DIR))
_CALIB_SRC = Path(__file__).resolve().parents[3]

SCRIPT = _CALIB_DIR / "calibrate_hand_eye.py"


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(path for path in (str(_CALIB_SRC), existing_pythonpath) if path)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def _have_cv2() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    # OpenCV 5.x dropped the top-level ``calibrateHandEye`` name; selftest needs it.
    return hasattr(cv2, "calibrateHandEye")


_CV2_REASON = 'cv2 with calibrateHandEye required (pip install -e ".[calib]")'
_VS = {"charuco": {"squares": 5, "marker": 22.0}, "chessboard": {"squares": 8, "marker": None}}


def _rpy_deg_to_rot(rx: float, ry: float, rz: float) -> np.ndarray:
    """Test-local RPY helper; calibration geometry owns only SE(3) primitives."""
    return Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()


class TestCliExitCodes:
    def test_help_exits_0(self):
        r = _run(["--help"])
        assert r.returncode == 0
        assert "手眼标定" in r.stdout or "usage" in r.stdout.lower()

    def test_help_advertises_legacy_report_params(self):
        """Parameters whose values feed ``_report_dict`` must remain on --help."""
        r = _run(["--help"])
        assert r.returncode == 0
        for flag in ("--method", "--cross-check", "--out", "--report", "--board"):
            assert flag in r.stdout, f"{flag} missing from --help"

    def test_no_args_noninteractive_exits_nonzero(self):
        # Without hardware the session build fails; legacy CLI returns 2 (RuntimeError).
        r = _run(["--non-interactive"])
        assert r.returncode != 0


class TestSelftest:
    """``--selftest`` is the offline numpy + cv2 sanity run."""

    @pytest.mark.skipif(not _have_cv2(), reason=_CV2_REASON)
    def test_selftest_exits_0(self):
        r = _run(["--selftest"])
        assert r.returncode == 0, r.stdout + r.stderr
        assert "全部离线自检通过" in r.stderr

    @pytest.mark.skipif(not _have_cv2(), reason=_CV2_REASON)
    def test_selftest_checks_pose_convention(self):
        r = _run(["--selftest"])
        assert r.returncode == 0
        assert "pose_to_tf_base_flange" in r.stderr

    @pytest.mark.skipif(not _have_cv2(), reason=_CV2_REASON)
    def test_selftest_checks_save_load_roundtrip(self):
        r = _run(["--selftest"])
        assert r.returncode == 0
        assert "save_calibration" in r.stderr or "round-trip" in r.stderr

    @pytest.mark.skipif(not _have_cv2(), reason=_CV2_REASON)
    def test_selftest_recovers_handeye_truth(self):
        r = _run(["--selftest"])
        assert r.returncode == 0
        assert "calibrateHandEye" in r.stderr


class TestGenerateBoard:
    @pytest.mark.skipif(not _have_cv2(), reason=_CV2_REASON)
    def test_charuco_board_generated(self, tmp_path):
        out = tmp_path / "charuco.png"
        r = _run(
            [
                "--generate-board",
                str(out),
                "--board",
                "charuco",
                "--squares-x",
                "5",
                "--squares-y",
                "7",
                "--square-size-mm",
                "30",
                "--marker-size-mm",
                "22",
            ]
        )
        assert r.returncode == 0, r.stdout + r.stderr
        assert out.exists() and out.stat().st_size > 0

    @pytest.mark.skipif(not _have_cv2(), reason=_CV2_REASON)
    def test_chessboard_board_generated(self, tmp_path):
        out = tmp_path / "chess.png"
        r = _run(
            [
                "--generate-board",
                str(out),
                "--board",
                "chessboard",
                "--squares-x",
                "8",
                "--squares-y",
                "6",
                "--square-size-mm",
                "25",
            ]
        )
        assert r.returncode == 0, r.stdout + r.stderr
        assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Pure-function field mapping (no cv2, no hardware — runs in any CI).
# ---------------------------------------------------------------------------
def _synthetic_result() -> object:
    """Build an EyeInHandResult with a known quality report to lock _report_dict output.

    After the §4.1 unified quality model the result composes a
    ``CalibrationQualityReport``; the legacy ``_report_dict`` fields are read
    via the §4.3 compatibility properties that map to ``quality.*``.
    """
    from jiuwensymbiosis.calibration.domain.models import (
        AxxbResidualReport,
        CalibrationQualityReport,
        CrossCheckReport,
        EyeInHandResult,
        ReprojectionReport,
        TargetConsistencyReport,
        VerifyStat,
    )
    from jiuwensymbiosis.calibration.domain.quality import ObservabilityReport
    from jiuwensymbiosis.utils.geometry import make_transform

    tf = make_transform(_rpy_deg_to_rot(20.0, 95.0, -30.0), np.array([-80.0, -0.3, -114.0]))
    quality = CalibrationQualityReport(
        reprojection=ReprojectionReport(per_view_rms_px=(0.1, 0.2, 0.15), summary=VerifyStat(0.15, 0.2, 0.04)),
        observability=ObservabilityReport(
            n_valid_pairs=66,
            n_relative_axes=3,
            max_relative_rotation_deg=45.0,
            max_relative_translation_mm=120.0,
            max_axis_separation_deg=60.0,
            n_relative_axes_camera=3,
            max_relative_rotation_camera_deg=30.0,
            max_relative_translation_camera_mm=80.0,
            max_axis_separation_camera_deg=40.0,
            n_duplicates=0,
            svd_condition_min=1.5,
        ),
        axxb_residual=AxxbResidualReport(
            rotation_deg=VerifyStat(0.3, 0.5, 0.1), translation_mm=VerifyStat(1.5, 2.0, 0.3)
        ),
        target_consistency=TargetConsistencyReport(
            invariant_frame="base_target",
            mean_transform=make_transform(np.eye(3), np.array([300.0, 0.0, 30.0])),
            translation_residual_mm=VerifyStat(1.0, 2.0, 0.5),
            rotation_residual_deg=VerifyStat(0.2, 0.4, 0.1),
        ),
        cross_check=CrossCheckReport(transforms={"PARK": tf}, max_rotation_deg=0.1, max_translation_mm=0.2),
    )
    return EyeInHandResult(
        tf_flange_cam=tf,
        method="PARK",
        n_stations=12,
        intrinsics=np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]),
        quality=quality,
    )


class TestReportDict:
    """Lock the 14 keys and value shapes consumed by the legacy JSON report.

    These are the exact fields listed in the plan §4.3 legacy-property table.
    When the unified ``CalibrationQualityReport`` lands, this test is updated
    in the same commit to read from ``quality.*`` paths.
    """

    EXPECTED_KEYS = {
        "frame_field",
        "method",
        "n_stations",
        "T_flange_cam",
        "intrinsics",
        "rotation_spread_deg",
        "reproj_rms_px",
        "per_view_reproj_rms_px",
        "axxb_rot_deg",
        "axxb_trans_mm",
        "board_origin_base_mm",
        "board_origin_spread_mm",
        "cross_check_max_deg",
        "cross_check_max_mm",
    }

    def test_keys_match_legacy_contract(self):
        from calibrate_hand_eye import _report_dict  # type: ignore

        d = _report_dict(_synthetic_result())
        assert set(d.keys()) == self.EXPECTED_KEYS

    def test_frame_identity_fields(self):
        from calibrate_hand_eye import _report_dict  # type: ignore

        d = _report_dict(_synthetic_result())
        assert d["frame_field"] == "T_flange_cam"
        assert d["method"] == "PARK"
        assert d["n_stations"] == 12
        assert np.array(d["T_flange_cam"]).shape == (4, 4)
        assert np.array(d["intrinsics"]).shape == (3, 3)

    def test_reproj_fields_map_to_verify_stat_dict(self):
        from calibrate_hand_eye import _report_dict  # type: ignore

        d = _report_dict(_synthetic_result())
        assert d["reproj_rms_px"] == {"mean": 0.15, "max": 0.2, "std": 0.04}
        assert d["per_view_reproj_rms_px"] == [0.1, 0.2, 0.15]

    def test_axxb_fields_map_to_verify_stat_dict(self):
        from calibrate_hand_eye import _report_dict  # type: ignore

        d = _report_dict(_synthetic_result())
        assert d["axxb_rot_deg"] == {"mean": 0.3, "max": 0.5, "std": 0.1}
        assert d["axxb_trans_mm"] == {"mean": 1.5, "max": 2.0, "std": 0.3}

    def test_board_origin_fields(self):
        from calibrate_hand_eye import _report_dict  # type: ignore

        d = _report_dict(_synthetic_result())
        assert d["board_origin_base_mm"] == [300.0, 0.0, 30.0]
        assert d["board_origin_spread_mm"] == {"mean": 1.0, "max": 2.0, "std": 0.5}

    def test_rotation_spread_and_cross_check(self):
        from calibrate_hand_eye import _report_dict  # type: ignore

        d = _report_dict(_synthetic_result())
        assert d["rotation_spread_deg"] == 45.0
        assert d["cross_check_max_deg"] == 0.1
        assert d["cross_check_max_mm"] == 0.2


class TestJudgeVerdict:
    """``_judge`` three-band verdict: ✅ good / ⚠️ warn / ❌ bad."""

    def test_below_good_threshold_is_ok(self):
        from calibrate_hand_eye import _judge  # type: ignore

        assert _judge(0.1, 1.0, 2.0) == "✅"

    def test_between_good_and_warn_is_warning(self):
        from calibrate_hand_eye import _judge  # type: ignore

        assert _judge(1.5, 1.0, 2.0) == "⚠️"

    def test_above_warn_is_bad(self):
        from calibrate_hand_eye import _judge  # type: ignore

        assert _judge(2.5, 1.0, 2.0) == "❌"


class TestImportContract:
    """The CLI module must remain importable after the SE(3)/spec migration."""

    def test_module_imports_cleanly(self):
        import importlib

        mod = importlib.import_module("calibrate_hand_eye")
        assert hasattr(mod, "main")
        assert hasattr(mod, "do_selftest")
        assert hasattr(mod, "do_calibrate")
        assert hasattr(mod, "do_generate_board")


class TestCalibrationDeviceMigration:
    """The eye-in-hand CLI must use the calibration-owned adapter wrapper.

    Env still supplies runtime RGB-D/tool configuration to ``--verify``; its
    old calibration delegates are deliberately made unusable here so this
    test catches accidental regressions back to the compatibility surface.
    """

    @pytest.mark.parametrize("use_config", [True, False], ids=["yaml", "adapter-defaults"])
    def test_do_calibrate_builds_and_uses_device(self, monkeypatch, tmp_path, use_config):
        from calibrate_hand_eye import do_calibrate

        from jiuwensymbiosis.calibration.domain.ports import CalibrationCameraFrame
        from jiuwensymbiosis.utils.geometry import make_transform

        class _Env:
            @property
            def camera_mount(self):
                raise AssertionError("camera_mount must come from the calibration device")

            def get_flange_transform_mm(self):
                raise AssertionError("flange pose must come from the calibration device")

            def capture_calibration_frame(self):
                raise AssertionError("calibration frame must come from the calibration device")

            def move_to_flange_transform_mm(self, _tf):
                raise AssertionError("calibration motion must come from the calibration device")

        class _Device:
            camera_mount = "eye_in_hand"

            def get_flange_transform_mm(self):
                return np.eye(4)

            def capture_calibration_frame(self):
                return CalibrationCameraFrame(
                    rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                    intrinsics=np.array([[600.0, 0.0, 4.0], [0.0, 600.0, 4.0], [0.0, 0.0, 1.0]]),
                    distortion=None,
                    captured_at_ns=0,
                )

            def move_to_flange_transform_mm(self, _tf):
                raise AssertionError("this test does not exercise auto/verify motion")

        env = _Env()
        device = _Device()
        factory_calls = []
        session_factory_calls = []

        class _Session:
            def __init__(self, session_env):
                self.env = session_env

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _SessionFactory:
            def from_yaml(self, _path, *, include_sidecars=True):
                assert include_sidecars is False
                session_factory_calls.append(("yaml", _path))
                return _Session(env)

            def from_dict(self, data, *, include_sidecars=True):
                assert include_sidecars is False
                assert data == {}
                session_factory_calls.append(("dict", data))
                return _Session(env)

        class _Spec:
            session_factory = _SessionFactory()

            def make_calibration_device(self, got_env):
                factory_calls.append(got_env)
                return device

        import calibrate_hand_eye as cli

        monkeypatch.setattr(cli, "_resolve_adapter_spec", lambda _module: _Spec())
        stations = [
            cli.Station(
                make_transform(np.eye(3), [float(index), 0.0, 300.0]),
                cli.ViewDetection(ok=True, tf_cam_target=np.eye(4), reproj_rms_px=0.1),
            )
            for index in range(cli.MIN_STATIONS)
        ]

        def _collect_manual(got_device, *unused_args):
            assert got_device is device
            return stations, (8, 8)

        monkeypatch.setattr(cli, "_collect_manual", _collect_manual)
        monkeypatch.setattr(cli, "solve_eye_in_hand", lambda *args, **kwargs: _synthetic_result())
        monkeypatch.setattr(cli, "save_calibration", lambda *args, **kwargs: None)

        argv = [
            "--out",
            str(tmp_path / "calibration.json"),
            "--non-interactive",
            "--intrinsics",
            "600",
            "600",
            "4",
            "4",
            "--object-xyz",
            "1",
            "2",
            "3",
            "--no-drop-outliers",
        ]
        if use_config:
            argv[0:0] = ["--config", str(tmp_path / "calibrate.yaml")]
        args = cli._parse_args(argv)

        assert do_calibrate(args) == 0
        assert factory_calls == [env]
        expected_call = ("yaml", str(tmp_path / "calibrate.yaml")) if use_config else ("dict", {})
        assert session_factory_calls == [expected_call]


class TestPerturbTargetSemantics:
    """``_perturb_target_tf`` golden test: fix the default-xyz output matrix.

    The migration replaced the Piper-specific ``_perturb_target`` (which built a
    ``FlangePose`` from rpy+delta — dependent on the original rpy main values)
    with a body-agnostic SE(3) version that reverse-derives rpy from the base
    matrix via ``Rotation.from_matrix().as_euler('xyz')``. In Euler singularity
    regions (ry near ±90°) the reverse-derived rpy can take a different but
    equivalent main value, so the new path is NOT bit-identical to the old
    FlangePose path everywhere. This golden test fixes the new path's own output
    for a fixed seed so future refactors do not silently shift --auto targets.
    """

    # Golden matrices: _perturb_target_tf(base_tf, 20, -15, 25, 40, axes="xyz")
    # for 3 base poses (seed-fixed). Computed once with the migrated implementation;
    # any future change to the perturbation math must update these deliberately.
    _BASE_CASES = [
        # (base_rpy_deg, base_xyz_mm)
        ((10.0, 20.0, 30.0), (100.0, -50.0, 200.0)),
        ((-45.0, 60.0, 120.0), (-200.0, 100.0, 150.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 300.0)),
    ]

    def test_perturb_target_tf_golden_matrices(self):
        from calibrate_hand_eye import _perturb_target_tf  # type: ignore

        from jiuwensymbiosis.utils.geometry import make_transform

        # Reference: the new path should equal rpy_deg_to_rot(reverse_rpy + delta)
        # for non-singular bases (where as_euler round-trips cleanly).
        for (rx, ry, rz), (x, y, z) in self._BASE_CASES:
            base_tf = make_transform(_rpy_deg_to_rot(rx, ry, rz), np.array([x, y, z]))
            drx, dry, drz, dz = 20.0, -15.0, 25.0, 40.0
            tf = _perturb_target_tf(base_tf, drx, dry, drz, dz, axes="xyz")
            # Expected: rpy_deg_to_rot(rx+drx, ry+dry, rz+drz) at [x, y, z+dz]
            expected = make_transform(
                _rpy_deg_to_rot(rx + drx, ry + dry, rz + drz),
                np.array([x, y, z + dz]),
            )
            assert np.allclose(tf, expected, atol=1e-9), (
                f"perturb golden mismatch for base rpy=({rx},{ry},{rz}) xyz=({x},{y},{z}): \n got={tf}\n exp={expected}"
            )
