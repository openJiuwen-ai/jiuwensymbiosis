from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.calibrate import so101_eye_calib


class _AutoLowLevel:
    def __init__(self, move_exc: Exception | None = None) -> None:
        self.move_exc = move_exc
        self.q = np.asarray([3.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.grab_count = 0
        self.sent_actions: list[dict] = []
        self.enable_calls: list[tuple] = []

    def move_joint_blocking(self, _q: list[float]) -> None:
        if self.move_exc is not None:
            raise self.move_exc

    def get_angles(self) -> list[float]:
        return self.q.tolist()

    def grab_frames(self):
        self.grab_count += 1
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.zeros((8, 8), dtype=np.uint16)
        return rgb, depth

    def send_action(self, action: dict) -> None:
        self.sent_actions.append(action)

    def disable_arm_torque(self) -> None:
        self.enable_calls.append(("disable", so101_eye_calib.ARM_JOINT_ORDER))

    def enable_arm_torque(self) -> None:
        self.enable_calls.append(("enable", so101_eye_calib.ARM_JOINT_ORDER))

    def preset_current_joint_goal(self) -> None:
        q = np.asarray(self.get_angles(), dtype=np.float64)
        if q.shape != (len(so101_eye_calib.ARM_JOINT_ORDER),) or not np.all(np.isfinite(q)):
            raise ValueError(f"invalid joint vector: {q}")
        action = {f"{name}.pos": float(value) for name, value in zip(so101_eye_calib.ARM_JOINT_ORDER, q, strict=True)}
        self.send_action(action)

    def restore_all_torque(self) -> None:
        self.enable_calls.append(("enable", ()))


def _cfg() -> SimpleNamespace:
    names = so101_eye_calib.ARM_JOINT_ORDER
    return SimpleNamespace(
        port="/dev/ttyACM0",
        home_joints_deg=[0.0] * 5,
        joint_limits=dict.fromkeys(names, (-500.0, 500.0)),
        z_min_safe_mm=-50.0,
        workspace_bounds=(0.0, -100.0, 200.0, 100.0),
        camera_serial="260322272909",
        disable_torque_on_disconnect=False,
    )


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        bounds_file=None,
        min_pose_gap_deg=0.0,
        n_stations=2,
    )


def _capture_args() -> SimpleNamespace:
    return SimpleNamespace(
        settle_dwell_s=0.0,
        settle_max_delta_deg=0.05,
        settle_samples=4,
        settle_period_s=0.05,
        settle_timeout_s=3.0,
        frames_per_station=1,
        max_capture_joint_delta_deg=0.1,
        min_corners=18,
        max_reproj_px=0.8,
        max_board_tilt_deg=45.0,
    )


def _capture_context(args=None) -> so101_eye_calib._CaptureContext:
    return so101_eye_calib._CaptureContext(
        board=None,
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion_coeffs=None,
        args=args or _capture_args(),
    )


def _successful_detection() -> so101_eye_calib.ViewDetection:
    return so101_eye_calib.ViewDetection(
        ok=True,
        object_points=np.zeros((18, 3), dtype=np.float64),
        image_points=np.zeros((18, 2), dtype=np.float64),
        tf_cam_target=np.eye(4, dtype=np.float64),
        reproj_rms_px=0.1,
    )


def test_standalone_calibration_config_does_not_need_yaml():
    cfg = so101_eye_calib._standalone_calibration_config(
        SimpleNamespace(
            port="/dev/ttyACM9",
            robot_id="test_arm",
            calibration_dir=None,
            urdf_path=None,
            camera_serial="camera-123",
            move_timeout_s=None,
        )
    )

    assert cfg.port == "/dev/ttyACM9"
    assert cfg.robot_id == "test_arm"
    assert cfg.camera_serial == "camera-123"
    assert cfg.home_use_init_pose is True
    assert cfg.safety_validated is True
    assert cfg.z_min_safe_mm == -1_000_000.0
    assert cfg.workspace_bounds is None
    assert cfg.move_timeout_s == 5.0
    assert set(cfg.joint_limits) == set(so101_eye_calib.ARM_JOINT_ORDER)
    assert all(bounds == (-180.0, 180.0) for bounds in cfg.joint_limits.values())


def test_interpolate_taught_poses_retains_vertices_and_balances_segment_steps():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [30.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    poses = so101_eye_calib._interpolate_taught_poses(points, 7, min_gap_deg=5.0)

    assert len(poses) == 7
    assert any(np.allclose(pose, points[0]) for pose in poses)
    assert any(np.allclose(pose, points[1]) for pose in poses)
    assert any(np.allclose(pose, points[2]) for pose in poses)
    steps = [so101_eye_calib._joint_angle_deg(poses[index + 1], poses[index]) for index in range(len(poses) - 1)]
    assert max(steps) <= 10.0


def test_bounds_auto_mode_uses_ordered_points_instead_of_random_box(tmp_path):
    points = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [20.0, 40.0, 60.0, 80.0, 100.0],
        ],
        dtype=np.float64,
    )
    bounds_path = tmp_path / "bounds.npz"
    np.savez(bounds_path, points=points, lo=points.min(axis=0), hi=points.max(axis=0))
    args = _args()
    args.bounds_file = str(bounds_path)
    args.n_stations = 9

    poses = so101_eye_calib._generate_joint_poses(args)

    assert len(poses) == 9
    assert any(np.allclose(pose, points[0]) for pose in poses)
    assert any(np.allclose(pose, points[1]) for pose in poses)
    assert any(np.allclose(pose, points[2]) for pose in poses)


def test_auto_capture_accepts_stable_actual_pose_after_move_target_timeout(monkeypatch, caplog):
    ll = _AutoLowLevel(TimeoutError("did not reach target"))
    settled = iter([True, True])  # initial station gate, then per-frame gate
    monkeypatch.setattr(so101_eye_calib.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(so101_eye_calib, "_wait_until_settled", lambda *_args, **_kwargs: next(settled))
    monkeypatch.setattr(so101_eye_calib, "detect_board", lambda *_args, **_kwargs: _successful_detection())
    monkeypatch.setattr(so101_eye_calib, "_fk_from_joint_midpoint", lambda *_args: np.eye(4, dtype=np.float64))
    caplog.set_level(logging.INFO)

    result = so101_eye_calib._collect_auto_station(
        ll,
        _capture_context(),
        np.zeros(5, dtype=np.float64),
        1,
        1,
    )

    assert result is not None
    assert ll.grab_count == 1
    output = caplog.text
    assert "stable off-target pose accepted" in output
    assert "max target error=3.00deg" in output


def test_relaxed_gates_accept_valid_low_corner_high_error_detection(monkeypatch):
    ll = _AutoLowLevel()
    args = _capture_args()
    args.relaxed_gates = True
    args.min_corners = 6
    seen_min_corners = []
    tilted = np.eye(4, dtype=np.float64)
    tilted[:3, :3] = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    detection = so101_eye_calib.ViewDetection(
        ok=True,
        object_points=np.zeros((6, 3), dtype=np.float64),
        image_points=np.zeros((6, 2), dtype=np.float64),
        tf_cam_target=tilted,
        reproj_rms_px=99.0,
    )

    def detect(*_args, **kwargs):
        seen_min_corners.append(kwargs["min_corners"])
        return detection

    monkeypatch.setattr(so101_eye_calib.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(so101_eye_calib, "_wait_until_settled", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(so101_eye_calib, "detect_board", detect)
    monkeypatch.setattr(so101_eye_calib, "_fk_from_joint_midpoint", lambda *_args: np.eye(4, dtype=np.float64))

    result = so101_eye_calib._collect_auto_station(
        ll,
        _capture_context(args),
        np.zeros(5, dtype=np.float64),
        1,
        1,
    )

    assert result is not None
    assert seen_min_corners == [6]


def test_auto_capture_rejects_move_timeout_when_encoder_never_settles(monkeypatch, caplog):
    ll = _AutoLowLevel(TimeoutError("did not reach target"))
    monkeypatch.setattr(so101_eye_calib.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(so101_eye_calib, "_wait_until_settled", lambda *_args, **_kwargs: False)
    caplog.set_level(logging.INFO)

    result = so101_eye_calib._collect_auto_station(
        ll,
        _capture_context(),
        np.zeros(5, dtype=np.float64),
        1,
        1,
    )

    assert result is None
    assert ll.grab_count == 0
    assert "encoder did not settle after move target timeout" in caplog.text


def test_auto_capture_rejects_safety_error_without_stability_or_capture(monkeypatch, caplog):
    ll = _AutoLowLevel(ValueError("joint waypoint FK below driver z_min_safe"))

    def unexpected_settle(*_args, **_kwargs):
        raise AssertionError("safety rejection must not proceed to settle/capture")

    monkeypatch.setattr(so101_eye_calib, "_wait_until_settled", unexpected_settle)
    caplog.set_level(logging.INFO)

    result = so101_eye_calib._collect_auto_station(
        ll,
        _capture_context(),
        np.zeros(5, dtype=np.float64),
        1,
        1,
    )

    assert result is None
    assert ll.grab_count == 0
    assert "SKIPPED unsafe/invalid move_joint" in caplog.text


def test_auto_capture_rejects_settle_drift_without_capture(monkeypatch, caplog):
    ll = _AutoLowLevel(RuntimeError("SO-101 settle drift"))
    caplog.set_level(logging.INFO)

    result = so101_eye_calib._collect_auto_station(
        ll,
        _capture_context(),
        np.zeros(5, dtype=np.float64),
        1,
        1,
    )

    assert result is None
    assert ll.grab_count == 0
    assert "SKIPPED unstable move_joint" in caplog.text


def test_joint_midpoint_uses_shortest_wrapped_arc():
    midpoint = so101_eye_calib._joint_midpoint_deg(
        np.asarray([179.0, -10.0]),
        np.asarray([-179.0, 10.0]),
    )

    np.testing.assert_allclose(midpoint, [180.0, 0.0])


# --------------------------------------------------------------------------- parser
def _parse(argv: list[str]):
    return so101_eye_calib._build_parser().parse_args(argv)


def test_no_mode_is_required_error():
    with pytest.raises(SystemExit):
        _parse([])


def test_modes_mutually_exclusive():
    with pytest.raises(SystemExit):
        _parse(["--auto", "--replay", "x.npz"])
    with pytest.raises(SystemExit):
        _parse(["--release-torque", "--collect-bounds"])


def test_release_torque_does_not_require_square_size():
    args = _parse(["--release-torque"])
    assert args.release_torque is True
    assert args.square_size_mm is None  # release must not demand a board spec


def test_replay_omits_square_size_ok():
    args = _parse(["--replay", "x.npz"])
    assert args.replay == "x.npz"


# --------------------------------------------------------------------------- validation
def _validate(argv: list[str]):
    ap = so101_eye_calib._build_parser()
    args = ap.parse_args(argv)
    so101_eye_calib._validate_args(ap, args)
    return args


def test_release_torque_validation_skips_board_and_n_stations(tmp_path):
    # No --square-size-mm, no --bounds-file, n_stations tiny — all fine for release.
    args = _validate(["--release-torque", "--n-stations", "1"])
    assert args.release_torque is True


def test_auto_requires_bounds_file(tmp_path):
    with pytest.raises(SystemExit):
        _validate(["--auto", "--square-size-mm", "15.28"])


def test_auto_requires_existing_bounds_file(tmp_path):
    missing = tmp_path / "missing.npz"
    with pytest.raises(SystemExit):
        _validate(["--auto", "--square-size-mm", "15.28", "--bounds-file", str(missing)])


def test_auto_n_stations_floor_enforced(tmp_path):
    bounds = tmp_path / "b.npz"
    np.savez(bounds, points=np.zeros((2, 5)), lo=np.zeros(5), hi=np.zeros(5))
    with pytest.raises(SystemExit):
        _validate(["--auto", "--square-size-mm", "15.28", "--bounds-file", str(bounds), "--n-stations", "1"])


def test_collect_bounds_requires_bounds_file():
    with pytest.raises(SystemExit):
        _validate(["--collect-bounds", "--square-size-mm", "15.28"])


def test_collect_bounds_does_not_require_n_stations_floor(tmp_path):
    bounds = tmp_path / "b.npz"
    np.savez(bounds, points=np.zeros((2, 5)), lo=np.zeros(5), hi=np.zeros(5))
    args = _validate(["--collect-bounds", "--square-size-mm", "15.28", "--bounds-file", str(bounds)])
    assert args.collect_bounds is True


# --------------------------------------------------------------------------- live lifecycle
def test_run_live_collection_connect_failure_preserves_original_error(monkeypatch):
    calls = []

    class _Env:
        def __init__(self, _cfg):
            pass

        def connect(self):
            calls.append("connect")
            raise OSError("serial port not found")

        @property
        def low_level(self):
            pytest.fail("low_level accessed after connect failure")

        def disconnect(self):
            calls.append("disconnect")

    monkeypatch.setattr(so101_eye_calib, "So101Env", _Env)
    monkeypatch.setattr(
        so101_eye_calib,
        "_enable_all_torque",
        lambda _ll: pytest.fail("torque restore attempted without a driver"),
    )

    with pytest.raises(OSError, match="serial port not found"):
        so101_eye_calib._run_live_collection(_cfg(), None, SimpleNamespace())

    assert calls == ["connect", "disconnect"]


def test_run_live_collection_rejects_missing_driver_after_connect(monkeypatch):
    calls = []

    class _Env:
        def __init__(self, _cfg):
            self.low_level = None

        def connect(self):
            calls.append("connect")

        def disconnect(self):
            calls.append("disconnect")

    monkeypatch.setattr(so101_eye_calib, "So101Env", _Env)
    monkeypatch.setattr(
        so101_eye_calib,
        "_enable_all_torque",
        lambda _ll: pytest.fail("torque restore attempted without a driver"),
    )

    with pytest.raises(RuntimeError, match="connected without a low-level driver"):
        so101_eye_calib._run_live_collection(_cfg(), None, SimpleNamespace())

    assert calls == ["connect", "disconnect"]


# --------------------------------------------------------------------------- release-torque helpers
def _arm_ll(q=None):
    ll = _AutoLowLevel()
    if q is not None:
        ll.q = np.asarray(q, dtype=np.float64)
    return ll


def test_preset_current_pose_goal_writes_goal_and_enables_nothing_on_success():
    ll = _arm_ll(q=[10.0, 20.0, 30.0, 40.0, 50.0])
    assert so101_eye_calib._preset_current_pose_goal(ll) is True
    assert len(ll.sent_actions) == 1
    assert ll.enable_calls == []


def test_preset_current_pose_goal_failure_enables_nothing(monkeypatch, caplog):
    ll = _arm_ll()

    def boom(_action):
        raise OSError("bus write failed")

    monkeypatch.setattr(ll, "send_action", boom)
    assert so101_eye_calib._preset_current_pose_goal(ll) is False
    assert ll.enable_calls == []
    assert "failed to preset current joint goal" in caplog.text


def test_preset_current_pose_goal_rejects_nonfinite_or_bad_shape(monkeypatch, caplog):
    ll = _arm_ll()
    caplog.set_level(logging.ERROR)

    for bad in ([0.0, 0.0, 0.0, np.nan, 0.0], [0.0, 0.0, 0.0]):  # NaN, then wrong length
        ll = _arm_ll(q=bad)
        ll.sent_actions.clear()
        assert so101_eye_calib._preset_current_pose_goal(ll) is False
        assert ll.sent_actions == []
    assert "invalid joint vector" in caplog.text


def test_safe_restore_torque_returns_true_on_full_success(monkeypatch):
    ll = _arm_ll()
    monkeypatch.setattr(so101_eye_calib, "_preset_current_pose_goal", lambda _ll: True)
    monkeypatch.setattr(so101_eye_calib, "_enable_all_torque", lambda _ll: True)
    assert so101_eye_calib._safe_restore_torque(ll) is True


def test_safe_restore_torque_preset_failure_keeps_arm_loose(monkeypatch, caplog):
    ll = _arm_ll()
    enabled = []
    monkeypatch.setattr(so101_eye_calib, "_preset_current_pose_goal", lambda _ll: False)
    monkeypatch.setattr(so101_eye_calib, "_enable_all_torque", lambda _ll: enabled.append(True) or True)
    caplog.set_level(logging.ERROR)
    assert so101_eye_calib._safe_restore_torque(ll) is False
    assert enabled == []
    assert "保持松开" in caplog.text and "托臂" in caplog.text


def test_safe_restore_torque_enable_failure_does_not_report_restored(monkeypatch, caplog):
    ll = _arm_ll()
    monkeypatch.setattr(so101_eye_calib, "_preset_current_pose_goal", lambda _ll: True)
    monkeypatch.setattr(so101_eye_calib, "_enable_all_torque", lambda _ll: False)
    caplog.set_level(logging.ERROR)
    assert so101_eye_calib._safe_restore_torque(ll) is False
    assert "力矩已恢复" not in caplog.text
    assert "恢复力矩失败" in caplog.text and "检查总线" in caplog.text


# --------------------------------------------------------------------------- _release_torque_config
def test_release_torque_config_disables_camera_and_disconnect_torque_off():
    base = so101_eye_calib._standalone_calibration_config(
        SimpleNamespace(
            port="/dev/ttyACM0",
            robot_id="so101_left",
            calibration_dir=None,
            urdf_path=None,
            camera_serial="260322272909",
            move_timeout_s=None,
        )
    )
    base.disable_torque_on_disconnect = True  # simulate a YAML that turns it on
    before = (base.camera_serial, base.disable_torque_on_disconnect)

    new = so101_eye_calib._release_torque_config(base)

    assert new.camera_serial is None
    assert new.disable_torque_on_disconnect is False
    # original cfg is NOT mutated
    assert (base.camera_serial, base.disable_torque_on_disconnect) == before


# --------------------------------------------------------------------------- _release_torque end-to-end
def _patch_env(monkeypatch, ll):
    """Stub So101Env and _release_torque_config for an end-to-end release test.

    So101Env(...) -> env whose connect() sets low_level=ll; disconnect() records call.
    _release_torque_config returns the cfg unchanged (the test cfg is a SimpleNamespace,
    not a dataclass, and So101Env itself is stubbed so the real config fields do not matter).
    """

    class _Env:
        def __init__(self, _cfg):
            self.low_level = None
            self.disconnected = False

        def connect(self):
            self.low_level = ll

        def disconnect(self):
            self.disconnected = True

    monkeypatch.setattr(so101_eye_calib, "So101Env", _Env)
    monkeypatch.setattr(so101_eye_calib, "_release_torque_config", lambda _cfg: _cfg)
    return _Env


def test_release_torque_normal_path_presets_then_restores(monkeypatch, caplog):
    ll = _arm_ll()
    env_cls = _patch_env(monkeypatch, ll)
    calls = []
    monkeypatch.setattr(so101_eye_calib, "_disable_arm_torque", lambda _ll: calls.append("disable") or True)
    monkeypatch.setattr(so101_eye_calib, "_prompt", lambda _msg: calls.append("prompt") or "")
    monkeypatch.setattr(so101_eye_calib, "_safe_restore_torque", lambda _ll: calls.append("restore") or True)
    caplog.set_level(logging.INFO)

    restored = so101_eye_calib._release_torque(_cfg())
    assert restored is True
    assert calls == ["disable", "prompt", "restore"]
    assert env_cls
    assert "disconnected." in caplog.text


def test_release_torque_q_also_presets_and_restores(monkeypatch):
    ll = _arm_ll()
    _patch_env(monkeypatch, ll)
    calls = []
    monkeypatch.setattr(so101_eye_calib, "_disable_arm_torque", lambda _ll: calls.append("disable") or True)
    monkeypatch.setattr(so101_eye_calib, "_prompt", lambda _msg: calls.append("prompt") or "q")
    monkeypatch.setattr(so101_eye_calib, "_safe_restore_torque", lambda _ll: calls.append("restore") or True)

    assert so101_eye_calib._release_torque(_cfg()) is True
    assert calls == ["disable", "prompt", "restore"]  # q goes through the same restore


def test_release_torque_eof_presets_and_restores(monkeypatch):
    ll = _arm_ll()
    _patch_env(monkeypatch, ll)
    calls = []
    monkeypatch.setattr(so101_eye_calib, "_disable_arm_torque", lambda _ll: calls.append("disable") or True)
    monkeypatch.setattr(so101_eye_calib, "_prompt", lambda _msg: calls.append("prompt") or None)  # EOF
    monkeypatch.setattr(so101_eye_calib, "_safe_restore_torque", lambda _ll: calls.append("restore") or True)

    assert so101_eye_calib._release_torque(_cfg()) is True
    assert calls == ["disable", "prompt", "restore"]


def test_release_torque_restore_failure_returns_false(monkeypatch, caplog):
    ll = _arm_ll()
    _patch_env(monkeypatch, ll)
    monkeypatch.setattr(so101_eye_calib, "_disable_arm_torque", lambda _ll: True)
    monkeypatch.setattr(so101_eye_calib, "_prompt", lambda _msg: "")
    monkeypatch.setattr(so101_eye_calib, "_safe_restore_torque", lambda _ll: False)
    caplog.set_level(logging.ERROR)

    assert so101_eye_calib._release_torque(_cfg()) is False
    assert "力矩未确认恢复" in caplog.text and "托臂" in caplog.text


def test_release_torque_disable_failure_attempts_safe_restore(monkeypatch):
    ll = _arm_ll()
    _patch_env(monkeypatch, ll)
    calls = []
    monkeypatch.setattr(so101_eye_calib, "_disable_arm_torque", lambda _ll: calls.append("disable") or False)
    monkeypatch.setattr(so101_eye_calib, "_safe_restore_torque", lambda _ll: calls.append("restore") or True)
    # _prompt must NOT be called when disable failed (no torque to release, arm unmoved)
    monkeypatch.setattr(so101_eye_calib, "_prompt", lambda _msg: pytest.fail("prompt called on disable failure"))

    assert so101_eye_calib._release_torque(_cfg()) is True
    assert calls == ["disable", "restore"]


def test_release_torque_exception_preserves_original_error_and_restores(monkeypatch):
    ll = _arm_ll()
    _patch_env(monkeypatch, ll)
    restored = []
    monkeypatch.setattr(so101_eye_calib, "_disable_arm_torque", lambda _ll: True)

    def boom(_msg):
        raise KeyboardInterrupt

    monkeypatch.setattr(so101_eye_calib, "_prompt", boom)
    monkeypatch.setattr(so101_eye_calib, "_safe_restore_torque", lambda _ll: restored.append(True) or True)

    with pytest.raises(KeyboardInterrupt):
        so101_eye_calib._release_torque(_cfg())
    assert restored == [True]  # best-effort recovery ran before re-raising


def test_release_torque_connect_exception_does_not_swallow(monkeypatch):
    class _Env:
        def __init__(self, _cfg):
            self.low_level = None

        def connect(self):
            raise OSError("serial port not found")

        def disconnect(self):
            # So101Env.disconnect is idempotent; release-torque's finally calls it for
            # resource cleanup even when connect() raised. That is acceptable.
            pass

    monkeypatch.setattr(so101_eye_calib, "So101Env", _Env)
    monkeypatch.setattr(so101_eye_calib, "_release_torque_config", lambda _cfg: _cfg)
    monkeypatch.setattr(so101_eye_calib, "_disable_arm_torque", lambda _ll: pytest.fail("disable after connect fail"))
    monkeypatch.setattr(so101_eye_calib, "_safe_restore_torque", lambda _ll: pytest.fail("restore after connect fail"))

    with pytest.raises(OSError, match="serial port not found"):
        so101_eye_calib._release_torque(_cfg())


# --------------------------------------------------------------------------- main dispatch
def _patch_main(monkeypatch, argv, *, restored: bool):
    """Wire sys.argv, stub config/validation, and stub _release_torque's return."""
    import sys

    monkeypatch.setattr(sys, "argv", ["so101_eye_calib", *argv])
    monkeypatch.setattr(so101_eye_calib, "_standalone_calibration_config", lambda _args: _cfg())
    monkeypatch.setattr(so101_eye_calib, "_validate_args", lambda _ap, _args: None)
    monkeypatch.setattr(so101_eye_calib, "_release_torque", lambda _cfg: restored)


def test_release_torque_main_exits_nonzero_on_restore_failure(monkeypatch):
    _patch_main(monkeypatch, ["--release-torque"], restored=False)
    with pytest.raises(SystemExit) as excinfo:
        so101_eye_calib.main()
    assert excinfo.value.code == 1


def test_release_torque_main_returns_on_success(monkeypatch, capsys):
    _patch_main(monkeypatch, ["--release-torque"], restored=True)
    so101_eye_calib.main()
    out = capsys.readouterr().out
    assert "RELEASE-TORQUE" in out  # _mode_label printed at startup


def test_release_torque_skips_board_and_run_live_collection(monkeypatch, capsys):
    _patch_main(monkeypatch, ["--release-torque"], restored=True)
    monkeypatch.setattr(so101_eye_calib, "_build_board", lambda _args: pytest.fail("board built in release mode"))
    monkeypatch.setattr(
        so101_eye_calib, "_run_live_collection", lambda *_a: pytest.fail("live collection in release mode")
    )
    so101_eye_calib.main()
