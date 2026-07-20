# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.so101.lowlevel (no LeRobot dependency).

The driver's LeRobot surface is injected via fakes (``lowlevel_helpers``) so
these tests run in the standard unit-test environment. They cover:

- ``set_gripper``: sends ONLY ``{"gripper.pos": target}`` (no arm keys), waits
  ``gripper_settle_s`` via an injected fake sleep, records the send_action return.
- ``move_joint_blocking``: linear interpolation, pre-validates all waypoints,
  rejects out-of-limit / non-finite before the first send_action, settles by
  polling real observation.
- ``connect``: calibration-file preload, action_features validation, kinematics
  build with the configured target frame.
- Reachability: IK residual rejection only when an explicit tolerance is set.
"""

from __future__ import annotations

import numpy as np
import pytest

from jiuwensymbiosis.adapters.so101.config import So101Config
from jiuwensymbiosis.adapters.so101.geometry import So101Pose, matrix_m_to_pose_mm_deg, position_error_mm
from jiuwensymbiosis.adapters.so101.lowlevel import ARM_JOINT_ORDER, So101Driver, So101PoseConvergenceError

from .lowlevel_helpers import FakeFollower, FakeKinematics, fake_lerobot_import, make_calib_file

_ARM_LIMITS = {
    "shoulder_pan": (-90.0, 90.0),
    "shoulder_lift": (-90.0, 90.0),
    "elbow_flex": (-90.0, 90.0),
    "wrist_flex": (-90.0, 90.0),
    "wrist_roll": (-180.0, 180.0),
}


def _make_cfg(**overrides) -> So101Config:
    base: dict = {
        "port": "/dev/fake",
        "home_joints_deg": [0.0, 0.0, 0.0, 0.0, 0.0],
        "joint_limits": _ARM_LIMITS,
        "max_relative_target": 5.0,
        "gripper_settle_s": 0.0,  # avoid real sleep in tests by default
        "trajectory_hz": 1000.0,  # near-zero period so settle loop is fast
        "settle_samples": 1,
        "move_timeout_s": 5.0,
        "max_joint_step_deg": 2.0,
        "joint_tolerance_deg": 0.5,
        # FakeKinematics maps elbow_flex=0 to z=0.  Motion-specific floor
        # tests override this explicitly; the generic control-flow tests use a
        # deliberately disabled floor so they exercise interpolation/settle.
        "z_min_safe_mm": -500.0,
        "safety_validated": True,  # tests use validated configs; connect() is fail-closed
    }
    base.update(overrides)
    return So101Config(**base)


def _make_driver(cfg: So101Config, tmp_path, follower: FakeFollower | None = None):
    calib = make_calib_file(tmp_path)
    if follower is None:
        follower = FakeFollower(config=None)
    follower.calibration_fpath = calib
    sleep_log: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_log.append(float(seconds))

    driver = So101Driver(
        cfg,
        sleep=fake_sleep,
        so_follower_factory=lambda robot_cfg: follower,
        kinematics_factory=lambda urdf, target_frame_name="gripper_frame_link", joint_names=None: FakeKinematics(
            urdf, target_frame_name, joint_names
        ),
        lerobot_import=fake_lerobot_import,
    )
    return driver, follower, sleep_log


class TestSetGripper:
    def test_close_sends_only_gripper_key(self, tmp_path):
        cfg = _make_cfg(gripper_settle_s=0.1)
        driver, follower, sleep_log = _make_driver(cfg, tmp_path)
        driver.connect()

        driver.set_gripper(on=True)

        assert len(follower.sent_actions) == 1
        action = follower.sent_actions[0]
        assert set(action.keys()) == {"gripper.pos"}
        assert action["gripper.pos"] == cfg.gripper_close_pos
        # Waited the configured settle time via the injected sleep.
        assert sleep_log == [pytest.approx(0.1, abs=1e-9)]

    def test_open_sends_open_target(self, tmp_path):
        cfg = _make_cfg(gripper_settle_s=0.0)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        driver.set_gripper(on=False)

        action = follower.sent_actions[0]
        assert action["gripper.pos"] == cfg.gripper_open_pos

    def test_no_arm_keys_in_gripper_action(self, tmp_path):
        """Critical: gripper action must never carry arm joint keys."""
        cfg = _make_cfg()
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        driver.set_gripper(on=True)

        action = follower.sent_actions[0]
        for arm_name in ARM_JOINT_ORDER:
            assert f"{arm_name}.pos" not in action

    def test_clipped_target_re_sent_until_converged(self, tmp_path):
        """Under the default max_relative_target a single send_action cannot move
        the gripper across its full range; set_gripper must re-send the target and
        poll the real gripper observation until it converges within
        gripper_tolerance."""
        cfg = _make_cfg(gripper_tolerance=2.0, settle_samples=1, gripper_settle_s=0.0)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        # FakeFollower starts the gripper at 50.0. Closing -> 0.0 but each
        # send_action is clipped to a 5-unit step toward the request (simulating
        # max_relative_target). FakeFollower tracks the actual (clipped) value,
        # so set_gripper must re-send ceil(50/5)=10 times to converge.
        step = 50.0

        def clip(action):
            nonlocal step
            req = action.get("gripper.pos", 0.0)
            # Clip to at most 5 units toward the requested target.
            if req >= step:
                actual = min(req, step + 5.0)
            else:
                actual = max(req, step - 5.0)
            step = actual
            return {"gripper.pos": actual}

        follower.clip_fn = clip
        driver.set_gripper(on=True)  # close -> gripper_close_pos=0.0

        assert driver._last_sent_action is not None
        # The gripper converged to the close target (0.0) within tolerance.
        assert abs(driver._last_sent_action["gripper.pos"] - 0.0) <= cfg.gripper_tolerance
        # Multiple sends happened (single-send could not reach 0 from 50).
        assert len(follower.sent_actions) > 1

    def test_gripper_stall_times_out(self, tmp_path):
        """If the gripper never converges (stall), set_gripper must raise
        TimeoutError instead of looping forever."""
        cfg = _make_cfg(gripper_timeout_s=0.05, gripper_settle_s=0.0)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        # Gripper observation never reflects the sent target -> stall.
        follower.track = False

        with pytest.raises(TimeoutError, match="gripper timeout"):
            driver.set_gripper(on=True)
        # The last recorded action is the requested gripper target.
        assert driver._last_sent_action is not None
        assert "gripper.pos" in driver._last_sent_action

    def test_settle_loop_is_throttled_by_trajectory_hz(self, tmp_path):
        """The gripper settle loop must NOT hammer the serial bus at full speed:
        each poll waits one ``trajectory_hz`` period. Injected fake sleep records
        every wait; this asserts the throttle fires once per poll (so a real serial
        bus is never saturated) — the alternative, an unthrottled loop, recorded
        ~34k sends in 50ms before the throttle was added."""
        cfg = _make_cfg(
            trajectory_hz=100.0,
            gripper_timeout_s=0.05,
            gripper_settle_s=0.0,
            settle_samples=1,
        )
        driver, follower, sleep_log = _make_driver(cfg, tmp_path)
        driver.connect()
        follower.track = False  # stall so the loop keeps polling until timeout

        with pytest.raises(TimeoutError, match="gripper timeout"):
            driver.set_gripper(on=True)

        # Every poll waits one trajectory_hz period (0.01s here). The loop polls
        # once per send, so a throttle sleep fires after every poll — proving each
        # poll is throttled rather than busy-spinning. (Fake sleep records the call
        # without advancing the clock, so the send count is NOT bounded here; on
        # real hardware the blocking sleep advances wall-clock and the timeout
        # fires after ~timeout/period sends.)
        period = 1.0 / cfg.trajectory_hz
        poll_sleeps = [s for s in sleep_log if abs(s - period) < 1e-9]
        assert len(poll_sleeps) >= len(follower.sent_actions) - 1, (
            f"expected a throttle sleep after every poll; got {len(poll_sleeps)} "
            f"sleeps vs {len(follower.sent_actions)} sends"
        )


class TestMoveJointBlocking:
    def test_fk_cartesian_floor_checked_before_joint_dispatch(self, tmp_path):
        cfg = _make_cfg(z_min_safe_mm=30.0, settle_overcompensate=False)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        # FakeKinematics maps elbow_flex to z=10*elbow_flex.  The target itself
        # is inside all joint limits, but the first interpolation waypoint is
        # below the configured floor.
        follower._arm = [0.0, 0.0, 4.0, 0.0, 0.0]
        sent_before = len(follower.sent_actions)
        with pytest.raises(ValueError, match="below driver z_min_safe"):
            driver.move_joint_blocking([0.0, 0.0, 0.0, 0.0, 0.0])
        assert len(follower.sent_actions) == sent_before

    def test_reaches_target_and_settles(self, tmp_path):
        cfg = _make_cfg()
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        # Small move within limits; target reachable in one interpolation step.
        driver.move_joint_blocking([1.0, 0.0, 0.0, 0.0, 0.0])

        # FakeFollower tracks, so the last sent action equals the target.
        last = follower.sent_actions[-1]
        assert last["shoulder_pan.pos"] == pytest.approx(1.0, abs=1e-9)

    def test_out_of_limit_target_rejected_before_first_action(self, tmp_path):
        cfg = _make_cfg()
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        sent_before = len(follower.sent_actions)

        # shoulder_pan limit is [-90, 90]; 95 exceeds it.
        with pytest.raises(ValueError, match="out of soft limits"):
            driver.move_joint_blocking([95.0, 0.0, 0.0, 0.0, 0.0])

        # No action was sent (pre-validation before first send_action).
        assert len(follower.sent_actions) == sent_before

    def test_non_finite_target_rejected_before_first_action(self, tmp_path):
        cfg = _make_cfg()
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        sent_before = len(follower.sent_actions)

        with pytest.raises(ValueError, match="finite"):
            driver.move_joint_blocking([float("nan"), 0.0, 0.0, 0.0, 0.0])

        assert len(follower.sent_actions) == sent_before

    def test_wrong_length_rejected(self, tmp_path):
        cfg = _make_cfg()
        driver, _, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        with pytest.raises(ValueError, match="5 joints"):
            driver.move_joint_blocking([0.0, 0.0, 0.0])  # only 3

    def test_interpolation_respects_max_joint_step(self, tmp_path):
        cfg = _make_cfg(max_joint_step_deg=2.0)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        # 6-degree move on shoulder_pan with 2deg steps -> ceil(6/2)=3 waypoints.
        driver.move_joint_blocking([6.0, 0.0, 0.0, 0.0, 0.0])

        # The interpolation waypoints appear as intermediate sent actions.
        sp_values = [a["shoulder_pan.pos"] for a in follower.sent_actions]
        # Expect at least the 3 interpolation waypoints before settle re-sends.
        assert sp_values[0] == pytest.approx(2.0, abs=1e-9)
        assert sp_values[1] == pytest.approx(4.0, abs=1e-9)
        assert sp_values[2] == pytest.approx(6.0, abs=1e-9)


class TestSettleEdgeCases:
    """Plan §A3: record send_action's actual (clipped) target, re-send it when
    the arm hasn't converged, and time out (stall) instead of looping forever."""

    def test_clipped_final_target_is_recorded_and_re_sent(self, tmp_path):
        """LeRobot's send_action may clip via max_relative_target. The driver must
        (a) record the actual (clipped) target it got back, and (b) keep re-sending
        the requested final target (each re-send is itself clipped) rather than
        relying on observation polling alone. Here clipping creates a gap the arm
        physically cannot close, so the settle loop correctly times out — but not
        before re-sending the target repeatedly, and recording the clipped actual
        each time."""
        # Tolerance (0.5) < the 1.0 clip gap -> arm can't reach the requested
        # 6.0, so the settle loop re-sends until move_timeout_s fires. Drift abort
        # is disabled (0): the error here is a stable 1.0 (not growing), so this
        # test exercises the timeout path, not the drift-abort path. Over-
        # compensation is OFF so the settle loop re-sends the bare target (this
        # test asserts the legacy "re-send + record clipped actual" contract;
        # over-compensation's behavior under clipping is covered separately).
        cfg = _make_cfg(
            joint_tolerance_deg=0.5,
            move_timeout_s=0.05,
            settle_drift_abort_samples=0,
            settle_overcompensate=False,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        # Simulate LeRobot clipping shoulder_pan from the requested 6.0 to 5.0.
        def clip(action):
            req = action.get("shoulder_pan.pos")
            if req is not None and abs(req - 6.0) < 1e-9:
                clipped = dict(action)
                clipped["shoulder_pan.pos"] = 5.0
                return clipped
            return dict(action)

        follower.clip_fn = clip

        with pytest.raises(TimeoutError, match="move timeout"):
            driver.move_joint_blocking([6.0, 0.0, 0.0, 0.0, 0.0])

        # (a) The driver recorded the *actual* (clipped) last target returned by
        # send_action, not the requested one — clipping is observable.
        assert driver._last_sent_action is not None
        assert driver._last_sent_action["shoulder_pan.pos"] == pytest.approx(5.0, abs=1e-9)
        # (b) The settle loop re-sent the requested final target (6.0 -> clipped
        # to 5.0) at least once after the interpolation sweep; the sweep itself
        # sends the last waypoint exactly once, so >=2 clipped actions means a
        # re-send happened in the settle loop.
        clipped_sends = [
            a["shoulder_pan.pos"] for a in follower.sent_actions if abs(a["shoulder_pan.pos"] - 5.0) < 1e-9
        ]
        assert len(clipped_sends) >= 2, "expected the clipped target re-sent in the settle loop"

    def test_stall_times_out_instead_of_looping(self, tmp_path):
        """If the arm never converges (stall), the settle loop must raise
        TimeoutError rather than loop forever. Uses a tiny move_timeout_s."""
        # Drift abort disabled: a stall has a constant error (not growing), so it
        # must hit the timeout, not the drift-abort path. Over-compensation OFF so
        # the settle loop re-sends the bare requested target (the recorded action
        # is the target itself); over-compensation's stall behavior (drift-abort
        # or timeout) is covered in TestSettleOvercompensate.
        cfg = _make_cfg(
            move_timeout_s=0.05,
            settle_samples=1,
            settle_drift_abort_samples=0,
            settle_overcompensate=False,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        # Disable tracking so observation never reflects the sent target -> stall.
        follower.track = False

        with pytest.raises(TimeoutError, match="move timeout"):
            driver.move_joint_blocking([1.0, 0.0, 0.0, 0.0, 0.0])
        # The last recorded action is the requested final target (no clip here).
        assert driver._last_sent_action is not None
        assert driver._last_sent_action["shoulder_pan.pos"] == pytest.approx(1.0, abs=1e-9)

    def test_settle_aborts_on_drift_instead_of_pushing_to_limit(self, tmp_path):
        """A gravity-loaded servo that drifts AWAY from the target (error grows
        each re-send) must trip the drift abort and raise RuntimeError, not loop
        until move_timeout_s — re-sending toward a drifting joint historically
        pushed it toward a mechanical limit. This is the regression test for the
        real-robot elbow divergence (see so101-settle-loop-issue memory)."""
        # Long timeout so the drift abort (not the timeout) is what fires.
        cfg = _make_cfg(
            joint_tolerance_deg=0.5,
            move_timeout_s=30.0,
            settle_drift_abort_samples=5,
            settle_resend_period_s=0.0,  # legacy rate; drift logic is rate-independent
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        # Simulate elbow_flex drifting toward larger angles (away from the
        # target) under gravity: the hook receives the prior real position
        # (prev_arm), so adding 0.5 deg each round accumulates — elbow creeps
        # away from the -2.0 target, so the settle error grows every re-send.
        def drift(prev_arm, target_arm):  # noqa: ARG001 - target unused; drift is uncontrolled
            prev_arm[2] = prev_arm[2] + 0.5
            return prev_arm

        follower.drift_fn = drift

        import time as _time

        t0 = _time.monotonic()
        with pytest.raises(RuntimeError, match="settle drift"):
            # Move elbow_flex (index 2) down toward -2.0; drift pushes it up,
            # so the error grows every settle re-send.
            driver.move_joint_blocking([0.0, 0.0, -2.0, 0.0, 0.0])
        elapsed = _time.monotonic() - t0
        # Aborted quickly via drift detection, NOT by burning the 30s timeout.
        assert elapsed < 5.0, f"drift abort should fire fast, took {elapsed:.1f}s"
        # The driver did not keep pushing elbow toward a limit: the largest
        # elbow value ever sent is the requested target (-2.0), never a runaway.
        elbow_sent = [a["elbow_flex.pos"] for a in follower.sent_actions]
        assert max(elbow_sent) <= 0.0 + 1e-6, f"elbow was pushed up: {elbow_sent}"

    def test_settle_resend_period_throttles_resend_rate(self, tmp_path):
        """The settle re-send rate is capped by settle_resend_period_s (not the
        1/trajectory_hz interpolation period). Verifies the low-frequency re-send
        that stops overdriving gravity-loaded servos."""
        # Clipped target creates a stable error so the settle loop re-sends until
        # timeout; the sleep intervals in between must match settle_resend_period_s.
        cfg = _make_cfg(
            joint_tolerance_deg=0.5,
            move_timeout_s=1.0,
            settle_resend_period_s=0.05,
            settle_drift_abort_samples=0,  # stable error -> don't abort, hit timeout
            trajectory_hz=1000.0,  # interpolation period 0.001s, much smaller than resend
        )
        driver, follower, sleep_log = _make_driver(cfg, tmp_path)
        driver.connect()

        def clip(action):
            req = action.get("shoulder_pan.pos")
            if req is not None and abs(req - 6.0) < 1e-9:
                clipped = dict(action)
                clipped["shoulder_pan.pos"] = 5.0
                return clipped
            return dict(action)

        follower.clip_fn = clip

        with pytest.raises(TimeoutError, match="move timeout"):
            driver.move_joint_blocking([6.0, 0.0, 0.0, 0.0, 0.0])

        # Settle re-sends (after the interpolation sweep) should sleep ~0.05s
        # each, not the 0.001s interpolation period. Filter out the tiny
        # interpolation sleeps and assert the settle sleeps are at the throttle.
        settle_sleeps = [s for s in sleep_log if s >= 0.04]
        assert len(settle_sleeps) >= 2, f"expected throttled settle re-sends, got {sleep_log}"
        for s in settle_sleeps:
            assert 0.04 <= s <= 0.06, f"settle sleep {s} not at throttle 0.05s"


class TestConnect:
    def test_unvalidated_config_refuses_before_import_or_serial_open(self):
        cfg = _make_cfg(safety_validated=False)
        driver = So101Driver(cfg)

        def unexpected_import():
            raise AssertionError("LeRobot import must not run before the safety gate")

        driver._import_lerobot = unexpected_import
        with pytest.raises(RuntimeError, match="not safety-validated"):
            driver.connect()
        assert driver._connected is False

    def test_missing_calibration_file_raises(self, tmp_path):
        cfg = _make_cfg()
        follower = FakeFollower(config=None)
        follower.calibration_fpath = "/nonexistent/does_not_exist.json"
        driver = So101Driver(
            cfg,
            so_follower_factory=lambda robot_cfg: follower,
            kinematics_factory=FakeKinematics,
            lerobot_import=fake_lerobot_import,
        )
        with pytest.raises(RuntimeError, match="calibration file not found"):
            driver.connect()
        assert driver._connected is False

    def test_missing_action_feature_raises(self, tmp_path):
        cfg = _make_cfg()
        follower = FakeFollower(config=None)
        follower.calibration_fpath = make_calib_file(tmp_path)
        follower.action_features.pop("wrist_roll.pos")
        driver = So101Driver(
            cfg,
            so_follower_factory=lambda robot_cfg: follower,
            kinematics_factory=FakeKinematics,
            lerobot_import=fake_lerobot_import,
        )
        with pytest.raises(RuntimeError, match="action_features missing"):
            driver.connect()
        assert driver._connected is False

    def test_configured_camera_start_failure_aborts_connection(self, tmp_path, monkeypatch):
        from jiuwensymbiosis.perception import camera as camera_module

        class FailedCamera:
            def __init__(self, **kwargs):
                pass

            def start(self):
                return False

            def stop(self):
                pass

        monkeypatch.setattr(camera_module, "RealSenseCamera", FailedCamera)
        cfg = _make_cfg(camera_serial="missing-camera")
        driver, follower, _ = _make_driver(cfg, tmp_path)

        with pytest.raises(RuntimeError, match="configured camera.*failed to start"):
            driver.connect()
        assert driver._connected is False
        assert follower.connected is False

    def test_disconnect_is_idempotent(self, tmp_path):
        cfg = _make_cfg()
        driver, _, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        driver.disconnect()
        # Calling again must not raise.
        driver.disconnect()
        driver.close()  # alias path also safe

    def test_kinematics_failure_closes_opened_bus(self, tmp_path):
        """P1.2: if a post-connect step (kinematics build) fails, the already-opened
        follower must be torn down — no serial/torque leak."""
        cfg = _make_cfg()
        follower = FakeFollower(config=None)
        follower.calibration_fpath = make_calib_file(tmp_path)
        disconnect_calls = []
        orig_disconnect = follower.disconnect

        def tracking_disconnect():
            disconnect_calls.append(True)
            return orig_disconnect()

        follower.disconnect = tracking_disconnect

        class _BadKin:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("kinematics build failed")

        driver = So101Driver(
            cfg,
            so_follower_factory=lambda robot_cfg: follower,
            kinematics_factory=_BadKin,
            lerobot_import=fake_lerobot_import,
        )
        with pytest.raises(RuntimeError, match="kinematics build failed"):
            driver.connect()
        # The follower's bus was opened in step 4 and MUST be closed when the
        # step-7 kinematics build fails — otherwise the hardware leaks.
        assert disconnect_calls, "follower.disconnect() was never called after a post-connect failure"
        assert follower.connected is False
        assert driver._connected is False

    def test_kinematics_built_with_configured_frame(self, tmp_path):
        cfg = _make_cfg(ik_target_frame="gripper_frame_link")
        driver, _, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        assert driver._kin.target_frame_name == "gripper_frame_link"


class TestReachability:
    def test_move_to_pose_rejects_position_residual_when_unreachable(self, tmp_path):
        # FakeKinematics IK is exact for x/y/z but a mismatch on orientation is
        # inherent for 5-DoF. Use a config with a tight position tolerance and a
        # target the fake IK CANNOT reach (orientation-only residual is still
        # computed). Here we make position tolerance tiny so any nonzero FK
        # residual from rounding is rejected — but since FakeKinematics IK is
        # exact, residual is ~0 and we must force it via orientation instead.
        cfg = _make_cfg(
            ik_position_tolerance_mm=0.001,
            ik_orientation_tolerance_deg=None,  # record only
            z_min_safe_mm=10.0,  # keep the whole path above the floor
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        # Safe start above the z-floor so the interpolated path stays in bounds
        # (the new planner validates every waypoint's FK z, not just the endpoint).
        follower._arm = [1.0, 2.0, 10.0, 0.0, 0.0]  # FK z = 100 mm

        # FakeKinematics IK inverts x/y/z/10 exactly, so position residual ~0;
        # orientation is exact too. This should NOT raise.
        driver.move_to_pose_blocking(So101Pose(10.0, 20.0, 300.0, 0, 0, 0))

    def test_orientation_tolerance_none_does_not_reject(self, tmp_path):
        cfg = _make_cfg(ik_orientation_tolerance_deg=None, z_min_safe_mm=10.0)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        # Safe start above the z-floor (new planner validates every waypoint FK z).
        follower._arm = [1.0, 2.0, 10.0, 0.0, 0.0]  # FK z = 100 mm

        # With None tolerance, even a large orientation residual is only recorded.
        # FakeKinematics produces zero residual, so this just completes.
        driver.move_to_pose_blocking(So101Pose(10.0, 20.0, 300.0, 0, 0, 0))

    def test_orientation_tolerance_explicit_rejects_on_excess(self, tmp_path):
        cfg = _make_cfg(ik_orientation_tolerance_deg=0.001, z_min_safe_mm=10.0)  # 0.001 deg
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        sent_before = len(follower.sent_actions)

        # Force an orientation mismatch: target has nonzero Euler rz, but
        # FakeKinematics IK always zeroes rx/ry/rz, so FK residual = the full
        # target orientation magnitude, exceeding 0.001 deg at every waypoint.
        # The orientation failure is a subdivisible candidate, but the fake IK
        # can't improve with a closer seed, so the cap is hit and the path is
        # rejected. Either the orientation-residual message or the final cap
        # message is acceptable; the contract is "reject before dispatch".
        with pytest.raises(ValueError, match="orientation residual|unreachable via a continuous"):
            driver.move_to_pose_blocking(So101Pose(10.0, 20.0, 300.0, 0, 0, 45.0))
        assert len(follower.sent_actions) == sent_before

    def test_position_residual_rejected_when_over_tolerance(self, tmp_path):
        cfg = _make_cfg(ik_position_tolerance_mm=0.0001, z_min_safe_mm=10.0)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        sent_before = len(follower.sent_actions)

        # Sabotage the fake IK so it returns a joint config that FK cannot match
        # the desired position. IK always returns zeros — FK of zeros = origin,
        # far from any non-origin desired. The position failure is subdivisible,
        # but the fake IK can't improve with a closer seed, so the cap is hit and
        # the path is rejected before dispatch.
        follower._arm = [0.0, 0.0, 0.0, 0.0, 0.0]

        class BadKin(FakeKinematics):
            def inverse_kinematics(self, current, desired, position_weight=1.0, orientation_weight=0.01):
                # Always return zeros — FK of zeros = origin, far from desired.
                return np.zeros(5)

        driver._kin = BadKin("fake", "gripper_frame_link", list(ARM_JOINT_ORDER))
        with pytest.raises(ValueError, match="position residual|unreachable via a continuous"):
            driver.move_to_pose_blocking(So101Pose(100.0, 200.0, 300.0, 0, 0, 0))
        assert len(follower.sent_actions) == sent_before


class TestCartesianSafetyChecks:
    """P1.3: the driver repeats Z-floor + XY-bound checks before dispatching,
    and pre-validates the whole interpolated path — not just the endpoint."""

    def test_target_below_z_floor_rejected_before_send(self, tmp_path):
        cfg = _make_cfg(z_min_safe_mm=30.0)  # default, made explicit
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        sent_before = len(follower.sent_actions)

        # Target z=20 < z_min_safe=30. The driver must reject before any action.
        with pytest.raises(ValueError, match="below driver z_min_safe"):
            driver.move_to_pose_blocking(So101Pose(0.0, 0.0, 20.0, 0, 0, 0))
        assert len(follower.sent_actions) == sent_before

    def test_target_out_of_xy_bounds_rejected(self, tmp_path):
        cfg = _make_cfg(workspace_bounds=(0.0, -300.0, 500.0, 300.0), z_min_safe_mm=10.0)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        sent_before = len(follower.sent_actions)

        # x=600 > xmax=500.
        with pytest.raises(ValueError, match="out of workspace x"):
            driver.move_to_pose_blocking(So101Pose(600.0, 0.0, 50.0, 0, 0, 0))
        assert len(follower.sent_actions) == sent_before

    def test_ik_endpoint_below_z_floor_rejected(self, tmp_path):
        """Target is safe (z=26, above floor 25) and the IK residual is within
        tolerance (2 mm < 3 mm), but the IK solution's FK z (24 mm) is below the
        floor. This is the 5-DoF case the boundary check exists for: a target
        that is itself safe but whose reachable IK endpoint is not. The boundary
        check must fire before dispatch."""
        cfg = _make_cfg(z_min_safe_mm=25.0, ik_position_tolerance_mm=3.0)

        class _DriftKin(FakeKinematics):
            # IK solves to a config 2 mm below the commanded waypoint z (within
            # the 3 mm residual tolerance, but below the 25 mm floor for the
            # endpoint at z=26 -> FK z=24).
            def inverse_kinematics(self, current, desired, position_weight=1.0, orientation_weight=0.01):
                wp = matrix_m_to_pose_mm_deg(np.asarray(desired, dtype=float))
                return np.array([wp.x / 10.0, wp.y / 10.0, (wp.z - 2.0) / 10.0, 0.0, 0.0], dtype=float)

        driver, follower, _ = _make_driver(cfg, tmp_path, follower=None)
        driver.connect()
        # Safe start (FK z = 5*10 = 50 mm). _DriftKin drifts 2 mm below each wp;
        # for the endpoint wp z=26 -> FK z=24 < 25.
        follower._arm = [0.0, 0.0, 5.0, 0.0, 0.0]
        driver._kin = _DriftKin("fake", "gripper_frame_link", list(ARM_JOINT_ORDER))
        sent_before = len(follower.sent_actions)

        with pytest.raises(ValueError, match="below driver z_min_safe|unreachable via a continuous"):
            driver.move_to_pose_blocking(So101Pose(0.0, 0.0, 26.0, 0, 0, 0))
        assert len(follower.sent_actions) == sent_before

    def test_safe_target_dispatches_and_settles(self, tmp_path):
        """Sanity: a fully-safe target (z, xy within bounds) does dispatch."""
        cfg = _make_cfg(z_min_safe_mm=10.0, workspace_bounds=(0.0, -300.0, 500.0, 300.0))
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        # Safe start above the z-floor so the interpolated path stays in bounds
        # (FakeKinematics FK z = joint[2] * 10 mm). The new EE-interpolation
        # planner validates EVERY waypoint's FK z, not just the endpoint.
        follower._arm = [1.0, 2.0, 5.0, 0.0, 0.0]  # FK z = 50 mm, xy=(10,20)
        sent_before = len(follower.sent_actions)

        driver.move_to_pose_blocking(So101Pose(10.0, 20.0, 50.0, 0, 0, 0))
        # Dispatched at least one action.
        assert len(follower.sent_actions) > sent_before

    def test_cartesian_path_respects_interp_step(self, tmp_path):
        """Contract: the dispatched Cartesian path is a sequence of IK waypoints
        spaced by ``cartesian_interp_step_mm`` (EE translation), one IK per step
        seeded by the previous solution. FakeKinematics IK is exact, so adjacent
        joint deltas = total joint delta / steps and must stay small."""
        # Use a large interp step so the waypoint count is predictable and small.
        # 10 mm/step over a 200 mm move => ~20 waypoints.
        cfg = _make_cfg(z_min_safe_mm=10.0, max_joint_step_deg=2.0, cartesian_interp_step_mm=10.0)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        # Safe start above the z-floor (FakeKinematics FK z = joint[2] * 10 mm);
        # start at z=100 so the lerp to z=300 stays >= 10 mm throughout.
        follower._arm = [0.0, 0.0, 10.0, 0.0, 0.0]
        sent_before = len(follower.sent_actions)

        # FakeKinematics: FK(q) maps q_i -> position_i*10 mm; IK inverts exactly.
        # Target (0, 0, 300) -> IK joint 2 = 30. From start joint 2 = 10, the
        # joint delta is 20 deg spread over ~20 steps => ~1 deg/step < max_joint_step.
        driver.move_to_pose_blocking(So101Pose(0.0, 0.0, 300.0, 0, 0, 0))

        arm_actions = follower.sent_actions[sent_before:]
        joint_seq = [
            [a[f"{n}.pos"] for n in ARM_JOINT_ORDER]
            for a in arm_actions
            if all(f"{n}.pos" in a for n in ARM_JOINT_ORDER)
        ]
        assert len(joint_seq) >= 15, f"expected >= 15 IK waypoints for a 200mm move at 10mm/step, got {len(joint_seq)}"
        # Every adjacent dispatched joint delta must respect the joint-step cap.
        for prev, cur in zip(joint_seq, joint_seq[1:], strict=False):
            max_delta = max(abs(c - p) for p, c in zip(prev, cur, strict=False))
            assert max_delta <= cfg.max_joint_step_deg + 1e-6, (
                f"adjacent joint delta {max_delta} exceeds max_joint_step_deg {cfg.max_joint_step_deg}"
            )

    def test_cartesian_rejects_ik_outside_joint_limits(self, tmp_path):
        """If IK returns a joint config outside the soft limits, the planner must
        reject before any dispatch (no bisecting away from a limit-violating IK —
        the seed chain already gives the best seed, so a limit violation means the
        target genuinely requires an out-of-bounds joint)."""
        cfg = _make_cfg(z_min_safe_mm=10.0, max_joint_step_deg=2.0, cartesian_interp_step_mm=10.0)

        class _OverLimitKin(FakeKinematics):
            """IK that maps the target normally but pushes shoulder_pan past its
            soft limit (+-90 deg) by adding a 100 deg offset."""

            def inverse_kinematics(self, current, desired, position_weight=1.0, orientation_weight=0.01):
                target_pose = matrix_m_to_pose_mm_deg(np.asarray(desired, dtype=float))
                return np.array(
                    [target_pose.x / 10.0 + 100.0, target_pose.y / 10.0, target_pose.z / 10.0, 0.0, 0.0],
                    dtype=float,
                )

        driver, follower, _ = _make_driver(cfg, tmp_path, follower=None)
        driver.connect()
        follower._arm = [0.0, 0.0, 5.0, 0.0, 0.0]  # safe start (FK z=50)
        driver._kin = _OverLimitKin("fake", "gripper_frame_link", list(ARM_JOINT_ORDER))
        sent_before = len(follower.sent_actions)

        # Every IK waypoint has shoulder_pan ~ 100 deg > +90 soft limit, so the
        # planner must reject on the first waypoint and dispatch nothing.
        with pytest.raises(ValueError, match="out of soft limits"):
            driver.move_to_pose_blocking(So101Pose(10.0, 20.0, 50.0, 0, 0, 0))
        assert len(follower.sent_actions) == sent_before


class TestSettleOvercompensate:
    """Settle real-time over-compensation (software I term for STS3215 PD).

    With ``settle_overcompensate=True`` the settle loop re-sends ``target + e``
    (``e = target - actual``, fresh from the encoder each round) instead of the
    bare ``target``: the servo (PD, no firmware I term) parks at ``target - e``,
    so over-commanding makes it park AT ``target``. FakeFollower.steady_offset
    simulates the servo parking at ``command + offset`` (the steady-state error).
    """

    def test_overcompensate_reaches_target_under_steady_offset(self, tmp_path):
        """A constant elbow steady-state offset (2 deg) would leave the arm 2 deg
        short of the target. With over-compensation ON, the settle loop re-sends
        ``target + e`` so the servo parks AT the target (err ~0), landing inside
        a tight ``joint_tolerance_deg``. Without it (OFF) the residual stays at the
        full offset (>= tolerance -> timeout)."""
        # Tolerance 0.5 < offset 2.0 so a bare-target re-send cannot converge;
        # over-compensation must close the 2 deg gap. Long timeout so the
        # convergence (not a timeout) is what succeeds.
        cfg = _make_cfg(
            joint_tolerance_deg=0.5,
            move_timeout_s=5.0,
            settle_overcompensate=True,
            settle_drift_abort_samples=0,  # offset is constant -> err shrinks, no drift
            trajectory_hz=1000.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        follower.steady_offset = [0.0, 0.0, 2.0, 0.0, 0.0]  # elbow +2 deg steady error

        # Move elbow from 0 to 5 deg; servo would park at 7 (5+2) without over-comp.
        driver.move_joint_blocking([0.0, 0.0, 5.0, 0.0, 0.0])

        actual = np.asarray(driver.get_angles(), dtype=float)
        assert abs(actual[2] - 5.0) <= cfg.joint_tolerance_deg + 1e-6, (
            f"over-compensation did not reach target: elbow actual={actual[2]:.3f} vs 5.0"
        )

    def test_overcompensate_disabled_leaves_steady_residual(self, tmp_path):
        """With over-compensation OFF and a tight tolerance (< offset), the settle
        loop re-sends the bare target and times out — the legacy behavior the
        ``joint_tolerance_deg >= 3.5`` workaround was for."""
        cfg = _make_cfg(
            joint_tolerance_deg=0.5,
            move_timeout_s=0.2,
            settle_overcompensate=False,
            settle_drift_abort_samples=0,
            trajectory_hz=1000.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        follower.steady_offset = [0.0, 0.0, 2.0, 0.0, 0.0]

        with pytest.raises(TimeoutError, match="move timeout"):
            driver.move_joint_blocking([0.0, 0.0, 5.0, 0.0, 0.0])

    def test_overcompensate_fails_closed_when_command_breaks_limit(self, tmp_path):
        """If the over-command ``target + e`` would break a soft limit, the settle
        loop falls back to the bare target (fail-closed: keeps the residual but
        stays in bounds) instead of raising or pushing past the limit. Because the
        residual then can't be closed (bare target can't beat PD error), the loop
        hits the move timeout — the contract is "never break the limit, never raise
        on the limit rejection" rather than reaching the target."""
        limits = {
            "shoulder_pan": (-90.0, 90.0),
            "shoulder_lift": (-90.0, 90.0),
            "elbow_flex": (-86.0, 86.0),
            "wrist_flex": (-85.0, 85.0),
            "wrist_roll": (-180.0, 180.0),
        }
        cfg = _make_cfg(
            joint_limits=limits,
            joint_tolerance_deg=0.5,
            move_timeout_s=0.3,
            settle_overcompensate=True,
            settle_drift_abort_samples=0,  # offset constant -> err flat, no drift
            trajectory_hz=1000.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        # Target elbow = 85 (within [-86,86]); steady_offset -3 -> servo parks at
        # 82; e = 85-82 = +3; over-command = 85+3 = 88 > 86 -> must fall back to
        # bare target each round (no raise, no limit break) until timeout.
        follower.steady_offset = [0.0, 0.0, -3.0, 0.0, 0.0]

        with pytest.raises(TimeoutError, match="move timeout"):
            driver.move_joint_blocking([0.0, 0.0, 85.0, 0.0, 0.0])

        # No dispatched action ever commanded elbow past the soft limit — the
        # over-command was rejected before send_action each round.
        elbow_sent = [a["elbow_flex.pos"] for a in follower.sent_actions]
        assert max(elbow_sent) <= 86.0 + 1e-6, f"over-command broke limit: {elbow_sent}"

    def test_overcompensate_drift_abort_still_fires_on_divergence(self, tmp_path):
        """A gravity-loaded servo that drifts AWAY (err grows each re-send) must
        still trip the drift abort even with over-compensation ON — over-comp
        does not mask a real settle failure. drift_fn accumulates away from the
        target faster than over-comp can correct, so err grows."""
        cfg = _make_cfg(
            joint_tolerance_deg=0.5,
            move_timeout_s=30.0,  # long so drift abort (not timeout) fires
            settle_overcompensate=True,
            settle_drift_abort_samples=5,
            settle_resend_period_s=0.0,  # drift logic is rate-independent
            trajectory_hz=1000.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        # Drift: elbow creeps +0.5 deg per send (away from the -2.0 target), so
        # err grows every settle re-send regardless of over-compensation.
        def drift(prev_arm, target_arm):  # noqa: ARG001
            prev_arm[2] = prev_arm[2] + 0.5
            return prev_arm

        follower.drift_fn = drift

        import time as _time

        t0 = _time.monotonic()
        with pytest.raises(RuntimeError, match="settle drift"):
            driver.move_joint_blocking([0.0, 0.0, -2.0, 0.0, 0.0])
        elapsed = _time.monotonic() - t0
        assert elapsed < 5.0, f"drift abort should fire fast, took {elapsed:.1f}s"


class TestPoseConvergence:
    """Joint-space convergence trim that compensates the STS3215 PD steady-state
    error (firmware I term is inert). FakeFollower.steady_offset simulates the
    servo parking at command+offset instead of command, leaving a Cartesian
    residual the convergence loop closes by over-commanding q_target + accum_e
    (re-solving NO IK).
    """

    def test_convergence_compensates_constant_offset(self, tmp_path):
        """A constant joint steady-state offset (elbow 2 deg -> 20 mm z residual
        under FakeKinematics FK=joint*10mm) must be compensated by the convergence
        loop: after the first planned move the arm is 20 mm short, the loop
        over-commands q_target - offset, the servo parks at q_target, and the
        residual drops to ~0 within the iteration budget. ``joint_tolerance_deg``
        is set above 2 deg so the single-move settle converges (no timeout)."""
        cfg = _make_cfg(
            joint_tolerance_deg=3.0,  # > 2.0 deg offset so settle converges
            z_min_safe_mm=10.0,
            pose_convergence_max_iters=3,
            pose_convergence_tolerance_mm=1.0,
            max_joint_step_deg=2.0,
            cartesian_interp_step_mm=10.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        # Safe start above the z-floor (FakeKinematics FK z = joint[2] * 10 mm).
        follower._arm = [0.0, 0.0, 10.0, 0.0, 0.0]  # FK z = 100 mm
        # STS3215 PD steady-state error: servo parks 2 deg off the commanded elbow.
        follower.steady_offset = [0.0, 0.0, 2.0, 0.0, 0.0]

        target = So101Pose(0.0, 0.0, 300.0, 0, 0, 0)
        driver.move_to_pose_blocking(target)

        # After convergence the real pose (FK of encoder joints) is within tol of target.
        actual = driver.get_pose()
        assert position_error_mm(actual, target) <= cfg.pose_convergence_tolerance_mm + 1e-6, (
            f"convergence failed: residual {position_error_mm(actual, target):.3f} mm"
        )

    def test_convergence_disabled_when_max_iters_zero(self, tmp_path):
        """``pose_convergence_max_iters=0`` restores the legacy single-move
        behavior: no convergence trim, so a steady offset leaves the full residual."""
        cfg = _make_cfg(
            joint_tolerance_deg=3.0,
            z_min_safe_mm=10.0,
            pose_convergence_max_iters=0,
            pose_convergence_tolerance_mm=1.0,
            max_joint_step_deg=2.0,
            cartesian_interp_step_mm=10.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        follower._arm = [0.0, 0.0, 10.0, 0.0, 0.0]
        follower.steady_offset = [0.0, 0.0, 2.0, 0.0, 0.0]  # 20 mm z residual

        target = So101Pose(0.0, 0.0, 300.0, 0, 0, 0)
        comp_calls = 0
        orig_move_joint = driver.move_joint_blocking

        def counting_move_joint(q, *a, **kw):
            nonlocal comp_calls
            comp_calls += 1
            return orig_move_joint(q, *a, **kw)

        driver.move_joint_blocking = counting_move_joint
        driver.move_to_pose_blocking(target)

        # No convergence compensation: residual stays at the full 20 mm.
        actual = driver.get_pose()
        assert position_error_mm(actual, target) > 10.0, "expected no compensation with max_iters=0"
        # The convergence loop never ran (max_iters=0), so no compensation move.
        assert comp_calls == 0, f"expected no compensation with max_iters=0, got {comp_calls}"

    def test_convergence_small_residual_one_shot(self, tmp_path):
        """A residual already within ``pose_convergence_tolerance_mm`` must stop on
        iteration 1 with NO compensation move (the over-command path is skipped).

        Detection: wrap ``move_joint_blocking`` to count compensation calls. The
        planned move uses ``_dispatch_prevalidated_waypoints`` directly (not
        ``move_joint_blocking``), so any call to ``move_joint_blocking`` during
        ``move_to_pose_blocking`` is a convergence compensation move."""
        cfg = _make_cfg(
            joint_tolerance_deg=3.0,
            z_min_safe_mm=10.0,
            pose_convergence_max_iters=3,
            pose_convergence_tolerance_mm=1.0,
            max_joint_step_deg=2.0,
            cartesian_interp_step_mm=10.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        follower._arm = [0.0, 0.0, 10.0, 0.0, 0.0]
        # 0.05 deg offset -> 0.5 mm residual < tol 1.0 mm: already "close enough".
        follower.steady_offset = [0.0, 0.0, 0.05, 0.0, 0.0]

        comp_calls = 0
        orig_move_joint = driver.move_joint_blocking

        def counting_move_joint(q, *a, **kw):
            nonlocal comp_calls
            comp_calls += 1
            return orig_move_joint(q, *a, **kw)

        driver.move_joint_blocking = counting_move_joint

        target = So101Pose(0.0, 0.0, 300.0, 0, 0, 0)
        driver.move_to_pose_blocking(target)

        actual = driver.get_pose()
        assert position_error_mm(actual, target) <= 1.0
        # Residual already within tol on iteration 1 -> no compensation move fired.
        assert comp_calls == 0, f"expected no compensation for within-tol residual, got {comp_calls} comp moves"

    def test_convergence_fail_closed_when_over_command_breaks_joint_limit(self, tmp_path):
        """An over-command that would break a soft limit must stop fail-closed
        (no raise): the arm stays at its current safe real pose rather than
        breaking the limit or triggering RecoveryRail."""
        # Tight elbow limit [-86, 86]; target places q_target.elbow near 85, and
        # the 2 deg steady offset pushes the over-command to 87 > 86 -> rejected.
        limits = {
            "shoulder_pan": (-90.0, 90.0),
            "shoulder_lift": (-90.0, 90.0),
            "elbow_flex": (-86.0, 86.0),
            "wrist_flex": (-85.0, 85.0),
            "wrist_roll": (-180.0, 180.0),
        }
        cfg = _make_cfg(
            joint_limits=limits,
            joint_tolerance_deg=3.0,
            z_min_safe_mm=10.0,
            pose_convergence_max_iters=3,
            pose_convergence_tolerance_mm=1.0,
            max_joint_step_deg=2.0,
            cartesian_interp_step_mm=10.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        follower._arm = [0.0, 0.0, 10.0, 0.0, 0.0]
        # q_target elbow = 84 (z=840). A -2.5 deg steady offset parks the servo at
        # 81.5 (err 2.5 < joint_tolerance 3 -> settles). The convergence loop then
        # computes accum_e = 84 - 81.5 = +2.5, cmd_q elbow = 86.5 > 86 -> the
        # over-command breaks the soft limit, so the loop stops fail-closed (no raise).
        follower.steady_offset = [0.0, 0.0, -2.5, 0.0, 0.0]

        target = So101Pose(0.0, 0.0, 840.0, 0, 0, 0)  # q_target elbow = 84
        # The compensation is rejected before dispatch and is surfaced as a
        # typed not-reached failure; RecoveryRail can distinguish it from a
        # transport failure and leave the last safe real pose in place.
        with pytest.raises(So101PoseConvergenceError, match="not reached"):
            driver.move_to_pose_blocking(target)

        actual = driver.get_pose()
        # The arm never broke the elbow soft limit (real elbow <= 86).
        assert actual is not None  # get_pose did not raise
        # The last commanded elbow value never exceeded the limit.
        elbow_sent = [a["elbow_flex.pos"] for a in follower.sent_actions]
        assert max(elbow_sent) <= 86.0 + 1e-6, f"elbow over-command broke limit: {elbow_sent}"

    def test_convergence_max_iters_exhausted_stops_safely(self, tmp_path):
        """Exhaustion with a residual above tolerance is an explicit failure."""
        cfg = _make_cfg(
            joint_tolerance_deg=3.0,
            z_min_safe_mm=-10.0,
            pose_convergence_max_iters=1,  # only ONE compensation attempt allowed
            pose_convergence_tolerance_mm=1.0,
            max_joint_step_deg=2.0,
            cartesian_interp_step_mm=10.0,
        )
        driver, _, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        target = So101Pose(0.0, 0.0, 300.0, 0, 0, 0)
        driver.get_pose = lambda: So101Pose(0.0, 0.0, 0.0, 0, 0, 0)
        driver.get_angles = lambda: [0.0] * 5
        driver.move_joint_blocking = lambda *args, **kwargs: None

        with pytest.raises(So101PoseConvergenceError, match="iterations exhausted") as exc_info:
            driver._converge_to_pose(target, np.zeros(5), timeout_s=None)
        assert exc_info.value.residual_mm == pytest.approx(300.0)


class TestHome:
    def test_home_moves_to_configured_joints(self, tmp_path):
        cfg = _make_cfg(home_joints_deg=[5.0, -5.0, 10.0, -10.0, 0.0])
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        driver.home()

        last = follower.sent_actions[-1]
        for i, name in enumerate(ARM_JOINT_ORDER):
            assert last[f"{name}.pos"] == pytest.approx(cfg.home_joints_deg[i], abs=1e-9)

    def test_home_pose_reports_fk_of_home_joints(self, tmp_path):
        cfg = _make_cfg(home_joints_deg=[1.0, 2.0, 3.0, 4.0, 5.0])
        driver, _, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        pose = driver.home_pose
        # FakeKinematics: FK maps joint i to position i*10 mm.
        assert pose.x == pytest.approx(10.0, abs=1e-6)
        assert pose.y == pytest.approx(20.0, abs=1e-6)
        assert pose.z == pytest.approx(30.0, abs=1e-6)
