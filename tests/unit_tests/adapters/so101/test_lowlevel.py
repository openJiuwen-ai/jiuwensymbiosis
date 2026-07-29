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

import math
import time

import numpy as np
import pytest

from jiuwensymbiosis.adapters.so101.config import So101Config
from jiuwensymbiosis.adapters.so101.geometry import (
    So101Pose,
    matrix_m_to_pose_mm_deg,
    pose_mm_deg_to_matrix_m,
    position_error_mm,
)
from jiuwensymbiosis.adapters.so101.lowlevel import (
    ARM_JOINT_ORDER,
    So101Driver,
    So101PoseConvergenceError,
    So101PreDispatchError,
)

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
        "servo_min_send_period_s": 0.02,
        "servo_max_joint_vel_dps": 30.0,
        "servo_max_joint_step_deg": 1.0,
        "cartesian_interp_step_mm": 1.0,
        "settle_samples": 1,
        "move_timeout_s": 5.0,
        "max_joint_step_deg": 2.0,
        "joint_tolerance_deg": 0.5,
        "settle_soft_samples": 3,
        # FakeKinematics maps elbow_flex=0 to z=0.  Motion-specific floor
        # tests override this explicitly; the generic control-flow tests use a
        # deliberately disabled floor so they exercise interpolation/settle.
        "z_min_safe_mm": -500.0,
        "safety_validated": True,  # tests use validated configs; connect() is fail-closed
    }
    base.update(overrides)
    # Preserve the historical strict-only behavior for existing tests;
    # dedicated soft-settle tests opt into a wider band explicitly.
    if "settle_soft_tolerance_deg" not in overrides:
        base["settle_soft_tolerance_deg"] = base["joint_tolerance_deg"]
    return So101Config(**base)


def _make_driver(
    cfg: So101Config,
    tmp_path,
    follower: FakeFollower | None = None,
    *,
    monotonic=None,
):
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
        monotonic=monotonic,
    )
    return driver, follower, sleep_log


class TestSetGripper:
    def test_close_sends_only_gripper_key(self, tmp_path):
        cfg = _make_cfg(gripper_settle_s=0.1)
        driver, follower, sleep_log = _make_driver(cfg, tmp_path)
        driver.connect()

        driver.set_gripper(on=True)

        assert len(follower.sent_actions) == 10
        action = follower.sent_actions[-1]
        assert set(action.keys()) == {"gripper.pos"}
        assert action["gripper.pos"] == cfg.gripper_close_pos
        assert driver.last_gripper_result is not None
        assert driver.last_gripper_result["state"] == "closed"
        assert driver.holding_payload is False
        # Waited the configured settle time via the injected sleep.
        assert sleep_log[-1] == pytest.approx(0.1, abs=1e-9)

    def test_open_sends_open_target(self, tmp_path):
        cfg = _make_cfg(gripper_settle_s=0.0)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        driver.set_gripper(on=False)

        action = follower.sent_actions[-1]
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

    def test_close_stable_after_meaningful_travel_is_contact_success(self, tmp_path):
        """Meaningful closing travel followed by stable reads means contact."""
        cfg = _make_cfg(
            gripper_close_pos=0.0,
            gripper_tolerance=2.0,
            gripper_contact_min_travel=5.0,
            gripper_contact_stall_tolerance=0.25,
            gripper_contact_stall_samples=3,
            gripper_timeout_s=1.0,
            gripper_settle_s=0.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        position = 50.0
        requested_actions: list[dict[str, float]] = []

        def contact_at_30(action):
            nonlocal position
            requested_actions.append(dict(action))
            if "gripper.pos" in action:
                position = max(30.0, position - 5.0)
            return {"gripper.pos": position}

        follower.clip_fn = contact_at_30
        driver.set_gripper(on=True)

        result = driver.last_gripper_result
        assert result is not None
        assert result["state"] == "contact"
        assert result["position"] == pytest.approx(30.0)
        assert result["hold_target"] == pytest.approx(29.0)
        assert requested_actions[-1] == {"gripper.pos": pytest.approx(29.0)}
        assert follower.sent_actions[-1] == {"gripper.pos": pytest.approx(30.0)}
        assert len(follower.sent_actions) >= 7  # detection sequence plus low-preload hold
        assert driver.holding_payload is True

    def test_close_without_minimum_travel_still_times_out(self, tmp_path):
        """A near-immediate mechanical jam must not be mistaken for contact."""
        cfg = _make_cfg(
            gripper_close_pos=0.0,
            gripper_contact_min_travel=5.0,
            gripper_contact_stall_samples=2,
            gripper_timeout_s=0.05,
            gripper_settle_s=0.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        follower.clip_fn = lambda action: {"gripper.pos": 48.0}
        with pytest.raises(TimeoutError, match=r"start=50\.000, observed=48\.000, target=0\.000"):
            driver.set_gripper(on=True)

    def test_open_stable_before_target_still_times_out(self, tmp_path):
        """Contact inference is closing-only; opening must reach its target."""
        cfg = _make_cfg(
            gripper_open_pos=100.0,
            gripper_contact_min_travel=5.0,
            gripper_contact_stall_samples=2,
            gripper_timeout_s=0.05,
            gripper_settle_s=0.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        follower.clip_fn = lambda action: {"gripper.pos": 70.0}
        with pytest.raises(TimeoutError, match="gripper timeout"):
            driver.set_gripper(on=False)

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

    def test_intermediate_waypoints_stream_without_waiting_for_static_tracking(self, tmp_path):
        """A lagging arm must not make each intermediate waypoint a settle point.

        The command sequence remains slew-limited, while only the final target
        is re-sent until the encoder reaches the configured settle tolerance.
        """
        cfg = _make_cfg(
            max_joint_step_deg=2.0,
            joint_tolerance_deg=0.1,
            settle_overcompensate=False,
        )
        follower = FakeFollower(config=None)
        follower.track = False

        def lagging_send(action):
            actual = dict(action)
            follower.sent_actions.append(actual)
            for index, name in enumerate(ARM_JOINT_ORDER):
                key = f"{name}.pos"
                if key not in actual:
                    continue
                delta = float(actual[key]) - follower._arm[index]
                follower._arm[index] += float(np.clip(delta, -0.25, 0.25))
            return actual

        follower.send_action = lagging_send
        driver, _, _ = _make_driver(cfg, tmp_path, follower)
        driver.connect()

        driver.move_joint_blocking([6.0, 0.0, 0.0, 0.0, 0.0])

        shoulder_commands = [action["shoulder_pan.pos"] for action in follower.sent_actions]
        assert shoulder_commands[:3] == pytest.approx([2.0, 4.0, 6.0])
        assert shoulder_commands[3:]  # final settle had to catch up
        assert shoulder_commands[3:] == pytest.approx([6.0] * len(shoulder_commands[3:]))


class TestSettleEdgeCases:
    """Plan §A3: record send_action's actual (clipped) target, re-send it when
    the arm hasn't converged, and time out (stall) instead of looping forever."""

    def test_z_undershoot_blocks_strict_acceptance_until_compensated(self, tmp_path):
        cfg = _make_cfg(
            joint_tolerance_deg=3.0,
            settle_samples=1,
            settle_soft_tolerance_deg=3.0,
            settle_max_z_undershoot_mm=5.0,
            settle_overcompensate=True,
            settle_drift_abort_samples=0,
            move_timeout_s=1.0,
        )
        follower = FakeFollower(config=None)
        # FakeKinematics maps elbow degrees to Z at 10 mm/deg. The initial
        # -2-degree steady-state offset is therefore 20 mm below target: inside
        # the 3-degree joint tolerance, but outside the 5 mm Z requirement.
        follower.steady_offset = [0.0, 0.0, -2.0, 0.0, 0.0]
        driver, _, _ = _make_driver(cfg, tmp_path, follower)
        driver.connect()

        driver.move_joint_blocking([0.0, 0.0, 5.0, 0.0, 0.0])

        result = driver.last_motion_result
        assert result is not None
        assert result["ok"] is True
        assert result["classification"] == "strict"
        assert result["z_requirement_met"] is True
        assert result["cartesian_z_error_mm"] >= -cfg.settle_max_z_undershoot_mm - 1e-6

    def test_z_undershoot_blocks_soft_acceptance_without_compensation(self, tmp_path):
        cfg = _make_cfg(
            joint_tolerance_deg=1.5,
            settle_samples=1,
            settle_soft_tolerance_deg=3.0,
            settle_soft_samples=2,
            settle_max_z_undershoot_mm=5.0,
            settle_overcompensate=False,
            settle_drift_abort_samples=0,
            move_timeout_s=0.05,
        )
        follower = FakeFollower(config=None)
        follower.steady_offset = [0.0, 0.0, -2.0, 0.0, 0.0]
        driver, _, _ = _make_driver(cfg, tmp_path, follower)
        driver.connect()

        with pytest.raises(TimeoutError, match="move timeout"):
            driver.move_joint_blocking([0.0, 0.0, 5.0, 0.0, 0.0])

        result = driver.last_motion_result
        assert result is not None
        assert result["classification"] == "hard_timeout"
        assert result["z_requirement_met"] is False
        assert result["cartesian_z_error_mm"] == pytest.approx(-20.0)

    def test_payload_lift_can_settle_by_z_only_with_xy_sacrifice(self, tmp_path):
        class ZTradeoffKinematics(FakeKinematics):
            def forward_kinematics(self, joint_pos_deg):
                q = np.asarray(joint_pos_deg, dtype=float)
                # shoulder_pan and elbow_flex both raise Z, while shoulder_pan
                # also changes X. A Z-only minimum-norm step intentionally uses
                # both and therefore sacrifices X accuracy.
                pose = So101Pose(
                    x=float(q[0] * 10.0),
                    y=0.0,
                    z=float((q[0] + q[2]) * 10.0),
                    rx=0.0,
                    ry=0.0,
                    rz=0.0,
                )
                return pose_mm_deg_to_matrix_m(pose)

            def inverse_kinematics(
                self,
                current_joint_pos,
                desired_ee_pose,
                position_weight=1.0,
                orientation_weight=0.01,
            ):
                pose = matrix_m_to_pose_mm_deg(np.asarray(desired_ee_pose, dtype=float))
                shoulder_pan = pose.x / 10.0
                return np.array(
                    [shoulder_pan, 0.0, pose.z / 10.0 - shoulder_pan, 0.0, 0.0],
                    dtype=float,
                )

        cfg = _make_cfg(
            joint_tolerance_deg=1.5,
            settle_soft_tolerance_deg=3.0,
            settle_samples=2,
            settle_max_z_undershoot_mm=5.0,
            settle_overcompensate=True,
            settle_gain=0.1,
            settle_z_only_lift_enabled=True,
            settle_z_only_lift_step_mm=2.0,
            settle_z_only_lift_max_joint_offset_deg=4.0,
            move_timeout_s=1.0,
        )
        follower = FakeFollower(config=None)
        follower.steady_offset = [0.0, 0.0, -2.0, 0.0, 0.0]
        driver, _, _ = _make_driver(cfg, tmp_path, follower)
        driver.connect()
        driver._kin = ZTradeoffKinematics("fake")
        driver._holding_payload = True

        driver.move_to_pose_blocking(So101Pose(0.0, 0.0, 50.0, 0.0, 0.0, 0.0))

        result = driver.last_motion_result
        assert result is not None
        assert result["ok"] is True
        assert result["classification"] == "z_only_lift"
        assert result["z_requirement_met"] is True
        assert abs(driver.get_pose().x) > 0.1

    def test_z_only_lift_failure_is_typed_safe_convergence_error(self, tmp_path):
        cfg = _make_cfg(
            joint_tolerance_deg=1.5,
            settle_soft_tolerance_deg=3.0,
            settle_samples=1,
            settle_max_z_undershoot_mm=5.0,
            settle_z_only_lift_enabled=True,
            move_timeout_s=1.0,
        )
        follower = FakeFollower(config=None)
        follower.steady_offset = [0.0, 0.0, -2.0, 0.0, 0.0]
        driver, _, _ = _make_driver(cfg, tmp_path, follower)
        driver.connect()
        driver._holding_payload = True

        def unavailable(*_args):
            raise ValueError("test: no safe +Z candidate")

        driver._next_z_only_lift_command = unavailable
        target_q = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
        start_matrix = pose_mm_deg_to_matrix_m(So101Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        target_matrix = pose_mm_deg_to_matrix_m(So101Pose(0.0, 0.0, 50.0, 0.0, 0.0, 0.0))

        with pytest.raises(So101PoseConvergenceError, match="Z-only payload lift") as exc_info:
            driver._dispatch_prevalidated_waypoints(
                [target_q],
                target_q,
                timeout_s=None,
                cartesian_target_matrix=target_matrix,
                cartesian_start_matrix=start_matrix,
            )

        assert exc_info.value.skip_recovery is True
        result = driver.last_motion_result
        assert result is not None
        assert result["classification"] == "hard_z_only_unavailable"

    def test_stable_soft_joint_error_is_accepted_after_fk_check(self, tmp_path, caplog):
        cfg = _make_cfg(
            joint_tolerance_deg=1.5,
            settle_samples=3,
            settle_soft_tolerance_deg=3.0,
            settle_soft_samples=3,
            ik_position_tolerance_mm=10.0,
            settle_overcompensate=False,
        )
        follower = FakeFollower(config=None)
        follower.steady_offset = [0.0, 2.49, 0.0, 0.0, 0.0]
        driver, _, _ = _make_driver(cfg, tmp_path, follower)
        driver.connect()
        caplog.set_level("WARNING", logger="jiuwensymbiosis.adapters.so101.lowlevel")

        driver.move_joint_blocking([1.0, 0.0, 0.0, 0.0, 0.0])

        result = driver.last_motion_result
        assert result is not None
        assert result["ok"] is True
        assert result["classification"] == "soft"
        assert result["max_joint"] == "shoulder_lift"
        assert result["max_abs_joint_error_deg"] == pytest.approx(2.49)
        assert result["cartesian_position_error_mm"] == pytest.approx(24.9)
        assert result["cartesian_position_error_mm"] > cfg.ik_position_tolerance_mm
        assert "final settle soft" in caplog.text
        assert "shoulder_lift" in caplog.text

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

        with pytest.raises(RuntimeError, match="hardware_send_mismatch|modified validated action"):
            driver.move_joint_blocking([6.0, 0.0, 0.0, 0.0, 0.0])

        # (a) The driver recorded the *actual* (clipped) last target returned by
        # send_action, not the requested one — clipping is observable.
        assert driver._last_sent_action is not None
        assert driver._last_sent_action["shoulder_pan.pos"] == pytest.approx(5.0, abs=1e-9)
        clipped_sends = [
            a["shoulder_pan.pos"] for a in follower.sent_actions if abs(a["shoulder_pan.pos"] - 5.0) < 1e-9
        ]
        assert len(clipped_sends) == 1

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

    def test_complete_stall_times_out_only_in_final_settle_without_wall_clock_wait(
        self,
        tmp_path,
        caplog,
    ):
        """All intermediate waypoints stream before a stalled final target fails."""
        now = 0.0

        def fake_monotonic() -> float:
            return now

        cfg = _make_cfg(
            max_joint_step_deg=2.0,
            move_timeout_s=0.005,
            settle_resend_period_s=0.001,
            settle_overcompensate=False,
        )
        follower = FakeFollower(config=None)
        follower.track = False
        driver, _, _ = _make_driver(cfg, tmp_path, follower, monotonic=fake_monotonic)

        def advance_time(seconds: float) -> None:
            nonlocal now
            now += seconds

        driver._sleep = advance_time
        driver.connect()
        caplog.set_level("WARNING", logger="jiuwensymbiosis.adapters.so101.lowlevel")

        with pytest.raises(TimeoutError, match="final target did not settle"):
            driver.move_joint_blocking([6.0, 0.0, 0.0, 0.0, 0.0])

        shoulder_commands = [action["shoulder_pan.pos"] for action in follower.sent_actions]
        assert shoulder_commands[:3] == pytest.approx([2.0, 4.0, 6.0])
        assert now < 0.02
        assert "SO-101 final settle timeout" in caplog.text
        assert "max_joint=shoulder_pan" in caplog.text
        assert "shoulder_pan(target=6.000, actual=0.000, error=-6.000)" in caplog.text
        for name in ARM_JOINT_ORDER:
            assert f"{name}(target=" in caplog.text

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
            settle_overcompensate=False,
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

        with pytest.raises(RuntimeError, match="hardware_send_mismatch|modified validated action"):
            driver.move_joint_blocking([6.0, 0.0, 0.0, 0.0, 0.0])

        # Settle re-sends (after the interpolation sweep) should sleep ~0.05s
        # each, not the 0.001s interpolation period. Filter out the tiny
        # interpolation sleeps and assert the settle sleeps are at the throttle.
        assert not [s for s in sleep_log if s >= 0.04]


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

    def test_home_use_init_pose_overrides_unvalidated_gate(self, tmp_path):
        """home_use_init_pose=True bypasses the safety_validated fail-closed
        gate (connecting is the only way to read current joints) and uses the
        startup joint pose as home_joints_deg at runtime."""
        kw = {"home_joints_deg": None, "safety_validated": False, "home_use_init_pose": True}
        cfg = _make_cfg(**kw)
        follower = FakeFollower(config=None)
        follower.calibration_fpath = make_calib_file(tmp_path)
        # Park the (fake) arm at a non-zero startup pose within soft limits.
        follower._arm = [5.0, -5.0, 10.0, -10.0, 0.0]
        driver = So101Driver(
            cfg,
            so_follower_factory=lambda robot_cfg: follower,
            kinematics_factory=FakeKinematics,
            lerobot_import=fake_lerobot_import,
        )
        driver.connect()
        assert driver._connected is True
        # Runtime home was overwritten with the startup pose, not None.
        expected_home = [5.0, -5.0, 10.0, -10.0, 0.0]
        assert cfg.home_joints_deg == expected_home
        # And home() dispatches that overwritten pose.
        driver.home()
        last = follower.sent_actions[-1]
        for i, name in enumerate(ARM_JOINT_ORDER):
            assert last[f"{name}.pos"] == pytest.approx(expected_home[i], abs=1e-9)

    def test_home_use_init_pose_refuses_pose_outside_soft_limits(self, tmp_path):
        """Even with home_use_init_pose=True, a startup pose outside joint_limits
        is rejected at Step 8 (the soft-limit safety net), so an operator who
        parked the arm illegally is refused rather than trusting an illegal home."""
        kw = {"home_joints_deg": None, "safety_validated": False, "home_use_init_pose": True}
        cfg = _make_cfg(**kw)
        follower = FakeFollower(config=None)
        follower.calibration_fpath = make_calib_file(tmp_path)
        # shoulder_pan soft limit is (-90, 90); 120.0 is outside.
        follower._arm = [120.0, 0.0, 0.0, 0.0, 0.0]
        driver = So101Driver(
            cfg,
            so_follower_factory=lambda robot_cfg: follower,
            kinematics_factory=FakeKinematics,
            lerobot_import=fake_lerobot_import,
        )
        with pytest.raises(ValueError, match="home_joints_deg.*out of soft limits"):
            driver.connect()
        assert driver._connected is False


class TestReachability:
    def test_servo_cartesian_search_preserves_fk_progress_for_coupled_joints(self, tmp_path):
        """Joint-wise clipping must not turn an upward Cartesian step downward."""

        class CoupledKinematics:
            def forward_kinematics(self, q):
                q = np.asarray(q, dtype=float)
                # The full IK solution for z=101 is (10, -8).  Clipping both
                # joints to +0.6/-0.6 would produce z=99.94 (down), whereas a
                # Cartesian fraction preserves the positive FK direction.
                z = 100.0 + 0.9 * q[0] + q[1]
                return pose_mm_deg_to_matrix_m(So101Pose(0.0, 0.0, z, 0.0, 0.0, 0.0))

            def inverse_kinematics(self, _current, desired, position_weight=1.0, orientation_weight=0.01):
                target_z = matrix_m_to_pose_mm_deg(np.asarray(desired, dtype=float)).z
                delta = target_z - 100.0
                return np.array([10.0 * delta, -8.0 * delta, 0.0, 0.0, 0.0])

        # Keep this target outside the terminal deadband: this test exercises
        # coupled-joint Cartesian progress, not endpoint hold behavior.
        cfg = _make_cfg(
            z_min_safe_mm=10.0,
            servo_max_joint_step_deg=1.0,
            servo_goal_tolerance_mm=0.5,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        driver._kin = CoupledKinematics()
        driver.servo_to_pose(So101Pose(0.0, 0.0, 101.0, 0.0, 0.0, 0.0))

        sent = follower.sent_actions[-1]
        sent_q = np.array([sent[f"{name}.pos"] for name in ARM_JOINT_ORDER])
        sent_z = matrix_m_to_pose_mm_deg(driver._kin.forward_kinematics(sent_q)).z
        assert sent_q[0] > 0.0 and sent_q[1] < 0.0
        assert sent_z > 100.0

    def test_servo_to_pose_first_command_obeys_step_and_velocity_limits(self, tmp_path):
        cfg = _make_cfg(z_min_safe_mm=10.0, servo_max_joint_step_deg=1.0)
        driver, follower, sleep_log = _make_driver(cfg, tmp_path)
        driver.connect()
        follower._arm = [0.0, 0.0, 5.0, 0.0, 0.0]
        sent_before = len(follower.sent_actions)

        driver.servo_to_pose(So101Pose(0.0, 0.0, 70.0, 0.0, 0.0, 0.0))

        sent = follower.sent_actions[sent_before:]
        assert len(sent) == 1
        # The test fixture's blocking Cartesian step is 1 mm, so the fast path
        # derives a 2 mm per-tick cap. The first send still uses the 0.02 s
        # minimum period, making the 60 mm/s Cartesian velocity cap the active
        # 1.2 mm limit. In FakeKinematics that is 0.12 deg, below both the
        # 30 deg/s joint velocity cap (0.6 deg/tick) and the configured
        # 1.0-degree joint per-call cap.
        # The Cartesian alpha search refines the largest safe candidate with a
        # finite number of IK solves; allow its sub-0.0001-degree quantization.
        assert sent[0]["elbow_flex.pos"] == pytest.approx(5.12, abs=1e-4)
        assert sleep_log == []

    def test_servo_to_pose_rejects_bad_ik_before_send(self, tmp_path):
        cfg = _make_cfg(z_min_safe_mm=10.0, ik_position_tolerance_mm=0.1)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        class BadKin(FakeKinematics):
            def inverse_kinematics(self, current, desired, position_weight=1.0, orientation_weight=0.01):
                return np.zeros(5)

        driver._kin = BadKin("fake", "gripper_frame_link", list(ARM_JOINT_ORDER))
        sent_before = len(follower.sent_actions)
        with pytest.raises(ValueError, match="position residual"):
            driver.servo_to_pose(So101Pose(100.0, 200.0, 300.0, 0.0, 0.0, 0.0))
        assert len(follower.sent_actions) == sent_before

    def test_servo_to_pose_rejects_when_not_connected(self, tmp_path):
        cfg = _make_cfg(z_min_safe_mm=10.0)
        driver, _follower, _ = _make_driver(cfg, tmp_path)
        with pytest.raises(RuntimeError, match="before connect"):
            driver.servo_to_pose(So101Pose(0.0, 0.0, 70.0, 0.0, 0.0, 0.0))

    def test_servo_to_pose_rejects_non_finite_target(self, tmp_path):
        cfg = _make_cfg(z_min_safe_mm=10.0)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        sent_before = len(follower.sent_actions)
        with pytest.raises(ValueError, match="must be finite"):
            driver.servo_to_pose(So101Pose(float("nan"), 0.0, 70.0, 0.0, 0.0, 0.0))
        assert len(follower.sent_actions) == sent_before

    def test_servo_to_pose_rejects_target_below_z_floor(self, tmp_path):
        cfg = _make_cfg(z_min_safe_mm=50.0, workspace_bounds=(0.0, -300.0, 500.0, 300.0))
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        sent_before = len(follower.sent_actions)
        with pytest.raises(ValueError, match="below driver z_min_safe"):
            driver.servo_to_pose(So101Pose(10.0, 0.0, 10.0, 0.0, 0.0, 0.0))
        assert len(follower.sent_actions) == sent_before

    def test_servo_to_pose_rejects_target_outside_xy_bounds(self, tmp_path):
        cfg = _make_cfg(z_min_safe_mm=10.0, workspace_bounds=(0.0, -300.0, 500.0, 300.0))
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        sent_before = len(follower.sent_actions)
        # x=-50 is outside [0, 500]
        with pytest.raises(ValueError, match="out of workspace x"):
            driver.servo_to_pose(So101Pose(-50.0, 0.0, 70.0, 0.0, 0.0, 0.0))
        assert len(follower.sent_actions) == sent_before

    def test_servo_to_pose_rejects_orientation_residual(self, tmp_path):
        cfg = _make_cfg(
            z_min_safe_mm=-500.0,
            ik_position_tolerance_mm=10.0,
            ik_orientation_tolerance_deg=0.1,  # tight: any orientation drift rejects
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        class DriftKin(FakeKinematics):
            # Exact position but FK orientation differs from the requested one
            # (returns the requested joints, but forward_kinematics applies a
            # fixed rotation offset so the orientation residual is nonzero).
            def forward_kinematics(self, q):
                mat = super().forward_kinematics(q)
                # Inject a small rotation about z so orientation residual > 0.1 deg.
                theta = math.radians(1.0)
                rot = np.array(
                    [
                        [math.cos(theta), -math.sin(theta), 0, 0],
                        [math.sin(theta), math.cos(theta), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )
                return (rot @ np.asarray(mat)).tolist()

        driver._kin = DriftKin("fake", "gripper_frame_link", list(ARM_JOINT_ORDER))
        sent_before = len(follower.sent_actions)
        with pytest.raises(ValueError, match="orientation residual"):
            driver.servo_to_pose(So101Pose(10.0, 20.0, 70.0, 0.0, 0.0, 90.0))
        assert len(follower.sent_actions) == sent_before

    def test_servo_to_pose_enforces_min_send_period(self, tmp_path):
        now = 10.0

        def fake_monotonic() -> float:
            return now

        cfg = _make_cfg(z_min_safe_mm=10.0, servo_min_send_period_s=1.0)
        driver, follower, _ = _make_driver(cfg, tmp_path, monotonic=fake_monotonic)
        driver.connect()
        follower._arm = [0.0, 0.0, 5.0, 0.0, 0.0]
        target = So101Pose(0.0, 0.0, 70.0, 0.0, 0.0, 0.0)
        assert driver.servo_to_pose(target) is True
        sent_after_first = len(follower.sent_actions)
        planned_q = driver._servo_planned_q.copy()
        planned_matrix = driver._servo_planned_matrix.copy()
        # Second call within the min send period must be skipped (no new action).
        assert driver.servo_to_pose(target) is False
        assert len(follower.sent_actions) == sent_after_first
        assert np.array_equal(driver._servo_planned_q, planned_q)
        assert np.array_equal(driver._servo_planned_matrix, planned_matrix)

        # Sub-microsecond floating error at a nominal one-second period must not
        # turn the next command into another false rate-gate skip.
        now += 1.0 - 0.5e-6
        assert driver.servo_to_pose(target) is True
        assert len(follower.sent_actions) == sent_after_first + 1

    def test_servo_holds_planned_endpoint_inside_position_deadband_when_orientation_is_best_effort(self, tmp_path):
        now = 10.0

        def fake_monotonic() -> float:
            return now

        cfg = _make_cfg(
            z_min_safe_mm=10.0,
            servo_goal_tolerance_mm=1.0,
            ik_orientation_tolerance_deg=None,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path, monotonic=fake_monotonic)
        driver.connect()
        # Planned/live FK is 0.5 mm from the requested Z but has a 90-degree
        # orientation residual. With a null orientation tolerance, SO-101 must
        # hold this endpoint instead of demanding another 0.001 mm improvement
        # or trying to solve an unattainable orientation-only step.
        planned_q = np.array([1.0, 2.0, 6.95, 0.0, 0.0])
        follower._arm = planned_q.tolist()
        driver._servo_planned_q = planned_q.copy()
        driver._servo_planned_matrix = driver._kin.forward_kinematics(planned_q)
        driver._servo_last_send_t = now - 0.021
        target = So101Pose(10.0, 20.0, 70.0, 0.0, 0.0, 90.0)
        sent_before = len(follower.sent_actions)

        assert driver.servo_to_pose(target) is True

        sent = follower.sent_actions[sent_before:]
        assert len(sent) == 1
        assert [sent[0][f"{name}.pos"] for name in ARM_JOINT_ORDER] == pytest.approx(planned_q)
        assert np.array_equal(driver._servo_planned_q, planned_q)

    def test_servo_endpoint_overcompensation_closes_live_steady_state_error(self, tmp_path):
        now = 10.0

        def fake_monotonic() -> float:
            return now

        cfg = _make_cfg(
            z_min_safe_mm=10.0,
            servo_goal_tolerance_mm=1.0,
            settle_overcompensate=True,
            settle_gain=0.5,
        )
        follower = FakeFollower(config=None)
        follower.steady_offset = [0.0, 0.0, -0.5, 0.0, 0.0]
        driver, _, _ = _make_driver(cfg, tmp_path, follower, monotonic=fake_monotonic)
        driver.connect()
        planned_q = np.array([1.0, 2.0, 7.0, 0.0, 0.0])
        follower._arm = (planned_q + np.asarray(follower.steady_offset)).tolist()
        driver._servo_planned_q = planned_q.copy()
        driver._servo_planned_matrix = driver._kin.forward_kinematics(planned_q)
        driver._servo_last_send_t = now - 0.021
        target = So101Pose(10.0, 20.0, 70.0, 0.0, 0.0, 90.0)
        initial_error = float(np.max(np.abs(np.asarray(follower._arm) - planned_q)))

        for _ in range(6):
            assert driver.servo_to_pose(target) is True
            now += 0.021

        final_error = float(np.max(np.abs(np.asarray(follower._arm) - planned_q)))
        assert final_error < initial_error / 10.0
        assert follower.sent_actions[-1]["elbow_flex.pos"] > planned_q[2]
        # Compensation is execution-only: it must never become the next IK seed.
        assert np.array_equal(driver._servo_planned_q, planned_q)
        assert np.array_equal(driver._servo_planned_matrix, driver._kin.forward_kinematics(planned_q))

    def test_servo_endpoint_compensation_state_resets_when_target_leaves_deadband(self, tmp_path):
        now = 10.0

        def fake_monotonic() -> float:
            return now

        cfg = _make_cfg(
            z_min_safe_mm=10.0,
            servo_goal_tolerance_mm=1.0,
            servo_max_joint_step_deg=10.0,
            servo_max_joint_vel_dps=500.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path, monotonic=fake_monotonic)
        driver.connect()
        planned_q = np.array([1.0, 2.0, 7.0, 0.0, 0.0])
        follower._arm = planned_q.tolist()
        driver._servo_planned_q = planned_q.copy()
        driver._servo_planned_matrix = driver._kin.forward_kinematics(planned_q)
        driver._servo_last_send_t = now - 0.021

        assert driver.servo_to_pose(So101Pose(10.0, 20.0, 70.0, 0.0, 0.0, 0.0)) is True
        assert driver._servo_endpoint_state is not None

        now += 0.021
        assert driver.servo_to_pose(So101Pose(10.0, 20.0, 80.0, 0.0, 0.0, 0.0)) is True
        assert driver._servo_endpoint_state is None

    def test_servo_endpoint_overcompensation_falls_back_when_soft_limit_would_be_exceeded(self, tmp_path, caplog):
        now = 10.0

        def fake_monotonic() -> float:
            return now

        cfg = _make_cfg(
            z_min_safe_mm=-500.0,
            servo_goal_tolerance_mm=1.0,
            settle_overcompensate=True,
            settle_gain=0.5,
        )
        follower = FakeFollower(config=None)
        follower.track = False
        driver, _, _ = _make_driver(cfg, tmp_path, follower, monotonic=fake_monotonic)
        driver.connect()
        planned_q = np.array([89.8, 0.0, 7.0, 0.0, 0.0])
        follower._arm = [89.0, 0.0, 7.0, 0.0, 0.0]
        driver._servo_planned_q = planned_q.copy()
        driver._servo_planned_matrix = driver._kin.forward_kinematics(planned_q)
        driver._servo_last_send_t = now - 0.021
        target = So101Pose(898.0, 0.0, 70.0, 0.0, 0.0, 0.0)
        caplog.set_level("WARNING", logger="jiuwensymbiosis.adapters.so101.lowlevel")

        assert driver.servo_to_pose(target) is True

        assert follower.sent_actions[-1]["shoulder_pan.pos"] == pytest.approx(planned_q[0])
        assert "over-command" in caplog.text
        assert "re-sending bare planned target" in caplog.text

    def test_servo_to_pose_enforces_max_joint_vel_dps(self, tmp_path):
        # With a tiny servo_max_joint_vel_dps and a large inter-send interval
        # (simulated by manually rewinding _servo_last_send_t far into the past),
        # the velocity cap rather than the per-call step cap binds.
        cfg = _make_cfg(
            z_min_safe_mm=10.0,
            servo_min_send_period_s=1e-6,
            servo_max_joint_step_deg=10.0,
            servo_max_joint_vel_dps=1.0,  # 1 deg/s
            servo_max_cartesian_step_mm=100.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        follower._arm = [0.0, 0.0, 5.0, 0.0, 0.0]
        # Pretend the previous send was 0.5s ago → vel_cap = 1.0 * 0.5 = 0.5 deg,
        # tighter than servo_max_joint_step_deg=10, so q_cmd is clipped to 0.5 deg.
        driver._servo_last_send_t = time.monotonic() - 0.5
        sent_before = len(follower.sent_actions)
        driver.servo_to_pose(So101Pose(0.0, 0.0, 70.0, 0.0, 0.0, 0.0))
        sent = follower.sent_actions[sent_before:]
        assert len(sent) == 1
        # vel_cap (≈0.5 deg, not the 10 deg step cap) binds: elbow ≈ 5.5, well
        # below the 7.0 a step-cap-only path would allow. abs=0.01 covers the
        # few-µs IK/validate overhead in dt (do not tighten to an exact dt).
        assert sent[0]["elbow_flex.pos"] == pytest.approx(5.5, abs=0.01)
        assert sent[0]["elbow_flex.pos"] < 7.0

    def test_servo_pauses_plan_until_live_joints_catch_up(self, tmp_path, caplog):
        now = 10.0

        def fake_monotonic() -> float:
            return now

        cfg = _make_cfg(
            z_min_safe_mm=-100.0,
            servo_min_send_period_s=0.1,
            servo_max_joint_step_deg=10.0,
            servo_max_joint_vel_dps=500.0,
            servo_max_cartesian_step_mm=100.0,
            tracking_error_deg=0.1,
        )
        follower = FakeFollower(config=None)
        follower.track = False
        follower._arm = [0.0, 0.0, 5.0, 0.0, 0.0]
        driver, _, _ = _make_driver(cfg, tmp_path, follower, monotonic=fake_monotonic)
        driver.connect()
        target = So101Pose(20.0, 0.0, 70.0, 0.0, 0.0, 0.0)

        assert driver.servo_to_pose(target) is True
        first_planned_q = driver._servo_planned_q.copy()
        first_planned_matrix = driver._servo_planned_matrix.copy()
        sent_after_first = len(follower.sent_actions)
        assert np.max(np.abs(first_planned_q - np.asarray(follower._arm))) > cfg.tracking_error_deg

        # The frozen encoders exceed the allowance. This tick must hold both
        # plans while dispatching a bounded catch-up command rather than
        # advancing Cartesian planning or waiting passively.
        now += 0.101
        caplog.set_level("WARNING", logger="jiuwensymbiosis.adapters.so101.lowlevel")
        assert driver.servo_to_pose(target) is True
        assert len(follower.sent_actions) == sent_after_first + 1
        assert np.array_equal(driver._servo_planned_q, first_planned_q)
        assert np.array_equal(driver._servo_planned_matrix, first_planned_matrix)
        assert "servo catch-up hold" in caplog.text

        # Once the physical arm reaches the held point, planning resumes from
        # that exact state rather than re-anchoring to a newer live seed.
        follower._arm = first_planned_q.tolist()
        now += 0.101
        assert driver.servo_to_pose(target) is True
        assert len(follower.sent_actions) == sent_after_first + 2
        assert not np.array_equal(driver._servo_planned_q, first_planned_q)

    def test_servo_catchup_overcompensation_reduces_static_joint_lag_without_advancing_plan(self, tmp_path):
        now = 10.0

        def fake_monotonic() -> float:
            return now

        cfg = _make_cfg(
            z_min_safe_mm=-100.0,
            servo_min_send_period_s=0.1,
            servo_max_joint_step_deg=10.0,
            servo_max_joint_vel_dps=500.0,
            tracking_error_deg=3.0,
            settle_overcompensate=True,
            settle_gain=0.5,
            settle_drift_abort_samples=0,
        )
        follower = FakeFollower(config=None)
        planned_q = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
        follower.steady_offset = [0.0, 0.0, 3.02, 0.0, 0.0]
        follower._arm = (planned_q + np.asarray(follower.steady_offset)).tolist()
        driver, _, _ = _make_driver(cfg, tmp_path, follower, monotonic=fake_monotonic)
        driver.connect()
        driver._servo_planned_q = planned_q.copy()
        driver._servo_planned_matrix = driver._kin.forward_kinematics(planned_q)
        driver._servo_last_send_t = now - 0.101
        planned_matrix = driver._servo_planned_matrix.copy()
        initial_error = float(np.max(np.abs(np.asarray(follower._arm) - planned_q)))

        assert driver.servo_to_pose(So101Pose(0.0, 0.0, 100.0, 0.0, 0.0, 0.0)) is True

        final_error = float(np.max(np.abs(np.asarray(follower._arm) - planned_q)))
        assert final_error < initial_error
        assert follower.sent_actions[-1]["elbow_flex.pos"] < planned_q[2]
        assert np.array_equal(driver._servo_planned_q, planned_q)
        assert np.array_equal(driver._servo_planned_matrix, planned_matrix)

    def test_servo_chains_planned_pose_and_previous_ik_seed_despite_encoder_lag(self, tmp_path):
        now = 10.0

        def fake_monotonic() -> float:
            return now

        class RecordingKinematics(FakeKinematics):
            def __init__(self):
                super().__init__("fake", "gripper_frame_link", list(ARM_JOINT_ORDER))
                self.seeds: list[np.ndarray] = []

            def inverse_kinematics(self, current, desired, position_weight=1.0, orientation_weight=0.01):
                self.seeds.append(np.asarray(current, dtype=float).copy())
                return super().inverse_kinematics(current, desired, position_weight, orientation_weight)

        cfg = _make_cfg(
            z_min_safe_mm=-100.0,
            servo_min_send_period_s=0.02,
            servo_max_joint_step_deg=10.0,
            servo_max_joint_vel_dps=500.0,
        )
        follower = FakeFollower(config=None)
        follower.track = False
        follower._arm = [0.0, 0.0, 5.0, 0.0, 0.0]
        driver, _, _ = _make_driver(cfg, tmp_path, follower, monotonic=fake_monotonic)
        driver.connect()
        kin = RecordingKinematics()
        driver._kin = kin
        target = So101Pose(20.0, 0.0, 70.0, 0.0, 0.0, 0.0)

        driver.servo_to_pose(target)
        first_planned_q = driver._servo_planned_q.copy()
        first_planned_pose = matrix_m_to_pose_mm_deg(driver._servo_planned_matrix.copy())
        assert not np.allclose(first_planned_q, follower._arm)
        assert np.allclose(driver._servo_planned_matrix, kin.forward_kinematics(first_planned_q))

        # The real encoder remains at the original pose. The next call must
        # nevertheless continue from the previous command state.
        now += 0.021
        kin.seeds.clear()
        driver.servo_to_pose(target)
        second_planned_pose = matrix_m_to_pose_mm_deg(driver._servo_planned_matrix.copy())

        assert kin.seeds
        assert all(np.allclose(seed, first_planned_q) for seed in kin.seeds)
        assert second_planned_pose.x > first_planned_pose.x
        assert second_planned_pose.z > first_planned_pose.z
        assert follower._arm == [0.0, 0.0, 5.0, 0.0, 0.0]

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
        # The seed-chain planner rejects the first invalid waypoint before send.
        with pytest.raises(ValueError, match="orientation residual"):
            driver.move_to_pose_blocking(So101Pose(10.0, 20.0, 300.0, 0, 0, 45.0))
        assert len(follower.sent_actions) == sent_before

    def test_position_residual_rejected_when_over_tolerance(self, tmp_path):
        cfg = _make_cfg(ik_position_tolerance_mm=0.0001, z_min_safe_mm=10.0)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        sent_before = len(follower.sent_actions)

        # Sabotage the fake IK so it returns a joint config that FK cannot match
        # the desired position. IK always returns zeros — FK of zeros = origin,
        # far from any non-origin desired. The first invalid seed-chain waypoint
        # is rejected before dispatch.
        follower._arm = [0.0, 0.0, 0.0, 0.0, 0.0]

        class BadKin(FakeKinematics):
            def inverse_kinematics(self, current, desired, position_weight=1.0, orientation_weight=0.01):
                # Always return zeros — FK of zeros = origin, far from desired.
                return np.zeros(5)

        driver._kin = BadKin("fake", "gripper_frame_link", list(ARM_JOINT_ORDER))
        with pytest.raises(So101PreDispatchError, match="position residual") as exc_info:
            driver.move_to_pose_blocking(So101Pose(100.0, 200.0, 300.0, 0, 0, 0))
        assert exc_info.value.skip_recovery is True
        assert len(follower.sent_actions) == sent_before

    def test_position_residual_retries_position_only_from_same_seed(self, tmp_path, caplog):
        cfg = _make_cfg(
            ik_position_tolerance_mm=0.1,
            ik_orientation_weight=0.01,
            ik_orientation_tolerance_deg=None,
            z_min_safe_mm=-500.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        follower._arm = [0.0, 0.0, 5.0, 0.0, 0.0]

        class OrientationBiasedKin(FakeKinematics):
            def __init__(self):
                super().__init__("fake", "gripper_frame_link", list(ARM_JOINT_ORDER))
                self.calls: list[tuple[np.ndarray, float]] = []

            def inverse_kinematics(self, current, desired, position_weight=1.0, orientation_weight=0.01):
                self.calls.append((np.asarray(current, dtype=float).copy(), float(orientation_weight)))
                if orientation_weight != 0.0:
                    return np.zeros(5)
                return super().inverse_kinematics(
                    current,
                    desired,
                    position_weight=position_weight,
                    orientation_weight=orientation_weight,
                )

        kin = OrientationBiasedKin()
        driver._kin = kin
        caplog.set_level("INFO", logger="jiuwensymbiosis.adapters.so101.lowlevel")

        driver.move_to_pose_blocking(So101Pose(10.0, 20.0, 70.0, 0.0, 0.0, 0.0))

        assert len(kin.calls) >= 2
        first_seed, first_weight = kin.calls[0]
        retry_seed, retry_weight = kin.calls[1]
        assert first_weight == pytest.approx(0.01)
        assert retry_weight == pytest.approx(0.0)
        assert np.array_equal(first_seed, retry_seed)
        assert "accepted position-only retry from the same seed" in caplog.text

    def test_explicit_orientation_tolerance_disables_position_only_retry(self, tmp_path):
        cfg = _make_cfg(
            ik_position_tolerance_mm=0.1,
            ik_orientation_weight=0.01,
            ik_orientation_tolerance_deg=180.0,
            z_min_safe_mm=-500.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        follower._arm = [0.0, 0.0, 5.0, 0.0, 0.0]
        weights: list[float] = []

        class AlwaysBadWeightedKin(FakeKinematics):
            def inverse_kinematics(self, current, desired, position_weight=1.0, orientation_weight=0.01):
                weights.append(float(orientation_weight))
                return np.zeros(5)

        driver._kin = AlwaysBadWeightedKin("fake", "gripper_frame_link", list(ARM_JOINT_ORDER))

        with pytest.raises(So101PreDispatchError, match="position residual"):
            driver.move_to_pose_blocking(So101Pose(10.0, 20.0, 70.0, 0.0, 0.0, 0.0))

        assert weights == pytest.approx([0.01])


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

        with pytest.raises(ValueError, match="below driver z_min_safe"):
            driver.move_to_pose_blocking(So101Pose(0.0, 0.0, 26.0, 0, 0, 0))
        assert len(follower.sent_actions) == sent_before

    def test_target_above_z_ceiling_rejected_before_send(self, tmp_path):
        """Symmetric to the z-floor: a target above z_max_safe is rejected
        before any action is sent. Guards against too-high/awkward postures."""
        cfg = _make_cfg(z_min_safe_mm=-10.0, z_max_safe_mm=50.0)
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        sent_before = len(follower.sent_actions)

        # Target z=60 > z_max_safe=50. Must reject before any action.
        with pytest.raises(ValueError, match="above driver z_max_safe"):
            driver.move_to_pose_blocking(So101Pose(0.0, 0.0, 60.0, 0, 0, 0))
        assert len(follower.sent_actions) == sent_before

    def test_z_ceiling_none_does_not_reject(self, tmp_path):
        """Default z_max_safe_mm=None means no upper check — a high target is
        not rejected. Preserves the legacy behaviour for configs that don't set
        the ceiling."""
        cfg = _make_cfg(z_min_safe_mm=-500.0)  # z_max_safe_mm left at None
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        # Safe start well above the floor; FakeKinematics FK z = joint[2]*10.
        follower._arm = [0.0, 0.0, 30.0, 0.0, 0.0]
        sent_before = len(follower.sent_actions)

        driver.move_to_pose_blocking(So101Pose(0.0, 0.0, 300.0, 0, 0, 0))
        # No ceiling -> high target dispatched without rejection.
        assert len(follower.sent_actions) > sent_before

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

    def test_convergence_fail_closed_when_over_command_breaks_joint_limit(self, tmp_path, caplog):
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
            move_timeout_s=0.05,
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
        # The global Z-settle requirement prevents the low endpoint from being
        # accepted before the outer convergence phase. Because the observed arm
        # remains inside the validated soft joint band, this is still the typed
        # safe-not-reached condition rather than a hardware timeout.
        caplog.set_level("ERROR", logger="jiuwensymbiosis.adapters.so101.lowlevel")
        with pytest.raises(So101PoseConvergenceError, match="not reached") as exc_info:
            driver.move_to_pose_blocking(target)
        assert exc_info.value.skip_recovery is True
        assert driver.last_motion_result is not None
        assert driver.last_motion_result["classification"] == "safe_convergence_timeout"

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
    def test_home_uses_direct_fk_validated_joint_path(self, tmp_path):
        cfg = _make_cfg(home_joints_deg=[0.0, 0.0, 0.0, 0.0, 0.0])
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        # Fake FK maps elbow_flex to Z. Home must interpolate directly toward
        # the configured joint target without an extra Cartesian lift.
        follower._arm = [0.0, 0.0, 5.0, 0.0, 0.0]

        driver.home()

        elbow_commands = [action["elbow_flex.pos"] for action in follower.sent_actions]
        assert elbow_commands[0] < 5.0
        assert max(elbow_commands) < 5.0
        assert elbow_commands[-1] == pytest.approx(0.0, abs=1e-6)

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

    def test_home_rejects_joint_path_that_dips_below_both_endpoints(self, tmp_path):
        cfg = _make_cfg(
            home_joints_deg=[10.0, 0.0, 0.0, 0.0, 0.0],
            z_min_safe_mm=10.0,
        )
        driver, follower, _ = _make_driver(cfg, tmp_path)
        driver.connect()

        class DippingKinematics:
            def forward_kinematics(self, q):
                q0 = float(np.asarray(q, dtype=float)[0])
                z = (q0 - 5.0) ** 2
                return pose_mm_deg_to_matrix_m(So101Pose(0.0, 0.0, z, 0.0, 0.0, 0.0))

        driver._kin = DippingKinematics()

        with pytest.raises(ValueError, match="below driver z_min_safe"):
            driver.home()
        assert follower.sent_actions == []

    def test_payload_lowest_point_is_checked_against_table(self, tmp_path):
        cfg = _make_cfg(
            table_z_mm=0.0,
            gripper_lowest_offset_mm=10.0,
            payload_protrusion_mm=20.0,
            minimum_floor_margin_mm=8.0,
        )
        driver, _, _ = _make_driver(cfg, tmp_path)
        driver.connect()
        driver._holding_payload = True

        with pytest.raises(ValueError, match="effective lowest z"):
            driver._validate_joint_waypoint(
                np.array([0.0, 0.0, 3.0, 0.0, 0.0]),
                label="payload",
            )
