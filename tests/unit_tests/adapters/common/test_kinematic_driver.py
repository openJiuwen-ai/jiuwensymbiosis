# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.kinematic_driver."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jiuwensymbiosis.adapters._common.kinematic_driver import KinematicArmDriver, KinematicSpec

from ._kinematic_fakes import FakeJointTransport, GantryBackend, advancing_clock

_NAMES = ("jx", "jy", "jz")
_LIMITS = {"jx": (-500.0, 500.0), "jy": (-500.0, 500.0), "jz": (0.0, 500.0)}


def _spec(**over) -> KinematicSpec:
    base = {
        "arm_joint_names": _NAMES,
        "home_joints": (0.0, 0.0, 150.0),
        "joint_limits": _LIMITS,
        "max_joint_step_deg": 50.0,
        "max_cartesian_step_mm": 50.0,
        "max_ik_jump_deg": 1000.0,  # gantry joints are mm; keep guard above the step
        "joint_tolerance_deg": 0.5,
        "settle_samples": 2,
        "move_timeout_s": 100.0,
        "z_min_safe_mm": 0.0,
    }
    base.update(over)
    return KinematicSpec(**base)


def _driver(transport, spec=None, *, sleep=None, clock=None) -> KinematicArmDriver:
    return KinematicArmDriver(
        transport,
        GantryBackend(),
        spec or _spec(),
        sleep=sleep or (lambda _s: None),
        clock=clock or (lambda: 0.0),
    )


class TestSpecValidation:
    def test_home_length_mismatch(self):
        with pytest.raises(ValueError):
            KinematicSpec(arm_joint_names=_NAMES, home_joints=(0.0, 0.0))

    def test_settle_samples_floor(self):
        with pytest.raises(ValueError):
            _spec(settle_samples=0)

    def test_negative_resend_period_rejected(self):
        with pytest.raises(ValueError):
            _spec(settle_resend_period_s=-1.0)

    def test_negative_overcompensate_gain_rejected(self):
        with pytest.raises(ValueError):
            _spec(settle_overcompensate_gain=-0.1)

    def test_non_positive_effector_tolerance_rejected(self):
        with pytest.raises(ValueError):
            _spec(effector_tolerance=0.0, effector_timeout_s=1.0)


class TestSafetyGate:
    def test_refuses_connect_when_required_and_unvalidated(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t, _spec(requires_safety_validation=True, safety_validated=False))
        with pytest.raises(RuntimeError, match="not safety-validated"):
            d.connect()
        assert t.opened is False  # refused BEFORE opening the transport

    def test_connects_when_required_and_validated(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t, _spec(requires_safety_validation=True, safety_validated=True))
        d.connect()
        assert t.opened is True

    def test_no_gate_when_not_required(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t, _spec(safety_validated=False))  # requires_safety_validation defaults False
        d.connect()
        assert t.opened is True


class TestConnect:
    def test_connect_success(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t)
        d.connect()
        assert t.opened is True
        assert d.get_angles() == [0.0, 0.0, 150.0]

    def test_connect_is_atomic_on_precheck_failure(self):
        t = FakeJointTransport([0.0, 0.0, 150.0], fail_precheck=True)
        d = _driver(t)
        with pytest.raises(RuntimeError, match="precheck"):
            d.connect()
        assert t.close_count >= 1
        with pytest.raises(RuntimeError, match="not connected"):
            d.get_angles()

    def test_disconnect_idempotent(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t)
        d.connect()
        d.disconnect()
        d.disconnect()
        d.close()

    def test_guard_before_connect(self):
        d = _driver(FakeJointTransport([0.0, 0.0, 150.0]))
        with pytest.raises(RuntimeError, match="not connected"):
            d.get_pose()


class TestReads:
    def test_home_pose_is_fk_of_home(self):
        d = _driver(FakeJointTransport([0.0, 0.0, 150.0]))
        d.connect()
        assert d.home_pose.z == pytest.approx(150.0)

    def test_get_pose_after_connect(self):
        d = _driver(FakeJointTransport([10.0, 20.0, 300.0]))
        d.connect()
        p = d.get_pose()
        assert (p.x, p.y, p.z) == pytest.approx((10.0, 20.0, 300.0))

    def test_safety_floors(self):
        d = _driver(FakeJointTransport([0.0, 0.0, 150.0]), _spec(z_min_safe_mm=30.0, tool_offset_mm=5.0))
        assert d.z_min_safe == pytest.approx(30.0)
        assert d.flange_z_min_safe == pytest.approx(35.0)
        assert d.tool_offset_mm == pytest.approx(5.0)


class TestMotion:
    def test_move_joint_reaches_target(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t)
        d.connect()
        d.move_joint_blocking([10.0, 20.0, 30.0])
        assert d.get_angles() == pytest.approx([10.0, 20.0, 30.0])
        assert t.sent_arm

    def test_move_to_pose_reaches_target(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t)
        d.connect()
        d.move_to_pose_blocking(SimpleNamespace(x=100.0, y=0.0, z=100.0, rx=0.0, ry=0.0, rz=0.0))
        assert d.get_angles() == pytest.approx([100.0, 0.0, 100.0])

    def test_out_of_limit_rejected_before_send(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t)
        d.connect()
        with pytest.raises(ValueError):
            d.move_joint_blocking([0.0, 0.0, 999.0])
        assert t.sent_arm == []

    def test_below_floor_rejected_before_send(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t, _spec(z_min_safe_mm=50.0))
        d.connect()
        with pytest.raises(ValueError, match="floor"):
            d.move_to_pose_blocking(SimpleNamespace(x=0.0, y=0.0, z=10.0, rx=0.0, ry=0.0, rz=0.0))
        assert t.sent_arm == []

    def test_stall_times_out(self):
        t = FakeJointTransport([0.0, 0.0, 150.0], follows=False)
        d = _driver(t, _spec(move_timeout_s=5.0), clock=advancing_clock(dt=1.0))
        d.connect()
        with pytest.raises(TimeoutError):
            d.move_joint_blocking([10.0, 20.0, 30.0])


class TestSettleOvercompensate:
    """Steady-state PD droop: servo parks at command + offset (inert I term)."""

    def test_reaches_target_under_steady_offset(self):
        # jz droops +2 after each send; gain=1 over-commands to park AT target.
        t = FakeJointTransport([0.0, 0.0, 150.0], steady_offset=[0.0, 0.0, 2.0])
        spec = _spec(settle_samples=1, joint_tolerance_deg=0.5, settle_overcompensate_gain=1.0)
        d = _driver(t, spec)
        d.connect()
        d.move_joint_blocking([0.0, 0.0, 100.0])
        assert abs(d.get_angles()[2] - 100.0) <= 0.5

    def test_disabled_leaves_steady_residual_and_times_out(self):
        # gain=0: bare resend cannot close the droop -> never settles within tol.
        t = FakeJointTransport([0.0, 0.0, 150.0], steady_offset=[0.0, 0.0, 2.0])
        spec = _spec(settle_samples=1, joint_tolerance_deg=0.5, settle_overcompensate_gain=0.0, move_timeout_s=5.0)
        d = _driver(t, spec, clock=advancing_clock(dt=1.0))
        d.connect()
        with pytest.raises(TimeoutError):
            d.move_joint_blocking([0.0, 0.0, 100.0])

    def test_fails_closed_when_overcommand_breaks_limit(self):
        # jz limit (0,100); target 98, droop -3 -> desired over-command 101 > limit,
        # so it must fall back to the bare target and never send above the limit.
        limits = {"jx": (-500.0, 500.0), "jy": (-500.0, 500.0), "jz": (0.0, 100.0)}
        t = FakeJointTransport([0.0, 0.0, 50.0], steady_offset=[0.0, 0.0, -3.0])
        spec = _spec(
            joint_limits=limits,
            settle_samples=1,
            joint_tolerance_deg=0.5,
            settle_overcompensate_gain=1.0,
            move_timeout_s=5.0,
        )
        d = _driver(t, spec, clock=advancing_clock(dt=1.0))
        d.connect()
        with pytest.raises(TimeoutError):
            d.move_joint_blocking([0.0, 0.0, 98.0])
        assert all(cmd[2] <= 100.0 + 1e-9 for cmd in t.sent_arm), t.sent_arm

    def test_gain_does_not_leak_into_interp_sweep(self):
        # gain=1 but perfect follow: the sweep must be plain stepped interpolation;
        # an overcompensation spike would exceed max_joint_step_deg between sends.
        t = FakeJointTransport([0.0, 0.0, 150.0])
        spec = _spec(max_joint_step_deg=20.0, settle_samples=1, settle_overcompensate_gain=1.0)
        d = _driver(t, spec)
        d.connect()
        d.move_joint_blocking([120.0, 0.0, 150.0])  # jx 0->120 needs 6 steps of 20
        for a, b in zip(t.sent_arm, t.sent_arm[1:]):
            step = max(abs(x - y) for x, y in zip(a, b))
            assert step <= 20.0 + 1e-9, (a, b)


class TestSettleDriftAbort:
    def test_aborts_when_error_grows(self):
        # Gravity-loaded servo driven the wrong way: jz creeps +5 away each send.
        def drift(current, _requested):
            return [current[0], current[1], current[2] + 5.0]

        t = FakeJointTransport([0.0, 0.0, 150.0], drift_fn=drift)
        spec = _spec(settle_samples=1, joint_tolerance_deg=0.5, settle_drift_abort_samples=3)
        d = _driver(t, spec)
        d.connect()
        with pytest.raises(RuntimeError, match="settle drift"):
            d.move_joint_blocking([0.0, 0.0, 150.0])
        # The driver only ever commanded the bare target; drift is the servo's.
        assert all(cmd[2] == pytest.approx(150.0) for cmd in t.sent_arm), t.sent_arm

    def test_drift_abort_fires_even_with_overcompensation(self):
        def drift(current, _requested):
            return [current[0], current[1], current[2] + 5.0]

        t = FakeJointTransport([0.0, 0.0, 150.0], drift_fn=drift)
        spec = _spec(
            settle_samples=1,
            joint_tolerance_deg=0.5,
            settle_drift_abort_samples=3,
            settle_overcompensate_gain=1.0,
        )
        d = _driver(t, spec)
        d.connect()
        with pytest.raises(RuntimeError, match="settle drift"):
            d.move_joint_blocking([0.0, 0.0, 150.0])


class TestSettleThrottle:
    def test_resend_period_throttles_settle_rate(self):
        recorded: list[float] = []
        t = FakeJointTransport([0.0, 0.0, 150.0], follows=False)  # stuck -> resends until timeout
        spec = _spec(
            trajectory_hz=1000.0,  # interp period 0.001s
            settle_resend_period_s=0.05,
            settle_drift_abort_samples=0,  # stable error -> don't abort, hit timeout
            move_timeout_s=0.3,
            joint_tolerance_deg=0.5,
        )
        d = _driver(t, spec, sleep=recorded.append, clock=advancing_clock(dt=0.1))
        d.connect()
        with pytest.raises(TimeoutError):
            d.move_joint_blocking([0.0, 0.0, 200.0])
        assert any(abs(s - 0.05) < 1e-9 for s in recorded), recorded  # settle throttled at 0.05, not 0.001


class TestEffector:
    def test_gripper_close_engages_and_leaves_arm_untouched(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        recorded: list[float] = []
        d = _driver(t, sleep=recorded.append)
        d.connect()
        d.set_gripper(True)
        assert t.sent_effector[-1] == pytest.approx(0.0)  # effector_engaged_pos
        assert d.gripper_state is True
        assert t.sent_arm == []  # effector channel never touches arm joints
        assert 0.4 in recorded  # effector_settle_s slept

    def test_gripper_open_releases(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t)
        d.connect()
        d.set_gripper(False)
        assert t.sent_effector[-1] == pytest.approx(100.0)  # effector_released_pos
        assert d.gripper_state is False

    def test_suction_shares_the_same_two_state_channel(self):
        # A suction cup reuses the effector channel: engaged=on → engaged_pos.
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t)
        d.connect()
        d.set_suction(True)
        assert t.sent_effector[-1] == pytest.approx(0.0)
        assert d.suction_state is True
        assert t.sent_arm == []
        d.set_suction(False)
        assert t.sent_effector[-1] == pytest.approx(100.0)
        assert d.suction_state is False


class TestEffectorConvergence:
    """Closed-loop effector: poll real feedback when an SDK clips per-send deltas."""

    @staticmethod
    def _clip_toward(step: float):
        return lambda req, cur: cur + max(-step, min(step, req - cur))

    def test_converges_despite_per_send_clip(self):
        # SDK moves the effector at most 5/step; a single send can't go 100 -> 0.
        t = FakeJointTransport([0.0, 0.0, 150.0], effector=100.0, effector_clip_fn=self._clip_toward(5.0))
        spec = _spec(effector_tolerance=2.0, effector_timeout_s=100.0, settle_samples=1, effector_settle_s=0.0)
        d = _driver(t, spec)
        d.connect()
        d.set_gripper(True)  # -> effector_engaged_pos 0.0
        assert abs(t.read_effector() - 0.0) <= 2.0
        assert len(t.sent_effector) > 1  # required multiple clipped resends

    def test_stall_times_out(self):
        t = FakeJointTransport([0.0, 0.0, 150.0], effector=100.0, effector_clip_fn=lambda req, cur: cur)
        spec = _spec(effector_tolerance=2.0, effector_timeout_s=5.0, settle_samples=1)
        d = _driver(t, spec, clock=advancing_clock(dt=1.0))
        d.connect()
        with pytest.raises(TimeoutError, match="effector"):
            d.set_gripper(True)

    def test_throttled_by_resend_period(self):
        recorded: list[float] = []
        t = FakeJointTransport([0.0, 0.0, 150.0], effector=100.0, effector_clip_fn=lambda req, cur: cur)
        spec = _spec(
            effector_tolerance=2.0,
            effector_timeout_s=0.3,
            settle_resend_period_s=0.05,
            trajectory_hz=1000.0,
            settle_samples=1,
        )
        d = _driver(t, spec, sleep=recorded.append, clock=advancing_clock(dt=0.1))
        d.connect()
        with pytest.raises(TimeoutError):
            d.set_gripper(True)
        assert any(abs(s - 0.05) < 1e-9 for s in recorded), recorded

    def test_send_once_when_not_configured(self):
        # No tolerance/timeout -> legacy send-once + dwell, single send.
        t = FakeJointTransport([0.0, 0.0, 150.0], effector=100.0, effector_clip_fn=lambda req, cur: cur)
        d = _driver(t)  # default spec: effector_tolerance is None
        d.connect()
        d.set_gripper(True)
        assert t.sent_effector == [0.0]  # exactly one send, no polling


class TestServo:
    # Bounded Cartesian servo: one tick dispatches the largest path fraction that
    # stays within the joint- and Cartesian-velocity caps and reduces error.
    _FAST = {"servo_max_joint_step_deg": 1.0e6, "servo_max_joint_vel_dps": 1.0e9}
    _TARGET = SimpleNamespace(x=40.0, y=0.0, z=120.0, rx=0.0, ry=0.0, rz=0.0)

    def test_servo_one_tick_makes_bounded_progress(self):
        import math

        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t)  # default caps -> velocity-limited fraction
        d.connect()
        d.servo_to_pose(self._TARGET)
        assert len(t.sent_arm) == 1  # single non-blocking step, no arrival loop
        sent = t.sent_arm[0]
        assert sent != pytest.approx([40.0, 0.0, 120.0])  # bounded: not the full target
        # GantryBackend joints ARE xyz(mm), so FK error = distance in joint space.
        assert math.dist(sent, [40.0, 0.0, 120.0]) < math.dist([0.0, 0.0, 150.0], [40.0, 0.0, 120.0])

    def test_servo_reaches_target_over_ticks(self):
        import math

        from jiuwensymbiosis.adapters._common.kinematics import CartesianServoError

        t = FakeJointTransport([0.0, 0.0, 150.0])
        # Generous joint cap + large inter-tick dt -> the Cartesian cap covers the
        # whole remaining move within a couple of ticks (the first tick uses
        # min_period, so convergence takes more than one call).
        d = _driver(t, _spec(**self._FAST), clock=advancing_clock(dt=10.0))
        d.connect()
        for _ in range(5):
            try:
                d.servo_to_pose(self._TARGET)
            except CartesianServoError:
                break  # no further Cartesian progress -> arrived (real loop stops here)
        assert math.dist(t.sent_arm[-1], [40.0, 0.0, 120.0]) < 1e-6

    def test_servo_rate_skips_call_within_min_period(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t, _spec(**self._FAST), clock=lambda: 1.0)  # frozen clock
        d.connect()
        d.servo_to_pose(self._TARGET)
        d.servo_to_pose(SimpleNamespace(x=41.0, y=0.0, z=120.0, rx=0.0, ry=0.0, rz=0.0))
        assert len(t.sent_arm) == 1  # 2nd call at the same instant is skipped (non-blocking)

    def test_servo_sends_again_after_min_period(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t, _spec(**self._FAST), clock=advancing_clock(dt=0.03))  # 0.03 > min_period 0.02
        d.connect()
        d.servo_to_pose(self._TARGET)
        d.servo_to_pose(self._TARGET)
        assert len(t.sent_arm) == 2  # enough elapsed -> second tick dispatches

    def test_servo_rejects_below_floor_before_send(self):
        from jiuwensymbiosis.adapters._common.kinematics import CartesianServoError

        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t, _spec(z_min_safe_mm=50.0))
        d.connect()
        with pytest.raises(CartesianServoError, match="cartesian_bounds_rejected"):
            d.servo_to_pose(SimpleNamespace(x=0.0, y=0.0, z=10.0, rx=0.0, ry=0.0, rz=0.0))
        assert t.sent_arm == []


class TestVision:
    def test_forwards_camera_when_transport_has_one(self):
        frames = ("rgb", "depth")
        d = _driver(FakeJointTransport([0.0, 0.0, 150.0], frames=frames))
        d.connect()
        assert d.grab_frames() == frames

    def test_vision_surface_absent_is_none(self):
        d = _driver(FakeJointTransport([0.0, 0.0, 150.0]))
        d.connect()
        assert d.grab_frames() is None
        assert d.intrinsics is None
        assert d.tf_flange_cam is None
        assert d.calibration is None


class TestNewSafetyAndHome:
    def test_move_rejected_above_z_ceiling(self):
        t = FakeJointTransport([0.0, 0.0, 150.0])
        d = _driver(t, _spec(z_max_safe_mm=200.0))
        d.connect()
        with pytest.raises(ValueError, match="ceiling"):
            d.move_to_pose_blocking(SimpleNamespace(x=0.0, y=0.0, z=250.0, rx=0.0, ry=0.0, rz=0.0))

    def test_home_from_startup_captures_live_pose(self):
        t = FakeJointTransport([10.0, 20.0, 130.0])
        d = _driver(t, _spec(home_joints=(), home_from_startup=True))
        d.connect()
        hp = d.home_pose  # FK of the live startup joints (GantryBackend: joints ARE xyz mm)
        assert (hp.x, hp.y, hp.z) == pytest.approx([10.0, 20.0, 130.0])

    def test_recovery_home_returns_to_home(self):
        t = FakeJointTransport([50.0, 50.0, 200.0])
        d = _driver(t, _spec(home_joints=(0.0, 0.0, 150.0), settle_samples=1))
        d.connect()
        d.recovery_home()
        assert t.sent_arm[-1] == pytest.approx([0.0, 0.0, 150.0])


class TestContactGripper:
    def test_contact_inferred_on_mid_close_stall(self):
        # A fixture blocks the closing gripper at 30 (100 -> 0 target); the stall
        # is read as object contact and a hold preload is applied.
        def clip(pos: float, _cur: float) -> float:
            return max(pos, 30.0)  # cannot close past 30

        t = FakeJointTransport([0.0, 0.0, 150.0], effector=100.0, effector_clip_fn=clip)
        spec = _spec(
            effector_engaged_pos=0.0,
            effector_released_pos=100.0,
            effector_tolerance=2.0,
            effector_timeout_s=5.0,
            gripper_contact_min_travel=5.0,
            gripper_contact_stall_samples=3,
            gripper_contact_stall_tolerance=0.5,
            gripper_contact_hold_offset=1.0,
            settle_samples=1,
        )
        d = _driver(t, spec)
        d.connect()
        d.set_gripper(True)  # close -> stalls at 30 -> contact
        res = d.last_gripper_result
        assert res is not None
        assert res["state"] == "contact"
        assert res["contact_position"] == pytest.approx(30.0)
        assert res["hold_target"] == pytest.approx(29.0)  # contact + (-1)*min(hold_offset=1, remaining=30)

    def test_full_close_without_obstacle_reports_closed(self):
        t = FakeJointTransport([0.0, 0.0, 150.0], effector=100.0)
        spec = _spec(
            effector_engaged_pos=0.0,
            effector_released_pos=100.0,
            effector_tolerance=2.0,
            effector_timeout_s=5.0,
            gripper_contact_min_travel=5.0,
            settle_samples=1,
        )
        d = _driver(t, spec)
        d.connect()
        d.set_gripper(True)  # no obstacle -> reaches 0
        assert d.last_gripper_result["state"] == "closed"
