# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SO-101 + real LeRobot 0.7.0 RobotKinematics FK/IK smoke (needs the so101 extra).

These tests are SKIPPED when LeRobot is not installed (the ``so101`` extra is
optional). When installed, they verify the packaged URDF loads, the 5 arm
joints map to the configured ``gripper_frame_link`` control frame, and that
FK/IK round-trip converges for a reachable target — the no-hardware smoke the
plan §A2 promises. No serial port / hardware is touched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("lerobot")

from lerobot.model.kinematics import RobotKinematics  # noqa: E402

from jiuwensymbiosis.adapters.so101.geometry import (  # noqa: E402
    So101Pose,
    matrix_m_to_pose_mm_deg,
    pose_mm_deg_to_matrix_m,
    position_error_mm,
)
from jiuwensymbiosis.adapters.so101.lowlevel import ARM_JOINT_ORDER, So101Driver  # noqa: E402
from jiuwensymbiosis.agent.fast.realtime.servo import _slew  # noqa: E402

URDF = "jiuwensymbiosis/adapters/so101/description/so101_new_calib.urdf"


@pytest.fixture(scope="function")
def kin() -> RobotKinematics:
    return RobotKinematics(
        URDF,
        target_frame_name="gripper_frame_link",
        joint_names=list(ARM_JOINT_ORDER),
    )


def _make_planner_driver(kin: RobotKinematics, **cfg_overrides) -> So101Driver:
    """Build a So101Driver with a real RobotKinematics injected, NOT connected.

    The planner (_plan_cartesian_waypoints) only uses self._kin / self._cfg and
    the _check_* helpers — no serial bus — so it can be exercised without
    LeRobot's SOFollower hardware.
    """
    from jiuwensymbiosis.adapters.so101.config import So101Config

    base: dict[str, Any] = {
        "port": "/dev/fake",
        "home_joints_deg": [0.0, 0.0, 0.0, 0.0, 0.0],
        "joint_limits": {
            "shoulder_pan": (-90.0, 90.0),
            "shoulder_lift": (-90.0, 90.0),
            "elbow_flex": (-120.0, 120.0),
            "wrist_flex": (-120.0, 120.0),
            "wrist_roll": (-180.0, 180.0),
        },
        "z_min_safe_mm": -500.0,  # disable the floor for the planner-only test
        "max_joint_step_deg": 2.0,
    }
    base.update(cfg_overrides)
    cfg = So101Config(**base)
    driver = So101Driver(cfg)
    driver._kin = kin  # inject real kinematics without connect()
    return driver


class TestUrdfLoads:
    def test_kinematics_builds_with_gripper_frame(self, kin):
        # Building at all means the URDF + gripper_frame_link resolve.
        assert kin is not None

    def test_fk_returns_4x4(self, kin):
        q = np.zeros(5)
        fk = np.asarray(kin.forward_kinematics(q), dtype=float)
        assert fk.shape == (4, 4)
        # All-zero joints should give a finite, reachable pose near the arm's
        # stowed configuration (no NaN/Inf).
        assert np.all(np.isfinite(fk))

    def test_fk_changes_with_joint_input(self, kin):
        q0 = np.zeros(5)
        q1 = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        fk0 = np.asarray(kin.forward_kinematics(q0), dtype=float)
        fk1 = np.asarray(kin.forward_kinematics(q1), dtype=float)
        # Moving shoulder_pan must change the translation or rotation.
        assert not np.allclose(fk0, fk1)


class TestFkIkRoundTrip:
    def test_reachable_pose_round_trips(self, kin):
        # Pick a reachable target by FK from a known joint config, then IK back
        # to it and verify the recovered joints reproduce a close pose.
        q_seed = np.array([5.0, -5.0, 10.0, -10.0, 0.0])
        fk = np.asarray(kin.forward_kinematics(q_seed), dtype=float)
        target_pose = matrix_m_to_pose_mm_deg(fk)

        # IK from a slightly perturbed seed toward the FK target.
        q_perturbed = q_seed + np.array([1.0, -1.0, 1.0, -1.0, 0.5])
        q_ik = np.asarray(
            kin.inverse_kinematics(q_perturbed, fk, position_weight=1.0, orientation_weight=0.01),
            dtype=float,
        )
        fk_ik = np.asarray(kin.forward_kinematics(q_ik), dtype=float)
        recovered_pose = matrix_m_to_pose_mm_deg(fk_ik)

        # 5-DoF underactuation: position should converge well, orientation only
        # best-effort. Assert position is close (the strong constraint).
        pos_err = position_error_mm(recovered_pose, target_pose)
        assert pos_err < 10.0, f"IK position residual {pos_err:.3f} mm too large"

    def test_ik_from_current_converges_to_self(self, kin):
        """IK seeded at the current config should keep it near the current pose."""
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        fk = np.asarray(kin.forward_kinematics(q), dtype=float)
        q_ik = np.asarray(
            kin.inverse_kinematics(q, fk, position_weight=1.0, orientation_weight=0.01),
            dtype=float,
        )
        fk_ik = np.asarray(kin.forward_kinematics(q_ik), dtype=float)
        pose_seed = matrix_m_to_pose_mm_deg(fk)
        pose_ik = matrix_m_to_pose_mm_deg(fk_ik)
        assert position_error_mm(pose_ik, pose_seed) < 5.0


class TestBananaServoRegression:
    START = np.array([-1.2747, -18.6374, 12.2198, 72.7912, -109.6703])
    TARGET_XYZ = (291.37, -107.22, 30.0)

    def test_real_trace_uses_bounded_cartesian_candidate(self, kin):
        from jiuwensymbiosis.adapters.so101.config import So101Config

        limits = dict.fromkeys(ARM_JOINT_ORDER, (-180.0, 180.0))
        cfg = So101Config(
            port="/dev/fake",
            home_joints_deg=[0.0] * 5,
            joint_limits=limits,
            z_min_safe_mm=-10.0,
            ik_position_tolerance_mm=3.0,
            servo_max_joint_step_deg=2.0,
            servo_min_send_period_s=0.1,
            servo_max_joint_vel_dps=20.0,
        )
        current_matrix = np.asarray(kin.forward_kinematics(self.START), dtype=float)
        current_pose = matrix_m_to_pose_mm_deg(current_matrix)
        target = So101Pose(*self.TARGET_XYZ, current_pose.rx, current_pose.ry, current_pose.rz)
        target_matrix = pose_mm_deg_to_matrix_m(target)

        old_kin = RobotKinematics(
            URDF,
            target_frame_name="gripper_frame_link",
            joint_names=list(ARM_JOINT_ORDER),
        )
        full_ik = np.asarray(
            old_kin.inverse_kinematics(
                self.START,
                target_matrix,
                position_weight=1.0,
                orientation_weight=0.01,
            ),
            dtype=float,
        )
        old_command = self.START + np.clip(full_ik - self.START, -2.0, 2.0)
        old_pose = matrix_m_to_pose_mm_deg(np.asarray(old_kin.forward_kinematics(old_command), dtype=float))

        class RecordingRobot:
            def __init__(self) -> None:
                self.sent: list[dict[str, float]] = []

            def get_observation(self) -> dict[str, float]:
                return {f"{name}.pos": float(self.START[index]) for index, name in enumerate(ARM_JOINT_ORDER)}

            def send_action(self, action: dict[str, float]) -> dict[str, float]:
                self.sent.append(dict(action))
                return dict(action)

        robot = RecordingRobot()
        robot.START = self.START
        driver = So101Driver(cfg)
        driver._kin = RobotKinematics(
            URDF,
            target_frame_name="gripper_frame_link",
            joint_names=list(ARM_JOINT_ORDER),
        )
        driver._robot = robot
        driver._connected = True
        driver.servo_to_pose(target)

        sent_q = np.array([robot.sent[-1][f"{name}.pos"] for name in ARM_JOINT_ORDER])
        sent_pose = matrix_m_to_pose_mm_deg(np.asarray(driver._kin.forward_kinematics(sent_q), dtype=float))
        cartesian_cap = cfg.servo_max_cartesian_vel_mm_s * cfg.servo_min_send_period_s
        assert position_error_mm(current_pose, old_pose) > cartesian_cap
        assert position_error_mm(current_pose, sent_pose) <= cartesian_cap + 1e-3
        assert position_error_mm(sent_pose, target) < position_error_mm(current_pose, target)

    def test_latest_fast_trace_keeps_outer_and_inner_plans_synchronized(self, kin):
        """Rate-gate skips must keep the outer and inner plans synchronized."""
        from jiuwensymbiosis.adapters.so101.config import So101Config

        start_q = np.array(
            [-2.241758241758242, -55.120879120879124, 39.824175824175825, 74.72527472527473, -92.96703296703296]
        )
        now = 100.0

        def fake_monotonic() -> float:
            return now

        class TrackingRobot:
            def __init__(self) -> None:
                self.q = start_q.copy()
                self.sent: list[dict[str, float]] = []

            def get_observation(self) -> dict[str, float]:
                return {f"{name}.pos": float(self.q[index]) for index, name in enumerate(ARM_JOINT_ORDER)}

            def send_action(self, action: dict[str, float]) -> dict[str, float]:
                actual = dict(action)
                self.sent.append(actual)
                self.q = np.array([actual[f"{name}.pos"] for name in ARM_JOINT_ORDER], dtype=float)
                return actual

        cfg = So101Config(
            port="/dev/fake",
            home_joints_deg=start_q.tolist(),
            joint_limits=dict.fromkeys(ARM_JOINT_ORDER, (-180.0, 180.0)),
            trajectory_hz=20.0,
            fast_control_hz=20.0,
            z_min_safe_mm=-10.0,
            ik_position_tolerance_mm=10.0,
        )
        robot = TrackingRobot()
        driver = So101Driver(cfg, monotonic=fake_monotonic)
        driver._kin = kin
        driver._robot = robot
        driver._connected = True

        start_pose = matrix_m_to_pose_mm_deg(np.asarray(kin.forward_kinematics(start_q), dtype=float))
        last_command = {
            "x": start_pose.x,
            "y": start_pose.y,
            "z": start_pose.z,
            "rx": start_pose.rx,
            "ry": start_pose.ry,
            "rz": start_pose.rz,
        }
        latest_approach = {
            "x": 366.57356976147474,
            "y": 28.624912543729707,
            "z": 72.96866932779551,
            "rz": start_pose.rz,
        }
        skipped = 0
        reached = False
        # The 3 mm hard Cartesian cap no longer permits a delayed tick to make
        # a >3 mm catch-up command. Allow extra ticks for the bounded plan to
        # converge while retaining deliberate 49 ms rate-gate skips.
        for _ in range(240):
            actual_pose = matrix_m_to_pose_mm_deg(np.asarray(kin.forward_kinematics(robot.q), dtype=float))
            actual_xyz = np.array([actual_pose.x, actual_pose.y, actual_pose.z], dtype=float)
            target_xyz = np.array(
                [latest_approach["x"], latest_approach["y"], latest_approach["z"]],
                dtype=float,
            )
            rz_error = abs((actual_pose.rz - latest_approach["rz"] + 180.0) % 360.0 - 180.0)
            if float(np.linalg.norm(actual_xyz - target_xyz)) <= 4.0 and rz_error <= 3.0:
                reached = True
                break
            step = _slew(last_command, latest_approach, max_lin=3.0, max_ang=5.0)
            now += 0.049
            dispatched = driver.servo_to_pose(step)
            if dispatched is False:
                skipped += 1
            else:
                last_command = step

        final_pose = matrix_m_to_pose_mm_deg(np.asarray(kin.forward_kinematics(robot.q), dtype=float))
        target_pose = So101Pose(
            latest_approach["x"],
            latest_approach["y"],
            latest_approach["z"],
            start_pose.rx,
            start_pose.ry,
            start_pose.rz,
        )
        assert skipped > 0
        assert reached is True
        assert position_error_mm(final_pose, target_pose) < 4.0


class TestRealPlanner:
    """Regression tests for _plan_cartesian_waypoints against the REAL LeRobot
    RobotKinematics, which is sensitive to the IK seed. FakeKinematics in the
    unit tests is linear/exact and hides this.

    LeRobot 0.7 runs eight placo Newton steps per ``inverse_kinematics`` call
    (0.6 ran one), so a direct solve seeded far from the target now converges.
    The planner's safety contract is still the residual check: a target outside
    the reachable workspace must be rejected before any action is dispatched.
    The START -> END path is now plannable and remains the positive real-IK
    control. The FakeKinematics unit tests cover the residual/branching logic
    deterministically.
    """

    # Start and end joint configs (deg) — each reachable, FK-validated.
    START = np.array([0.0, -10.0, 20.0, -10.0, 0.0])
    END = np.array([20.0, -25.0, 35.0, -20.0, 15.0])

    def test_direct_ik_converges_for_far_seed(self, kin):
        """LeRobot 0.7's multi-step IK converges from START to the END pose;
        the planner can therefore rely on it for a full Cartesian path."""
        target_m = np.asarray(kin.forward_kinematics(self.END), dtype=float)
        q_direct = np.asarray(
            kin.inverse_kinematics(self.START, target_m, position_weight=1.0, orientation_weight=0.01),
            dtype=float,
        )
        fk_direct = np.asarray(kin.forward_kinematics(q_direct), dtype=float)
        residual = position_error_mm(matrix_m_to_pose_mm_deg(fk_direct), matrix_m_to_pose_mm_deg(target_m))
        assert residual < 1.0, f"expected direct IK to converge (<1 mm); got {residual:.3f}"

    def test_planner_rejects_unreachable_target(self, kin):
        """A target 300 mm beyond the arm's reach has no IK solution within the
        residual tolerance. The planner must raise and never dispatch an action;
        a reachable pose is not enough if the requested Cartesian endpoint cannot
        be reached. This is the pre-dispatch safety contract."""
        driver = _make_planner_driver(kin, ik_position_tolerance_mm=3.0)
        start_q = self.START
        start_m = np.asarray(kin.forward_kinematics(start_q), dtype=float)
        target_m = np.asarray(kin.forward_kinematics(self.END), dtype=float)
        target_m[0, 3] += 0.3  # 300 mm beyond the endpoint; outside the workspace.
        target_pose = matrix_m_to_pose_mm_deg(target_m)

        with pytest.raises(ValueError, match="IK position residual.*exceeds tolerance"):
            driver._plan_cartesian_waypoints(start_q, start_m, target_m, target_pose)

    def test_planner_accepts_reachable_cartesian_path(self, kin):
        """The START -> END path is reachable under LeRobot 0.7's multi-step IK;
        the planner must return a residual-validated waypoint chain."""
        driver = _make_planner_driver(kin, ik_position_tolerance_mm=3.0)
        start_q = self.START
        start_m = np.asarray(kin.forward_kinematics(start_q), dtype=float)
        target_m = np.asarray(kin.forward_kinematics(self.END), dtype=float)
        target_pose = matrix_m_to_pose_mm_deg(target_m)

        waypoints = driver._plan_cartesian_waypoints(start_q, start_m, target_m, target_pose)
        assert len(waypoints) > 0
        final_pose = matrix_m_to_pose_mm_deg(np.asarray(kin.forward_kinematics(waypoints[-1]), dtype=float))
        assert position_error_mm(final_pose, target_pose) <= 3.0
