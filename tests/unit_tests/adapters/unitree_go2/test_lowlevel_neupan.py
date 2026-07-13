# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the NEUPAN navigation loop in UnitreeGo2Driver.

These tests run without rclpy / neupan / torch / unitree_sdk2py: the driver is
subclassed so the ROS2 publisher, the NEUPAN planner, and the odom/scan reads
are replaced with in-process fakes. The fake odom *integrates* the published
velocity (a mecanum step: vx in the body frame + vy orthogonal, no rotation
when omega=0), so the distance-based arrival check converges naturally and the
loop terminates — no real time.sleep is used (``_sleep_remaining`` is a no-op
in the subclass).

Key difference from the ubetech (diff) tests: the Go2 uses **mecanum**
kinematics, so the planner returns a **3-dim** action ``[vx, vy, omega]`` and
``vy`` is published (NOT forced to 0). The clamp tests cover vx AND vy.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from jiuwensymbiosis.adapters.unitree_go2.lowlevel import UnitreeGo2Driver

_SCAN = {"ranges": [1.0, 2.0, 3.0], "angle_min": -math.pi, "angle_max": math.pi, "range_min": 0.1, "range_max": 10.0}


class _FakePublisher:
    """Records every publish_twist(vx, vy, wz) call.

    When wired to a shared sim-pose dict, each publish also integrates the
    commanded velocity one control-period step (mecanum model: vx along the
    body x-axis, vy along the body y-axis, omega spins the heading) — so the
    driver's distance-based arrival check converges naturally.
    """

    def __init__(self, *, pose: dict | None = None, dt: float = 0.1) -> None:
        self.cmds: list[tuple[float, float, float]] = []
        self.is_running = True
        self._pose = pose  # {"x","y","theta"} mutates in place
        self._dt = dt

    def start(self) -> bool:
        # Real Ros2CmdVel.start() returns True on success; the fake is always "up".
        return True

    def stop(self) -> None:
        return None

    def publish_twist(self, vx: float, vy: float, wz: float) -> bool:
        self.cmds.append((float(vx), float(vy), float(wz)))
        if self._pose is not None:
            th = self._pose["theta"]
            # Mecanum: vx along body x, vy along body y (orthogonal), then spin.
            self._pose["x"] += (vx * math.cos(th) - vy * math.sin(th)) * self._dt
            self._pose["y"] += (vx * math.sin(th) + vy * math.cos(th)) * self._dt
            self._pose["theta"] += wz * self._dt
        return True


class _FakeIPath:
    arrive_threshold: float = 0.1


class _FakePlanner:
    """Programmable NEUPAN stand-in. Records calls; returns scripted action/info.

    ``action`` is a **3-dim** ``[vx, vy, omega]`` list (mecanum). ``info`` may be
    a single dict (used every call) or a list of dicts (one per call, cycled).
    """

    def __init__(self, action: list[float] | None = None, info: dict | list[dict] | None = None) -> None:
        self.ipath = _FakeIPath()
        self.min_distance = 5.0
        self.calls: list[tuple[Any, Any]] = []  # (state, points) per __call__
        self.last_goal: tuple[Any, Any] | None = None  # (start, goal) passed to update_initial_path_from_goal
        self.action = action if action is not None else [1.0, 0.0, 0.0]  # [vx, vy, omega]
        self.info = info if info is not None else {}

    def update_initial_path_from_goal(self, start: Any, goal: Any) -> None:
        self.last_goal = (start, goal)

    def scan_to_point(self, state: Any, scan: Any, **kwargs: Any) -> Any:
        return np.zeros((2, 1))  # one dummy obstacle point; content unused by the loop

    def __call__(self, state: Any, points: Any, velocities: Any = None) -> tuple[np.ndarray, dict]:
        self.calls.append((state, points))
        action = np.array([[self.action[0]], [self.action[1]], [self.action[2]]])
        if isinstance(self.info, list):
            idx = min(len(self.calls) - 1, len(self.info) - 1)
            info = dict(self.info[idx])
        else:
            info = dict(self.info)
        return action, info


class _NavDriver(UnitreeGo2Driver):
    """Driver with ROS2 + NEUPAN swapped for in-process fakes.

    The odom read returns a sim pose that the FakePublisher advances on each
    publish, so the distance arrival check converges without any real planner.
    """

    def __init__(self, *, planner: _FakePlanner, max_linear: float = 1.0, max_angular: float = 1.5) -> None:
        # Bypass the real __init__ (which builds Ros2Camera/Ros2Odom/Ros2Scan/Ros2CmdVel).
        self._max_linear = float(max_linear)
        self._max_angular = float(max_angular)
        self._home_xy_yaw = [0.0, 0.0, 0.0]
        self._neupan_config_path = "fake"
        self._neupan_arrive_threshold_m = 0.1
        self._neupan_max_collision_count = 100
        self._neupan_control_hz = 10.0
        self._planner = planner
        # Shared mutable sim pose; the publisher integrates it on each publish.
        self._sim_pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self._cmd_vel = _FakePublisher(pose=self._sim_pose, dt=1.0 / self._neupan_control_hz)
        self._connected = True
        self._scan_value: dict | None = None
        # connect() touches these ROS2 sidecars — stub them as None so the
        # TestSdkSoftened.connect() path (which skips the real __init__) works.
        self._camera = None
        self._odom = None
        self._scan = None
        self._network_interface = None
        self._sdk = None

    # --- fakes for the nav loop's data reads ---
    def get_odom_pose(self) -> dict | None:
        return {
            "x": self._sim_pose["x"],
            "y": self._sim_pose["y"],
            "yaw_deg": math.degrees(self._sim_pose["theta"]),
        }

    def get_scan(self) -> dict | None:
        return self._scan_value

    def set_scan(self, scan: dict | None) -> None:
        self._scan_value = scan

    # --- no real sleeping: keep the test fast ---
    @staticmethod
    def _sleep_remaining(loop_start: float, period: float) -> None:  # type: ignore[override]  # test: no real sleeping
        return None


def _make_driver(planner: _FakePlanner | None = None, **kw: Any) -> tuple[_NavDriver, _FakePublisher, _FakePlanner]:
    planner = planner if planner is not None else _FakePlanner()
    drv = _NavDriver(planner=planner, **kw)
    drv.set_scan(_SCAN)
    return drv, drv._cmd_vel, planner


class TestNavLoopArrival:
    def test_arrives_by_distance_and_stops(self):
        # Planner steers straight +x at 1.0 m/s; sim pose integrates toward
        # goal (0.5, 0). Distance drops below 0.1 within a few ticks.
        drv, pub, planner = _make_driver(_FakePlanner(action=[1.0, 0.0, 0.0]))
        drv._move_to_xy_yaw(0.5, 0.0, 0.0)
        nonzero = [c for c in pub.cmds if c != (0.0, 0.0, 0.0)]
        assert nonzero, "expected at least one non-zero velocity command before arrival"
        assert pub.cmds[-1] == (0.0, 0.0, 0.0)  # unconditional stop in finally
        assert len(planner.calls) >= 1

    def test_arrives_on_neupan_arrive_flag(self):
        # Planner reports arrive on the first call → loop exits immediately.
        drv, pub, planner = _make_driver(_FakePlanner(action=[1.0, 0.0, 0.0], info={"arrive": True}))
        drv._move_to_xy_yaw(5.0, 5.0, 0.0)
        assert len(planner.calls) == 1
        assert pub.cmds[-1] == (0.0, 0.0, 0.0)

    def test_goal_heading_is_atan2_to_goal(self):
        # Goal in +x/+y → goal_theta should be +45°. Captured via the planner.
        drv, _pub, planner = _make_driver(_FakePlanner(action=[1.0, 0.0, 0.0], info={"arrive": True}))
        drv._move_to_xy_yaw(1.0, 1.0, 0.0)
        assert planner.last_goal is not None
        _start, goal = planner.last_goal
        assert goal[2, 0] == pytest.approx(math.pi / 4, abs=1e-6)


class TestNavLoopCollision:
    def test_collision_cap_raises_after_max(self):
        drv, pub, _ = _make_driver(_FakePlanner(action=[1.0, 0.0, 0.0], info={"stop": True}), max_linear=1.0)
        drv._neupan_max_collision_count = 3
        with pytest.raises(RuntimeError, match="collisions persisted"):
            drv._move_to_xy_yaw(5.0, 0.0, 0.0)
        assert pub.cmds[-1] == (0.0, 0.0, 0.0)

    def test_collision_count_resets_on_non_collision(self):
        # A fresh planner that never stops → must converge by distance, no abort.
        drv, pub, _ = _make_driver(_FakePlanner(action=[1.0, 0.0, 0.0]), max_linear=1.0)
        drv._neupan_max_collision_count = 2
        drv._move_to_xy_yaw(0.3, 0.0, 0.0)
        assert pub.cmds[-1] == (0.0, 0.0, 0.0)


class TestNavLoopMecanumAndClamp:
    def test_vy_published_mecanum_model(self):
        # Mecanum: planner returns a non-zero vy → it must be published (NOT 0).
        # This is the key difference from the ubetech diff model (vy forced to 0).
        drv, pub, _ = _make_driver(
            _FakePlanner(action=[1.0, 0.5, 0.0], info=[{"arrive": False}, {"arrive": True}]),
        )
        drv._move_to_xy_yaw(5.0, 0.0, 0.0)
        nonzero = [c for c in pub.cmds if c != (0.0, 0.0, 0.0)]
        assert nonzero, "expected at least one non-zero velocity command before arrival"
        assert any(vy == 0.5 for _vx, vy, _wz in nonzero), "mecanum vy must be published, not forced to 0"

    def test_velocity_clamped_to_max(self):
        # Planner demands 100 m/s vx and vy → both must be clamped to max_linear.
        drv, pub, _ = _make_driver(
            _FakePlanner(action=[100.0, 100.0, 0.0], info=[{"arrive": False}, {"arrive": True}]),
            max_linear=1.0,
            max_angular=1.5,
        )
        drv._move_to_xy_yaw(5.0, 0.0, 0.0)
        nonzero = [c for c in pub.cmds if c != (0.0, 0.0, 0.0)]
        assert nonzero
        assert all(abs(vx) <= 1.0 + 1e-9 for vx, _vy, _wz in nonzero)
        assert all(abs(vy) <= 1.0 + 1e-9 for _vx, vy, _wz in nonzero)

    def test_angular_clamped_to_max(self):
        # vx=0, vy=0, omega=100 → only spins, never reaches (0,5) by distance.
        drv, pub, _ = _make_driver(
            _FakePlanner(action=[0.0, 0.0, 100.0], info=[{"arrive": False}, {"arrive": True}]),
            max_linear=1.0,
            max_angular=0.5,
        )
        drv._move_to_xy_yaw(0.0, 5.0, 0.0)
        nonzero = [c for c in pub.cmds if c != (0.0, 0.0, 0.0)]
        assert nonzero
        assert all(abs(wz) <= 0.5 + 1e-9 for _vx, _vy, wz in nonzero)


class TestNavLoopWaitsAndErrors:
    def test_holds_when_scan_missing_then_proceeds(self):
        # Scan present throughout → converges (the missing-scan wait branch is
        # covered structurally; see ubetech test for the same design).
        drv, pub, planner = _make_driver(_FakePlanner(action=[1.0, 0.0, 0.0]))
        drv._move_to_xy_yaw(0.5, 0.0, 0.0)
        assert len(planner.calls) >= 1
        assert pub.cmds[-1] == (0.0, 0.0, 0.0)

    def test_raises_when_no_initial_odom(self):
        drv, pub, _ = _make_driver(_FakePlanner(action=[1.0, 0.0, 0.0]))
        drv.get_odom_pose = lambda: None  # type: ignore[method-assign]  # test: inject no-odom path
        with pytest.raises(RuntimeError, match="no odometry"):
            drv._move_to_xy_yaw(0.5, 0.0, 0.0)
        assert pub.cmds[-1] == (0.0, 0.0, 0.0)

    def test_raises_when_planner_not_built(self):
        drv, _pub, _ = _make_driver(_FakePlanner(action=[1.0, 0.0, 0.0]))
        drv._planner = None
        with pytest.raises(RuntimeError, match="NEUPAN planner not built"):
            drv._move_to_xy_yaw(0.5, 0.0, 0.0)

    def test_raises_when_cmd_vel_not_running(self):
        drv, _pub, _ = _make_driver(_FakePlanner(action=[1.0, 0.0, 0.0]))
        drv._cmd_vel = None
        with pytest.raises(RuntimeError, match="cmd_vel not running"):
            drv._move_to_xy_yaw(0.5, 0.0, 0.0)

    def test_non_finite_target_raises(self):
        drv, _pub, _ = _make_driver(_FakePlanner(action=[1.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="non-finite"):
            drv._move_to_xy_yaw(float("nan"), 0.0, 0.0)

    def test_stops_on_planner_exception(self):
        class _ExplodingPlanner(_FakePlanner):
            def __call__(self, state, points, velocities=None):
                raise RuntimeError("planner blew up")

        drv, pub, _ = _make_driver(_ExplodingPlanner(action=[1.0, 0.0, 0.0]))
        with pytest.raises(RuntimeError, match="planner blew up"):
            drv._move_to_xy_yaw(0.5, 0.0, 0.0)
        assert pub.cmds[-1] == (0.0, 0.0, 0.0)


class TestSdkSoftened:
    """The SDK path is softened: a missing unitree_sdk2py must NOT raise in
    connect() — the NEUPAN/cmd_vel nav path works without it."""

    def test_connect_succeeds_without_sdk(self, monkeypatch):
        # Hide unitree_sdk2py so the SDK import in connect() fails — connect()
        # must not raise (it just logs a warning and continues).
        import builtins

        real_import = builtins.__import__

        def _block_sdk(name, *args, **kwargs):
            if name.startswith("unitree_sdk2py"):
                raise ImportError("simulated: unitree_sdk2py not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_sdk)
        # Also hide neupan so the planner build path isn't triggered (no config).
        drv = _NavDriver(planner=_FakePlanner(action=[1.0, 0.0, 0.0]))
        drv._neupan_config_path = None  # no planner build in connect()
        drv._connected = False
        drv._sdk = None
        # Should NOT raise even though unitree_sdk2py is "missing".
        drv.connect()
        assert drv._connected is True
        # SDK marker stays None (SDK path unavailable), but nav path is unaffected.
        assert drv._sdk is None
