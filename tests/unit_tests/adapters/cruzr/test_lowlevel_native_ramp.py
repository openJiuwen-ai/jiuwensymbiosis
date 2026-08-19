# coding: utf-8
"""The native rclpy backend must ramp, not snap.

Regression for the dangerous arm jump seen after switching to a conda env whose
Python matches the machine's ROS (so rclpy imports in-process and the driver
picks the native backend). The native path used to publish the raw target once,
slewing the arm to it at max rate. It must smoothstep from the measured joints
instead — a dense duration-based trajectory over ramp_duration_s.
"""
import jiuwensymbiosis.adapters.cruzr.lowlevel as ll_mod
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel


def _driver(monkeypatch):
    # No __init__ (no rclpy on the test path); the ramp helper under test only reads
    # cfg and calls get_joint_positions / publish_joint_positions, both stubbed below.
    ll = object.__new__(CruzrLowLevel)
    ll.cfg = CruzrConfig(control_rate_hz=250.0, ramp_duration_s=1.5)
    monkeypatch.setattr(ll_mod.time, "sleep", lambda *a, **k: None)
    return ll


def test_native_move_ramps_from_measured_no_jump(monkeypatch):
    ll = _driver(monkeypatch)
    start = {"j1": 0.0, "j2": 1.0}
    target = {"j1": 1.0, "j2": -0.5}
    setpoints: list[dict] = []
    monkeypatch.setattr(ll, "get_joint_positions", lambda: dict(start))
    monkeypatch.setattr(ll, "publish_joint_positions", lambda d: setpoints.append(dict(d)))

    assert ll._ramp_to_targets_native(dict(target)) is True
    assert len(setpoints) == 375  # ramp_duration_s * control_rate_hz = 1.5 * 250
    # eases from the measured start (no first-setpoint snap) to EXACTLY the target
    assert abs(setpoints[0]["j1"] - start["j1"]) < 0.02
    assert abs(setpoints[0]["j2"] - start["j2"]) < 0.02
    assert setpoints[-1] == target
    # every consecutive step is tiny — the signature of "no jump"
    maxstep = max(abs(setpoints[i][j] - setpoints[i - 1][j])
                  for i in range(1, len(setpoints)) for j in target)
    assert maxstep < 0.02


def test_native_move_ramp_duration_override_shortens_trajectory(monkeypatch):
    # The light head uses a short per-call ramp (head_ramp_duration_s) so a "look" is
    # quick, without touching the gentle box-carrying default. Override → fewer setpoints,
    # still eased from measured start to exactly the target (no jump).
    ll = _driver(monkeypatch)  # cfg default ramp_duration_s = 1.5 → 375 setpoints
    start = {"j1": 0.0, "j2": 1.0}
    target = {"j1": 1.0, "j2": -0.5}
    setpoints: list[dict] = []
    monkeypatch.setattr(ll, "get_joint_positions", lambda: dict(start))
    monkeypatch.setattr(ll, "publish_joint_positions", lambda d: setpoints.append(dict(d)))

    assert ll._ramp_to_targets_native(dict(target), ramp_duration_s=0.4) is True
    assert len(setpoints) == 100  # 0.4 * 250, not the 375 of the default
    assert abs(setpoints[0]["j1"] - start["j1"]) < 0.03
    assert setpoints[-1] == target
    maxstep = max(abs(setpoints[i][j] - setpoints[i - 1][j])
                  for i in range(1, len(setpoints)) for j in target)
    assert maxstep < 0.05  # still smooth, just fewer/larger-than-1.5s steps


def test_move_joints_blocking_passes_ramp_override(monkeypatch):
    # set_head relies on move_joints_blocking forwarding ramp_duration_s to the ramp helper.
    ll = _driver(monkeypatch)
    seen: dict = {}
    monkeypatch.setattr(ll, "_ramp_to_targets_native",
                        lambda targets, *, ramp_duration_s=None: seen.update(dur=ramp_duration_s) or True)
    monkeypatch.setattr(ll, "get_joint_positions", lambda: {"h_yaw": 0.0})
    monkeypatch.setattr(ll, "publish_joint_positions", lambda d: None)
    monkeypatch.setattr(ll, "_targets_reached", lambda t: True)

    ll.move_joints_blocking({"h_yaw": 0.3}, ramp_duration_s=0.4)
    assert seen["dur"] == 0.4


def test_native_move_missing_state_holds_at_target(monkeypatch):
    # No live state -> we cannot know the start; ramping from 0.0 would itself jump.
    # The helper must hold at the target and report ramp_from_state=False.
    ll = _driver(monkeypatch)
    setpoints: list[dict] = []
    monkeypatch.setattr(ll, "get_joint_positions", lambda: {})
    monkeypatch.setattr(ll, "publish_joint_positions", lambda d: setpoints.append(dict(d)))

    target = {"j1": 0.3, "j2": -0.2}
    assert ll._ramp_to_targets_native(dict(target)) is False
    assert setpoints and all(sp == target for sp in setpoints)
