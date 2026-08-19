# coding: utf-8
import jiuwensymbiosis.adapters.cruzr.lowlevel as ll_mod
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel


def _driver(monkeypatch):
    """A CruzrLowLevel with NO __init__ (no rclpy on the test path)."""
    ll = object.__new__(CruzrLowLevel)
    ll.cfg = CruzrConfig()
    monkeypatch.setattr(ll_mod.time, "sleep", lambda *a, **k: None)
    return ll


def test_move_joints_blocking_ramps_then_publishes_targets(monkeypatch):
    ll = _driver(monkeypatch)
    ramped = {}
    ll.get_joint_positions = lambda: {}          # open-loop (no live state)
    ll._targets_reached = lambda t: True          # break the hold loop immediately
    ll._ramp_to_targets_native = lambda t, **kw: (ramped.update(t), True)[1]
    ll.publish_joint_positions = lambda t: None

    out = ll.move_joints_blocking({"L_shoulder_pitch_joint": 0.2, "L_elbow_roll_joint": -0.3})
    assert out["ok"] and out["ramp_from_state"] is True
    # targets normalized to str/float and handed to the native ramp
    assert ramped == {"L_shoulder_pitch_joint": 0.2, "L_elbow_roll_joint": -0.3}


def test_set_lifter_uses_move_joints_robotcommand_path():
    # The lifter is driven by the same SDK RobotCommand path as the arms
    # (/mc/sdk/robot_command via move_joints_blocking), NOT a separate JointCommand topic.
    ll = object.__new__(CruzrLowLevel)
    seen = []
    ll.move_joints_blocking = lambda targets, **kw: (seen.append(dict(targets)), {"ok": True})[1]

    out = ll.set_lifter({"lifter_pitch_1_joint": -0.3, "lifter_pitch_2_joint": -0.3,
                         "lifter_pitch_3_joint": 0.6})
    assert out["ok"]
    assert seen[0] == {"lifter_pitch_1_joint": -0.3, "lifter_pitch_2_joint": -0.3,
                       "lifter_pitch_3_joint": 0.6}


def test_set_lifter_defaults_to_lifter_ramp_duration(monkeypatch):
    # set_lifter with no explicit ramp uses the dedicated cfg.lifter_ramp_duration_s knob.
    ll = _driver(monkeypatch)
    seen = {}
    ll.move_joints_blocking = lambda targets, **kw: (seen.update(kw), {"ok": True})[1]
    ll.cfg.lifter_ramp_duration_s = 1.2
    ll.set_lifter({"lifter_pitch_1_joint": 0.0})
    assert seen["ramp_duration_s"] == 1.2


def test_set_lifter_explicit_ramp_overrides_cfg(monkeypatch):
    ll = _driver(monkeypatch)
    seen = {}
    ll.move_joints_blocking = lambda targets, **kw: (seen.update(kw), {"ok": True})[1]
    ll.cfg.lifter_ramp_duration_s = 1.2
    ll.set_lifter({"lifter_pitch_1_joint": 0.0}, ramp_duration_s=0.5)
    assert seen["ramp_duration_s"] == 0.5
