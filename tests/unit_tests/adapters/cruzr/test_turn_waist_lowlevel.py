# coding: utf-8
"""Low-level waist turn: ONE gentle move that holds the arms (no per-degree chunks).

Chunking the turn per-degree spawned the ramp worker dozens of times on the
subprocess backend (~ramp_duration_s each) and made a large turn crawl for a
minute. turn_waist_blocking now issues a single move_joints_blocking; the
backend's smoothstep ramp does the whole turn.
"""

from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel


def _new_ll():
    """A CruzrLowLevel with NO __init__ (no ROS), stubbed move I/O."""
    ll = object.__new__(CruzrLowLevel)
    calls = []
    ll.move_joints_blocking = lambda targets, **kw: (
        calls.append(dict(targets)),
        {"ok": True, "readback": dict(targets)},
    )[1]
    return ll, calls


def test_turn_waist_blocking_is_a_single_move_holding_arms():
    ll, calls = _new_ll()
    hold = {"L_shoulder_pitch_joint": 0.3, "R_shoulder_pitch_joint": -0.2}
    out = ll.turn_waist_blocking(0.5, hold=hold, waist_joint="waist_yaw_joint")

    assert len(calls) == 1                              # ONE command, not per-degree chunks
    cmd = calls[0]
    assert cmd["waist_yaw_joint"] == 0.5               # waist commanded to the absolute target
    assert cmd["L_shoulder_pitch_joint"] == 0.3        # arms held (sent in the same command)
    assert cmd["R_shoulder_pitch_joint"] == -0.2
    assert out["ok"] and out["joint"] == "waist_yaw_joint" and out["target_rad"] == 0.5


def test_turn_waist_blocking_empty_hold():
    ll, calls = _new_ll()
    ll.turn_waist_blocking(-0.3, hold={}, waist_joint="waist_yaw_joint")
    assert len(calls) == 1
    assert calls[0] == {"waist_yaw_joint": -0.3}       # just the waist target when no arms to hold
