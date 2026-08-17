# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for CruzrNav wheel-velocity base move (worker faked) + tool wiring."""

from __future__ import annotations

import json
from types import SimpleNamespace

from jiuwensymbiosis.adapters.cruzr import lowlevel as nav_mod
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrNav


def _fake_proc(stdout: str, rc: int = 0):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr="")


def test_navigate_relative_parses_result(monkeypatch):
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _fake_proc(json.dumps({"ok": True, "yaw_reached": 0.1, "dist_traveled": 0.4}))

    monkeypatch.setattr(nav_mod.subprocess, "run", _fake_run)
    out = CruzrNav(CruzrConfig()).navigate_relative(0.4, 0.0, 0.1)
    assert out["ok"] is True
    assert out["dist_traveled"] == 0.4
    cmd = captured["cmd"]
    assert cmd[cmd.index("--dx") + 1] == "0.4"
    assert cmd[cmd.index("--dyaw") + 1] == "0.1"
    # wheel-velocity backend: SDK command topic + wheel joints, NOT nav2
    assert cmd[cmd.index("--command-topic") + 1] == "/mc/sdk/robot_command"
    assert cmd[cmd.index("--left-wheel-joint") + 1] == "driving_wheel_left_joint"
    assert "--frame" not in cmd
    # rotation deceleration params are wired through (accurate turn: slow near target)
    assert "--k-rot-min" in cmd and "--k-rot-slow-rad" in cmd
    # forward deceleration params too (approach the box without overshooting into it)
    assert cmd[cmd.index("--k-fwd-min") + 1] == str(CruzrConfig().base_k_fwd_min)
    assert cmd[cmd.index("--k-fwd-slow-m") + 1] == str(CruzrConfig().base_k_fwd_slow_m)


def test_navigate_relative_gain_overrides_reach_worker(monkeypatch):
    """approach fine-positioning passes gentle k_rot/k_rot_slow_rad/k_fwd overrides; they must land
    in the wheel-worker cmd. Omitted (None) → the global base_k_* defaults are used instead."""
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _fake_proc(json.dumps({"ok": True, "yaw_reached": 0.0, "dist_traveled": 0.0}))

    monkeypatch.setattr(nav_mod.subprocess, "run", _fake_run)
    cfg = CruzrConfig()
    nav = CruzrNav(cfg)
    # overrides supplied → used verbatim
    nav.navigate_relative(0.4, 0.0, 0.1, k_rot=1.5, k_rot_slow_rad=0.5, k_fwd=0.8)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--k-rot") + 1] == "1.5"
    assert cmd[cmd.index("--k-rot-slow-rad") + 1] == "0.5"
    assert cmd[cmd.index("--k-fwd") + 1] == "0.8"
    # omitted → fall back to the global config gains (search big-turn values)
    nav.navigate_relative(0.4, 0.0, 0.1)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--k-rot") + 1] == str(cfg.base_k_rot)
    assert cmd[cmd.index("--k-fwd") + 1] == str(cfg.base_k_fwd)


def test_navigate_relative_lidar_blocked(monkeypatch):
    monkeypatch.setattr(nav_mod.subprocess, "run",
                        lambda cmd, **k: _fake_proc(json.dumps(
                            {"ok": False, "reason": "lidar_blocked", "range": 0.3})))
    out = CruzrNav(CruzrConfig()).navigate_relative(0.4)
    assert out["ok"] is False
    assert out["reason"] == "lidar_blocked"


def test_navigate_relative_worker_failure(monkeypatch):
    monkeypatch.setattr(nav_mod.subprocess, "run",
                        lambda cmd, **k: _fake_proc("", rc=1))
    out = CruzrNav(CruzrConfig()).navigate_relative(0.4)
    assert out["ok"] is False
    assert out["reason"] == "wheel_worker_failed"


def test_navigate_relative_no_output(monkeypatch):
    monkeypatch.setattr(nav_mod.subprocess, "run", lambda cmd, **k: _fake_proc(""))
    out = CruzrNav(CruzrConfig()).navigate_relative(0.4)
    assert out["ok"] is False
    assert out["reason"] == "wheel_no_output"


def test_worker_help_runs_without_rclpy():
    import subprocess as sp
    import sys
    from importlib.util import find_spec
    from pathlib import Path

    worker = Path(find_spec("jiuwensymbiosis.adapters.cruzr.ros2.wheel_worker").origin)
    proc = sp.run([sys.executable, str(worker), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "--dx" in proc.stdout and "--dyaw" in proc.stdout
    # continuous-search spin flags are exposed (and importable without rclpy)
    assert "--spin" in proc.stdout and "--spin-max-rad" in proc.stdout
    # continuous-approach forward flags are exposed too
    assert "--forward" in proc.stdout and "--fwd-max-m" in proc.stdout
    # forward-drive deceleration flags (discrete approach: slow near target)
    assert "--k-fwd-min" in proc.stdout and "--k-fwd-slow-m" in proc.stdout


# ---------------------------------------------------------------- continuous-search spin

class _FakePopen:
    def __init__(self, cmd, **kw):
        self.cmd = cmd
        self.kw = kw
        self._polled = None            # None = running
        self.communicated = "unset"

    def poll(self):
        return self._polled

    def communicate(self, input=None, timeout=None):
        self.communicated = input
        return (json.dumps({"ok": True, "stopped": True, "yaw_turned": 1.23}), "")

    def terminate(self):
        self._polled = 0

    def kill(self):
        self._polled = -9


def test_start_spin_builds_interruptible_worker(monkeypatch):
    captured = {}
    monkeypatch.setattr(nav_mod.subprocess, "Popen",
                        lambda cmd, **kw: captured.update(cmd=cmd, kw=kw) or _FakePopen(cmd, **kw))
    CruzrNav(CruzrConfig()).start_spin(direction=1.0)
    cmd = captured["cmd"]
    assert "--spin" in cmd
    assert cmd[cmd.index("--dyaw") + 1] == "1.0"                 # CCW/left
    assert cmd[cmd.index("--spin-max-rad") + 1] == str(CruzrConfig().search_spin_max_rad)
    assert cmd[cmd.index("--k-rot") + 1] == str(CruzrConfig().search_spin_wheel_rs)
    assert cmd[cmd.index("--command-topic") + 1] == "/mc/sdk/robot_command"
    # non-blocking with piped stdin so the parent can send the 'stop' sentinel
    assert captured["kw"].get("stdin") is nav_mod.subprocess.PIPE


def test_start_spin_direction_sign():
    captured = {}
    import unittest.mock as m
    with m.patch.object(nav_mod.subprocess, "Popen",
                        side_effect=lambda cmd, **kw: captured.update(cmd=cmd) or _FakePopen(cmd)):
        CruzrNav(CruzrConfig()).start_spin(direction=-2.0)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--dyaw") + 1] == "-1.0"                # CW/right


def test_spin_running_reflects_poll():
    p = _FakePopen(["x"])
    p._polled = None
    assert CruzrNav.spin_running(p) is True
    p._polled = 0
    assert CruzrNav.spin_running(p) is False


def test_stop_spin_sends_sentinel_when_running():
    p = _FakePopen(["x"])
    p._polled = None                       # still spinning
    out = CruzrNav.stop_spin(p)
    assert p.communicated == "stop\n"      # clean stop via stdin sentinel
    assert out["ok"] and out["yaw_turned"] == 1.23


def test_stop_spin_drains_when_already_finished():
    p = _FakePopen(["x"])
    p._polled = 0                          # worker already exited (full revolution, not found)
    out = CruzrNav.stop_spin(p)
    assert p.communicated is None          # no sentinel needed
    assert out["yaw_turned"] == 1.23


def test_lowlevel_base_spin_delegates():
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel

    ll = CruzrLowLevel.__new__(CruzrLowLevel)
    ll._nav_obj = None
    calls = {}

    class _FakeNav:
        def start_spin(self, direction=1.0):
            calls["start"] = direction
            return "H"

        def spin_running(self, h):
            calls["running"] = h
            return True

        def stop_spin(self, h):
            calls["stop"] = h
            return {"ok": True, "yaw_turned": 2.0}

    ll._nav_obj = _FakeNav()
    assert ll.start_base_spin(1.0) == "H" and calls["start"] == 1.0
    assert ll.base_spin_running("H") is True and calls["running"] == "H"
    assert ll.stop_base_spin("H")["yaw_turned"] == 2.0 and calls["stop"] == "H"


def test_lowlevel_navigate_relative_delegates():
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel

    ll = CruzrLowLevel.__new__(CruzrLowLevel)
    ll._nav_obj = None

    class _FakeNav:
        def navigate_relative(self, dx, dy=0.0, dyaw=0.0, *, k_rot=None, k_rot_slow_rad=None, k_fwd=None):
            return {"ok": True, "args": [dx, dy, dyaw], "gains": [k_rot, k_rot_slow_rad, k_fwd]}

    ll._nav_obj = _FakeNav()
    out = ll.navigate_relative(0.5, 0.0, -0.2)
    assert out["ok"] and out["args"] == [0.5, 0.0, -0.2]
    assert out["gains"] == [None, None, None]                 # no override → global base_k_* used downstream
    out = ll.navigate_relative(0.5, 0.0, -0.2, k_rot=1.5, k_rot_slow_rad=0.5, k_fwd=0.8)
    assert out["gains"] == [1.5, 0.5, 0.8]                     # gentle approach gains threaded through


# ---------------------------------------------------------------- continuous-approach forward drive

class _FakeDrivePopen(_FakePopen):
    def communicate(self, input=None, timeout=None):
        self.communicated = input
        return (json.dumps({"ok": True, "stopped": True, "dist_traveled": 0.9}), "")


def test_start_drive_builds_interruptible_worker(monkeypatch):
    captured = {}
    monkeypatch.setattr(nav_mod.subprocess, "Popen",
                        lambda cmd, **kw: captured.update(cmd=cmd, kw=kw) or _FakeDrivePopen(cmd, **kw))
    CruzrNav(CruzrConfig()).start_drive()
    cmd = captured["cmd"]
    assert "--forward" in cmd
    assert cmd[cmd.index("--k-fwd") + 1] == str(CruzrConfig().approach_drive_wheel_rs)
    assert cmd[cmd.index("--fwd-max-m") + 1] == str(CruzrConfig().approach_drive_max_m)
    assert cmd[cmd.index("--timeout") + 1] == str(CruzrConfig().approach_drive_timeout_s)
    # steering (differential wheels toward the live head bearing) wired through
    assert cmd[cmd.index("--k-steer") + 1] == str(CruzrConfig().approach_drive_steer_gain)
    assert cmd[cmd.index("--steer-max") + 1] == str(CruzrConfig().approach_drive_steer_max_rs)
    assert cmd[cmd.index("--command-topic") + 1] == "/mc/sdk/robot_command"
    # front-lidar e-stop wired through for the forward creep
    assert cmd[cmd.index("--safe-dist") + 1] == str(CruzrConfig().base_safe_dist_m)
    # non-blocking with piped stdin so the parent can send the 'stop' sentinel
    assert captured["kw"].get("stdin") is nav_mod.subprocess.PIPE


def test_start_drive_overrides_creep_and_self_bound(monkeypatch):
    """The fine-grasp servo passes a slower creep (``k_fwd``) + a tighter forward self-bound
    (``fwd_max_m``); both must reach the worker cmd. Omitted (None) → the config approach-drive defaults
    (see test_start_drive_builds_interruptible_worker) are used, so the coarse approach is unchanged."""
    captured = {}
    monkeypatch.setattr(nav_mod.subprocess, "Popen",
                        lambda cmd, **kw: captured.update(cmd=cmd, kw=kw) or _FakeDrivePopen(cmd, **kw))
    CruzrNav(CruzrConfig()).start_drive(k_fwd=0.6, fwd_max_m=0.4)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--k-fwd") + 1] == "0.6"      # slower creep than approach_drive_wheel_rs
    assert cmd[cmd.index("--fwd-max-m") + 1] == "0.4"  # tighter self-bound than approach_drive_max_m


def test_drive_running_reflects_poll():
    p = _FakeDrivePopen(["x"])
    p._polled = None
    assert CruzrNav.drive_running(p) is True
    p._polled = 0
    assert CruzrNav.drive_running(p) is False


class _StdinRec:
    """Records lines written to a fake worker stdin (for steer_drive)."""
    def __init__(self):
        self.lines = []
    def write(self, s):
        self.lines.append(s)
    def flush(self):
        pass


def test_steer_drive_feeds_bearing_to_running_worker():
    p = _FakeDrivePopen(["x"])
    p._polled = None                       # still driving
    p.stdin = _StdinRec()
    CruzrNav.steer_drive(p, 0.3)
    assert p.stdin.lines == ["0.3000\n"]   # bearing forwarded over stdin, curves toward target


def test_steer_drive_noop_when_worker_finished():
    p = _FakeDrivePopen(["x"])
    p._polled = 0                          # worker already exited (self-bound) → don't write
    p.stdin = _StdinRec()
    CruzrNav.steer_drive(p, 0.3)
    assert p.stdin.lines == []             # no write to a dead worker; drive self-bounds


def test_steer_drive_swallows_broken_pipe():
    class _DeadStdin:
        def write(self, s):
            raise BrokenPipeError("closed")
        def flush(self):
            pass
    p = _FakeDrivePopen(["x"])
    p._polled = None
    p.stdin = _DeadStdin()
    CruzrNav.steer_drive(p, 0.3)           # best-effort: broken pipe is swallowed, no raise


def test_forward_parser_accepts_steer_flags():
    from jiuwensymbiosis.adapters.cruzr.ros2.wheel_worker import _build_parser

    a = _build_parser().parse_args(
        ["--forward", "--k-fwd", "0.5", "--k-steer", "0.6", "--steer-max", "0.4"])
    assert a.forward is True
    assert a.k_steer == 0.6 and a.steer_max == 0.4


def test_lowlevel_steer_base_drive_delegates():
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel

    ll = CruzrLowLevel.__new__(CruzrLowLevel)
    ll._nav_obj = None
    calls = {}

    class _FakeNav:
        def steer_drive(self, handle, bearing_rad):
            calls["args"] = (handle, bearing_rad)

    ll._nav_obj = _FakeNav()
    ll.steer_base_drive("H", 0.25)
    assert calls["args"] == ("H", 0.25)


def test_stop_drive_sends_sentinel_when_running():
    p = _FakeDrivePopen(["x"])
    p._polled = None                       # still driving
    out = CruzrNav.stop_drive(p)
    assert p.communicated == "stop\n"      # clean stop via stdin sentinel
    assert out["ok"] and out["dist_traveled"] == 0.9


def test_stop_drive_drains_when_already_finished():
    p = _FakeDrivePopen(["x"])
    p._polled = 0                          # worker already exited (hit its self-bound, parent late)
    out = CruzrNav.stop_drive(p)
    assert p.communicated is None          # no sentinel needed
    assert out["dist_traveled"] == 0.9


def test_lowlevel_base_drive_delegates():
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel

    ll = CruzrLowLevel.__new__(CruzrLowLevel)
    ll._nav_obj = None
    calls = {}

    class _FakeNav:
        def start_drive(self, *, k_fwd=None, fwd_max_m=None):
            calls["start"] = {"k_fwd": k_fwd, "fwd_max_m": fwd_max_m}
            return "H"

        def drive_running(self, h):
            calls["running"] = h
            return True

        def stop_drive(self, h):
            calls["stop"] = h
            return {"ok": True, "dist_traveled": 1.5}

    ll._nav_obj = _FakeNav()
    assert ll.start_base_drive() == "H" and calls["start"] == {"k_fwd": None, "fwd_max_m": None}
    # fine-grasp servo threads a slower creep + tighter forward self-bound through to the worker
    assert ll.start_base_drive(k_fwd=0.6, fwd_max_m=0.4) == "H"
    assert calls["start"] == {"k_fwd": 0.6, "fwd_max_m": 0.4}
    assert ll.base_drive_running("H") is True and calls["running"] == "H"
    assert ll.stop_base_drive("H")["dist_traveled"] == 1.5 and calls["stop"] == "H"


def test_api_navigate_relative_tool(monkeypatch):
    from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
    from jiuwensymbiosis.tools.builder import list_tool_meta

    class _LL:
        def navigate_relative(self, dx_m, dy_m=0.0, dyaw_rad=0.0):
            return {"ok": True, "got": [dx_m, dy_m, dyaw_rad]}

    class _Env:
        low_level = _LL()

        class cfg:
            default_arm = "left"

    api = CruzrApi(_Env())
    out = api.navigate_relative(0.4, 0.0, 0.1)
    assert out["ok"] and out["got"] == [0.4, 0.0, 0.1]
    meta = {m["name"]: m for m in list_tool_meta(api)}
    assert "navigate_relative" in meta
    assert "motion" in meta["navigate_relative"].get("tags", [])
