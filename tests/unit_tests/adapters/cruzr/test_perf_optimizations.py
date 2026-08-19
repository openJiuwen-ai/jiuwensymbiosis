# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cruzr latency optimizations: connect-time warm-ups (IK / camera streams) and the
``resident_workers`` gate that routes camera/wheel calls through persistent --serve workers
instead of a fresh subprocess per call. resident_workers=False must keep the one-shot path."""

from __future__ import annotations

from pathlib import Path

import jiuwensymbiosis.adapters.cruzr.lowlevel as ll_mod
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv
from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrCamera, CruzrNav
from jiuwensymbiosis.kinematics import ik_pinocchio

# ---- optimization 2: connect-time warm-ups -------------------------------------------------

def test_ik_warm_is_noop_without_pinocchio(monkeypatch):
    monkeypatch.setattr(ik_pinocchio, "_PIN_OK", False)
    assert ik_pinocchio.warm("/nonexistent.urdf") is False


def test_warm_ik_async_calls_warm_with_urdf(monkeypatch):
    env = CruzrEnv(CruzrConfig())
    seen = {}
    monkeypatch.setattr(ik_pinocchio, "warm", lambda urdf: seen.setdefault("urdf", urdf))
    env._warm_ik_async()
    if env._warm_ik_thread is not None:
        env._warm_ik_thread.join(timeout=2.0)
    assert seen.get("urdf") == env.cfg.urdf_path


def test_warm_camera_async_grabs_waist_and_head():
    env = CruzrEnv(CruzrConfig())
    grabbed = []

    class _Inner:
        def grab_frames(self, camera="waist"):
            grabbed.append(("waist", camera))

        def grab_head_frame(self):
            grabbed.append(("head",))

    env._inner = _Inner()
    env._warm_camera_async()
    if env._warm_camera_thread is not None:
        env._warm_camera_thread.join(timeout=2.0)
    assert ("waist", "waist") in grabbed and ("head",) in grabbed


# ---- optimization 4: resident_workers routing ----------------------------------------------

class _FakeProc:
    returncode = 0
    stdout = '{"ok": true, "dist_traveled": 0.5}'
    stderr = ""


def _nav(cfg):
    nav = CruzrNav.__new__(CruzrNav)
    nav.cfg = cfg
    nav._worker = Path("/tmp/wheel_worker.py")
    nav._resident_move = None
    return nav


def test_navigate_relative_uses_resident_when_enabled():
    cfg = CruzrConfig()
    cfg.resident_workers = True
    nav = _nav(cfg)
    seen = {}
    nav._resident_move_request = lambda dx, dyaw, kr, krs, kf: (
        seen.update(dx=dx, dyaw=dyaw), {"ok": True})[1]
    out = nav.navigate_relative(0.5, 0.0, 0.2)
    assert out == {"ok": True}
    assert seen == {"dx": 0.5, "dyaw": 0.2}


def test_navigate_relative_uses_oneshot_when_disabled(monkeypatch):
    cfg = CruzrConfig()  # resident_workers defaults to False
    nav = _nav(cfg)
    called = {"resident": False, "run": False}
    nav._resident_move_request = lambda *a, **k: called.__setitem__("resident", True) or {"ok": True}
    monkeypatch.setattr(ll_mod.subprocess, "run",
                        lambda *a, **k: (called.__setitem__("run", True), _FakeProc())[1])
    nav.navigate_relative(0.5, 0.0, 0.0)
    assert called["run"] is True and called["resident"] is False


def test_camera_grab_uses_resident_when_enabled():
    cfg = CruzrConfig()
    cfg.resident_workers = True
    cam = CruzrCamera.__new__(CruzrCamera)
    cam.cfg = cfg
    cam._resident = {}
    routed = {}
    cam._resident_grab = lambda build_cmd, reader: (routed.setdefault("hit", True), "frame")[1]
    r = cam._run_and_read_once(lambda out: ["cmd"], lambda o, m: "frame")
    assert routed.get("hit") is True and r == "frame"
