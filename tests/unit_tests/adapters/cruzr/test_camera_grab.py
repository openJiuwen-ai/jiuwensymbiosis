# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for CruzrCamera subprocess orchestration (worker is faked)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from jiuwensymbiosis.adapters.cruzr import lowlevel as camera_mod
from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrCamera
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig


def _output_dir_from_cmd(cmd):
    return Path(cmd[cmd.index("--output-dir") + 1])


def _fake_run_factory(*, ok=True, has_depth=True, k=None, rc=0):
    def _fake_run(cmd, **kwargs):
        out = _output_dir_from_cmd(cmd)
        if ok:
            np.save(out / "color.npy", np.zeros((2, 2, 3), dtype=np.uint8))
            if has_depth:
                np.save(out / "depth.npy", np.ones((2, 2), dtype=np.float32))
        meta = {"ok": ok, "has_depth": has_depth, "K": k}
        (out / "meta.json").write_text(json.dumps(meta))
        return SimpleNamespace(returncode=rc, stdout="", stderr="")
    return _fake_run


def test_grab_returns_frame(monkeypatch):
    monkeypatch.setattr(camera_mod.subprocess, "run",
                        _fake_run_factory(k=[[100, 0, 1], [0, 100, 1], [0, 0, 1]]))
    frame = CruzrCamera(CruzrConfig()).grab()
    assert frame is not None
    assert frame.rgb.shape == (2, 2, 3)
    assert frame.depth_m.shape == (2, 2)
    assert frame.intrinsics.shape == (3, 3)


def test_grab_without_depth_or_k(monkeypatch):
    monkeypatch.setattr(camera_mod.subprocess, "run",
                        _fake_run_factory(has_depth=False, k=None))
    frame = CruzrCamera(CruzrConfig()).grab()
    assert frame is not None
    assert frame.depth_m is None
    assert frame.intrinsics is None


def test_grab_returns_none_on_worker_failure(monkeypatch):
    monkeypatch.setattr(camera_mod.time, "sleep", lambda *_a, **_k: None)  # skip retry settle
    monkeypatch.setattr(camera_mod.subprocess, "run", _fake_run_factory(ok=False))
    assert CruzrCamera(CruzrConfig()).grab() is None


def test_grab_returns_none_on_nonzero_rc(monkeypatch):
    monkeypatch.setattr(camera_mod.time, "sleep", lambda *_a, **_k: None)  # skip retry settle

    def _boom(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")
    monkeypatch.setattr(camera_mod.subprocess, "run", _boom)
    assert CruzrCamera(CruzrConfig()).grab() is None


def test_grab_retries_with_fresh_worker_until_frame(monkeypatch):
    # Head-camera startup warmup: the first attempt's fresh worker finishes DDS discovery too late to
    # catch a frame (no_color_frame); a second fresh worker, discovery now done, gets one. grab() must
    # transparently retry and succeed — no per-frame no-op leaking to the caller.
    monkeypatch.setattr(camera_mod.time, "sleep", lambda *_a, **_k: None)  # skip retry settle
    calls = {"n": 0}
    good = _fake_run_factory(k=[[100, 0, 1], [0, 100, 1], [0, 0, 1]])

    def _fail_then_ok(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            out = _output_dir_from_cmd(cmd)          # worker ran but caught no frame
            (out / "meta.json").write_text(json.dumps({"ok": False, "reason": "no_color_frame"}))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return good(cmd, **kwargs)

    monkeypatch.setattr(camera_mod.subprocess, "run", _fail_then_ok)
    frame = CruzrCamera(CruzrConfig()).grab()
    assert frame is not None
    assert frame.rgb.shape == (2, 2, 3)
    assert calls["n"] == 2                            # one retry with a fresh worker


def test_grab_no_retry_when_disabled(monkeypatch):
    # camera_grab_retries=0 → exactly one attempt, no retry (opt-out honored).
    calls = {"n": 0}

    def _count(cmd, **kwargs):
        calls["n"] += 1
        return _fake_run_factory(ok=False)(cmd, **kwargs)

    monkeypatch.setattr(camera_mod.subprocess, "run", _count)
    cfg = CruzrConfig()
    cfg.camera_grab_retries = 0
    assert CruzrCamera(cfg).grab() is None
    assert calls["n"] == 1


def test_worker_help_runs_without_rclpy():
    import subprocess as sp
    import sys
    from importlib.util import find_spec
    from pathlib import Path

    worker = Path(find_spec("jiuwensymbiosis.adapters.cruzr.ros2.camera_worker").origin)
    proc = sp.run([sys.executable, str(worker), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "--color-topic" in proc.stdout


def test_head_cloud_worker_uses_aligned_color_topic():
    # Legacy head-cloud grab (kept for debug_align; the 2-D search no longer uses it): still the
    # left rect-color compressed stream + its point cloud, per head_aligned_color_topic.
    cmd = CruzrCamera(CruzrConfig())._head_cloud_cmd(Path("/tmp/head-cloud-test"))
    assert cmd[cmd.index("--color-topic") + 1] == \
        "/sensor/camera/stereo/left/image_rect_color/compressed"
    assert cmd[cmd.index("--color-msg-type") + 1] == "sensor_msgs/msg/CompressedImage"
    assert "--rectify-cruzr-stereo-left" not in cmd


def test_head_cloud_worker_uses_default_transports_env(monkeypatch):
    # camera_fastdds_profiles_file="" (default transports): _worker_env sets the RMW and cleans the
    # discovery env, and actively POPS any stale Fast DDS profile / CycloneDDS / localhost-only vars.
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        out = _output_dir_from_cmd(cmd)
        np.save(out / "color.npy", np.zeros((2, 2, 3), dtype=np.uint8))
        (out / "meta.json").write_text(json.dumps(
            {"ok": True, "has_cloud": False, "tf_base_cam": None}
        ))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    monkeypatch.setenv("FASTRTPS_DEFAULT_PROFILES_FILE", "/tmp/stale-profiles.xml")
    monkeypatch.setenv("FASTDDS_DEFAULT_PROFILES_FILE", "/tmp/stale-profiles.xml")
    monkeypatch.setenv("CYCLONEDDS_URI", "file:///tmp/stale-cyclonedds.xml")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setattr(camera_mod.subprocess, "run", _fake_run)

    result = CruzrCamera(CruzrConfig()).grab_head_frame()

    assert result is not None
    assert captured["env"]["RMW_IMPLEMENTATION"] == "rmw_fastrtps_cpp"
    assert "FASTRTPS_DEFAULT_PROFILES_FILE" not in captured["env"]
    assert "FASTDDS_DEFAULT_PROFILES_FILE" not in captured["env"]
    assert "CYCLONEDDS_URI" not in captured["env"]
    assert captured["env"]["ROS_DOMAIN_ID"] == "0"
    assert captured["env"]["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "SUBNET"
    assert "ROS_LOCALHOST_ONLY" not in captured["env"]


def test_head_cloud_worker_can_rectify_vendor_rgb_as_fallback():
    cfg = CruzrConfig()
    cfg.head_aligned_color_topic = cfg.head_left_topic
    cfg.head_aligned_color_msg_type = cfg.head_color_msg_type
    cmd = CruzrCamera(cfg)._head_cloud_cmd(Path("/tmp/head-cloud-test"))
    assert "--rectify-cruzr-stereo-left" in cmd


def test_cruzr_stereo_rectification_shape():
    from jiuwensymbiosis.adapters.cruzr.ros2.camera_worker import _rectify_cruzr_stereo_left

    rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    rectified = _rectify_cruzr_stereo_left(rgb)
    assert rectified.shape == (360, 640, 3)
    assert rectified.dtype == np.uint8


def test_lowlevel_grab_frames_delegates_to_camera():
    from jiuwensymbiosis.perception.frame import CameraFrame
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel

    ll = CruzrLowLevel.__new__(CruzrLowLevel)  # 跳过 __init__（不连 ROS）
    ll._camera_obj = None

    class _FakeCam:
        def grab(self, camera="waist"):
            return CameraFrame(rgb=np.zeros((2, 2, 3), np.uint8),
                               depth_m=np.ones((2, 2), np.float32),
                               intrinsics=np.eye(3))

    ll._camera_obj = _FakeCam()
    rgb, depth, k, tf = ll.grab_frames()
    assert rgb.shape == (2, 2, 3)
    assert depth.shape == (2, 2)
    assert k.shape == (3, 3)
    assert tf is None


def test_lowlevel_grab_frames_none_when_no_frame():
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel

    ll = CruzrLowLevel.__new__(CruzrLowLevel)

    class _NoCam:
        def grab(self, camera="waist"):
            return None

    ll._camera_obj = _NoCam()
    assert ll.grab_frames() is None


def test_lowlevel_head_frame_falls_back_to_joint_fk(monkeypatch):
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel

    ll = CruzrLowLevel.__new__(CruzrLowLevel)

    class _HeadCam:
        def grab_head_frame(self):
            return (
                np.zeros((2, 3, 3), np.uint8),
                np.zeros((1, 1, 3), np.float32),
                None,
            )

    ll._camera_obj = _HeadCam()
    fallback = np.eye(4, dtype=np.float64)
    fallback[2, 3] = 1431.0
    monkeypatch.setattr(ll, "_head_tf_from_joint_state", lambda: fallback)

    rgb, cloud, tf = ll.grab_head_frame()
    assert rgb.shape == (2, 3, 3)
    assert cloud.shape == (1, 1, 3)
    assert np.array_equal(tf, fallback)


def test_grab_parses_tf_base_cam(tmp_path, monkeypatch):
    import json
    import numpy as np
    from jiuwensymbiosis.adapters.cruzr import lowlevel as cam_mod

    out = tmp_path
    np.save(out / "color.npy", np.zeros((4, 4, 3), dtype=np.uint8))
    np.save(out / "depth.npy", np.ones((4, 4), dtype=np.float32))
    tf = [[1, 0, 0, 10.0], [0, 1, 0, 20.0], [0, 0, 1, 30.0], [0, 0, 0, 1]]
    (out / "meta.json").write_text(json.dumps(
        {"ok": True, "has_depth": True, "K": [[345, 0, 320], [0, 345, 180], [0, 0, 1]],
         "tf_base_cam": tf, "width": 4, "height": 4}))

    class _Cfg:
        ros_python = "/usr/bin/python3"
        waist_color_topic = waist_depth_topic = waist_camera_info_topic = "t"
        color_msg_type = depth_msg_type = "shm_msgs/msg/Image1m"
        depth_scale = 0.001
        camera_grab_timeout_s = 1.0
        base_frame = "base_link"
        camera_optical_frame = "waist_front_rgbd_color_optical_frame"

    def _fake_run(*a, **k):
        class P:
            returncode = 0
            stderr = ""
            stdout = ""
        return P()

    monkeypatch.setattr(cam_mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(cam_mod.tempfile, "TemporaryDirectory",
                        lambda *a, **k: _DirCtx(str(out)))
    frame = cam_mod.CruzrCamera(_Cfg()).grab("waist")
    assert frame is not None
    assert frame.tf_base_cam is not None
    assert frame.tf_base_cam[0, 3] == 10.0


class _DirCtx:
    def __init__(self, p): self.p = p
    def __enter__(self): return self.p
    def __exit__(self, *a): return False


def test_grab_head_builds_head_cmd_and_returns_rgb(monkeypatch):
    """camera='head' grabs the RIGHT-EYE raw image; no depth/info/TF."""
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        out = _output_dir_from_cmd(cmd)
        np.save(out / "color.npy", np.zeros((3, 5, 3), dtype=np.uint8))
        (out / "meta.json").write_text(json.dumps(
            {"ok": True, "has_depth": False, "K": None,
             "tf_base_cam": None, "width": 5, "height": 3}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(camera_mod.subprocess, "run", _fake_run)
    frame = CruzrCamera(CruzrConfig()).grab("head")
    assert frame is not None
    assert frame.rgb.shape == (3, 5, 3)
    assert frame.depth_m is None
    assert frame.intrinsics is None
    assert frame.tf_base_cam is None
    cmd = captured["cmd"]
    assert "--color-topic" in cmd
    assert cmd[cmd.index("--color-topic") + 1] == "/sensor/camera/stereo_right/image/raw"
    assert cmd[cmd.index("--color-msg-type") + 1] == "shm_msgs/msg/Image2m"
    assert captured["env"]["RMW_IMPLEMENTATION"] == "rmw_fastrtps_cpp"
    # Default transports (camera_fastdds_profiles_file=""): no Fast DDS profile is wired in.
    assert "FASTRTPS_DEFAULT_PROFILES_FILE" not in captured["env"]
    assert "FASTDDS_DEFAULT_PROFILES_FILE" not in captured["env"]
    assert "CYCLONEDDS_URI" not in captured["env"]
    assert "--depth-topic" not in cmd            # head has no depth
    assert "--ensure-rgb" in cmd                 # grayscale-safe
    # TF is skipped for the head: optical frame passed empty
    assert cmd[cmd.index("--camera-optical-frame") + 1] == ""


def test_lowlevel_grab_frames_head_passthrough():
    from jiuwensymbiosis.perception.frame import CameraFrame
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel

    ll = CruzrLowLevel.__new__(CruzrLowLevel)
    ll._camera_obj = None

    class _HeadCam:
        def grab(self, camera="waist"):
            assert camera == "head"
            return CameraFrame(rgb=np.zeros((3, 5, 3), np.uint8))

    ll._camera_obj = _HeadCam()
    rgb, depth, k, tf = ll.grab_frames(camera="head")
    assert rgb.shape == (3, 5, 3)
    assert depth is None and k is None and tf is None
