"""Synchronized capture and interrupted recording through the public entry point."""

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.spatial_capture import (
    GenesisMemoryRecorder,
    capture_sequence,
    observation_from_capture,
)
from jiuwensymbiosis.spatial_memory.artifacts import read_observation
from tests.unit_tests.spatial_memory.test_adapters import capture


def moving_capture():
    raw = capture()
    raw.update(
        step=12,
        capture_mode="spatial_memory",
        synchronization={
            "kind": "single_physics_state",
            "rgb_step": 12,
            "depth_step": 12,
            "pose_step": 12,
        },
    )
    raw["base_sample"]["linear_velocity_m_s"] = [0.15, 0, 0]
    return raw


def test_explicit_memory_mode_accepts_synced_motion_without_weakening_action_capture(tmp_path):
    raw = moving_capture()
    with pytest.raises(ValueError, match="stationary"):
        observation_from_capture(raw, map_frame="room")
    recorder = GenesisMemoryRecorder(tmp_path / "capture", map_frame="room", allow_motion=True)
    obs = recorder.record(raw)
    assert obs.timestamp_s == raw["captured_at"]
    assert np.array_equal(obs.map_from_camera, raw["camera"]["T_world_camera"])
    saved = json.loads((tmp_path / "capture/000/capture.json").read_text())
    assert saved["synchronization"]["pose_step"] == saved["step"]
    assert saved["base_sample"]["linear_velocity_m_s"][0] == 0.15


@pytest.mark.parametrize(
    "field,value", [("pose_step", 13), ("depth_step", None), ("rgb_step", 12.0), ("kind", "latest_pose")]
)
def test_moving_frames_reject_unsynchronized_or_missing_state(field, value):
    raw = moving_capture()
    raw["synchronization"][field] = value
    with pytest.raises(ValueError, match="synchronized"):
        observation_from_capture(raw, map_frame="room", allow_motion=True)


@pytest.mark.parametrize("failure", ["capture", "cancel", None])
def test_recording_keeps_committed_frames_and_propagates_failure(tmp_path, failure):
    count = 0
    cancelled = threading.Event()

    def grab(name, **options):
        nonlocal count
        count += 1
        assert name == "head_cam" and options["capture_mode"] == "spatial_memory"
        if count == 2 and failure == "capture":
            raise RuntimeError("capture expired")
        raw = moving_capture()
        raw.update(id=str(count), captured_at=42.0 + count)
        if failure == "cancel":
            cancelled.set()
        return raw

    output = tmp_path / "capture"
    kwargs = {"map_frame": "room", "frames": 3, "interval_s": 0, "allow_motion": True, "cancelled": cancelled}
    preview = SimpleNamespace(capture_image=grab)
    if failure:
        with pytest.raises(RuntimeError):
            capture_sequence(preview, output, **kwargs)
    else:
        capture_sequence(preview, output, **kwargs)
    report = json.loads((output / "capture-report.json").read_text())
    names = json.loads((output / "sequence.json").read_text())["frames"]
    assert report["completed"] is (failure is None)
    assert len(names) == report["captured_frames"] == (1 if failure else 3)
    assert [read_observation(output / name).observation_id for name in names] == [str(i + 1) for i in range(len(names))]
