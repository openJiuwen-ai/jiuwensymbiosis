"""Capture coordinate semantics and optional model failure behavior."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.spatial_capture import GenesisMemoryRecorder, observation_from_capture
from jiuwensymbiosis.spatial_memory import SpatialMemory
from jiuwensymbiosis.spatial_memory.artifacts import write_observation
from jiuwensymbiosis.spatial_memory.replay import replay
from jiuwensymbiosis.spatial_memory.yoloe import YoloEPromptFree, label_kind
from tests.unit_tests.spatial_memory.test_memory import sample


def capture():
    obs, _ = sample()
    return {
        "id": "frame",
        "scene_instance_id": "epoch",
        "config_revision": 3,
        "captured_at": 42.0,
        "camera_name": "head_cam",
        "rgb": obs.rgb,
        "depth_m": obs.depth_m,
        "base_sample": {"linear_velocity_m_s": [0, 0, 0], "angular_velocity_rad_s": [0, 0, 0]},
        "camera": {
            "K": obs.intrinsics.tolist(),
            "D": [0] * 5,
            "width": 100,
            "height": 60,
            "T_world_camera": obs.map_from_camera.tolist(),
            "T_base_camera_mm": (obs.map_from_camera * 1000).tolist(),
            "depth_type": "optical_z",
            "depth_unit": "m",
            "distortion_model": "rectified_pinhole",
        },
    }


def test_genesis_capture_uses_frozen_world_camera_in_metres():
    raw = capture()
    raw["camera"]["T_world_camera"][0][3] = 2.5
    obs = observation_from_capture(raw, map_frame="room")
    assert obs.map_from_camera[0, 3] == 2.5
    assert obs.timestamp_s == 42 and obs.scene_epoch == "epoch:3"
    assert obs.pose_source == "simulator_readback"
    raw["camera"]["T_world_camera"][0][3] = 100
    assert obs.map_from_camera[0, 3] == 2.5


def test_moz1_recorder_publishes_replayable_sequence_and_rejects_reset(tmp_path):
    raw = capture()
    raw["step"] = 10
    recorder = GenesisMemoryRecorder(tmp_path / "capture", map_frame="room")
    recorder(raw)
    manifest = json.loads((tmp_path / "capture/sequence.json").read_text())
    assert manifest["capture_mode"] == "moz1_head_cam"
    assert manifest["frames"] == ["000"]
    with pytest.raises(ValueError, match="Duplicate"):
        recorder(raw)
    raw["id"] = "second"
    raw["config_revision"] += 1
    with pytest.raises(ValueError, match="scope"):
        recorder(raw)
    assert json.loads((tmp_path / "capture/sequence.json").read_text()) == manifest


@pytest.mark.parametrize("failure", ["moving", "distortion", "units"])
def test_capture_rejects_incompatible_or_moving_data(failure):
    raw = capture()
    if failure == "moving":
        raw["base_sample"]["linear_velocity_m_s"][0] = 0.1
    elif failure == "units":
        raw["camera"]["depth_unit"] = "mm"
    else:
        raw["camera"]["D"][0] = 0.1
    with pytest.raises(ValueError):
        observation_from_capture(raw, map_frame="room")


def test_detector_propagates_failure_and_preserves_all_instances(tmp_path, monkeypatch):
    obs, batch = sample()
    calls = []

    class Array:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Model:
        task = "segment"
        names = dict.fromkeys(range(4585), "chair")

        def __init__(self, path):
            self.fail = False

        def predict(self, image, **kwargs):
            if self.fail:
                raise RuntimeError("inference failed")
            calls.append((np.asarray(image), kwargs))
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(conf=Array(np.array([0.9, 0.8])), cls=Array(np.array([0, 1]))),
                    masks=SimpleNamespace(data=Array(np.stack([d.mask for d in batch.instances]))),
                    names=self.names,
                )
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLOE=Model))
    monkeypatch.setattr("importlib.metadata.version", lambda package: "fixture")
    monkeypatch.setenv("YOLO_OFFLINE", "false")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "true")
    weights = tmp_path / "fixture-pf.pt"
    weights.write_bytes(b"fixture")
    detector = YoloEPromptFree(weights, hashlib.sha256(b"fixture").hexdigest(), device="cpu")
    result = detector.detect(obs.rgb)
    assert len(result.instances) == 2
    assert np.array_equal(calls[0][0], obs.rgb)
    assert calls[0][1]["retina_masks"] is True
    assert result.instances[1].class_id == 1
    detector.model.fail = True
    with pytest.raises(RuntimeError, match="inference failed"):
        detector.detect(obs.rgb)
    with pytest.raises(ValueError, match="digest"):
        YoloEPromptFree(weights, "wrong")


def test_scene_surface_routing_is_explicit():
    assert label_kind("darkness") == label_kind("childs room") == "scene"
    assert label_kind("kitchen floor") == "surface"
    assert label_kind("chair") == "object"
    assert label_kind("playroom") == label_kind("home interior") == "scene"
    for label in ("bedroom window", "dining room table", "floor lamp", "wall clock", "ceiling light"):
        assert label_kind(label) == "object"


def test_real_replay_path_records_failure_instead_of_an_empty_frame(tmp_path):
    sequence = tmp_path / "sequence"
    obs, batch = sample()
    write_observation(sequence / "000", obs)
    (sequence / "sequence.json").write_text(json.dumps({"schema_version": 1, "frames": ["000"]}))

    class Failing:
        def detect(self, rgb):
            raise RuntimeError("unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        replay(sequence, tmp_path / "failed", detector=Failing())
    with SpatialMemory(tmp_path / "failed/memory.sqlite", readonly=True) as memory:
        assert memory.get_status()["observation_count"] == 0
    assert json.loads((tmp_path / "failed/failure.json").read_text())["committed_frames"] == 0

    class Detector:
        def detect(self, rgb):
            return batch

    first = replay(sequence, tmp_path / "live", detector=Detector())
    second = replay(sequence, tmp_path / "replay", frozen_detections=tmp_path / "live/detections")
    assert first["objects"] == second["objects"]
    assert (tmp_path / "live/review.html").is_file()


def test_cli_stdout_is_json_even_during_sdk_initialization(tmp_path):
    database = tmp_path / "memory.sqlite"
    with SpatialMemory(database):
        pass
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/spatial_memory.py"), "status", str(database)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout)["observation_count"] == 0
