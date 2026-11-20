"""Versioned optional descriptors survive replay and cannot silently mix spaces."""

import json
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jiuwensymbiosis.spatial_memory import Detections, MemoryPolicy, SpatialMemory
from jiuwensymbiosis.spatial_memory.appearance import object_crops
from jiuwensymbiosis.spatial_memory.artifacts import read_detections, write_detections, write_observation
from jiuwensymbiosis.spatial_memory.models import canonical
from jiuwensymbiosis.spatial_memory.replay import replay
from tests.unit_tests.spatial_memory.test_memory import sample


def enrich(batch, *, space="fixture-space"):
    provenance = json.loads(batch.provenance_json)
    provenance["appearance"] = {"space_id": space, "dimension": 3, "kind": "fixture"}
    return Detections(
        tuple(replace(d, embedding=np.array([1, 0, 0], dtype=np.float32)) for d in batch.instances),
        canonical(provenance),
    )


def test_features_are_frozen_and_bound_to_replay_evidence(tmp_path):
    obs, raw = sample()
    batch = enrich(raw)
    with pytest.raises(ValueError):
        batch.instances[0].embedding[0] = 0
    write_detections(tmp_path / "frozen", batch, obs.depth_m.shape)
    recovered = read_detections(tmp_path / "frozen")
    assert np.array_equal(recovered.instances[0].embedding, batch.instances[0].embedding)
    with SpatialMemory(tmp_path / "memory.sqlite") as memory:
        receipt = memory.ingest(obs, recovered)
        assert memory.ingest(obs, recovered)["duplicate"]
        with pytest.raises(ValueError, match="conflicts"):
            memory.ingest(obs, raw)
        assert "embedding" not in receipt["assignments"][0]
        _, saved = memory.get_observation(obs.observation_id)
        assert np.array_equal(saved.instances[0].embedding, batch.instances[0].embedding)
    with SpatialMemory(tmp_path / "memory.sqlite", readonly=True) as memory:
        assert memory.get_status()["appearance_kind"] == "fixture"


def test_model_space_changes_rejected_before_partial_update(tmp_path):
    obs, batch = sample()
    with SpatialMemory(tmp_path / "memory.sqlite") as memory:
        memory.ingest(obs, enrich(batch))
        next_obs = replace(obs, observation_id="next", timestamp_s=obs.timestamp_s + 1)
        with pytest.raises(ValueError, match="model"):
            memory.ingest(next_obs, enrich(batch, space="another-model"))
        assert memory.get_status()["observation_count"] == 1


def test_feature_provenance_and_dimensions_are_required():
    _, batch = sample()
    with pytest.raises(ValueError, match="provenance"):
        Detections(enrich(batch).instances, batch.provenance_json)
    with pytest.raises(ValueError, match="dimension"):
        Detections(batch.instances, enrich(batch).provenance_json)
    with pytest.raises(ValueError, match="unit vector"):
        replace(batch.instances[0], embedding=np.array([2, 0, 0]))
    with pytest.raises(ValueError, match="appearance weight"):
        MemoryPolicy(appearance_weight=1.1)


def test_crop_preserves_long_object_and_masks_background():
    rgb = np.full((20, 100, 3), 255, np.uint8)
    mask = np.zeros((20, 100), bool)
    mask[8:12, 1:99] = True
    rgb[mask] = [240, 10, 20]
    images, masks = object_crops(rgb, [mask])
    assert images[0].size == (224, 224)
    assert np.count_nonzero(masks[0]) > 0
    assert (np.asarray(images[0])[0, 0] == 127).all()
    columns = np.nonzero(masks[0] > 0.5)[1]
    assert columns.min() < 10 and columns.max() > 214


def test_live_features_and_frozen_features_replay_identically(tmp_path):
    obs, batch = sample()
    sequence = tmp_path / "sequence"
    write_observation(sequence / "000", obs)
    (sequence / "sequence.json").write_text(canonical({"schema_version": 1, "frames": ["000"]}))

    class Detector:
        def detect(self, rgb):
            return batch

    class Features:
        def enrich(self, rgb, detections):
            return enrich(detections)

    live = replay(sequence, tmp_path / "live", detector=Detector(), appearance=Features())
    frozen = replay(sequence, tmp_path / "frozen", frozen_detections=tmp_path / "live/detections")
    assert live == frozen


def test_benchmark_cannot_report_existing_clip_features_as_rgb_baseline(tmp_path, monkeypatch):
    obs, batch = sample()
    sequence = tmp_path / "sequence"
    write_observation(sequence / "000", obs)
    (sequence / "sequence.json").write_text(canonical({"schema_version": 1, "frames": ["000"]}))
    detections = tmp_path / "detections"
    write_detections(detections / "000", enrich(batch), obs.depth_m.shape)
    annotations = tmp_path / "annotations.json"
    annotations.write_text('{"reviewer":"fixture"}')
    output = tmp_path / "benchmark"
    script = Path(__file__).resolve().parents[3] / "scripts/spatial_memory_benchmark.py"
    monkeypatch.setattr("sys.argv", list(map(str, [script, sequence, detections, annotations, output])))
    with pytest.raises(ValueError, match="raw YOLOE"):
        runpy.run_path(str(script), run_name="__main__")
    assert not output.exists()
