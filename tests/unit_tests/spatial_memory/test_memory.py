"""Exercise public evidence/identity contracts with analytically known surfaces."""

from dataclasses import replace

import numpy as np
import pytest

from jiuwensymbiosis.spatial_memory import Detection, Detections, MemoryPolicy, Observation, SpatialMemory
from jiuwensymbiosis.spatial_memory.artifacts import (
    read_detections,
    read_observation,
    write_detections,
    write_observation,
)
from jiuwensymbiosis.spatial_memory.models import canonical


def sample(index=0, *, shift=0, boxes=((15, 35), (65, 85)), label="chair"):
    rgb = np.full((60, 100, 3), [120, 60, 30], np.uint8)
    depth = np.full((60, 100), np.nan, np.float32)
    detections = []
    for i, (start, end) in enumerate(boxes):
        mask = np.zeros(depth.shape, bool)
        mask[15:45, start - shift : end - shift] = True
        depth[mask] = 2
        detections.append(Detection(str(i), label, 0.8, mask))
    pose = np.eye(4)
    pose[0, 3] = shift * 2 / 100
    observation = Observation(
        str(index),
        "epoch",
        "map",
        "calibration",
        float(index),
        "clock",
        "test",
        rgb,
        depth,
        np.array([[100, 0, 50], [0, 100, 30], [0, 0, 1]]),
        pose,
    )
    batch = Detections(
        tuple(detections),
        canonical(
            {
                "model_id": "fixture",
                "weights_sha256": "test-digest",
                "vocabulary_sha256": "test-vocab",
                "inference": {"mode": "fixture"},
            }
        ),
    )
    return observation, batch


def test_translated_camera_reuses_ids_and_keeps_same_class_objects_separate(tmp_path):
    with SpatialMemory(tmp_path / "memory.db") as memory:
        first = memory.ingest(*sample())
        second = memory.ingest(*sample(1, shift=5, label="stool"))
        ids = [a["entity_id"] for a in first["assignments"]]
        assert len(set(ids)) == 2
        assert [a["entity_id"] for a in second["assignments"]] == ids
        assert all(a["status"] == "matched" for a in second["assignments"])
        assert all(a["score"] == 0.8 and a["association_score"] >= 0.4 for a in second["assignments"])
        objects = memory.query_objects(label="CHAIR")
        assert len(objects) == 2
        assert all(o["state"] == "confirmed" and o["observation_count"] == 2 for o in objects)
        assert memory.get_entity(ids[0])["observed_center_m"] == pytest.approx([-0.51, -0.01, 2], abs=0.03)
        assert len(memory.query_objects(near=(-0.5, 0, 2), radius_m=0.2)) == 1
        assert memory.get_status()["scan_complete"] is None


def test_duplicate_evidence_does_not_confirm_or_increment_and_conflict_is_atomic(tmp_path):
    obs, batch = sample()
    with SpatialMemory(tmp_path / "m.db") as memory:
        first = memory.ingest(obs, batch)
        assert memory.ingest(obs, batch)["duplicate"]
        assert memory.get_status()["observation_count"] == 1
        assert all(o["state"] == "provisional" and o["observation_count"] == 1 for o in memory.query_objects())
        changed = replace(obs, timestamp_s=99)
        with pytest.raises(ValueError, match="conflicts"):
            memory.ingest(changed, batch)
        assert memory.get_entity(first["assignments"][0]["entity_id"])["observation_count"] == 1


def test_same_pose_frames_and_empty_detection_do_not_confirm_or_delete(tmp_path):
    with SpatialMemory(tmp_path / "m.db") as memory:
        memory.ingest(*sample())
        memory.ingest(*sample(1))
        obs, batch = sample(2)
        memory.ingest(obs, replace(batch, instances=()))
        assert len(memory.query_objects()) == 2
        assert all(o["state"] == "provisional" for o in memory.query_objects())


@pytest.mark.parametrize(
    "field,value",
    [("scene_epoch", "reset"), ("map_frame", "other"), ("clock_id", "different"), ("calibration_id", "updated")],
)
def test_scope_conflicts_do_not_partially_commit(tmp_path, field, value):
    with SpatialMemory(tmp_path / "m.db") as memory:
        memory.ingest(*sample())
        before = memory.get_status()
        obs, batch = sample(1)
        with pytest.raises(ValueError, match="scope"):
            memory.ingest(replace(obs, **{field: value}), batch)
        assert memory.get_status() == before


def test_missing_depth_keeps_2d_evidence_and_asks_for_view(tmp_path):
    obs, batch = sample()
    obs = replace(obs, depth_m=np.full_like(obs.depth_m, np.nan))
    with SpatialMemory(tmp_path / "m.db") as memory:
        receipt = memory.ingest(obs, batch)
        assert len(receipt["needs_view"]) == 2
        assert {a["reason"] for a in receipt["needs_view"]} == {"insufficient_depth"}
        assert not memory.query_objects()
        loaded_obs, loaded_batch = memory.get_observation(obs.observation_id)
        assert np.isnan(loaded_obs.depth_m).all()
        assert np.array_equal(loaded_batch.instances[0].mask, batch.instances[0].mask)


def test_depth_quality_distinguishes_sampling_from_surface_resolution(tmp_path):
    obs, batch = sample(boxes=((20, 30),))
    # Enough measurements on a tiny object can still occupy too few map voxels.
    dense = replace(obs, intrinsics=np.array([[10000, 0, 50], [0, 10000, 30], [0, 0, 1]]))
    with SpatialMemory(tmp_path / "m.db") as memory:
        item = memory.ingest(dense, batch)["assignments"][0]
        assert item["reason"] == "insufficient_surface_support"
        quality = item["geometry_quality"]
        assert quality["valid_depth_pixels"] > 12 and quality["surface_voxels"] < 12
        assert quality["depth_coverage_fraction"] == 1
        assert quality["material_depth_verified"] is False


def test_latest_gaps_do_not_relabel_historical_gaps_as_open_tasks(tmp_path):
    obs, batch = sample()
    with SpatialMemory(tmp_path / "m.db") as memory:
        memory.ingest(replace(obs, depth_m=np.full_like(obs.depth_m, np.nan)), batch)
        memory.ingest(*sample(1))
        status = memory.get_status()
        assert status["needs_view_history_count"] == 2
        assert status["latest_observation_needs_view_count"] == 0
        assert status["needs_view_history_by_reason"] == {"insufficient_depth": 2}
        assert memory.query_objects()[0]["semantic_status"] == "unverified_model_labels"
        assert status["scan_complete"] is None


def test_duplicate_masks_and_merged_instances_are_not_forcibly_assigned(tmp_path):
    obs, batch = sample(boxes=((15, 35), (18, 38)))
    with SpatialMemory(tmp_path / "duplicate.db") as memory:
        receipt = memory.ingest(obs, batch)
        assert all(a["reason"] == "overlapping_masks" for a in receipt["assignments"])
        assert not memory.query_objects()
    with SpatialMemory(tmp_path / "merged.db") as memory:
        memory.ingest(*sample())
        obs, batch = sample(1, boxes=((15, 85),))
        receipt = memory.ingest(obs, batch)
        assert receipt["assignments"][0]["reason"] == "ambiguous_identity"
        assert len(memory.query_objects()) == 2


def test_oversegmentation_does_not_assign_one_entity_twice(tmp_path):
    with SpatialMemory(tmp_path / "m.db") as memory:
        memory.ingest(*sample(boxes=((15, 55),)))
        receipt = memory.ingest(*sample(1, boxes=((15, 34), (36, 55))))
        assert len(receipt["needs_view"]) == 2
        assert all(a["entity_id"] is None for a in receipt["assignments"])
        assert len(memory.query_objects()) == 1


def test_contained_small_object_does_not_steal_parent_identity(tmp_path):
    with SpatialMemory(tmp_path / "m.db") as memory:
        first = memory.ingest(*sample(boxes=((15, 85), (25, 35))))
        second = memory.ingest(*sample(1, shift=5, boxes=((15, 85), (25, 35))))
        assert [a["entity_id"] for a in first["assignments"]] == [a["entity_id"] for a in second["assignments"]]
        assert all(a["status"] == "matched" for a in second["assignments"])


def test_object_cropped_by_camera_fov_keeps_identity(tmp_path):
    with SpatialMemory(tmp_path / "m.db") as memory:
        first = memory.ingest(*sample(boxes=((5, 85),)))
        obs, batch = sample(1, boxes=((0, 5),))
        pose = obs.map_from_camera.copy()
        pose[0, 3] = 1.6  # 80 px shift: most of the object is outside the image.
        second = memory.ingest(replace(obs, map_from_camera=pose), batch)
        assert [a["entity_id"] for a in second["assignments"]] == [a["entity_id"] for a in first["assignments"]]
        assert all(a["status"] == "matched" for a in second["assignments"])


def test_restart_readonly_and_replay_preserve_identity(tmp_path):
    path = tmp_path / "m.db"
    with SpatialMemory(path) as memory:
        memory.ingest(*sample())
        memory.ingest(*sample(1, shift=5))
        expected = memory.query_objects()
        saved = [memory.get_observation(str(i)) for i in range(2)]
    before = path.read_bytes()
    with SpatialMemory(path, readonly=True) as memory:
        assert memory.query_objects() == expected
        assert len(memory.get_entity(expected[0]["entity_id"])["evidence"]) == 2
        with pytest.raises(ValueError, match="read-only"):
            memory.ingest(*sample(2))
    assert path.read_bytes() == before
    with SpatialMemory(tmp_path / "replay.db") as memory:
        for observation, batch in saved:
            memory.ingest(observation, batch)
        assert memory.query_objects() == expected


def test_previous_association_database_is_readable_but_cannot_mix_new_writes(tmp_path):
    path = tmp_path / "legacy.db"
    with SpatialMemory(path) as memory:
        memory.ingest(*sample())
        with memory.connection:
            memory.connection.execute(
                "UPDATE metadata SET value=? WHERE key='association'", ("symmetric-surface-rgb-histogram-v2",)
            )
    before = path.read_bytes()
    with SpatialMemory(path, readonly=True) as memory:
        assert len(memory.query_objects()) == 2
        assert memory.get_status()["association_version"] == "symmetric-surface-rgb-histogram-v2"
    with pytest.raises(ValueError, match="association version"):
        SpatialMemory(path)
    assert path.read_bytes() == before


def test_observation_and_detection_bundles_are_hash_checked(tmp_path):
    obs, batch = sample()
    write_observation(tmp_path / "frame", obs)
    write_detections(tmp_path / "detections", batch, obs.depth_m.shape)
    assert np.array_equal(read_observation(tmp_path / "frame").rgb, obs.rgb)
    assert read_detections(tmp_path / "detections").provenance_json == batch.provenance_json
    path = tmp_path / "frame" / "rgbd.npz"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        read_observation(tmp_path / "frame")


def test_frame_arrays_are_immutable_and_invalid_geometry_rejected():
    obs, _ = sample()
    with pytest.raises(ValueError):
        obs.rgb.setflags(write=True)
    bad = np.eye(4)
    bad[0, 0] = -1
    with pytest.raises(ValueError, match="rigid"):
        replace(obs, map_from_camera=bad)
    with pytest.raises(ValueError, match="Depth"):
        replace(obs, depth_m=np.zeros_like(obs.depth_m))
    with pytest.raises(ValueError, match="pinhole"):
        replace(obs, intrinsics=np.zeros((3, 3)))


def test_policy_changes_rejected_and_failed_ingest_rolls_back_entity_updates(tmp_path, monkeypatch):
    path = tmp_path / "m.db"
    with SpatialMemory(path) as memory:
        memory.ingest(*sample())
        before = memory.query_objects()

        # Real storage failure after association must roll back entity mutations.
        def fail(**arrays):
            if "rgb" in arrays:
                raise OSError("disk encoding failed")
            from jiuwensymbiosis.spatial_memory.models import pack

            return pack(**arrays)

        monkeypatch.setattr("jiuwensymbiosis.spatial_memory.memory.pack", fail)
        with pytest.raises(OSError):
            memory.ingest(*sample(1, shift=5))
        assert memory.query_objects() == before
        assert memory.get_status()["observation_count"] == 1
    with pytest.raises(ValueError, match="Policy"):
        SpatialMemory(path, policy=MemoryPolicy(voxel_m=0.1))


def test_mask_size_and_model_change_fail_without_losing_evidence(tmp_path):
    with SpatialMemory(tmp_path / "m.db") as memory:
        obs, batch = sample()
        memory.ingest(obs, batch)
        obs2, batch2 = sample(1)
        wrong = replace(batch2.instances[0], mask=np.ones((10, 10), bool))
        with pytest.raises(ValueError, match="dimensions"):
            memory.ingest(obs2, replace(batch2, instances=(wrong,)))
        provenance = __import__("json").loads(batch.provenance_json)
        provenance["weights_sha256"] = "different"
        with pytest.raises(ValueError, match="model"):
            memory.ingest(obs2, replace(batch2, provenance_json=canonical(provenance)))
        assert memory.get_status()["observation_count"] == 1
