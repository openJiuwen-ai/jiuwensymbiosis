"""Identity must survive an independently observed foreground occluder."""

from dataclasses import replace

import numpy as np

from jiuwensymbiosis.spatial_memory import SpatialMemory
from tests.unit_tests.spatial_memory.test_memory import sample


def test_foreground_occlusion_does_not_create_a_second_identity(tmp_path):
    first, batch = sample(boxes=((10, 90),))
    later, partial = sample(1, boxes=((10, 18),))
    depth = later.depth_m.copy()
    # The remaining 85% of the original object is demonstrably behind a closer
    # occluder, not simply outside this detection's mask.
    depth[15:45, 18:90] = 1.0
    later = replace(later, depth_m=depth)
    with SpatialMemory(tmp_path / "m.db") as memory:
        identity = memory.ingest(first, batch)["assignments"][0]["entity_id"]
        result = memory.ingest(later, partial)["assignments"][0]
        assert result["entity_id"] == identity
        assert result["status"] == "matched"
        assert memory.get_status()["entity_count"] == 1


def test_missing_depth_does_not_certify_occlusion(tmp_path):
    first, batch = sample(boxes=((10, 90),))
    later, partial = sample(1, boxes=((10, 18),))
    assert np.isnan(later.depth_m[20, 60])
    with SpatialMemory(tmp_path / "m.db") as memory:
        identity = memory.ingest(first, batch)["assignments"][0]["entity_id"]
        result = memory.ingest(later, partial)["assignments"][0]
        assert result["entity_id"] != identity


def test_redundant_mask_keeps_raw_evidence_but_only_one_identity_vote(tmp_path):
    obs, batch = sample(boxes=((15, 35), (15, 35)))
    with SpatialMemory(tmp_path / "m") as memory:
        result = memory.ingest(obs, batch)
        assert memory.get_status()["entity_count"] == 1
        assert len(result["assignments"]) == 2
        assert len(memory.get_observation(obs.observation_id)[1].instances) == 2
        assert len({a["entity_id"] for a in result["assignments"]}) == 1
        assert memory.query_objects()[0]["observation_count"] == 1
        assert memory.query_objects()[0]["state"] == "provisional"


def test_fully_occluded_revisit_is_not_counted_as_a_failed_opportunity():
    from jiuwensymbiosis.spatial_memory import MemoryPolicy
    from jiuwensymbiosis.spatial_memory.geometry import project
    from jiuwensymbiosis.spatial_memory.view_tasks import update_tasks

    first, batch = sample(boxes=((15, 35),))
    policy = MemoryPolicy()
    points, _, _ = project(first, batch.instances[0], policy)
    task = {
        "state": "open",
        "surface_m": points.tolist(),
        "last_attempt_pose": first.map_from_camera.tolist(),
        "attempts": 0,
    }
    later, _ = sample(1, shift=5)
    later = replace(later, depth_m=np.ones_like(later.depth_m))
    update_tasks([task], later, [], [], policy)
    assert task["attempts"] == 0
    assert task["state"] == "open"
