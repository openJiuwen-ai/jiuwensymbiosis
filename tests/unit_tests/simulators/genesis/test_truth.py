"""Truth snapshots retain their step and reject stale/reset references."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis import truth


@pytest.fixture
def registry():
    runtime = SimpleNamespace(
        source=None,
        robot=None,
        motion=None,
        bodies=[],
        cycle=0,
        steps=12,
        settings=SimpleNamespace(scene_id="demo", dt=0.01),
        info={"instance_id": "scene-1"},
    )
    return truth.EntityRegistry(runtime)


def test_reused_snapshot_keeps_original_step_and_explicit_truth_source(registry):
    first = registry.query("list_entities", {})
    registry.runtime.steps += 10
    entity = registry.query(
        "entity_state", {"entity_id": first["entities"][0]["entity_id"], "snapshot_id": first["snapshot_id"]}
    )
    assert entity["snapshot_id"] == first["snapshot_id"]
    assert entity["step"] == 12 and entity["sim_time"] == pytest.approx(0.12)
    assert entity["source"] == "simulation_ground_truth"
    assert entity["entity"]["frame"] == "world"
    assert registry.query("list_entities", {})["step"] == 22


@pytest.mark.parametrize("invalidate", ["reset", "expiry"])
def test_snapshot_rejected_after_reset_or_expiry(registry, monkeypatch, invalidate):
    monkeypatch.setattr(truth.time, "monotonic", lambda: 100.0)
    first = registry.query("list_entities", {})
    if invalidate == "reset":
        registry.runtime.cycle += 1
    else:
        monkeypatch.setattr(truth.time, "monotonic", lambda: 161.0)
    with pytest.raises(ValueError, match="snapshot_expired_or_scene_reset"):
        registry.query("list_entities", {"snapshot_id": first["snapshot_id"]})


@pytest.mark.parametrize("params", [{"limit": True}, {"limit": 0}, {"limit": 51}, {"offset": True}, {"offset": -1}])
def test_invalid_pagination_is_rejected(registry, params):
    with pytest.raises(ValueError, match="invalid_pagination"):
        registry.query("list_entities", params)


def test_empty_search_and_missing_robot_do_not_invent_state(registry):
    result = registry.query("list_entities", {"query": "no such object", "limit": 1})
    assert result["entities"] == [] and result["total"] == 0 and result["next_offset"] is None
    with pytest.raises(ValueError, match="mobile_robot_state_unavailable"):
        registry.query("robot_state", {})
    with pytest.raises(ValueError, match="unknown_entity_id"):
        registry.query("entity_state", {"entity_id": "missing"})


def test_base_coordinates_transform_origin_and_bounds_without_changing_world_entity():
    entity = {
        "position_m": [2.0, 3.0, 0.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "bounds_m": [[1.5, 2.0, 0.0], [2.5, 4.0, 1.0]],
        "frame": "world",
    }
    robot = {"position_m": [1.0, 1.0, 0.0], "body_yaw_rad": math.pi / 2}
    result = truth.in_base(entity, robot)
    np.testing.assert_allclose(result["position_m"], [2, -1, 0], atol=1e-12)
    np.testing.assert_allclose(result["bounds_m"], [[1, -1.5, 0], [3, -0.5, 1]], atol=1e-12)
    np.testing.assert_allclose(result["quaternion_wxyz"], [math.sqrt(0.5), 0, 0, -math.sqrt(0.5)], atol=1e-12)
    assert result["frame"] == "base" and entity["frame"] == "world"
    assert entity["position_m"] == [2.0, 3.0, 0.0]
