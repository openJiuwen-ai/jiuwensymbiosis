"""A published mobile spawn must be admitted by the very map used for execution."""

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis import navigation
from jiuwensymbiosis.simulators.genesis.indoor import supported_spawn
from tests.unit_tests.simulators.genesis.test_indoor import box


def test_provisional_spawn_is_relocated_for_measured_body_and_navigation():
    meshes = [box((6, 6, 0.1), (0, 0, -0.05), "floor"), box((0.8, 0.8, 1.5), (0, 0, 0.75), "furniture")]
    provisional = np.array(supported_spawn(meshes))
    position = provisional + [0, 0, 0.1]
    bounds = np.array([position + [-0.35, -0.33, -0.1], position + [0.35, 0.33, 1.7]])
    spawn, grid, envelope = navigation.admit_spawn(meshes, bounds, position, provisional[2])
    assert grid.clear(spawn)
    assert grid.plan(spawn, [*spawn[:2], 0])["footprint_radius_m"] == envelope["radius"]
    assert np.linalg.norm(np.asarray(spawn)[:2] - provisional[:2]) > 0.05


def test_usable_provisional_spawn_is_preserved():
    meshes = [box((6, 6, 0.1), (0, 0, -0.05), "floor")]
    position = np.array([0, 0, 0.1])
    spawn, grid, _ = navigation.admit_spawn(meshes, [[-0.35, -0.33, 0], [0.35, 0.33, 1.8]], position, 0)
    np.testing.assert_allclose(spawn, [0, 0, 0])
    assert grid.clear(spawn)


def test_runtime_retains_spawn_map_for_later_motion(monkeypatch):
    import importlib
    from types import SimpleNamespace
    from unittest.mock import Mock

    module = importlib.import_module("jiuwensymbiosis.simulators.genesis.scene_runtime")
    meshes = [box((6, 6, 0.1), (0, 0, -0.05), "floor"), box((0.8, 0.8, 1.5), (0, 0, 0.75), "furniture")]
    provisional = np.array(supported_spawn(meshes))
    position = provisional + [0, 0, 0.105]
    from jiuwensymbiosis.simulators.common import base_motion

    monkeypatch.setattr(base_motion, "measure", lambda _: {"position_m": position})
    runtime = module.RuntimeScene.__new__(module.RuntimeScene)
    runtime.info = {"spawn_ground_m": tuple(provisional)}
    runtime.indoor = SimpleNamespace(meshes=meshes)
    runtime.robot = SimpleNamespace(
        bounds=[position + [-0.35, -0.33, -0.1], position + [0.35, 0.33, 1.7]], entity=Mock()
    )
    runtime._admit_mobile_spawn()
    target = runtime.robot.entity.set_pos.call_args.args[0]
    assert runtime.navigation_map.clear(target)
    assert runtime.robot.entity.set_pos.call_args.kwargs["zero_velocity"]
    assert runtime.info["spawn_admission"]["map_fingerprint"] == runtime.navigation_map.fingerprint


@pytest.mark.parametrize("change", ["radius", "height", "open", "failed_rebuild"])
def test_navigation_rebuilds_spawn_map_after_arm_posture_changes(change, monkeypatch):
    from types import SimpleNamespace

    from jiuwensymbiosis.simulators.genesis.truth import EntityRegistry

    meshes = [box((6, 6, 0.1), (0, 0, -0.05), "floor")]
    if change == "radius":
        meshes += [box((0.2, 2.3, 2), (0, y, 1), f"wall-{y}") for y in (-1.85, 1.85)]
    elif change == "height":
        meshes += [box((0.2, 6, 0.2), (0, 0, 1.5), "overhead")]
    position = np.array([-1.8, 0, 0.1])
    bounds = np.array([position + [-0.25, -0.25, -0.1], position + [0.25, 0.25, 0.9]])
    envelope = navigation.whole_body_envelope(bounds, position, 0)
    grid = navigation.NavigationMap.from_meshes(meshes, floor_z=0, **envelope)
    registry = EntityRegistry.__new__(EntityRegistry)
    registry.navigation_map = None
    registry.runtime = SimpleNamespace(
        navigation_map=grid,
        robot=SimpleNamespace(bounds=bounds),
        indoor=SimpleNamespace(meshes=meshes),
        info={"spawn_ground_m": [-1.8, 0, 0]},
    )
    snapshot = {"robot": {"position_m": position.tolist()}, "entities": {}}
    registry.snapshot = lambda snapshot_id: snapshot
    goal = {"x_m": 1.8, "y_m": 0, "yaw_rad": 0}
    registry.query("plan_navigation", goal)
    assert registry.navigation_map is grid
    if change != "height":
        bounds[1, 0] = position[0] + 0.9
    else:
        bounds[1, 2] = 1.8
    if change == "open":
        result = registry.query("plan_navigation", goal)
        assert result["plan"]["footprint_radius_m"] > grid.radius
        assert registry.runtime.navigation_map is registry.navigation_map
        bounds[1, 0] = position[0] + 0.25
        assert registry.query("plan_navigation", goal)["plan"] == result["plan"]
        return
    if change == "failed_rebuild":

        def fail(*args, **kwargs):
            raise ValueError("map build failed")

        monkeypatch.setattr(navigation.NavigationMap, "from_meshes", fail)
        for _ in range(2):
            with pytest.raises(ValueError, match="map build failed"):
                registry.query("plan_navigation", goal)
            assert registry.navigation_map is None
            assert registry.runtime.navigation_map is None
        return
    with pytest.raises(ValueError, match="navigation_"):
        registry.query("plan_navigation", goal)
    assert registry.navigation_map is not grid
    assert registry.runtime.navigation_map is registry.navigation_map
    # A second planner must not resurrect the smaller spawn-time map.
    registry.navigation_map = None
    with pytest.raises(ValueError, match="navigation_"):
        registry.query("plan_navigation", goal)
    with pytest.raises(ValueError, match="fresh"):
        registry.query("plan_navigation", goal | {"snapshot_id": "old-posture"})
