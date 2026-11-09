"""A published mobile spawn must be admitted by the very map used for execution."""

import numpy as np

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
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock

    directory = Path(__file__).parents[4] / "jiuwensymbiosis/simulators/genesis"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("runtime_spawn_under_test", directory / "scene_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meshes = [box((6, 6, 0.1), (0, 0, -0.05), "floor"), box((0.8, 0.8, 1.5), (0, 0, 0.75), "furniture")]
    provisional = np.array(supported_spawn(meshes))
    position = provisional + [0, 0, 0.105]
    monkeypatch.setitem(sys.modules, "base_motion", SimpleNamespace(measure=lambda _: {"position_m": position}))
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
