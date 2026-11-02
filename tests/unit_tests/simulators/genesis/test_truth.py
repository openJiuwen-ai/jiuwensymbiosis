"""Truth snapshots are coherent, scoped, bounded and independent from control."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis import base_motion
from jiuwensymbiosis.simulators.genesis.agent import GenesisDriver, make_session
from jiuwensymbiosis.simulators.genesis.observation_policy import allowed_tools, history_for, policy_id
from jiuwensymbiosis.simulators.genesis.truth import EntityRegistry
from jiuwensymbiosis.tools.builder import list_tool_meta


class Reading:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, _):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return np.asarray(self.value).tolist()


class Body:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)

    def get_pos(self):
        return Reading(self.position)

    def get_quat(self):
        return Reading([1, 0, 0, 0])

    def get_AABB(self):
        return Reading([self.position - 0.1, self.position + 0.1])


@pytest.fixture
def registry(monkeypatch):
    robot = Body([1, 2, 0.1])
    runtime = SimpleNamespace(
        source=None,
        robot=SimpleNamespace(name="robot", entity=robot),
        bodies=[Body([2, 3, 0.2]), Body([4, 5, 0.5]), Body([6, 7, 0.7])],
        settings=SimpleNamespace(scene_id="demo", dt=0.01),
        motion=True,
        cycle=0,
        steps=10,
        info={"instance_id": "instance-a"},
    )
    monkeypatch.setattr(
        base_motion, "measure", lambda _: {"position_m": robot.position.tolist(), "body_yaw_rad": math.pi / 2}
    )
    return EntityRegistry(runtime)


def test_pagination_and_base_frame_share_one_snapshot(registry):
    first = registry.query("list_entities", {"limit": 2})
    items = first["entities"]
    offset = first["next_offset"]
    while offset is not None:
        page = registry.query("list_entities", {"limit": 2, "offset": offset, "snapshot_id": first["snapshot_id"]})
        items += page["entities"]
        offset = page["next_offset"]
    assert len(items) == first["total"] == 6
    assert len({item["entity_id"] for item in items}) == 6
    target = next(e for e in items if e["name"] == "金色球")
    registry.runtime.bodies[0].position += 10
    result = registry.query(
        "entity_state", {"entity_id": target["entity_id"], "frame": "base", "snapshot_id": first["snapshot_id"]}
    )
    assert result["entity"]["position_m"] == pytest.approx([1, -1, 0.1])
    assert result["source"] == "simulation_ground_truth" and result["step"] == 10
    fresh = registry.query("entity_state", {"entity_id": target["entity_id"]})
    assert fresh["entity"]["position_m"] == [12, 13, 10.2]


def test_reset_expiry_and_registry_rebuild(registry, monkeypatch):
    snapshot = registry.query("robot_state", {})
    ids = list(registry.entities)
    assert list(EntityRegistry(registry.runtime).entities) == ids
    registry.runtime.cycle += 1
    with pytest.raises(ValueError, match="expired_or_scene_reset"):
        registry.query("robot_state", {"snapshot_id": snapshot["snapshot_id"]})
    snapshot = registry.query("robot_state", {})
    monkeypatch.setattr("jiuwensymbiosis.simulators.genesis.truth.time.monotonic", lambda: snapshot["captured_at"] + 61)
    with pytest.raises(ValueError, match="expired_or_scene_reset"):
        registry.query("robot_state", {"snapshot_id": snapshot["snapshot_id"]})


def test_readonly_truth_never_acquires_motion_and_visual_cannot_query():
    preview = SimpleNamespace(
        acquire_motion=Mock(),
        query_scene=Mock(
            return_value={"source": "simulation_ground_truth", "scene_instance_id": "a", "reset_generation": 0}
        ),
    )
    session = make_session(preview, actions_enabled=False, observation_mode="ground_truth")
    session.strict_capabilities = True
    with session:
        assert session.api.sim_list_entities()["source"] == "simulation_ground_truth"
    preview.acquire_motion.assert_not_called()
    names = {m["name"] for m in list_tool_meta(session.api, env=session.env)}
    assert {"sim_list_entities", "sim_get_robot_state", "sim_get_entity_state"} <= names
    assert "navigate_to_entity" not in names
    with pytest.raises(RuntimeError, match="disabled"):
        GenesisDriver(preview).query_truth("list_entities", {})
    assert "sim_list_entities" not in allowed_tools("visual", True)
    assert not history_for(
        "visual", [{"role": "assistant", "content": "secret position", "observation_policy": policy_id("ground_truth")}]
    )


def test_truth_timeout_requires_fresh_measurement(registry):
    preview = SimpleNamespace(
        acquire_motion=Mock(return_value="owner"),
        query_scene=lambda op, params, **kw: registry.query(op, params),
        navigate_relative=Mock(return_value={"ok": False, "settled": True, "reason": "motion_timeout"}),
    )
    driver = GenesisDriver(preview, observation_mode="ground_truth")
    driver.connect()
    old = driver.query_truth("robot_state", {})
    driver.navigate_relative(0.1)
    driver.query_truth("robot_state", {"snapshot_id": old["snapshot_id"]})
    with pytest.raises(RuntimeError, match="sim_get_robot_state"):
        driver.require_motion()
    driver.query_truth("robot_state", {})
    driver.require_motion()
    assert not driver.cancelled.is_set()


def test_scene_change_during_agent_turn_is_terminal(registry):
    driver = GenesisDriver(
        SimpleNamespace(query_scene=lambda op, params, **kw: registry.query(op, params)),
        actions_enabled=False,
        observation_mode="ground_truth",
    )
    driver.query_truth("robot_state", {})
    registry.runtime.cycle += 1
    with pytest.raises(RuntimeError, match="Scene reset"):
        driver.query_truth("robot_state", {})
    assert driver.cancelled.is_set()


def test_authored_units_axis_duplicate_names_and_stable_ids(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    source = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    for path, x in (("/World/one/chair", 100), ("/World/two/chair", 300)):
        model = UsdGeom.Xform.Define(stage, path)
        model.GetPrim().SetMetadata("kind", "component")
        model.AddTranslateOp().Set(Gf.Vec3d(x, 200, 300))
        triangle = UsdGeom.Mesh.Define(stage, path + "/mesh")
        triangle.CreatePointsAttr([(0, 0, 0), (100, 0, 0), (0, 100, 0)])
        triangle.CreateFaceVertexCountsAttr([3])
        triangle.CreateFaceVertexIndicesAttr([0, 1, 2])
        triangle.CreateExtentAttr([(0, 0, 0), (100, 100, 0)])
    stage.GetRootLayer().Save()
    runtime = SimpleNamespace(
        source=source,
        robot=None,
        bodies=[],
        indoor=None,
        motion=None,
        cycle=0,
        steps=1,
        settings=SimpleNamespace(scene_id="chairs", dt=0.01),
        info={"instance_id": "a"},
    )
    registry = EntityRegistry(runtime)
    items = list(registry.entities.values())
    assert len(items) == 2 and items[0]["name"] == items[1]["name"] == "chair"
    assert items[0]["entity_id"] != items[1]["entity_id"]
    assert items[0]["position_m"] == [1, -3, 2]
    assert items[0]["bounds_m"] == [[1, -3, 2], [2, -3, 3]]
    assert list(EntityRegistry(runtime).entities) == list(registry.entities)


async def test_chat_switch_resets_history_without_changing_physics(monkeypatch):
    from jiuwensymbiosis.agent.config import ModelSpec
    from jiuwensymbiosis.simulators.genesis.chat import ChatError, GenesisChat
    from tests.unit_tests.simulators.genesis.test_agent_chat import mobile_ready

    preview = Mock()
    runner = Mock(return_value={"reply": "truth", "actions": []})
    monkeypatch.setattr("jiuwensymbiosis.simulators.genesis.agent.run_agent_query", runner)
    chat = GenesisChat(mobile_ready, preview=preview)
    chat.observation_mode = "ground_truth"
    await chat.send("查询", ModelSpec())
    assert runner.call_args.kwargs["observation_mode"] == "ground_truth"
    assert not runner.call_args.kwargs["actions_enabled"]
    assert chat._history
    chat.observation_mode = "visual"
    assert not chat._history and chat.last_execution is None
    preview.configure.assert_not_called()
    chat._pending = object()
    with pytest.raises(ChatError, match="等待"):
        chat.observation_mode = "ground_truth"
