"""Cache publication/invalidation contracts, without Genesis or a GPU."""

import copy
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis import indoor, scene_cache

OPTIONS = {"sim": {"dt": 1 / 120, "substeps": 4, "gravity": [0, 0, -9.81]}, "rigid": {"enable_self_collision": False}}


@pytest.fixture
def room_cache(monkeypatch, tmp_path):
    source = tmp_path / "room.usda"
    source.write_text("room-v1")
    dependency = tmp_path / "referenced.usd"
    dependency.write_text("geometry-v1")
    state = SimpleNamespace(reads=0, destroyed=0, versions={"genesis": "test"}, delay=0)
    mesh = SimpleNamespace(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), faces=np.array([[0, 1, 2]])
    )

    class Scene:
        def __init__(self, **kwargs):
            self.entities = []
            self.is_built = False

        def export(self, path):
            path.write_text(json.dumps({"count": len(self.entities)}))

        @classmethod
        def load(cls, path, **kwargs):
            scene = cls()
            scene.entities = [object() for _ in range(json.loads(path.read_text())["count"])]
            return scene

        def destroy(self):
            state.destroyed += 1

    def read_meshes(path):
        state.reads += 1
        time.sleep(state.delay)
        return [indoor.RoomMesh("/Room/Floor", mesh)]

    def build_room(self, scene, meshes, directory, *, progress=None):
        self.entities = [object() for _ in meshes]
        self.meshes = meshes
        self.paths = dict(zip(self.entities, [m.path for m in meshes], strict=True))
        scene.entities.extend(self.entities)
        if progress:
            progress(len(meshes), len(meshes))

    monkeypatch.setitem(
        sys.modules,
        "genesis",
        SimpleNamespace(Scene=Scene, options=SimpleNamespace(SimOptions=lambda **kw: kw, RigidOptions=lambda **kw: kw)),
    )
    monkeypatch.setattr(scene_cache, "_versions", lambda: state.versions.copy())
    monkeypatch.setattr(
        scene_cache,
        "_dependencies",
        lambda path: {"files": {str(p): scene_cache._digest(p) for p in [path, dependency]}, "unresolved": []},
    )
    monkeypatch.setattr(indoor, "read_room_meshes", read_meshes)
    monkeypatch.setattr(indoor.IndoorCollision, "__init__", build_room)
    # Geometry serialization is real numpy; the mesh reader's optional trimesh
    # dependency is replaced with its data constructor for hardware-free tests.
    monkeypatch.setitem(sys.modules, "trimesh", SimpleNamespace(Trimesh=lambda **kw: SimpleNamespace(**kw)))
    state.source, state.dependency, state.Scene = source, dependency, Scene
    state.cache = scene_cache.RoomSceneCache(tmp_path / "cache")
    state.prepare = lambda **kw: state.cache.prepare(source, copy.deepcopy(OPTIONS), tmp_path, **kw)
    return state


def test_room_hit_keeps_geometry_and_never_bakes_added_robots(room_cache):
    first = room_cache.prepare()
    first.scene.entities.append("robot-a")
    second = room_cache.prepare()
    assert first.cache["state"] == "built" and second.cache["state"] == "hit"
    assert first.cache["key"] == second.cache["key"]
    assert room_cache.reads == 1 and len(second.scene.entities) == 1
    assert not second.scene.is_built
    np.testing.assert_array_equal(first.indoor.meshes[0].mesh.vertices, second.indoor.meshes[0].mesh.vertices)
    np.testing.assert_array_equal(first.indoor.meshes[0].mesh.faces, second.indoor.meshes[0].mesh.faces)
    assert list(second.indoor.paths.values()) == ["/Room/Floor"]
    second.scene.entities.append("robot-b")
    assert len(room_cache.prepare().scene.entities) == 1


@pytest.mark.parametrize("changed", ["source", "dependency", "versions", "options"])
def test_scene_identity_changes_rebuild(room_cache, tmp_path, changed):
    before = room_cache.prepare()
    options = copy.deepcopy(OPTIONS)
    if changed in {"source", "dependency"}:
        getattr(room_cache, changed).write_text("changed-content")
    elif changed == "versions":
        room_cache.versions["genesis"] = "new-version"
    else:
        options["sim"]["dt"] = 1 / 240
    after = room_cache.cache.prepare(room_cache.source, options, tmp_path)
    assert after.cache["key"] != before.cache["key"]
    assert after.cache["state"] == "built" and room_cache.reads == 2


@pytest.mark.parametrize("filename", ["room.gscene", "meshes.npz", "manifest.json"])
def test_corrupt_entry_is_rejected_and_rebuilt(room_cache, filename):
    first = room_cache.prepare()
    from pathlib import Path

    (Path(first.cache["path"]) / filename).write_bytes(b"incomplete")
    rebuilt = room_cache.prepare()
    assert rebuilt.cache["state"] == "built" and rebuilt.cache["reason"].startswith("invalid:")
    assert room_cache.reads == 2
    assert room_cache.prepare().cache["state"] == "hit"


def test_failed_export_never_publishes_partial_entry(room_cache, monkeypatch):
    def failed_export(self, path):
        path.write_text("partial")
        raise RuntimeError("disk write failed")

    monkeypatch.setattr(room_cache.Scene, "export", failed_export)
    with pytest.raises(RuntimeError, match="disk write failed"):
        room_cache.prepare()
    assert not [p for p in room_cache.cache.root.iterdir() if p.is_dir()]
    assert room_cache.destroyed == 1


def test_change_during_build_rejects_stale_publication(room_cache, monkeypatch):
    original = room_cache.Scene.export

    def changing_export(self, path):
        original(self, path)
        room_cache.dependency.write_text("changed-during-build")

    monkeypatch.setattr(room_cache.Scene, "export", changing_export)
    with pytest.raises(RuntimeError, match="changed during"):
        room_cache.prepare()
    assert not [p for p in room_cache.cache.root.iterdir() if p.is_dir()]


def test_concurrent_preparation_publishes_once(room_cache):
    room_cache.delay = 0.05
    barrier = threading.Barrier(2)

    def prepare():
        barrier.wait()
        return room_cache.prepare()

    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(prepare) for _ in range(2)]
        results = [job.result(timeout=5) for job in jobs]
    assert sorted(r.cache["state"] for r in results) == ["built", "hit"]
    assert room_cache.reads == 1


def test_cache_progress_reports_completed_room_entities(room_cache):
    values = []
    room_cache.prepare(progress=lambda done, total: values.append((done, total)))
    assert values == [(1, 1)]


def test_relative_cache_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("JIUWENSYMBIOSIS_GENESIS_SCENE_CACHE", "relative-cache")
    with pytest.raises(ValueError, match="absolute"):
        scene_cache.cache_root()
