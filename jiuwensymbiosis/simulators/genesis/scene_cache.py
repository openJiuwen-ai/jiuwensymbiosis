"""Persistent, content-checked room preparation; no robot or built GPU world is cached.

Imported both by the isolated file worker and by CPU tests. SDK imports stay lazy.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("jiuwensymbiosis.simulators.genesis.scene_cache")
_SCHEMA = 1


def cache_root() -> Path:
    """Resolve the host/worker shared cache location without creating it."""
    configured = os.environ.get("JIUWENSYMBIOSIS_GENESIS_SCENE_CACHE")
    root = (
        Path(configured).expanduser()
        if configured
        else Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "jiuwensymbiosis/genesis/scenes"
    )
    if not root.is_absolute():
        raise ValueError("Genesis scene cache must be an absolute directory path")
    return root.resolve()


def _digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _dependencies(source: Path) -> dict:
    from pxr import Sdf, UsdUtils

    # USD layers can survive world replacement in a resident process. Refresh
    # them before hashing AND parsing, otherwise new bytes can name stale data.
    root = Sdf.Layer.FindOrOpen(str(source))
    if root is None:
        raise ValueError(f"Cannot open scene: {source}")
    root.Reload(True)
    layers, _, _ = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(source)))
    for layer in layers:
        layer.Reload(True)
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(source)))
    paths = {Path(layer.realPath).resolve() for layer in layers if layer.realPath}
    paths.update(Path(asset).resolve() for asset in assets if Path(asset).is_file())
    paths.add(source)
    return {"files": {str(p): _digest(p) for p in sorted(paths)}, "unresolved": sorted(map(str, unresolved))}


def _versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("genesis-world", "quadrants", "coacd", "trimesh", "usd-core", "numpy", "scipy")
    }


def _indoor_module():
    if __package__:
        from jiuwensymbiosis.simulators.genesis import indoor
    else:
        import indoor
    return indoor


@dataclass
class PreparedRoom:
    """An unbuilt scene and its original room geometry/contact attribution."""

    scene: Any
    indoor: Any
    cache: dict[str, Any]


class RoomSceneCache:
    """Prepare a room once; validate every hit and atomically replace corrupt entries."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or cache_root()).resolve()

    def prepare(
        self,
        source: Path,
        options: dict,
        directory: Path,
        *,
        phase: Callable[[str], AbstractContextManager] = lambda _: nullcontext(),
        progress: Callable[[int, int], None] | None = None,
    ) -> PreparedRoom:
        """Return only fixed room entities, before any robot/objects or scene.build().

        Source/config/SDK changes create a new content key. An incomplete or
        corrupt hit is rejected and rebuilt under the same per-key process lock.
        IO or source errors propagate rather than producing an unchecked hit.
        """
        import genesis as gs

        source = source.resolve(strict=True)
        indoor = _indoor_module()
        with phase("scene_cache_lookup"):
            dependencies = _dependencies(source)
            identity = {
                "schema": _SCHEMA,
                "source": str(source),
                "dependencies": dependencies,
                "options": options,
                "versions": _versions(),
                "processors": {p.name: _digest(p) for p in (Path(__file__), Path(indoor.__file__))},
            }
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        entry = self.root / key
        with (self.root / f"{key}.lock").open("a") as lock:
            with phase("scene_cache_wait"):
                fcntl.flock(lock, fcntl.LOCK_EX)
            reason = "missing"
            if entry.exists():
                with phase("scene_cache_load"):
                    scene = None
                    try:
                        manifest = json.loads((entry / "manifest.json").read_text())
                        if manifest["identity"] != identity:
                            raise ValueError("Scene cache identity mismatch")
                        for filename in ("room.gscene", "meshes.npz"):
                            if _digest(entry / filename) != manifest["files"][filename]:
                                raise ValueError(f"Scene cache checksum mismatch: {filename}")
                        meshes = self._read_meshes(entry, manifest["paths"])
                        scene = gs.Scene.load(entry / "room.gscene", show_viewer=False)
                        room = indoor.IndoorCollision.from_prepared(scene, meshes)
                        return PreparedRoom(scene, room, {"state": "hit", "key": key, "path": str(entry)})
                    except Exception as exc:
                        if scene is not None:
                            scene.destroy()
                        reason = f"invalid: {type(exc).__name__}"
                        logger.warning("Rejecting scene cache %s: %s", key, exc)
                        shutil.rmtree(entry)
            # TemporaryDirectory removes partial exports on error or interruption.
            with tempfile.TemporaryDirectory(prefix=f".{key}-", dir=self.root) as staging:
                stage = Path(staging)
                scene = gs.Scene(
                    sim_options=gs.options.SimOptions(**options["sim"]),
                    rigid_options=gs.options.RigidOptions(**options["rigid"]),
                    show_viewer=False,
                )
                try:
                    with phase("room_mesh_read"):
                        meshes = indoor.read_room_meshes(source)
                    with phase("collision_entities"):
                        room = indoor.IndoorCollision(scene, meshes, directory, progress=progress)
                    with phase("scene_cache_export"):
                        scene.export(stage / "room.gscene")
                        self._write_meshes(stage, meshes)
                        if _dependencies(source) != dependencies:
                            raise RuntimeError("Scene changed during cache preparation; retry loading the scene")
                        manifest = {
                            "identity": identity,
                            "paths": [mesh.path for mesh in meshes],
                            "files": {name: _digest(stage / name) for name in ("room.gscene", "meshes.npz")},
                        }
                        (stage / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
                        # Readers hold the same key lock; the entry is published only when complete.
                        os.replace(stage, entry)
                except BaseException:
                    scene.destroy()
                    raise
            return PreparedRoom(scene, room, {"state": "built", "reason": reason, "key": key, "path": str(entry)})

    @staticmethod
    def _write_meshes(directory: Path, meshes: list) -> None:
        import numpy as np

        arrays = {
            f"{kind}{i}": getattr(item.mesh, field)
            for i, item in enumerate(meshes)
            for kind, field in (("v", "vertices"), ("f", "faces"))
        }
        np.savez_compressed(directory / "meshes.npz", **arrays)

    @staticmethod
    def _read_meshes(directory: Path, paths: list[str]) -> list:
        import numpy as np
        import trimesh

        indoor = _indoor_module()
        with np.load(directory / "meshes.npz", allow_pickle=False) as arrays:
            return [
                indoor.RoomMesh(path, trimesh.Trimesh(vertices=arrays[f"v{i}"], faces=arrays[f"f{i}"], process=False))
                for i, path in enumerate(paths)
            ]
