"""Engine-neutral scene candidates from the shared local asset index."""

from __future__ import annotations

import sys
from pathlib import Path

if not __package__:
    import importlib.util

    package = "_jiuwensymbiosis_simulators"
    entry = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location(package, entry, submodule_search_locations=[str(entry.parent)])
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the simulator package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    __package__ = package + ".common"
    script_spec = importlib.util.spec_from_file_location(__package__ + ".scene_catalog", __file__)
    if script_spec is None:
        raise ImportError("Cannot identify the standalone simulator module")
    __spec__ = script_spec

from ..assets import AssetError, AssetLibrary, ResolvedAsset, validate_material_bundle
from .preview_backends import POLICY_MODES

DEMO = "demo"
__all__ = ["DEMO", "scene_catalog", "resolve_scene", "scene_path", "validate_material_bundle"]


def scene_catalog(backend: str | None = None) -> list[dict]:
    entries = [{"id": DEMO, "label": "动态物理演示", "backends": ["genesis", "mujoco"]}]
    library = AssetLibrary()
    if not library.index.is_file():
        return entries
    for mode in ("urdf", "static_usd", "mjcf", *POLICY_MODES):
        offset = 0
        while True:
            page = library.search(preview_mode=mode, limit=1000, offset=offset)
            entries.extend(
                {
                    "id": asset.id,
                    "preview_mode": mode,
                    "backends": ["mujoco"] if mode in ("mjcf", *POLICY_MODES) else ["genesis"],
                    "robot_mode": mode if mode in POLICY_MODES else "inspection",
                    "label": f"{asset.root_id} · {asset.name}" + (" · URDF 检视" if mode == "urdf" else ""),
                }
                for asset in page.items
            )
            offset += len(page.items)
            if offset >= page.total:
                break
    return [e for e in entries if backend is None or backend in e["backends"]]


def resolve_scene(scene_id: str, backend: str | None = None) -> ResolvedAsset:
    asset = AssetLibrary().resolve(scene_id)
    mode = asset.asset.preview_mode
    supported = "mujoco" if mode in ("mjcf", *POLICY_MODES) else "genesis"
    if mode not in ("urdf", "static_usd", "mjcf", *POLICY_MODES) or (backend and backend != supported):
        raise AssetError(f"场景不支持所选仿真后端：{backend or 'unknown'}")
    return asset


def scene_path(scene_id: str, backend: str | None = None):
    return resolve_scene(scene_id, backend).preview_path
