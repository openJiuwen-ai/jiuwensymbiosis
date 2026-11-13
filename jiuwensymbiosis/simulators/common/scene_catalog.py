"""Engine-neutral scene candidates from the shared local asset index."""

from __future__ import annotations

import sys
from pathlib import Path

if not __package__:
    import importlib.util

    _PACKAGE_NAME = "_jiuwensymbiosis_simulators"
    _PACKAGE_ENTRY = Path(__file__).resolve().parents[1] / "__init__.py"
    _PACKAGE_SPEC = importlib.util.spec_from_file_location(
        _PACKAGE_NAME, _PACKAGE_ENTRY, submodule_search_locations=[str(_PACKAGE_ENTRY.parent)]
    )
    if _PACKAGE_SPEC is None or _PACKAGE_SPEC.loader is None:
        raise ImportError("Cannot load the simulator package")
    _PACKAGE_MODULE = importlib.util.module_from_spec(_PACKAGE_SPEC)
    sys.modules[_PACKAGE_NAME] = _PACKAGE_MODULE
    _PACKAGE_SPEC.loader.exec_module(_PACKAGE_MODULE)
    _ENTRY_SPEC = importlib.util.spec_from_file_location(_PACKAGE_NAME + ".common.scene_catalog", __file__)
    if _ENTRY_SPEC is None:
        raise ImportError("Cannot identify the standalone simulator module")
    # Seed the file entry's import context from its loader, without core import hooks.
    globals().update({"__package__": _ENTRY_SPEC.parent, "__spec__": _ENTRY_SPEC})

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
