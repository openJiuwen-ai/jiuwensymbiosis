"""Genesis 只消费统一资产索引；动态演示仍由代码生成。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# worker.py 作为独立文件运行，不能触发核心包/openjiuwen 的全局导入钩子。
# 同一资产实现也可从 simulators 目录作为轻量包导入，不复制解析逻辑。
if TYPE_CHECKING or __package__:
    from jiuwensymbiosis.simulators.assets import AssetError, AssetLibrary, ResolvedAsset, validate_material_bundle
else:
    # Load this sibling package explicitly; never shadow an installed assets or
    # genesis package, or execute the core package's global import hooks.
    PACKAGE_NAME = "_jiuwensymbiosis_simulators"
    if PACKAGE_NAME not in sys.modules:
        entry = Path(__file__).resolve().parents[1] / "__init__.py"
        spec = importlib.util.spec_from_file_location(PACKAGE_NAME, entry)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load simulator assets from {entry}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for name in list(sys.modules):
                if name == PACKAGE_NAME or name.startswith(PACKAGE_NAME + "."):
                    del sys.modules[name]
            raise
    assets = importlib.import_module(PACKAGE_NAME + ".assets")
    AssetError = assets.AssetError
    AssetLibrary = assets.AssetLibrary
    ResolvedAsset = assets.ResolvedAsset
    validate_material_bundle = assets.validate_material_bundle

DEMO = "demo"
__all__ = ["DEMO", "resolve_scene", "scene_catalog", "scene_path", "validate_material_bundle"]


def scene_catalog() -> list[dict[str, str]]:
    """列出支持的预览候选；文件是否仍有效在加载前再次检查。"""
    entries = [{"id": DEMO, "label": "动态物理演示"}]
    library = AssetLibrary()
    for mode in ("urdf", "static_usd"):
        offset = 0
        while True:
            page = library.search(preview_mode=mode, limit=1000, offset=offset)
            entries.extend(
                {
                    "id": asset.id,
                    "preview_mode": mode,
                    "label": f"{asset.root_id} · {asset.name}" + (" · URDF 检视" if mode == "urdf" else ""),
                }
                for asset in page.items
            )
            offset += len(page.items)
            if offset >= page.total:
                break
    return entries


def resolve_scene(scene_id: str) -> ResolvedAsset:
    """只接受索引 ID/配置别名；不接受调用者直接提供路径或 URL。"""
    resolved = AssetLibrary().resolve(scene_id)
    if resolved.asset.preview_mode not in {"urdf", "static_usd"}:
        raise AssetError("该资产暂无 Genesis 预览；机器人需要 scanner 识别到关联的 URDF")
    return resolved


def scene_path(scene_id: str) -> Path:
    return resolve_scene(scene_id).preview_path
