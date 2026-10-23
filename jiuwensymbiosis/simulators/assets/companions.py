"""扫描时关联 USD、URDF 与材质包；运行时只使用索引保存的关联与摘要。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import Asset, AssetError, ResolvedAsset, local_file


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AssetError(f"无法读取资产清单 {path}：{exc}") from exc
    if not isinstance(data, dict):
        raise AssetError(f"资产清单必须是 JSON 对象：{path}")
    return data


def _reference(manifest: Path, value: Any, root: Path) -> Path:
    """清单可引用相邻源文件，绝对路径和上级路径均须留在资产根内。"""
    if not isinstance(value, str) or not value:
        raise AssetError(f"无效的本地清单引用：{manifest}: {value!r}")
    if ":" in value or "\\" in value:
        raise AssetError(f"无效的本地清单引用：{manifest}: {value!r}")
    path = (manifest.parent / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise AssetError(f"清单引用文件不存在或离开根目录：{manifest}: {value}")
    return path


def validate_material_bundle(resolved: ResolvedAsset) -> Path | None:
    """按 SQLite 内的材质清单快照校验实际文件，不重新解析源清单。"""
    paths = resolved.paths
    if "nyx_urdf" not in paths:
        return None
    manifest = resolved.asset.material_bundle
    if manifest.get("physics_urdf_sha256") != file_sha256(paths["genesis_urdf"]):
        raise AssetError("材质清单与物理 URDF 不匹配，请重新转换材质")
    directory = paths["material_manifest"].parent
    render = local_file(directory, manifest.get("render_urdf", ""))
    if render != paths["nyx_urdf"]:
        raise AssetError("索引的 nyx_urdf 与材质清单快照不一致")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, dict) or render.relative_to(directory).as_posix() not in hashes:
        raise AssetError("材质清单缺少渲染 URDF 文件摘要")
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise AssetError("材质清单的文件路径和摘要必须是字符串")
        file = local_file(directory, relative)
        if not file.is_file() or file_sha256(file) != expected:
            raise AssetError(f"材质文件缺失或摘要不匹配：{relative}")
    return render


def attach_companions(asset: Asset, root: Path) -> Asset:
    """优先读取匹配源 USD 的导出清单，否则只接受同名 URDF；歧义明确失败。"""
    if asset.kind != "robot":
        return asset
    source = local_file(root, asset.representations["source"])
    exports = []
    other_exports = set()
    for directory in (source.parent, source.parent / "urdf"):
        candidate = directory / "export_manifest.json"
        if not candidate.exists() and not candidate.is_symlink():
            continue
        manifest_path = local_file(root, candidate.relative_to(root).as_posix())
        data = _manifest(manifest_path)
        original = _reference(manifest_path, data.get("source"), root)
        urdf = _reference(manifest_path, data.get("urdf"), root)
        if urdf.suffix.lower() != ".urdf":
            raise AssetError(f"导出清单的 urdf 必须引用 URDF：{manifest_path}")
        if original == source:
            exports.append((manifest_path, urdf, data))
        else:
            other_exports.add(urdf)
    if len(exports) > 1:
        raise AssetError(f"同一 USD 存在多个导出清单，请清理重复导出：{source}")
    representations = dict(asset.representations)
    if exports:
        manifest_path, urdf, export = exports[0]
        representations["export_manifest"] = manifest_path.relative_to(root).as_posix()
        meshes_are_zup = export.get("file_meshes_are_zup", True)
    else:
        candidates = set()
        for directory in (source.parent, source.parent / "urdf"):
            candidate = directory / f"{source.stem}.urdf"
            if candidate.exists() or candidate.is_symlink():
                path = local_file(root, candidate.relative_to(root).as_posix())
                if not path.is_file():
                    raise AssetError(f"URDF 入口不是文件：{candidate}")
                candidates.add(path)
        if not candidates:
            return asset
        if len(candidates) > 1:
            raise AssetError(f"同一 USD 存在多个同名 URDF，需要唯一的导出清单：{source}")
        urdf = candidates.pop()
        meshes_are_zup = False
    if urdf in other_exports:
        raise AssetError(f"导出清单声明该 URDF 属于不同的源 USD：{urdf}")
    if not isinstance(meshes_are_zup, bool):
        raise AssetError(f"file_meshes_are_zup 必须为布尔值：{source}")
    representations["genesis_urdf"] = urdf.relative_to(root).as_posix()
    bundle = {}
    material_path = urdf.parent / "materials/manifest.json"
    if material_path.exists() or material_path.is_symlink():
        material_path = local_file(root, material_path.relative_to(root).as_posix())
        data = _manifest(material_path)
        if "source" in data and _reference(material_path, data["source"], root) != source:
            raise AssetError(f"材质清单引用了不同的源 USD：{material_path}")
        render_reference = data.get("render_urdf")
        if not isinstance(render_reference, str) or Path(render_reference).suffix.lower() != ".urdf":
            raise AssetError(f"材质清单缺少 render_urdf：{material_path}")
        render = local_file(material_path.parent, render_reference)
        representations.update(
            material_manifest=material_path.relative_to(root).as_posix(),
            nyx_urdf=render.relative_to(root).as_posix(),
        )
        bundle = {key: data.get(key) for key in ("physics_urdf_sha256", "render_urdf", "files_sha256")}
    elif any((urdf.parent / "materials").glob("*.urdf")):
        raise AssetError(f"渲染 URDF 缺少 materials/manifest.json：{urdf}")
    result = replace(
        asset,
        representations=representations,
        preview_mode="urdf",
        options={"file_meshes_are_zup": meshes_are_zup},
        material_bundle=bundle,
    )
    paths = {role: local_file(root, relative) for role, relative in representations.items()}
    validate_material_bundle(ResolvedAsset(result, paths, root))
    return result
