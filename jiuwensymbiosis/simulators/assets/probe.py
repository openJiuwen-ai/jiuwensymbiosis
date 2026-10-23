"""按需探测文件引用与 USD 元数据，证据级别与仿真运行验收明确分开。"""

from __future__ import annotations

import json
import shlex
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .companions import validate_material_bundle
from .index import AssetLibrary
from .models import AssetError


def _dependency(path: Path, reference: str, root: Path) -> Path:
    """URDF/OBJ/MTL 可向上引用，但必须留在该资产库内；首版不解析 package://。"""
    if not reference or ":" in reference or Path(reference).is_absolute():
        raise AssetError(f"仅支持本地相对文件引用：{path}: {reference}")
    target = (path.parent / reference).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise AssetError(f"依赖缺失或越界：{path}: {reference}")
    return target


def probe_urdf(path: Path, root: Path) -> dict[str, Any]:
    """检查 URDF → OBJ/MTL 或 glTF/GLB → 图片引用，不导入 Genesis。"""
    try:
        return _probe_urdf(path, root)
    except UnicodeError as exc:
        raise AssetError(f"资产文本不是有效 UTF-8：{path}：{exc}") from exc


def _probe_urdf(path: Path, root: Path) -> dict[str, Any]:
    xml = ET.parse(path).getroot()
    if xml.tag != "robot":
        raise AssetError(f"不是 URDF robot：{path}")
    dependencies = set()
    for node in [*xml.findall(".//mesh"), *xml.findall(".//texture")]:
        dependencies.add(_dependency(path, node.get("filename", ""), root))
    materials = set()
    for mesh in sorted(dependencies):
        if mesh.suffix.lower() == ".obj":
            with mesh.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip().startswith("mtllib "):
                        for reference in shlex.split(line)[1:]:
                            materials.add(_dependency(mesh, reference, root))
        elif mesh.suffix.lower() in {".gltf", ".glb"}:
            if mesh.suffix.lower() == ".gltf":
                content = json.loads(mesh.read_text(encoding="utf-8"))
            else:
                with mesh.open("rb") as stream:
                    header = stream.read(20)
                    if len(header) != 20:
                        raise AssetError(f"GLB 文件头不完整：{mesh}")
                    magic, version, length, size, chunk = struct.unpack("<4sIIII", header)
                    invalid_header = magic != b"glTF" or version != 2
                    if invalid_header or chunk != 0x4E4F534A or size > length - 20:
                        raise AssetError(f"无效的 GLB JSON chunk：{mesh}")
                    content = json.loads(stream.read(size))
            if not isinstance(content, dict):
                raise AssetError(f"glTF/GLB JSON 顶层必须是对象：{mesh}")
            for item in [*content.get("buffers", []), *content.get("images", [])]:
                reference = item.get("uri")
                if reference and not reference.startswith("data:"):
                    materials.add(_dependency(mesh, reference, root))
    dependencies.update(materials)
    for material in sorted(materials):
        if material.suffix.lower() != ".mtl":
            continue
        for line in material.read_text(encoding="utf-8").splitlines():
            tokens = shlex.split(line, comments=True)
            if tokens and (tokens[0].lower().startswith("map_") or tokens[0].lower() in {"bump", "disp", "decal"}):
                dependencies.add(_dependency(material, tokens[-1], root))
    return {
        "links": len(xml.findall("link")),
        "joints": len(xml.findall("joint")),
        "dependencies": [path.relative_to(root).as_posix() for path in sorted(dependencies)],
        "validation": "local_file_references",  # not geometry, kinematics, contacts or image comparison
    }


def probe_asset(library: AssetLibrary, identifier: str, *, usd: bool = False) -> dict[str, Any]:
    resolved = library.resolve(identifier)
    root = resolved.root
    report: dict[str, Any] = {
        "id": resolved.asset.id,
        "validation": "entry_files_only",
        "runtime_validated": False,
        "files": {role: str(path) for role, path in resolved.paths.items()},
    }
    urdfs = {role: probe_urdf(path, root) for role, path in resolved.paths.items() if role.endswith("urdf")}
    if urdfs:
        report["urdf"] = urdfs
        report["validation"] = "local_urdf_references"
    if validate_material_bundle(resolved):
        report["material_bundle"] = "sha256_verified"
    if usd:
        try:
            from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils
        except ImportError as exc:
            raise AssetError("USD 探测需安装 usd-core；普通构建/查询不需要此依赖") from exc
        source = resolved.paths["source"]
        stage = Usd.Stage.Open(str(source))
        if stage is None:
            raise AssetError(f"无法打开 USD：{source}")
        _, _, unresolved = UsdUtils.ComputeAllDependencies(str(source))
        prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
        report["usd"] = {
            "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "meshes": sum(p.IsA(UsdGeom.Mesh) for p in prims),
            "articulation_roots": sum(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in prims),
            "unresolved_dependencies": list(unresolved),
            "dependency_status": "incomplete" if unresolved else "resolved",
        }
        report["validation"] = "usd_inspected"
    return report
