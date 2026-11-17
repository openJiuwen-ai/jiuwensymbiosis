"""Local MJCF file references, inspected without loading the physics engine."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .models import AssetError, local_file


def mjcf_files(root: Path, entry: str) -> dict[str, str]:
    """Resolve includes and file assets within the configured library root."""
    files = {"source": entry}
    seen: set[Path] = set()
    active: set[Path] = set()
    source = local_file(root, entry)

    def checked(path: Path) -> Path:
        path = path.resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise AssetError(f"MJCF 引用缺失或离开资产根目录：{path}")
        return path

    def visit(path: Path, inherited: dict[str, str], *, first: bool = False) -> None:
        path = checked(path)
        if path in active:
            raise AssetError("MJCF include 存在循环")
        if path in seen:
            return
        active.add(path)
        seen.add(path)
        try:
            xml = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise AssetError(f"无效 MJCF XML：{path}") from exc
        if first and xml.tag != "mujoco":
            raise AssetError("MJCF 入口必须以 mujoco 为根元素")
        compiler = xml.find("compiler")
        options = {**inherited, **(compiler.attrib if compiler is not None else {})}
        for node in xml.iter():
            name = node.get("file")
            if not name:
                continue
            if node.tag == "include":
                reference = checked(path.parent / name)
                visit(reference, options)
            else:
                folder = options.get("meshdir" if node.tag == "mesh" else "texturedir", options.get("assetdir", ""))
                reference = checked(source.parent / folder / name)
            if reference != source:
                relative = reference.relative_to(root.resolve()).as_posix()
                if relative not in files.values():
                    files[f"reference_{len(files)}"] = relative
        active.remove(path)

    visit(source, {}, first=True)
    return files
