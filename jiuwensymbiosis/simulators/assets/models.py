"""资产身份、文件表示及查询结果；不包含仿真器运行能力。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote


class AssetError(ValueError):
    """资产配置、索引或文件不可用；调用方应显示原因。"""


def relative_path(value: str) -> str:
    """接受根目录内的规范 POSIX 相对路径，拒绝 URL 和目录穿越。"""
    path = PurePosixPath(value)
    invalid_syntax = not value or "\\" in value or ":" in value
    if invalid_syntax or path.is_absolute() or ".." in path.parts:
        raise AssetError(f"需要资产根目录内的相对路径：{value!r}")
    return path.as_posix()


def local_file(root: Path, relative: str) -> Path:
    """解析文件并检查符号链接不会离开配置的根目录。"""
    path = (root / relative_path(relative)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise AssetError(f"资产路径离开根目录：{relative}")
    return path


def asset_id(root_id: str, relative: str) -> str:
    """同一逻辑入口的 USD 后缀变体使用同一身份。"""
    stem = PurePosixPath(relative_path(relative)).with_suffix("").as_posix()
    return f"library://{root_id}/{quote(stem, safe='/_.-~')}"


@dataclass(frozen=True)
class Asset:
    """可检索的资产；路径均相对于 root_id，不将支持格式当作控制能力。"""

    id: str
    root_id: str
    name: str
    kind: str
    category: str
    representations: dict[str, str]
    preview_mode: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    material_bundle: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssetPage:
    total: int
    items: list[Asset]
    limit: int
    offset: int


@dataclass(frozen=True)
class ResolvedAsset:
    """加载前重新检查的文件路径；不声明物理或控制已经验收。"""

    asset: Asset
    paths: dict[str, Path]
    root: Path

    @property
    def preview_path(self) -> Path:
        return self.paths["genesis_urdf"] if self.asset.preview_mode == "urdf" else self.paths["source"]
