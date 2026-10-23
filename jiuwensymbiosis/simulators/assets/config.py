"""本仓库资产根和显式入口配置；数据库位置相对于配置文件解析。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

from .models import AssetError, relative_path

# The directory CLI imports assets without initializing the Agent package.
if TYPE_CHECKING or __package__ != "assets":
    from .._paths import simulator_root
else:
    from _paths import simulator_root

DEFAULT_CONFIG_PATH = simulator_root() / "configs/simulators/assets.yaml"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContentFamily(StrictModel):
    """目录树的一类入口；深度包含文件名，相对于该类的目录计算。"""

    kind: Literal["robot", "environment", "object", "sensor", "other"]
    min_depth: PositiveInt = 1
    preview_depth: PositiveInt = 2
    preview_depth_overrides: dict[str, PositiveInt] = Field(default_factory=dict)

    @field_validator("preview_depth_overrides")
    @classmethod
    def check_preview_dirs(cls, value: dict[str, int]) -> dict[str, int]:
        for directory in value:
            if relative_path(directory) == "." or len(Path(directory).parts) != 1:
                raise ValueError("预览深度覆盖必须指定一级子目录")
        return value


class RootSpec(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    path: str = Field(min_length=1)
    scanner: Literal["content_tree", "geniesim", "spiritai"]
    content_path: str = "."
    exclude: list[str] = Field(default_factory=list)
    families: dict[str, ContentFamily] = Field(default_factory=dict)

    @field_validator("content_path")
    @classmethod
    def check_content(cls, value: str) -> str:
        return relative_path(value)

    @model_validator(mode="after")
    def check_families(self) -> RootSpec:
        if self.scanner == "content_tree" and not self.families:
            raise ValueError("content_tree 必须声明 families 目录与分类")
        if self.scanner != "content_tree" and self.families:
            raise ValueError("families 只用于 content_tree 扫描器")
        for directory in self.families:
            relative_path(directory)
        return self


class LibrarySpec(StrictModel):
    version: Literal[1] = 1
    index: str
    roots: list[RootSpec] = Field(min_length=1)

    @field_validator("index")
    @classmethod
    def check_index(cls, value: str) -> str:
        if Path(value).suffix.lower() not in {".sqlite", ".db"}:
            raise ValueError("生成索引必须使用 .sqlite 或 .db 后缀")
        return value

    @model_validator(mode="after")
    def check_ids(self) -> LibrarySpec:
        roots = [root.id for root in self.roots]
        if len(set(roots)) != len(roots):
            raise ValueError("资产 root id 重复")
        return self


@dataclass(frozen=True)
class AssetConfig:
    source: Path
    index: Path
    spec: LibrarySpec
    roots: dict[str, Path]
    digest: str


def default_config_path() -> Path:
    """Resolve the local configuration under the selected simulator data root."""
    return DEFAULT_CONFIG_PATH


def initialize_config(path: Path | str | None = None) -> Path:
    """Copy the packaged example to a user-selected path without overwriting it."""
    source = Path(path or default_config_path()).expanduser().resolve()
    template = Path(__file__).with_name("data") / "assets.example.yaml"
    contents = template.read_bytes()
    source.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("xb") as output:
            output.write(contents)
    except FileExistsError:
        raise AssetError(f"资产配置已存在，未覆盖：{source}") from None
    return source


def load_config(path: Path | str | None = None) -> AssetConfig:
    """仅在构建时加载；目录位置与扫描规则的变化在重建后生效。"""
    source = Path(path or default_config_path()).expanduser().resolve()
    if path is None and not source.is_file():
        raise AssetError(
            f"未找到本机资产配置 {source}；请运行 python -m jiuwensymbiosis.simulators.assets init-config，再编辑资产路径"
        )
    try:
        spec = LibrarySpec.model_validate(yaml.safe_load(source.read_text(encoding="utf-8")))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise AssetError(f"无法读取资产配置 {source}：{exc}") from exc

    def absolute(value: str) -> Path:
        candidate = Path(value).expanduser()
        return (source.parent / candidate).resolve()

    roots = {root.id: absolute(root.path) for root in spec.roots}
    # 仅用于构建溯源；运行时不读取 YAML，也不将当前配置与该摘要比较。
    identity: dict[str, Any] = spec.model_dump(exclude={"index"})
    for root in identity["roots"]:
        root["path"] = str(roots[root["id"]])
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return AssetConfig(source, absolute(spec.index), spec, roots, digest)
