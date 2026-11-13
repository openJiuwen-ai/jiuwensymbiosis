"""三套本地资产的入口发现规则；不导入 USD/仿真 SDK，也不执行资产附带脚本。

参考 RoboSim 的入口过滤和目录约定，独立实现。扫描只给出文件级候选，
不从文件名推断机器人控制能力，不能代替加载、材质或物理验收。
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml

from .companions import attach_companions
from .config import RootSpec
from .models import Asset, AssetError, asset_id, local_file

USD_SUFFIXES = {".usd", ".usda", ".usdc"}
SUPPORT_DIRS = {
    "materials",
    "material",
    "textures",
    "texture",
    "meshes",
    "mesh",
    "looks",
    "geometry",
    "geometries",
    "collision",
    "collisions",
    "visual",
    "visuals",
    "physics",
    "configuration",
    "configurations",
    "animations",
    "skeletons",
}
SUPPORT_MARKERS = (
    "_collision",
    "_collisions",
    "_visual",
    "_visuals",
    "_physics",
    "_material",
    "_materials",
    "instanceable_mesh",
    "_robot_schema",
)
ROBOT_SUPPORT_DIRS = {"payload", "payloads", "props", "parts", "detailedprops", "highresprops", "legacy"}


@dataclass
class ScanResult:
    assets: list[Asset] = field(default_factory=list)
    diagnostics: list[dict[str, str]] = field(default_factory=list)


def _walk_usd(directory: Path) -> Iterator[Path]:
    if not directory.is_dir():
        return

    def fail(error: OSError) -> None:
        raise error

    for base, dirs, files in os.walk(directory, followlinks=False, onerror=fail):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d.lower() not in SUPPORT_DIRS)
        for name in sorted(files):
            path = Path(base, name)
            if not name.startswith(".") and path.suffix.lower() in USD_SUFFIXES:
                yield path


def _usd_variants(directory: Path) -> dict[str, dict[str, Path]]:
    variants: dict[str, dict[str, Path]] = {}
    if not directory.is_dir():
        return variants
    for path in sorted(directory.iterdir()):
        suffix = path.suffix.lower()
        if suffix not in USD_SUFFIXES or not path.is_file():
            continue
        entries = variants.setdefault(path.stem, {})
        # Prefer the canonical lowercase spelling if duplicate suffix variants exist.
        if suffix not in entries or path.suffix == suffix:
            entries[suffix] = path
    return variants


def _select(stem: str, variants: dict[str, Path]) -> Path | None:
    """Aligned 的 Z-up wrapper 优先；否则保留原始 USD，避免误选同名中间层。"""
    if stem == "Aligned" and ".usd" in variants and ".usda" in variants:
        with variants[".usda"].open(encoding="utf-8", errors="replace") as stream:
            header = stream.read(65536)
        reference = r"@(?:\./)?" + re.escape(variants[".usd"].name) + "@"
        canonical = re.search(r'upAxis\s*=\s*"Z"', header) and re.search(reference, header)
        return variants[".usda"] if canonical else variants[".usd"]
    return next((variants[s] for s in (".usda", ".usd", ".usdc") if s in variants), None)


def scan_root(spec: RootSpec, root: Path) -> ScanResult:
    """读取一个已配置的库；缺失根目录/内容目录会使整个构建失败。"""
    content = local_file(root, spec.content_path)
    if not content.is_dir():
        raise AssetError(f"资产根目录不存在：{spec.id}: {content}")
    if spec.scanner == "mjcf":
        from .mjcf import mjcf_files

        return ScanResult(
            assets=[
                Asset(
                    id=asset_id(spec.id, entry),
                    root_id=spec.id,
                    name=Path(entry).stem,
                    kind="environment",
                    category="mjcf",
                    preview_mode="mjcf",
                    representations=mjcf_files(root, entry),
                )
                for entry in spec.entries
            ]
        )
    if spec.scanner == "microduck":
        # Official microduck_rl layout. Indexing MJCF does not grant control capabilities.
        model_dir = "src/mjlab_microduck/robot/microduck"
        microduck_paths = {
            "source": f"{model_dir}/robot_walk.xml",
            "scene": f"{model_dir}/scene_walk.xml",
            "code_license": "LICENSE",
            "model_license": "LICENSE-MODELS.md",
        }
        for microduck_file in microduck_paths.values():
            if not local_file(root, microduck_file).is_file():
                raise AssetError(f"官方 Microduck 资产缺失：{microduck_file}")
        policy_paths = {
            "policy_manifest": "policies/manifest.json",
            "walking_policy": "policies/velstand.onnx",
            "policy_reference": "scripts/infer_policy.py",
            "runtime_scene": f"{model_dir}/scene.xml",
        }
        runnable = local_file(root, policy_paths["walking_policy"]).is_file()
        if runnable:
            for role, policy_relative in policy_paths.items():
                if not local_file(root, policy_relative).is_file():
                    raise AssetError(f"Microduck 策略配套文件缺失：{policy_relative}")
                microduck_paths[role] = policy_relative
        return ScanResult(
            assets=[
                Asset(
                    id=asset_id(spec.id, microduck_paths["source"]),
                    root_id=spec.id,
                    name="Microduck",
                    kind="robot",
                    category="biped",
                    representations=microduck_paths,
                    preview_mode="microduck" if runnable else None,
                )
            ]
        )
    if spec.scanner == "jumper":
        # Explicit upstream layout, not a guess based on a robot's filename.
        jumper_paths = {
            "source": "assets/jumper/jumper.xml",
            "policy_bundle": "out/bundle_example/jumper/bundle.json",
            "motor_config": "assets/jumper/motor/motor_config.yaml",
        }
        for jumper_file in jumper_paths.values():
            if not local_file(root, jumper_file).is_file():
                raise AssetError(f"官方 Jumper 资产缺失：{jumper_file}")
        return ScanResult(
            assets=[
                Asset(
                    id=asset_id(spec.id, jumper_paths["source"]),
                    root_id=spec.id,
                    name="Jumper",
                    kind="robot",
                    category="hexapod",
                    representations=jumper_paths,
                    preview_mode="jumper",
                )
            ]
        )
    result = ScanResult()
    selected: dict[str, Asset] = {}
    # Scope the cache to this build: enumerate each directory once, never across rebuilds.
    variants_in = cache(_usd_variants)

    def select(directory: Path, stem: str) -> Path | None:
        return _select(stem, variants_in(directory).get(stem, {}))

    def note(path: Path, reason: str) -> None:
        result.diagnostics.append({"root_id": spec.id, "path": path.relative_to(root).as_posix(), "reason": reason})

    def add(path: Path, kind: str, category: str, name: str | None = None) -> None:
        relative = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatchcase(relative, pattern) for pattern in spec.exclude):
            return
        try:
            resolved = local_file(root, relative)
        except AssetError as exc:
            note(path, str(exc))
            return
        if not resolved.is_file():
            note(path, "入口文件不存在")
            return
        identifier = asset_id(spec.id, relative)
        selected[identifier] = Asset(
            id=identifier,
            root_id=spec.id,
            name=name or path.stem,
            kind=kind,
            category=category,
            representations={"source": relative},
            preview_mode="static_usd" if kind == "environment" else None,
        )

    if spec.scanner == "content_tree":
        seen: set[Path] = set()
        for family, rule in spec.families.items():
            kind = rule.kind
            folder = local_file(content, family)
            for path in _walk_usd(folder):
                relative = path.relative_to(folder)
                if any(marker in path.stem.lower() for marker in SUPPORT_MARKERS):
                    continue
                if len(relative.parts) < rule.min_depth:
                    continue
                if kind == "robot" and any(part.lower() in ROBOT_SUPPORT_DIRS for part in relative.parts[:-1]):
                    continue
                key = path.with_suffix("")
                if key in seen:
                    continue
                seen.add(key)
                chosen = select(path.parent, path.stem)
                if chosen:
                    entry_kind = kind
                    if kind == "environment":
                        # Environment libraries also contain thousands of props/layers.
                        # Preview depth comes from this library's directory conventions.
                        depth = rule.preview_depth_overrides.get(relative.parts[0], rule.preview_depth)
                        if "props" in {part.lower() for part in relative.parts[:-1]}:
                            entry_kind = "object"
                        elif len(relative.parts) > depth:
                            entry_kind = "other"
                    add(chosen, entry_kind, "/".join((family, *relative.parts[:-1])))
    elif spec.scanner == "geniesim":
        robots = content / "robot"
        if robots.is_dir():
            for directory in sorted(robots.iterdir()):
                if directory.is_dir() and not directory.is_symlink() and not directory.name.startswith("."):
                    candidate = select(directory, "robot")
                    if candidate:
                        add(candidate, "robot", "robot", directory.name)
        backgrounds = content / "background"
        if backgrounds.is_dir():
            for directory in sorted(backgrounds.glob("*/*")):
                if (
                    directory.is_dir()
                    and not directory.is_symlink()
                    and not any(p.startswith(".") for p in directory.relative_to(backgrounds).parts)
                ):
                    candidate = select(directory, "background")
                    if candidate:
                        add(
                            candidate,
                            "environment",
                            directory.parent.name,
                            directory.relative_to(backgrounds).as_posix(),
                        )
        for family in ("objects", "real2sim/object_assets"):
            groups: dict[Path, list[Path]] = {}
            for path in _walk_usd(content / family):
                groups.setdefault(path.parent, []).append(path)
            for directory, paths in sorted(groups.items()):
                aligned = select(directory, "Aligned")
                stems = {p.stem for p in paths}
                if aligned:
                    chosen = aligned
                elif len(stems) == 1:
                    chosen = select(directory, paths[0].stem)
                else:
                    note(directory, "多个 USD 入口，缺少唯一入口或规范的 Aligned 文件")
                    continue
                if chosen:
                    add(chosen, "object", directory.relative_to(content).parent.as_posix(), directory.name)
        environment = content / "real2sim/environment_assets"
        if environment.is_dir():
            for directory in sorted(environment.iterdir()):
                if directory.is_dir() and not directory.is_symlink() and not directory.name.startswith("."):
                    candidate = select(directory, directory.name)
                    if candidate:
                        add(candidate, "environment", "real2sim", directory.name)
    else:  # SpiritAI: robot config is authoritative; ambiguous variants are not guessed.
        for directory in sorted(content.iterdir()):
            if not directory.is_dir() or directory.is_symlink() or directory.name.startswith("."):
                continue
            config = directory / "config.yaml"
            if config.is_file():
                try:
                    data = yaml.safe_load(config.read_text(encoding="utf-8"))
                    if not isinstance(data, dict) or not isinstance(data.get("asset_path"), str):
                        raise AssetError("机器人 config.yaml 缺少 asset_path")
                    path = local_file(directory, data["asset_path"])
                    if path.suffix.lower() not in USD_SUFFIXES or not path.is_file():
                        raise AssetError("config.yaml 指定的 USD 入口不可用")
                except (ValueError, OSError, yaml.YAMLError) as exc:
                    note(directory, str(exc))
                    continue
                add(path, "robot", directory.name, directory.name)
            else:
                paths = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in USD_SUFFIXES]
                if len({p.stem for p in paths}) == 1:
                    candidate = select(directory, paths[0].stem)
                    if candidate:
                        add(candidate, "robot", directory.name, directory.name)
                elif paths:
                    note(directory, "多个机器人 USD 入口，需在资产自身的 config.yaml 中指定 asset_path")
    result.assets = [attach_companions(asset, root) for asset in sorted(selected.values(), key=lambda asset: asset.id)]
    return result
