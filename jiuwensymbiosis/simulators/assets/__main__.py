"""显式资产库操作：python -m jiuwensymbiosis.simulators.assets --help。"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from . import AssetError, AssetLibrary, build_index, initialize_config, load_config, probe_asset
else:
    # Load a private namespace so the directory CLI never shadows installed modules.
    from types import ModuleType

    namespace = ModuleType("_jiuwensymbiosis_asset_cli")
    namespace.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[namespace.__name__] = namespace
    from _jiuwensymbiosis_asset_cli.assets import (
        AssetError,
        AssetLibrary,
        build_index,
        initialize_config,
        load_config,
        probe_asset,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本仓库的本地资产目录：通用目录树 / GenieSim / SpiritAI")
    parser.add_argument("--config", help="build / init-config 的 YAML；默认仿真数据根下 configs/simulators/assets.yaml")
    parser.add_argument("--index", help="查询使用的 SQLite；默认仿真数据根下 outputs/simulator-assets/index.sqlite")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-config", help="生成本机资产配置示例，不覆盖已有文件")
    commands.add_parser("build", help="扫描并原子替换索引；不修改源资产")
    commands.add_parser("status", help="构建摘要、诊断及验证边界")
    search = commands.add_parser("search", help="分页查询文件级候选")
    search.add_argument("query", nargs="?", default="")
    for name in ("root-id", "kind", "category", "preview-mode"):
        search.add_argument("--" + name)
    search.add_argument("--limit", type=int, default=30)
    search.add_argument("--offset", type=int, default=0)
    show = commands.add_parser("show", help="按 ID 查询并重新检查入口文件")
    show.add_argument("id")
    probe = commands.add_parser("probe", help="检查 URDF 引用与材质；可选 USD 元数据和引用探测")
    probe.add_argument("id")
    probe.add_argument("--usd", action="store_true")
    args = parser.parse_args(argv)
    if args.command not in {"build", "init-config"} and args.config:
        parser.error("查询不读取 YAML；请用 --index 指定 SQLite")
    if args.command in {"build", "init-config"} and args.index:
        parser.error("build 的索引输出位置由 YAML 的 index 指定")
    try:
        result: dict[str, Any]
        library = AssetLibrary(args.index)
        if args.command == "init-config":
            result = {"config": str(initialize_config(args.config)), "next_step": "编辑 roots[].path 后运行 build"}
        elif args.command == "build":
            result = build_index(load_config(args.config))
        elif args.command == "status":
            result = library.status()
        elif args.command == "search":
            result = asdict(
                library.search(
                    **{
                        key: getattr(args, key)
                        for key in ("query", "root_id", "kind", "category", "preview_mode", "limit", "offset")
                    }
                )
            )
        elif args.command == "show":
            resolved = library.resolve(args.id)
            result = {**resolved.asset.to_dict(), "paths": {role: str(path) for role, path in resolved.paths.items()}}
        else:
            result = probe_asset(library, args.id, usd=args.usd)
    except (AssetError, OSError, ET.ParseError, json.JSONDecodeError) as exc:
        # These streams carry the CLI JSON protocol, not prefixed log messages.
        json.dump({"error": str(exc)}, sys.stderr, ensure_ascii=False)
        return 1
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
