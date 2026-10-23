"""可重建的 SQLite 目录：显式构建，原子替换，只读查询，加载前检查。"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import AssetConfig, load_config, simulator_root
from .models import Asset, AssetError, AssetPage, ResolvedAsset, local_file
from .scanners import scan_root

SCHEMA_VERSION = 2
SCANNER_VERSION = 2
DEFAULT_INDEX_PATH = simulator_root() / "outputs/simulator-assets/index.sqlite"


def _fingerprint(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise AssetError(f"资产文件不存在：{path}")
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_index(config: AssetConfig | None = None) -> dict[str, Any]:
    """失败保留旧库；不写入源资产，不从兄弟项目读取索引。"""
    config = config or load_config()
    assets: dict[str, Asset] = {}
    diagnostics = []
    for spec in config.spec.roots:
        result = scan_root(spec, config.roots[spec.id])
        assets.update((asset.id, asset) for asset in result.assets)
        diagnostics.extend(result.diagnostics)
    if not assets:
        raise AssetError("没有发现可索引入口；保留原索引，请检查资产目录或扫描规则")
    for identifier, asset in assets.items():
        fingerprints = {
            role: _fingerprint(local_file(config.roots[asset.root_id], relative))
            for role, relative in asset.representations.items()
        }
        assets[identifier] = replace(asset, files=fingerprints)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scanner_version": SCANNER_VERSION,
        "config_digest": config.digest,
        "build_config": str(config.source),
        "built_at": datetime.now(UTC).isoformat(),
        "total": len(assets),
        "by_root": dict(Counter(asset.root_id for asset in assets.values())),
        "by_kind": dict(Counter(asset.kind for asset in assets.values())),
        "preview_candidates": dict(Counter(asset.preview_mode for asset in assets.values() if asset.preview_mode)),
        "validation": "entry_files_only",
        "diagnostics": diagnostics,
    }
    config.index.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".assets-", suffix=".sqlite", dir=config.index.parent)
    os.close(descriptor)
    try:
        with closing(sqlite3.connect(temporary)) as db, db:
            db.executescript("""
                PRAGMA foreign_keys = ON;
                CREATE TABLE roots (id TEXT PRIMARY KEY, path TEXT NOT NULL, scanner TEXT NOT NULL);
                CREATE TABLE assets (
                    id TEXT PRIMARY KEY, root_id TEXT NOT NULL REFERENCES roots(id), name TEXT NOT NULL,
                    kind TEXT NOT NULL, category TEXT NOT NULL, preview_mode TEXT,
                    record TEXT NOT NULL
                );
                CREATE INDEX asset_filters ON assets(root_id, kind, category);
                CREATE INDEX asset_preview ON assets(preview_mode);
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            db.executemany(
                "INSERT INTO roots VALUES (?, ?, ?)",
                [(root.id, str(config.roots[root.id]), root.scanner) for root in config.spec.roots],
            )
            db.executemany(
                "INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        a.id,
                        a.root_id,
                        a.name,
                        a.kind,
                        a.category,
                        a.preview_mode,
                        json.dumps(a.to_dict(), ensure_ascii=False),
                    )
                    for a in sorted(assets.values(), key=lambda asset: asset.id)
                ],
            )
            db.execute("INSERT INTO metadata VALUES ('manifest', ?)", (json.dumps(manifest, ensure_ascii=False),))
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise AssetError("新索引 SQLite 完整性检查失败")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, config.index)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return manifest


class AssetLibrary:
    """轻量查询入口；每次打开只读连接，热重建后下一次查询即读取新索引。"""

    def __init__(self, index: Path | str | None = None) -> None:
        self.index = Path(index if index is not None else DEFAULT_INDEX_PATH).expanduser().resolve()

    @contextmanager
    def _connection(self):
        if not self.index.is_file():
            raise AssetError("资产索引未构建，请先运行 python -m jiuwensymbiosis.simulators.assets build")
        db = None
        try:
            db = sqlite3.connect(self.index.as_uri() + "?mode=ro", uri=True)
            row = db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
            manifest = json.loads(row[0]) if row else {}
            if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
                raise AssetError("资产索引结构版本不兼容，请重新 build")
            yield db, manifest
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise AssetError(f"资产索引损坏或不可读，请重新 build：{exc}") from exc
        finally:
            if db is not None:
                db.close()

    def status(self) -> dict[str, Any]:
        with self._connection() as (_, manifest):
            return {**manifest, "index": str(self.index)}

    def search(
        self,
        *,
        query: str = "",
        root_id: str | None = None,
        kind: str | None = None,
        category: str | None = None,
        preview_mode: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> AssetPage:
        if not 1 <= limit <= 1000 or offset < 0:
            raise AssetError("limit 应为 1..1000，offset 不可为负数")
        filters, values = [], []
        for column, value in (
            ("root_id", root_id),
            ("kind", kind),
            ("category", category),
            ("preview_mode", preview_mode),
        ):
            if value is not None:
                filters.append(f"{column} = ?")
                values.append(value)
        if query:
            filters.append("instr(lower(id || ' ' || name || ' ' || category), lower(?)) > 0")
            values.append(query)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        with self._connection() as (db, _):
            total = db.execute("SELECT count(*) FROM assets" + where, values).fetchone()[0]
            rows = db.execute(
                "SELECT record FROM assets" + where + " ORDER BY id LIMIT ? OFFSET ?", [*values, limit, offset]
            )
            return AssetPage(total, [Asset(**json.loads(row[0])) for row in rows], limit, offset)

    def get(self, identifier: str) -> Asset:
        with self._connection() as (db, _):
            return self._get(db, identifier)

    @staticmethod
    def _get(db: sqlite3.Connection, identifier: str) -> Asset:
        row = db.execute("SELECT record FROM assets WHERE id = ?", (identifier,)).fetchone()
        if row is None:
            raise AssetError(f"资产 ID 未登记：{identifier}")
        return Asset(**json.loads(row[0]))

    def resolve(self, identifier: str) -> ResolvedAsset:
        # One connection keeps the asset and root mapping in the same published index.
        with self._connection() as (db, _):
            asset = self._get(db, identifier)
            row = db.execute("SELECT path FROM roots WHERE id = ?", (asset.root_id,)).fetchone()
            if row is None or not Path(row[0]).is_absolute():
                raise AssetError(f"索引缺少有效的资产根目录：{asset.root_id}")
            root = Path(row[0])
        paths = {}
        for role, relative in asset.representations.items():
            path = local_file(root, relative)
            # Source manifests are build inputs; their contents are already in SQLite.
            if role in {"source", "genesis_urdf", "nyx_urdf"} and _fingerprint(path) != asset.files[role]:
                raise AssetError(f"资产文件已变化，请重新 build：{path}")
            paths[role] = path
        return ResolvedAsset(asset, paths, root)
