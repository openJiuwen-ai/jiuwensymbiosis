"""Render all InteriorAgent scenes through automatic Nyx material repair.

Keeps independent per-scene logs and repair reports. Does not rebuild occupancy
maps or relax the normal preview triangle limit. Run with symbiosis_genesis.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[3]
WARNING = "only UsdPreviewSurface and OmniPBR are supported, using default values"


def run_one(scene_id: str, output: Path) -> None:
    if TYPE_CHECKING or __package__:
        from .genesis_interioragent_batch import render_scene
    else:
        from genesis_interioragent_batch import render_scene
    from material_repair import prepare_materials
    from scene_catalog import material_rules_for_asset, resolve_scene

    output.mkdir(parents=True, exist_ok=True)
    try:
        report = render_scene(scene_id, output)
        report["render_ok"] = True
        repair = report["appearance"]["material_repair"]
        if repair.get("report_path"):
            report["appearance"]["material_repair"] = json.loads(Path(repair["report_path"]).read_text())
    except Exception as exc:
        traceback.print_exc()
        # Conversion still covers a scene whose full render exceeds the cap.
        asset = resolve_scene(scene_id)
        path, repair = prepare_materials(
            asset.preview_path, output / "materials", rules=material_rules_for_asset(asset)
        )
        report = {
            "scene_id": scene_id,
            "render_ok": False,
            "render_error": str(exc),
            "compatible_stage": str(path),
            "appearance": {"material_repair": repair},
        }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "jiuwensymbiosis/simulators/genesis"))
    if args.scene:
        run_one(args.scene, args.output.resolve())
        return
    from scene_catalog import AssetLibrary

    assets = sorted(
        AssetLibrary().search(root_id="interioragent", kind="environment", limit=1000).items, key=lambda a: a.id
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, asset in enumerate(assets):
        print(f"[{index + 1}/{len(assets)}] {asset.name}", flush=True)
        directory = output / asset.name
        directory.mkdir(exist_ok=True)
        log_path = directory / "render.log"
        row = {"scene": asset.name, "scene_id": asset.id}
        with log_path.open("w") as log:
            try:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--scene", asset.id, "--output", str(directory)],
                    cwd=ROOT,
                    env={**os.environ, "OMP_NUM_THREADS": "8", "DO_NOT_TRACK": "1", "CODEGRAPH_TELEMETRY": "0"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                row["exit_code"] = result.returncode
            except subprocess.TimeoutExpired:
                row["error"] = "timeout"
        if (directory / "report.json").is_file() and row.get("exit_code") == 0:
            report = json.loads((directory / "report.json").read_text())
            repair = report["appearance"]["material_repair"]
            row.update(
                render_ok=report["render_ok"],
                converted=repair["converted"],
                textures=repair["textures"],
                remaining=len(repair["remaining"]),
                issues=len(repair["issues"]),
                baked_textures=len(repair["baked_textures"]),
            )
            if not report["render_ok"]:
                row["render_error"] = report["render_error"]
        row["material_fallback_warnings"] = log_path.read_text(errors="replace").count(WARNING)
        if args.baseline:
            baseline = args.baseline / asset.name / "batch.log"
            if baseline.is_file():
                row["baseline_material_fallback_warnings"] = baseline.read_text(errors="replace").count(WARNING)
        rows.append(row)
        (output / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
