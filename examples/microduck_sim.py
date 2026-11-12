"""Register a Microduck checkout plus official policies/velstand.onnx in the workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def register(source: Path, home: Path) -> dict:
    import yaml

    from jiuwensymbiosis.simulators.assets import build_index, load_config
    from jiuwensymbiosis.simulators.assets.config import RootSpec
    from jiuwensymbiosis.simulators.assets.scanners import scan_root

    source, home = source.resolve(), home.resolve()
    entry = {"id": "microduck", "path": str(source), "scanner": "microduck"}
    asset = scan_root(RootSpec.model_validate(entry), source).assets[0]
    if asset.preview_mode != "microduck":
        raise ValueError("请先下载官方 policies/velstand.onnx 和 policies/manifest.json")
    config = home / "configs/simulators/assets.yaml"
    if config.is_file():
        data = yaml.safe_load(config.read_text())
        existing = next((root for root in data["roots"] if root["id"] == "microduck"), None)
        if existing is not None and existing != entry:
            raise ValueError("Microduck root already points elsewhere; edit assets.yaml explicitly")
        if existing is None:
            data["roots"].append(entry)
    else:
        data = {"version": 1, "index": str(home / "outputs/simulator-assets/index.sqlite"), "roots": [entry]}
    config.parent.mkdir(parents=True, exist_ok=True)
    candidate = config.with_suffix(".candidate.yaml")
    try:
        candidate.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        load_config(candidate)
        candidate.replace(config)
    finally:
        candidate.unlink(missing_ok=True)
    build_index(load_config(config))
    return {"scene_id": asset.id, "robot_mode": "microduck", "config": str(config)}


def main():
    from jiuwensymbiosis.simulators._paths import simulator_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/vendor/microduck_rl"))
    parser.add_argument("--home", type=Path, default=simulator_root())
    args = parser.parse_args()
    json.dump(register(args.source, args.home), sys.stdout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
