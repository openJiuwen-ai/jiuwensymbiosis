"""Export an offline occupancy map for one indexed InteriorAgent scene.

Run with the Genesis Python environment. The source assets remain untouched.
This script uses the same world-space mesh reader as Genesis indoor collisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING or __package__:
    from ._bootstrap import configure_script_logging, load_genesis_module
else:
    from _bootstrap import configure_script_logging, load_genesis_module


logger = logging.getLogger("jiuwensymbiosis.scripts.genesis_scene_occupancy")


def export_map(scene_id: str, output: Path, resolution: float, min_height: float, max_height: float) -> dict:
    from PIL import Image
    from pxr import Usd, UsdGeom

    read_room_meshes = load_genesis_module("indoor").read_room_meshes
    build_occupancy = load_genesis_module("occupancy").build_occupancy
    resolve_scene = load_genesis_module("scene_catalog").resolve_scene

    asset = resolve_scene(scene_id)
    source = asset.preview_path
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError(f"Cannot open USD stage: {source}")
    if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("InteriorAgent rooms.json requires an authored metre/Z-up stage")
    rooms_path = source.parent / "rooms.json"
    rooms = json.loads(rooms_path.read_text())
    meshes = read_room_meshes(source)
    result = build_occupancy(meshes, rooms, resolution=resolution, min_height=min_height, max_height=max_height)

    output.mkdir(parents=True, exist_ok=True)
    # Validate before writing any artifact: a degenerate map must not leave a
    # half-exported directory that looks like a successful run.
    for name, value in (("free", 0), ("occupied", 100)):
        if not np.count_nonzero(result.cells == value):
            raise ValueError(f"Degenerate map: no {name} cells")
    image = np.full(result.cells.shape, 205, np.uint8)
    image[result.cells == 0] = 254
    image[result.cells == 100] = 0
    image = np.flipud(image)
    Image.fromarray(image).save(output / "occupancy.png")
    Image.fromarray(image).save(output / "occupancy.pgm")
    np.savez_compressed(
        output / "occupancy.npz",
        occupancy=result.cells,
        resolution_m=result.resolution,
        origin_xy_m=result.origin,
        floor_z_m=result.floor_z,
    )
    (output / "occupancy.yaml").write_text(
        f"image: occupancy.pgm\nmode: trinary\nresolution: {result.resolution}\n"
        f"origin: [{result.origin[0]}, {result.origin[1]}, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
    )
    dependencies = {}
    paths = {Path(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath} | {rooms_path}
    for path in sorted(paths):
        with path.open("rb") as file:
            dependencies[str(path)] = hashlib.file_digest(file, "sha256").hexdigest()
    report = {
        "ok": True,
        "scene_id": scene_id,
        "source": str(source),
        "geometry_and_room_sha256": dependencies,
        "versions": {name: version(name) for name in ("numpy", "trimesh", "usd-core", "opencv-python", "scipy")},
        "resolution_m": result.resolution,
        "origin_xy_m": result.origin,
        "floor_z_m": result.floor_z,
        "width": result.cells.shape[1],
        "height": result.cells.shape[0],
        "cells": {
            name: int(np.count_nonzero(result.cells == value))
            for name, value in (("unknown", -1), ("free", 0), ("occupied", 100))
        },
        **result.metadata,
    }
    (output / "map-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--min-height", type=float, default=0.06)
    parser.add_argument("--max-height", type=float, default=1.95)
    args = parser.parse_args()
    configure_script_logging(logger)
    report = export_map(args.scene, args.output.resolve(), args.resolution, args.min_height, args.max_height)
    summary = {key: report[key] for key in ("scene_id", "resolution_m", "width", "height", "cells")}
    logger.info("%s", json.dumps(summary))


if __name__ == "__main__":
    main()
