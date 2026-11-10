"""Load every indexed InteriorAgent scene in Genesis/Nyx, then build offline maps.

Each scene runs in a separate process so a failed SDK import cannot poison the
remaining scenes. Static visuals use the workbench SceneRenderer; collision
decomposition and robot navigation are explicitly outside this batch's evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from ._bootstrap import configure_script_logging, load_genesis_module
else:
    from _bootstrap import configure_script_logging, load_genesis_module


ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger("jiuwensymbiosis.scripts.genesis_interioragent_batch")


def _directory_name(asset_id: str) -> str:
    """One output directory per indexed asset: ids are unique, names are not."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in asset_id)


def inspection_camera(source: Path) -> dict:
    """Choose a clear inspection eye when the author's saved camera is outside."""
    import numpy as np

    read_room_meshes = load_genesis_module("indoor").read_room_meshes
    build_occupancy = load_genesis_module("occupancy").build_occupancy
    from scipy.ndimage import distance_transform_edt

    occupancy = build_occupancy(
        read_room_meshes(source), json.loads((source.parent / "rooms.json").read_text()), resolution=0.1
    )
    free = occupancy.cells == 0
    row, column = np.unravel_index(np.argmax(distance_transform_edt(free)), free.shape)
    if not free[row, column]:
        raise ValueError("No clear camera position in the room")
    eye = np.array([column, row], dtype=float)
    best = None
    for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        length = 0
        for distance in range(1, max(free.shape)):
            x, y = np.rint(eye + distance * direction).astype(int)
            if not (0 <= x < free.shape[1] and 0 <= y < free.shape[0]) or not free[y, x]:
                break
            length = distance
        if best is None or length > best[0]:
            best = (length, direction)
    if best is None:  # The fixed, nonempty angle sweep always selects a direction.
        raise RuntimeError("No free viewing direction found in the room")
    world_eye = np.asarray(occupancy.origin) + (eye + 0.5) * occupancy.resolution
    target = world_eye + max(1.0, best[0] * occupancy.resolution) * best[1]
    return {
        "pos": [*world_eye.tolist(), occupancy.floor_z + 1.5],
        "lookat": [*target.tolist(), occupancy.floor_z + 1.2],
        "up": [0, 0, 1],
    }


def render_scene(scene_id: str, output: Path) -> dict:
    # OpenCV reads this at FIRST import; EXR material textures need it before that.
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    import genesis as gs
    import gs_nyx.nyx_py_sdk as nps
    import numpy as np

    camera = load_genesis_module("camera")
    from PIL import Image
    from pxr import Usd

    rendering = load_genesis_module("rendering")
    catalog = load_genesis_module("scene_catalog")
    material_rules_for_asset, resolve_scene = catalog.material_rules_for_asset, catalog.resolve_scene
    usd_scene = load_genesis_module("usd_scene")
    initial_camera, prepare_static_stage = usd_scene.initial_camera, usd_scene.prepare_static_stage

    asset = resolve_scene(scene_id)
    source = asset.preview_path
    report = {
        "scene_id": scene_id,
        "source": str(source),
        "mode": "Genesis_scene_with_Nyx_static_USD_background",
        "collision": False,
        "robot_navigation_tested": False,
        "versions": {name: version(name) for name in ("genesis-world", "gs-nyx", "gs-nyx-plugin")},
    }
    report.update(prepare_static_stage(source, output / "static.usda"))
    bounds = report.get("bounds_m")
    if bounds is None:
        raise ValueError("Static stage preparation reported no scene bounds")
    stage = Usd.Stage.Open(str(source))
    authored = stage.GetRootLayer().customLayerData.get("cameraSettings", {}).get("Perspective", {})
    if "position" in authored and "target" in authored:
        pose = {"pos": list(authored["position"]), "lookat": list(authored["target"]), "up": [0, 0, 1]}
        report["camera_source"] = "authored_Perspective_metres_Z_up"
        if report.get("source_up_axis") != "Z" or report.get("source_meters_per_unit") != 1:
            raise ValueError("Authored camera needs explicit frame conversion")
    else:
        pose = camera.OrbitCamera(**initial_camera(scene_id, bounds)).snapshot()
        report["camera_source"] = "bounds_default"
    low, high = np.asarray(bounds)
    if np.any(np.asarray(pose["pos"]) < low - 0.5) or np.any(np.asarray(pose["pos"]) > high + 0.5):
        report["rejected_camera"] = pose
        pose = inspection_camera(source)
        report["camera_source"] = "clear_floor_cell_with_longest_free_view_direction"
    report["camera"] = pose
    settings = camera.PreviewSettings(scene_id=scene_id, spp=16)
    renderer = scene = None
    sdk_up = False
    started = time.monotonic()
    try:
        gs.init(backend=gs.cuda, logging_level="warning", performance_mode=True)
        nps.startup()
        sdk_up = True
        scene = gs.Scene(show_viewer=False)
        scene.build(n_envs=1)
        env = nps.EnvironmentMapAsset()
        hdr = output / "studio.hdr"
        if not cv2.imwrite(str(hdr), np.full((128, 256, 3), 0.3, dtype=np.float32)):
            raise RuntimeError("Cannot write preview environment")
        env.texture = str(hdr)
        env.layout = nps.EEnvMapLayout.LongLat
        renderer = rendering.SceneRenderer(
            scene,
            output / "static.usda",
            output,
            settings,
            pose,
            env,
            exposure=0,
            material_rules=material_rules_for_asset(asset),
        )
        for _ in range(8):
            rgb = renderer.read_rgb().cpu().numpy()
        Image.fromarray(rgb).save(output / "genesis.png")
        if float(rgb.std()) <= 2:
            raise ValueError("Rendered image has insufficient visible structure; inspect camera and lighting")
        report.update(
            ok=True,
            startup_and_capture_s=round(time.monotonic() - started, 3),
            rgb_std=float(rgb.std()),
            rgb_mean=float(rgb.mean()),
            appearance=renderer.appearance,
            material_equivalence_verified=False,
            lighting="fixed inspection light at authored camera plus constant environment",
        )
        return report
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:  # noqa: BLE001 - cleanup must continue to the scene and SDK
                traceback.print_exc()
        if scene is not None:
            scene.destroy()
        if sdk_up:
            gs.destroy()
            nps.shutdown()


def run_one(scene_id: str, output: Path) -> bool:
    """Render and map one scene; return whether both operations succeeded."""
    if TYPE_CHECKING or __package__:
        from .genesis_scene_occupancy import export_map
    else:
        from genesis_scene_occupancy import export_map

    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"scene_id": scene_id}
    for operation in ("render", "map"):
        started = time.monotonic()
        try:
            if operation == "render":
                render_dir = output / "genesis"
                render_dir.mkdir(exist_ok=True)
                report = render_scene(scene_id, render_dir)
                (render_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                report = export_map(scene_id, output, 0.05, 0.06, 1.95)
            result[operation] = {"ok": bool(report["ok"]), "seconds": round(time.monotonic() - started, 3)}
        except Exception as exc:
            traceback.print_exc()
            result[operation] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        (output / "result.json").write_text(json.dumps(result, indent=2))
    logger.info("%s", json.dumps(result))
    return all(result[operation]["ok"] for operation in ("render", "map"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", help="Internal child mode, or retry one indexed scene")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    configure_script_logging(logger)
    if args.scene:
        if not run_one(args.scene, args.output.resolve()):
            raise SystemExit(1)
        return
    catalog = load_genesis_module("scene_catalog")

    assets = catalog.AssetLibrary().search(root_id="interioragent", kind="environment", limit=1000).items
    assets = sorted(assets, key=lambda a: a.id)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = [{"id": a.id, "name": a.name} for a in assets]
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2))
    results = []
    for index, asset in enumerate(assets):
        target = output / _directory_name(asset.id)
        target.mkdir(exist_ok=True)
        # A previous run (or a killed retry) may have left a result.json behind;
        # reading it would report this run's failure as that old success.
        (target / "result.json").unlink(missing_ok=True)
        logger.info("[%d/%d] %s", index + 1, len(assets), asset.name)
        command = [sys.executable, str(Path(__file__).resolve()), "--scene", asset.id, "--output", str(target)]
        with (target / "batch.log").open("w") as log:
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env={**os.environ, "DO_NOT_TRACK": "1", "CODEGRAPH_TELEMETRY": "0", "OMP_NUM_THREADS": "8"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                status: dict[str, Any] = {"exit_code": completed.returncode}
            except subprocess.TimeoutExpired:
                status = {"error": "timeout", "timeout_s": args.timeout}
        result_path = target / "result.json"
        try:
            result = json.loads(result_path.read_text()) if result_path.exists() else {"scene_id": asset.id}
        except (json.JSONDecodeError, OSError):
            # A truncated write (killed child) must fail this scene, not the batch.
            result = {"scene_id": asset.id}
        result.update(process=status)
        results.append(result)
        (output / "batch-results.json").write_text(json.dumps(results, indent=2))
        logger.info("%s", json.dumps(result))
    if not all(r.get("render", {}).get("ok") and r.get("map", {}).get("ok") for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
