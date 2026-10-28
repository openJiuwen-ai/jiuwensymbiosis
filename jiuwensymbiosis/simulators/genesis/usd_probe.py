"""Standalone USD/Nyx acceptance, without importing the agent package.

Run ``python -m jiuwensymbiosis.simulators.genesis.usd_probe --help`` for usage.
python jiuwensymbiosis/simulators/genesis/usd_probe.py --inventory --output outputs/usd-inventory
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from .camera import OrbitCamera, PreviewSettings
    from .rendering import SceneRenderer
    from .scene_catalog import resolve_scene, scene_catalog
    from .usd_scene import initial_camera, inspect_stage, prepare_static_stage
else:
    from camera import OrbitCamera, PreviewSettings
    from rendering import SceneRenderer
    from scene_catalog import resolve_scene, scene_catalog
    from usd_scene import initial_camera, inspect_stage, prepare_static_stage


def render(scene_id: str, output: Path, *, spp: int = 64) -> dict:
    import cv2
    import genesis as gs
    import gs_nyx.nyx_py_sdk as nps
    import numpy as np
    import torch
    from PIL import Image

    asset = resolve_scene(scene_id)
    if asset.asset.preview_mode != "static_usd":
        raise ValueError("USD 探针仅用于静态场景；机器人引用检查请使用 simulators/assets probe。")
    source = asset.preview_path
    camera_key = asset.asset.category + "/" + asset.asset.name
    report: dict[str, Any] = {
        "scene_id": scene_id,
        "source": str(source),
        "mode": "static_background",
        "collision": False,
    }
    started = time.monotonic()
    report.update(prepare_static_stage(source, output / "static.usda"))
    pose = OrbitCamera(**initial_camera(camera_key, report["bounds_m"])).snapshot()
    settings = PreviewSettings(scene_id=scene_id, spp=spp)
    report["spp"] = spp
    gs.init(backend=gs.cuda, logging_level="warning")
    nps.startup()
    background = env = None
    try:
        scene = gs.Scene(show_viewer=False)
        scene.build(n_envs=1)
        env = nps.EnvironmentMapAsset()
        hdr = output / "studio.hdr"
        cv2.imwrite(str(hdr), np.full((128, 256, 3), 0.3, dtype=np.float32))
        env.texture = str(hdr)
        env.layout = nps.EEnvMapLayout.LongLat
        background = SceneRenderer(
            scene, output / "static.usda", output, settings, pose, env, exposure=report["preview_exposure_ev"]
        )
        report["startup_s"] = round(time.monotonic() - started, 3)
        render_ms = []
        for _ in range(60):
            start = time.monotonic()
            rgb = background.read_rgb()
            torch.cuda.synchronize()
            render_ms.append((time.monotonic() - start) * 1000)
        Image.fromarray(rgb.cpu().numpy()).save(output / "view.png")
        report.update(
            ok=True,
            camera=pose,
            render_ms_p50=float(np.median(render_ms[10:])),
            render_ms_p95=float(np.percentile(render_ms[10:], 95)),
        )
        # A second actual camera pose verifies that the renderer updates views.
        moved = OrbitCamera(**initial_camera(camera_key, report["bounds_m"])).command("orbit", dx=0.12).snapshot()
        background.update_camera_pose(pos=moved["pos"], lookat=moved["lookat"], up=moved["up"])
        for _ in range(10):
            rgb = background.read_rgb()
        Image.fromarray(rgb.cpu().numpy()).save(output / "orbit.png")
        return report
    finally:
        if background is not None:
            background.close()
        background = env = None
        gc.collect()
        gs.destroy()
        nps.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="library://geniesim/background/kitchen/kitchen_1/background")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--spp", type=int, default=64, help="Samples per pixel (1–64)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.inventory:
        rows = []
        for entry in scene_catalog():
            if entry["id"] == "demo":
                continue
            try:
                asset = resolve_scene(entry["id"])
                if asset.asset.preview_mode == "static_usd":
                    rows.append({**entry, **inspect_stage(asset.preview_path)})
            except Exception as exc:
                rows.append({**entry, "error": str(exc)})
        (output / "inventory.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    try:
        report = render(args.scene, output, spp=args.spp)
    except Exception as exc:
        report = {"scene_id": args.scene, "ok": False, "error": str(exc)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    # CLI result data, not a diagnostic log: preserve machine-readable stdout.
    json.dump(report, sys.stdout, ensure_ascii=False)
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
