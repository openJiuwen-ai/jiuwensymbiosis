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
from typing import Any

if not __package__:
    import importlib.util

    _PACKAGE_NAME = "_jiuwensymbiosis_simulators"
    _PACKAGE_ENTRY = Path(__file__).resolve().parents[1] / "__init__.py"
    _PACKAGE_SPEC = importlib.util.spec_from_file_location(
        _PACKAGE_NAME, _PACKAGE_ENTRY, submodule_search_locations=[str(_PACKAGE_ENTRY.parent)]
    )
    if _PACKAGE_SPEC is None or _PACKAGE_SPEC.loader is None:
        raise ImportError("Cannot load the simulator package")
    _PACKAGE_MODULE = importlib.util.module_from_spec(_PACKAGE_SPEC)
    sys.modules[_PACKAGE_NAME] = _PACKAGE_MODULE
    _PACKAGE_SPEC.loader.exec_module(_PACKAGE_MODULE)
    _ENTRY_SPEC = importlib.util.spec_from_file_location(_PACKAGE_NAME + ".genesis.usd_probe", __file__)
    if _ENTRY_SPEC is None:
        raise ImportError("Cannot identify the standalone simulator module")
    # Seed the file entry's import context from its loader, without core import hooks.
    globals().update({"__package__": _ENTRY_SPEC.parent, "__spec__": _ENTRY_SPEC})

from ..common.camera import OrbitCamera, PreviewSettings
from ..common.scene_catalog import resolve_scene, scene_catalog
from .rendering import SceneRenderer
from .scene_catalog import material_rules_for_asset
from .source_lighting import load_source_lighting, make_environment
from .usd_scene import initial_camera, inspect_stage, prepare_static_stage


def render(scene_id: str, output: Path) -> dict:
    import os

    # OpenCV reads this at FIRST import; EXR material textures need it before that.
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
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
    settings = PreviewSettings(scene_id=scene_id, spp=32)
    report["spp"] = 32
    gs.init(backend=gs.cpu, logging_level="warning")
    nps.startup()
    background = env = None
    try:
        scene = gs.Scene(show_viewer=False)
        scene.build(n_envs=1)
        lighting = load_source_lighting(source, directory=output)
        env = make_environment(lighting)
        report["lighting"] = lighting.summary()
        background = SceneRenderer(
            scene,
            output / "static.usda",
            output,
            settings,
            pose,
            env,
            exposure=lighting.exposure_ev,
            source_lighting=lighting,
            material_rules=material_rules_for_asset(asset),
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
        report = render(args.scene, output)
    except Exception as exc:
        report = {"scene_id": args.scene, "ok": False, "error": str(exc)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    # CLI result data, not a diagnostic log: preserve machine-readable stdout.
    json.dump(report, sys.stdout, ensure_ascii=False)
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
