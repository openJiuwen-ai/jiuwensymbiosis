"""Read SceneGen's existing USD presentation contract without importing its host.

The asset index owns identity and paths. This module only interprets authored
camera metadata and validates local dependencies; it never publishes a scene.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def dependencies(source: Path, *, root: Path | None = None) -> dict:
    """Resolve referenced layers, payloads and textures before GPU construction."""
    from pxr import UsdUtils

    layers, assets, missing = UsdUtils.ComputeAllDependencies(str(source))
    if missing:
        raise ValueError(f"Missing USD dependencies: {sorted(map(str, missing))}")
    paths = sorted(
        {str(Path(p).resolve()) for p in assets if p} | {str(Path(p.realPath).resolve()) for p in layers if p.realPath}
    )
    for path in paths:
        if not Path(path).is_file():
            raise ValueError(f"Missing USD dependency: {path}")
        if root is not None and not Path(path).is_relative_to(root.resolve()):
            raise ValueError(f"USD dependency outside asset root: {path}")
    return {"count": len(paths), "paths": paths, "missing": []}


def read_presentation(source: Path, *, root: Path | None = None) -> dict | None:
    """Opt in SceneGen USDs; unrelated indexed assets retain their preview policy."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError(f"Cannot open USD: {source}")
    cameras, marked = [], []
    generated = False
    for prim in stage.Traverse():
        generated |= bool(prim.GetCustomDataByKey("sceneGen:usdPrimPath"))
        if prim.IsA(UsdGeom.Camera) and UsdGeom.Imageable(prim).ComputeVisibility() != "invisible":
            cameras.append(prim)
            flag = prim.GetAttribute("userProperties:package_camera")
            if flag and flag.Get():
                marked.append(prim)
    manifest = source.parent / "manifest.json"
    record = None
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        if isinstance(data, dict) and data.get("artifact_type") in {"base", "revision"} and data.get("scene_id"):
            generated = True
            record = data.get("camera", {})
            if not isinstance(record, dict):
                raise ValueError("Invalid SceneGen manifest camera")
    if not (generated or marked):
        return None
    dependency_report = dependencies(source, root=root)
    if record is not None and record.get("usd_prim_path"):
        selected = [p for p in cameras if str(p.GetPath()) == record["usd_prim_path"]]
    else:
        selected = marked or cameras
    if len(selected) != 1:
        raise ValueError("SceneGen requires one explicit authored camera; missing or ambiguous camera")
    prim = selected[0]
    camera = UsdGeom.Camera(prim)
    if camera.GetProjectionAttr().Get() != "perspective":
        raise ValueError("Nyx source camera requires perspective projection")
    if camera.GetHorizontalApertureOffsetAttr().Get() or camera.GetVerticalApertureOffsetAttr().Get():
        raise ValueError("Nyx source camera does not support aperture offsets")
    focal = float(camera.GetFocalLengthAttr().Get())
    horizontal = float(camera.GetHorizontalApertureAttr().Get())
    vertical = float(camera.GetVerticalApertureAttr().Get())
    scale = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    if up_axis not in {"Y", "Z"} or not all(math.isfinite(v) and v > 0 for v in (focal, horizontal, vertical, scale)):
        raise ValueError("Invalid source camera units or intrinsics")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    def zup(vector):
        x, y, z = map(float, vector)
        return np.array((x, -z, y) if up_axis == "Y" else (x, y, z))

    pos = zup(matrix.Transform(Gf.Vec3d(0, 0, 0))) * scale
    forward = zup(matrix.TransformDir(Gf.Vec3d(0, 0, -1)))
    up = zup(matrix.TransformDir(Gf.Vec3d(0, 1, 0)))
    if not np.isfinite(np.r_[pos, forward, up]).all() or min(np.linalg.norm(forward), np.linalg.norm(up)) <= 0:
        raise ValueError("Invalid source camera transform")
    forward /= np.linalg.norm(forward)
    up /= np.linalg.norm(up)
    if abs(float(forward @ up)) > 1e-6:
        raise ValueError("Sheared source camera cannot be represented by Nyx")
    near, far = [float(v) * scale for v in camera.GetClippingRangeAttr().Get()]
    if not (math.isfinite(near + far) and 0 < near < far):
        raise ValueError("Invalid source camera clipping range")
    return {
        "camera_path": str(prim.GetPath()),
        "pos": pos.tolist(),
        "lookat": (pos + forward).tolist(),
        "up": up.tolist(),
        "fov_y": math.degrees(2 * math.atan(vertical / (2 * focal))),
        "aspect": horizontal / vertical,
        "near": near,
        "far": far,
        "dependencies": dependency_report,
        "quality_policy": "interactive resolution/spp; authored vertical FOV, pose and clipping",
    }
