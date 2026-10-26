"""USD inspection and a static preview layer. SDKs are loaded only by the worker.

Keep the source stage and textures untouched. A small override layer disables
authored rigid bodies/joints; Nyx consumes the composed stage as a static visual.
Materials remain authored USD, without claiming equivalence to the source renderer.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

MAX_PREVIEW_TRIANGLES = 5_000_000


def inspect_stage(source: Path) -> dict[str, Any]:
    """Measure visible geometry and resolved assets without importing Genesis."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(source))
    if not stage or not stage.GetDefaultPrim():
        raise ValueError("USD 缺少有效的 defaultPrim。")
    shaders: Counter[str] = Counter()
    missing: set[str] = set()
    mdl: set[str] = set()
    meshes = triangles = 0
    max_light = 0.0
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    # Lights and environment domes can be hundreds of metres wide. Use only
    # visible render meshes, or a room camera may spawn outside the building.
    box = Gf.Range3d()
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        intensity = prim.GetAttribute("inputs:intensity")
        if intensity and intensity.Get() is not None:
            max_light = max(max_light, float(intensity.Get()))
        if prim.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(prim).ComputeVisibility() != "invisible":
            if UsdGeom.Imageable(prim).ComputePurpose() not in {"default", "render"}:
                continue
            counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
            if counts is not None:
                triangles += sum(max(0, int(n) - 2) for n in counts)
                meshes += 1
                box.UnionWith(bounds.ComputeWorldBound(prim).ComputeAlignedRange())
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            shaders[str(shader.GetShaderId() or shader.GetSourceAssetSubIdentifier("mdl"))] += 1
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attribute.Get()
            if value and value.path and not value.resolvedPath:
                (mdl if value.path.lower().endswith(".mdl") else missing).add(value.path)
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    low, high = list(box.GetMin()), list(box.GetMax())
    if up_axis == "Y":
        low, high = [low[0], -high[2], low[1]], [high[0], -low[2], high[1]]
    return {
        "visible_meshes": meshes,
        "visible_triangles": triangles,
        "source_up_axis": up_axis,
        "source_meters_per_unit": scale,
        "bounds_m": [[float(v) * scale for v in xyz] for xyz in (low, high)],
        "shaders": dict(shaders),
        "missing_assets": sorted(missing),
        "mdl_modules": sorted(mdl),
        "preview_exposure_ev": -min(8.0, max(0.0, math.log2(max(max_light, 300) / 300))),
        "lighting": "preview environment + fixed inspection light; source UsdLux rig not reproduced",
    }


def prepare_static_stage(source: Path, output: Path) -> dict[str, Any]:
    """Create an override layer for a bounded, collision-disabled background preview."""
    from pxr import Usd, UsdGeom, UsdPhysics

    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("预览缓存不能覆盖源 USD。")
    report = inspect_stage(source)
    if not report["visible_meshes"] or not report["visible_triangles"]:
        raise ValueError("该 USD 没有可见网格，仅有隐藏碰撞体或空场景。")
    if report["visible_triangles"] > MAX_PREVIEW_TRIANGLES:
        raise ValueError(
            f"场景有 {report['visible_triangles']:,} 个可见三角形，超过首版预览的 "
            f"{MAX_PREVIEW_TRIANGLES:,} 上限；请先制作 LOD。"
        )
    original = Usd.Stage.Open(str(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths = [str(source)]
    stage.SetDefaultPrim(stage.GetPrimAtPath(original.GetDefaultPrim().GetPath()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
    # Remove rigid-body discovery rather than just disabling simulation: the USD
    # importer splits links by the applied schema, including disabled rigid bodies.
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    stage.GetRootLayer().Save()
    return report


def initial_camera(scene_id: str, bounds: list[list[float]]) -> dict[str, Any]:
    """An interior room view, or a full view of small open workcells, in metres."""
    low, high = bounds
    if any(not math.isfinite(v) or abs(v) > 10_000 for xyz in bounds for v in xyz):
        raise ValueError("USD 场景包围盒无效。")
    center = [(a + b) / 2 for a, b in zip(low, high, strict=True)]
    span = [b - a for a, b in zip(low, high, strict=True)]
    if scene_id.startswith("kitchen/"):
        pos, target = (-0.6, 0.4, 1.6), (0.9, -0.7, 1.1)
    elif scene_id.startswith("warehouse/"):
        pos, target = (2.8, -3.2, 2.2), (0.5, 0, 0.7)
    else:
        z = low[2] + min(1.6, max(0.6, span[2] * 0.6))
        pos = (center[0] + span[0] * 0.30, center[1] - span[1] * 0.30, z)
        target = (center[0] - span[0] * 0.20, center[1] + span[1] * 0.20, z - 0.25)
    offset = [a - b for a, b in zip(pos, target, strict=True)]
    distance = math.sqrt(sum(v * v for v in offset))
    return {
        "yaw": math.atan2(offset[1], offset[0]),
        "pitch": math.asin(offset[2] / distance),
        "distance": distance,
        "target": target,
        "center": (center[0], center[1], low[2]),
        "scene_scale": max(1.01, max(span) / 3),
    }
