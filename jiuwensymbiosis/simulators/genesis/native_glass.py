"""Bridge explicitly converted glass to Nyx's native double-sided transmission.

USD layers partition visibility only. Geometry, transforms and source files stay
unchanged; ordinary surfaces keep their imported material and sidedness.
"""

import json
from pathlib import Path


def prepare_glass_layers(source: Path, directory: Path):
    """Return the opaque layer, native glass groups and bounded audit counts."""
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(source))
    groups = {}
    skipped = []
    requested = any(p.IsA(UsdShade.Shader) and p.GetCustomDataByKey("symbiosis:nativeGlass") for p in stage.Traverse())
    if not requested:
        return source, [], {"group_count": 0, "mesh_count": 0, "skipped_subset_meshes": 0}
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    for prim in meshes:
        if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        binding = UsdShade.MaterialBindingAPI(prim)
        material = binding.ComputeBoundMaterial()[0]
        if not material:
            continue
        shader = material.ComputeSurfaceSource()[0]
        if not shader or not shader.GetPrim().GetCustomDataByKey("symbiosis:nativeGlass"):
            continue
        # Whole-mesh overrides cannot represent mixed face materials. Retain the
        # source binding and explicitly report this unsupported partition.
        if binding.GetMaterialBindSubsets():
            skipped.append(str(prim.GetPath()))
            continue
        color = tuple(shader.GetInput("diffuseColor").Get())
        roughness = float(shader.GetInput("roughness").Get())
        ior = float(shader.GetInput("ior").Get())
        groups.setdefault((color, roughness, ior), []).append(str(prim.GetPath()))
    audit = {
        "group_count": len(groups),
        "mesh_count": sum(map(len, groups.values())),
        "skipped_subset_meshes": len(skipped),
        "model": "double_sided_thin_transmission",
    }
    if not groups:
        return source, [], audit
    directory.mkdir(parents=True, exist_ok=True)
    all_targets = {p for paths in groups.values() for p in paths}

    def layer(name, visible):
        path = directory / name
        overlay = Usd.Stage.CreateInMemory()
        overlay.GetRootLayer().subLayerPaths = [str(source.resolve())]
        UsdGeom.SetStageUpAxis(overlay, UsdGeom.GetStageUpAxis(stage))
        UsdGeom.SetStageMetersPerUnit(overlay, UsdGeom.GetStageMetersPerUnit(stage))
        if stage.GetDefaultPrim():
            overlay.SetDefaultPrim(overlay.GetPrimAtPath(stage.GetDefaultPrim().GetPath()))
        for prim in meshes:
            if not visible(str(prim.GetPath())):
                UsdGeom.Imageable(overlay.GetPrimAtPath(prim.GetPath())).CreateVisibilityAttr("invisible")
        overlay.GetRootLayer().Export(str(path))
        return path

    base = layer("background.usda", lambda p: p not in all_targets)
    result = []
    for index, ((color, roughness, ior), paths) in enumerate(groups.items()):
        targets = set(paths)
        result.append(
            {
                "path": layer(f"glass-{index}.usda", targets.__contains__),
                "color": color,
                "roughness": roughness,
                "ior": ior,
            }
        )
    # Full paths belong in the file, not in worker IPC status.

    (directory / "glass-audit.json").write_text(
        json.dumps({**audit, "meshes": sorted(all_targets), "skipped": skipped}, indent=2)
    )
    return base, result, audit


def append_glass_instances(asset, groups):
    """Enable transmission and double-sided shading on glass instances only."""
    from gs_nyx import nyx_py_sdk as nps

    start = asset.instance_size()
    asset.instance_resize(start + len(groups))
    for index, group in enumerate(groups, start):
        instance = asset.get_instance(index)
        instance.type = nps.EInstanceType.Mesh
        instance.mesh_uri = str(group["path"])
        instance.uuid = nps.generate_uuid()
        instance.position = nps.float3(0, 0, 0)
        instance.rotation = nps.quaternion(0, 0, 0, 1)
        instance.scale = nps.float3(1, 1, 1)
        instance.convertAxis = False
        instance.enabled = True
        material = nps.MaterialAsset()
        material.opacity = nps.EMaterialOpacity.Transmissive
        material.sidedness = nps.EMaterialSidedness.Double
        material.normalMode = nps.ENormalMode.Flip
        material.albedoColor = nps.float4(*group["color"], 1.0)
        material.transmission = 1.0
        material.ior = group["ior"]
        material.thickness = 0.0
        material.perceptualRoughness = group["roughness"]
        material.metalness = 0.0
        material.set_properties(
            nps.EMaterialProperty.Opacity
            | nps.EMaterialProperty.Sidedness
            | nps.EMaterialProperty.NormalMode
            | nps.EMaterialProperty.AlbedoColor
            | nps.EMaterialProperty.Transmission
            | nps.EMaterialProperty.IOR
            | nps.EMaterialProperty.Thickness
            | nps.EMaterialProperty.PerceptualRoughness
            | nps.EMaterialProperty.Metalness
        )
        instance.matOverride = material
        asset.set_instance(index, instance)
