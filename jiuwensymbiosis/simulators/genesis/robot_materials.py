"""Restore USD PBR face bindings into a rendering companion for an exported URDF.

Physics keeps the original OBJ-based URDF. GLB visuals preserve baked link-frame
vertices, authored normals and material partitions; Nyx still drives URDF links.
This profile handles constant UsdPreviewSurface and glTF MDL materials. It rejects
texture networks rather than claiming to have recovered unsupported image maps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def material_parameters(material: Any) -> dict:
    """Read linear PBR values, including the USD spec's unauthored defaults."""
    from pxr import UsdShade

    if not material:
        return {
            "name": "unbound",
            "baseColorFactor": [0.18, 0.18, 0.18, 1],
            "metallicFactor": 0,
            "roughnessFactor": 0.5,
        }
    shader, _, _ = material.ComputeSurfaceSource()
    if not shader:
        shader, _, _ = material.ComputeSurfaceSource("mdl")
    if not shader:
        raise ValueError(f"No surface shader: {material.GetPath()}")

    def value(name: str, default: Any) -> Any:
        inp = shader.GetInput(name)
        seen = set()
        while inp and inp.HasConnectedSource():
            path = str(inp.GetAttr().GetPath())
            if path in seen:
                raise ValueError(f"Material input cycle: {path}")
            seen.add(path)
            connected, connected_name, kind = inp.GetConnectedSource()
            # USD materials often expose constant interface inputs. Following
            # these is essential: Get() on the shader input itself returns None.
            if kind != UsdShade.AttributeType.Input:
                raise ValueError(f"Texture/network requires explicit translation: {path}")
            inp = connected.GetInput(connected_name)
        result = inp.Get() if inp else None
        return default if result is None else result

    if shader.GetIdAttr().Get() == "UsdPreviewSurface":
        if value("useSpecularWorkflow", 0):
            raise ValueError("Specular workflow requires a separate material conversion")
        color = value("diffuseColor", (0.18, 0.18, 0.18))
        opacity = value("opacity", 1.0)
        metallic, roughness = value("metallic", 0.0), value("roughness", 0.5)
        emission = value("emissiveColor", (0, 0, 0))
    elif shader.GetSourceAssetSubIdentifier("mdl") == "gltf_material":
        color = value("base_color_factor", (1, 1, 1))
        opacity = value("base_alpha", 1.0)
        metallic, roughness = value("metallic_factor", 1.0), value("roughness_factor", 1.0)
        emission = [float(c) * value("emissive_strength", 1.0) for c in value("emissive_factor", (0, 0, 0))]
    else:
        raise ValueError(f"Unsupported shader: {shader.GetPath()}")
    return {
        "name": str(material.GetPath()),
        "baseColorFactor": [*map(float, color), float(opacity)],
        "metallicFactor": float(metallic),
        "roughnessFactor": float(roughness),
        "emissiveFactor": list(map(float, emission)),
    }


def mesh_payload(prim: Any, obj: Path) -> tuple[dict, dict]:
    """Recover source topology/materials on precisely the exported link-frame vertices."""
    import numpy as np
    from pxr import UsdGeom, UsdShade

    mesh = UsdGeom.Mesh(prim)
    source = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    with obj.open() as stream:
        points = np.array([list(map(float, line.split()[1:4])) for line in stream if line.startswith("v ")])
    if points.shape != source.shape:
        raise ValueError(f"Vertex correspondence lost: {obj}")
    # The exporter bakes each mesh into a URDF link frame. Recover that affine map from
    # its unchanged vertex order, and verify every vertex before using it on normals.
    design = np.column_stack((source, np.ones(len(source))))
    transform, _, rank, _ = np.linalg.lstsq(design, points, rcond=None)
    error = float(np.max(np.abs(design @ transform - points)))
    if rank != 4 or error > 1e-6:
        raise ValueError(f"Cannot recover unambiguous normal transform for {obj}: rank={rank}, error={error}")
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if np.any(counts < 3) or counts.sum() != len(indices):
        raise ValueError(f"Invalid topology: {prim.GetPath()}")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    if np.all(counts == 3):
        triangles = np.arange(len(indices)).reshape(-1, 3)
        owners = np.arange(len(counts))
    else:
        triangles = np.array(
            [(s, s + i, s + i + 1) for s, n in zip(starts, counts, strict=True) for i in range(1, n - 1)]
        )
        owners = np.repeat(np.arange(len(counts)), counts - 2)
    normals: np.ndarray | None
    normals_var = UsdGeom.PrimvarsAPI(prim).GetPrimvar("normals")
    if normals_var and normals_var.HasValue():
        normals = np.asarray(normals_var.ComputeFlattened(), dtype=np.float64)
        interpolation = normals_var.GetInterpolation()
    else:
        value = mesh.GetNormalsAttr().Get()
        normals = None if value is None else np.asarray(value, dtype=np.float64)
        interpolation = mesh.GetNormalsInterpolation()
    corners = points[indices]
    if normals is None:
        # Preserve hard faces when the source supplies no shading normals.
        corners = corners[triangles].reshape(-1, 3)
        cross = np.cross(corners[1::3] - corners[::3], corners[2::3] - corners[::3])
        normals = np.repeat(cross, 3, axis=0)
        triangles = np.arange(len(corners)).reshape(-1, 3)
    else:
        if interpolation in ("vertex", "varying"):
            normals = normals[indices]
        elif interpolation == "uniform":
            normals = np.repeat(normals, counts, axis=0)
        elif interpolation == "constant":
            normals = np.repeat(normals[:1], len(indices), axis=0)
        elif interpolation != "faceVarying":
            raise ValueError(f"Unsupported normal interpolation: {interpolation}")
        normals = normals @ np.linalg.inv(transform[:3]).T
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-30)
    if mesh.GetOrientationAttr().Get() == "leftHanded":
        triangles = triangles[:, [0, 2, 1]]
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    materials = [material_parameters(material)]
    assignment: Any = np.zeros(len(counts), dtype=np.int32)
    covered: Any = np.zeros(len(counts), dtype=bool)
    for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
        faces = np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int64)
        if np.any(faces < 0) or np.any(faces >= len(counts)) or np.any(covered[faces]):
            raise ValueError(f"Invalid or overlapping material subsets: {subset.GetPath()}")
        subset_material, _ = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
        materials.append(material_parameters(subset_material))
        assignment[faces] = len(materials) - 1
        covered[faces] = True
    groups = [(int(i), triangles[assignment[owners] == i]) for i in np.unique(assignment)]
    report = {
        "source_prim": str(prim.GetPath()),
        "vertex_error_m": error,
        "faces": len(counts),
        "triangles": len(triangles),
        "materials": [materials[i] for i, _ in groups],
        "subsets": len(materials) - 1,
        "authored_normals": normals_var.HasValue() if normals_var else mesh.GetNormalsAttr().HasValue(),
    }
    return {"positions": corners, "normals": normals, "groups": groups, "materials": materials}, report


def write_glb(path: Path, payload: dict) -> None:
    """Write lossless float PBR factors and per-face materials without 8-bit color quantization."""
    import numpy as np

    binary = bytearray()
    views, accessors = [], []

    def accessor(data: Any, component: int, kind: str, target: int) -> int:
        array = np.ascontiguousarray(data, dtype="<f4" if component == 5126 else "<u4")
        while len(binary) % 4:
            binary.append(0)
        start = len(binary)
        binary.extend(array.tobytes())
        views.append({"buffer": 0, "byteOffset": start, "byteLength": array.nbytes, "target": target})
        item = {"bufferView": len(views) - 1, "componentType": component, "count": len(array), "type": kind}
        if kind == "VEC3":
            item.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        accessors.append(item)
        return len(accessors) - 1

    positions = accessor(payload["positions"], 5126, "VEC3", 34962)
    normals = accessor(payload["normals"], 5126, "VEC3", 34962)
    primitives = [
        {
            "attributes": {"POSITION": positions, "NORMAL": normals},
            "indices": accessor(faces.reshape(-1), 5125, "SCALAR", 34963),
            "material": i,
        }
        for i, faces in payload["groups"]
    ]
    materials = []
    for mat in payload["materials"]:
        item = {
            "name": mat["name"],
            "pbrMetallicRoughness": {k: mat[k] for k in ("baseColorFactor", "metallicFactor", "roughnessFactor")},
            "doubleSided": True,
        }
        if mat["baseColorFactor"][3] < 1:
            item["alphaMode"] = "BLEND"
        emission = mat.get("emissiveFactor", [0, 0, 0])
        if max(emission) > 1:
            raise ValueError("HDR emission needs an explicit glTF emissive-strength extension")
        item["emissiveFactor"] = emission
        materials.append(item)
    document = {
        "asset": {"version": "2.0", "generator": "Symbiosis USD material bridge"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": primitives}],
        "materials": materials,
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\0" * (-len(binary) % 4)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary))
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def restore_materials(source: Path, urdf: Path, output: Path) -> dict:
    """Create a relocatable render-only companion; leave physics and source assets intact."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    source, urdf = source.resolve(strict=True), urdf.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    stage = Usd.Stage.Open(str(source))
    hashes = {p.realPath: sha256(Path(p.realPath)) for p in stage.GetUsedLayers() if p.realPath}
    # This asset has no image textures. Fail explicitly if a future revision adds them.
    assets = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() == Sdf.ValueTypeNames.Asset and attribute.Get():
                assets.append((attribute.GetPath(), attribute.Get()))
    images = [str(p) for p, v in assets if not v.path.endswith(".mdl")]
    if images:
        raise ValueError(f"Image texture restoration is not supported by this constant-material profile: {images}")
    tree = ET.parse(urdf)
    meshes = {}
    hidden = []
    provenance = urdf.parent / "export_manifest.json"
    link_paths = {}
    if provenance.exists():
        exported = json.loads(provenance.read_text())
        link_paths = {name: path for path, name in exported["compatibility"]["prim_names"].items()}
    owners = {visual: link for link in tree.getroot().findall("link") for visual in link.findall("visual")}
    for visual, link in owners.items():
        ref = visual.find("geometry/mesh")
        if ref is None:
            # The exporter also included wheel collision guides as visible spheres.
            # Respect the composed USD purpose/visibility without changing physics.
            root = stage.GetPrimAtPath(link_paths.get(link.get("name"), ""))
            if not root:
                raise ValueError(f"Missing primitive provenance: {link.get('name')}")
            iterator = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
            matches = []
            for prim in iterator:
                if prim != root and (prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint)):
                    iterator.PruneChildren()
                elif prim.IsA(UsdGeom.Gprim) and prim.GetName() == visual.get("name"):
                    matches.append(prim)
            if len(matches) != 1:
                raise ValueError(f"Ambiguous primitive provenance: {link.get('name')}")
            imageable = UsdGeom.Imageable(matches[0])
            if imageable.ComputePurpose() in ("guide", "proxy") or imageable.ComputeVisibility() == "invisible":
                link.remove(visual)
                hidden.append(str(matches[0].GetPath()))
            else:
                raise ValueError(f"Visible primitive material needs translation: {matches[0].GetPath()}")
            continue
        obj = (urdf.parent / ref.attrib["filename"]).resolve(strict=True)
        if obj not in meshes:
            with obj.open() as stream:
                header = stream.readline().strip()
            prefix = "# Exported from USD: "
            if not header.startswith(prefix):
                raise ValueError(f"Missing USD source provenance: {obj}")
            prefix_length = len(prefix)
            prim = stage.GetPrimAtPath(header[prefix_length:])
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputePurpose() in ("guide", "proxy") or imageable.ComputeVisibility() == "invisible":
                link.remove(visual)
                hidden.append(str(prim.GetPath()))
                continue
            payload, report = mesh_payload(prim, obj)
            name = f"{obj.stem}.glb"
            write_glb(output / name, payload)
            meshes[obj] = {"file": name, "obj_sha256": sha256(obj), **report}
        ref.set("filename", meshes[obj]["file"])
        # URDF per-visual overrides would mask the embedded face materials.
        for material in list(visual.findall("material")):
            visual.remove(material)
    for ref in tree.getroot().findall("link/collision/geometry/mesh"):
        mesh = (urdf.parent / ref.attrib["filename"]).resolve(strict=True)
        ref.set("filename", Path(os.path.relpath(mesh, output)).as_posix())
    target = output / "robot.urdf"
    ET.indent(tree, space="  ")
    tree.write(target, encoding="utf-8", xml_declaration=True)
    if any(sha256(Path(p)) != value for p, value in hashes.items()):
        raise RuntimeError("Source changed while restoring materials")
    report = {
        "source": str(source),
        "source_sha256": hashes,
        "source_unchanged": True,
        "physics_urdf_sha256": sha256(urdf),
        "render_urdf": target.name,
        "texture_images": 0,
        "hidden_visual_prims": hidden,
        "meshes": list(meshes.values()),
        "files_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def rendering_urdf(physics_urdf: Path) -> Path | None:
    """Use a companion only for the exact physical description it was generated from."""
    manifest = physics_urdf.parent / "materials" / "manifest.json"
    if not manifest.exists():
        return None
    report = json.loads(manifest.read_text())
    if report["physics_urdf_sha256"] != sha256(physics_urdf):
        raise ValueError("Robot material companion belongs to a different URDF; regenerate it")
    for name, digest in report["files_sha256"].items():
        if sha256(manifest.parent / name) != digest:
            raise ValueError(f"Robot material companion changed: {name}")
    return manifest.parent / str(report["render_urdf"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = restore_materials(args.source, args.urdf, args.output.resolve())
    try:
        summary = {
            "meshes": len(result["meshes"]),
            "texture_images": result["texture_images"],
            "output": str(args.output),
        }
    except KeyError as exc:
        raise ValueError("Incomplete robot material export report") from exc
    json.dump(summary, sys.stdout)
