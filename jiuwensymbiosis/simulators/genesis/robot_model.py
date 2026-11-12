"""URDF preparation contract shared by robot profiles and Genesis admission.

The source is immutable. All derived files belong to a caller-owned output
directory. Library calls know no robot names and never create or step a scene.
The optional CLI owns a separate, inspection-only Genesis process lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _module(name: str):
    # Workers also load these lightweight modules directly, without core/LLM dependencies.
    return import_module(f"{__package__}.{name}" if __package__ else name)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tool_versions() -> dict:
    result: dict[str, str | None] = {}
    for package in ("genesis-world", "trimesh", "coacd", "numpy", "mujoco", "fast-simplification"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _numbers(text: str, count: int, *, positive: bool = False) -> list[float]:
    values = [float(v) for v in text.split()]
    if len(values) != count or any(not math.isfinite(v) for v in values):
        raise ValueError(f"invalid URDF numeric vector: {text}")
    if positive and any(v <= 0 for v in values):
        raise ValueError(f"invalid URDF numeric vector: {text}")
    return values


def _origin(element: ET.Element) -> dict:
    origin = element.find("origin")
    attrs = {} if origin is None else origin.attrib
    return {key: _numbers(attrs.get(key, "0 0 0"), 3) for key in ("xyz", "rpy")}


def _limit_value(text: str) -> float | str:
    value = float(text)
    if math.isnan(value):
        raise ValueError("NaN joint limit")
    return value if math.isfinite(value) else ("+infinity" if value > 0 else "-infinity")


def _inspect_mesh(path: Path) -> dict:
    import numpy as np
    import trimesh

    if path.suffix.lower() not in {".obj", ".stl", ".ply"}:
        raise ValueError(f"unsupported collision mesh format: {path.suffix}")
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"collision resource is not a triangle mesh: {path}")
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    vertex_shape_ok = vertices.ndim == 2 and vertices.shape[1:] == (3,) and len(vertices) >= 3
    face_shape_ok = faces.ndim == 2 and faces.shape[1:] == (3,) and len(faces) > 0
    if not vertex_shape_ok or not face_shape_ok:
        raise ValueError(f"invalid collision triangle mesh: {path}")
    if not np.isfinite(vertices).all() or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"invalid collision triangle mesh: {path}")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"invalid collision triangle mesh: {path}")
    return {
        "vertices": len(vertices),
        "triangles": len(faces),
        "bounds": mesh.bounds.tolist(),
        "watertight": bool(mesh.is_watertight),
        "convex": bool(mesh.is_convex),
        "load_process": False,
    }


def inspect_structure(root: ET.Element) -> tuple[list[dict], list[dict]]:
    if root.tag != "robot":
        raise ValueError("expected a URDF robot")
    links: list[dict[str, Any]] = []
    joints: list[dict[str, Any]] = []
    for link in root.findall("link"):
        collisions = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError("each collision requires exactly one geometry")
            shape = geometry[0]
            attributes: dict[str, Any] = dict(shape.attrib)
            if shape.tag == "mesh":
                attributes["scale"] = _numbers(shape.get("scale", "1 1 1"), 3, positive=True)
            elif shape.tag == "box":
                attributes["size"] = _numbers(shape.attrib["size"], 3, positive=True)
            elif shape.tag in {"sphere", "cylinder"}:
                for key in ("radius",) if shape.tag == "sphere" else ("radius", "length"):
                    attributes[key] = _numbers(shape.attrib[key], 1, positive=True)[0]
            else:
                raise ValueError(f"unsupported URDF collision geometry: {shape.tag}")
            collisions.append({"origin": _origin(collision), "type": shape.tag, **attributes})
        inertial = link.find("inertial")
        links.append(
            {
                "name": link.attrib["name"],
                "collisions": collisions,
                "inertial_xml": None if inertial is None else ET.tostring(inertial, encoding="unicode"),
            }
        )
    names = [link["name"] for link in links]
    if not names or len(set(names)) != len(names):
        raise ValueError("missing or duplicate URDF links")
    children = set()
    for joint in root.findall("joint"):
        kind = joint.attrib["type"]
        if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
            raise ValueError(f"unsupported URDF joint type: {kind}")
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            raise ValueError("unresolved URDF joint link")
        if parent.get("link") not in names or child.get("link") not in names:
            raise ValueError("unresolved URDF joint link")
        if child.get("link") in children:
            raise ValueError("multiple parents/closed loops are unsupported")
        children.add(child.get("link"))
        mimic, limit, axis = joint.find("mimic"), joint.find("limit"), joint.find("axis")
        axis_values = _numbers("1 0 0" if axis is None else axis.get("xyz", "1 0 0"), 3)
        if kind != "fixed" and sum(v * v for v in axis_values) < 1e-12:
            raise ValueError("zero URDF joint axis")
        joints.append(
            {
                "name": joint.attrib["name"],
                "type": kind,
                "parent": parent.get("link"),
                "child": child.get("link"),
                "origin": _origin(joint),
                "axis": axis_values,
                "limit": {} if limit is None else {k: _limit_value(v) for k, v in limit.attrib.items()},
                "mimic": None
                if mimic is None
                else {
                    "joint": mimic.attrib["joint"],
                    "multiplier": _numbers(mimic.get("multiplier", "1"), 1)[0],
                    "offset": _numbers(mimic.get("offset", "0"), 1)[0],
                },
            }
        )
    joint_names = [j["name"] for j in joints]
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("duplicate URDF joint names")
    roots = set(names) - children
    if len(roots) != 1:
        raise ValueError("URDF must have one root")
    reached = set(roots)
    while True:
        expanded = reached | {j["child"] for j in joints if j["parent"] in reached}
        if expanded == reached:
            break
        reached = expanded
    if reached != set(names):
        raise ValueError("disconnected/cyclic URDF")
    resolved = {j["name"] for j in joints if j["mimic"] is None}
    while True:
        expanded = resolved | {j["name"] for j in joints if j["mimic"] and j["mimic"]["joint"] in resolved}
        if expanded == resolved:
            break
        resolved = expanded
    if resolved != set(joint_names):
        raise ValueError("unresolved or cyclic mimic joints")
    return links, joints


def prepare_robot_model(
    source: Path, directory: Path, *, policy: dict | None = None, packages: dict[str, Path] | None = None
) -> dict:
    """Resolve resources, validate structure, apply explicit policy, write manifest.

    URDF dimensions are metres/radians; mesh scale is retained. Package URIs
    require an explicit package-root mapping. No guessed paths or unit conversion.
    """
    source = Path(source).resolve(strict=True)
    directory = Path(directory).resolve()
    if source.suffix.lower() != ".urdf":
        raise ValueError("only URDF preparation is supported")
    if source.is_relative_to(directory):
        raise ValueError("derived output directory must not contain source assets")
    original = source.read_bytes()
    policy = deepcopy(policy)
    root = ET.fromstring(original)
    resources: dict[str, dict[str, Any]] = {}
    for mesh in root.findall(".//mesh"):
        uri = mesh.attrib["filename"]
        if uri.startswith("package://"):
            package, relative = uri.removeprefix("package://").split("/", 1)
            if not packages or package not in packages:
                raise ValueError(f"unresolved package URI: {uri}")
            base = Path(packages[package]).resolve()
            path = (base / relative).resolve(strict=True)
            if not path.is_relative_to(base):
                raise ValueError(f"package URI escapes root: {uri}")
        elif "://" in uri:
            raise ValueError(f"unsupported mesh URI: {uri}")
        else:
            path = (source.parent / uri).resolve(strict=True)
        resources[str(path)] = {"path": str(path), "sha256": file_hash(path)}
        mesh.set("filename", str(path))
    for mesh in root.findall(".//collision/geometry/mesh"):
        path = Path(mesh.attrib["filename"])
        if "mesh_validation" not in resources[str(path)]:
            resources[str(path)]["mesh_validation"] = _inspect_mesh(path)
    links, joints = inspect_structure(root)
    preparation = None
    if policy is not None:
        preparation = _module("collision_preparation").apply_collision_policy(root, directory, policy)
        links, joints = inspect_structure(root)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "robot.urdf"
    ET.indent(root)
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    for mesh in root.findall(".//collision/geometry/mesh"):
        path = Path(mesh.attrib["filename"])
        if str(path) not in resources:
            resources[str(path)] = {
                "path": str(path),
                "sha256": file_hash(path),
                "mesh_validation": _inspect_mesh(path),
            }
    manifest = {
        "schema_version": 1,
        "preparer_version": 1,
        "tool_versions": tool_versions(),
        "format": "urdf",
        "units": {"length": "m", "angle": "rad", "mass": "kg"},
        "source": {"path": str(source), "sha256": hashlib.sha256(original).hexdigest()},
        "urdf": str(target),
        "model_sha256": file_hash(target),
        "resources": list(resources.values()),
        "links": links,
        "joints": joints,
        "collision_preparation": preparation,
        "policy": policy,
    }
    verify_model_files(manifest)
    (directory / "model.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def verify_model_files(manifest: dict) -> None:
    """Reject changed source, derived model or dependencies before/after import."""
    records = [
        manifest["source"],
        *manifest["resources"],
        {"path": manifest["urdf"], "sha256": manifest["model_sha256"]},
    ]
    for record in records:
        if file_hash(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"model resource changed: {record['path']}")


def admission_morph_options(*, fixed: bool, decimate: bool) -> dict:
    """Pinned import settings shared by the CLI and scene controller."""
    return {
        "fixed": fixed,
        "align": False,
        "merge_fixed_links": False,
        "collision": True,
        "convexify": True,
        "decimate": decimate,
        "watertighten": 5,
        "decompose_robot_error_threshold": float("inf"),
    }


def main() -> None:
    """Explicit prepare/inspect CLI; no controllers, actions or simulation steps."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, help="YAML collision policy; omitted means no geometry replacement")
    parser.add_argument("--package", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--load-audit", action="store_true", help="Build one Genesis scene and audit; never step")
    parser.add_argument("--floating", action="store_true", help="Use floating rather than fixed base for inspection")
    parser.add_argument("--decimate", action="store_true", help="Reproduce the mobile runtime simplification")
    parser.add_argument("--self-collision", action="store_true", help="Enable robot self-collision in the audit scene")
    args = parser.parse_args()
    policy = None
    if args.policy:
        import yaml

        policy = yaml.safe_load(args.policy.read_text())
    packages = {}
    for item in args.package:
        name, separator, path = item.partition("=")
        if not separator or not name.strip() or not path.strip():
            parser.error("--package requires NAME=PATH with nonempty name and path")
        packages[name] = Path(path)
    manifest = prepare_robot_model(args.source, args.output, policy=policy, packages=packages)
    if args.load_audit:
        import genesis as gs

        gs.init(backend=gs.cpu, logging_level="warning")
        scene = None
        try:
            scene = gs.Scene(
                show_viewer=False, rigid_options=gs.options.RigidOptions(enable_self_collision=args.self_collision)
            )
            entity = scene.add_entity(
                gs.morphs.URDF(
                    file=manifest["urdf"], **admission_morph_options(fixed=not args.floating, decimate=args.decimate)
                )
            )
            scene.build()
            _module("collision_audit").audit_loaded_model(entity, manifest, args.output / "loaded-model.json")
        finally:
            try:
                if scene is not None:
                    scene.destroy()
            finally:
                gs.destroy()
    import logging

    logger = logging.getLogger("jiuwensymbiosis.simulators.genesis.robot_model")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.info("Model: %s", args.output / "model.json")


if __name__ == "__main__":
    main()
