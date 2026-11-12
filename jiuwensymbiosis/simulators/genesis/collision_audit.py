"""Read back built collision geometry without rendering, stepping, or driving.

Genesis 1.4.4's public ``RigidGeom.init_verts/init_faces`` are the processed
collision topology used to initialize the solver. ``get_verts()`` refreshes and
returns world collision vertices. Neither path reads visual geometry or infers
geometry from its metadata. Call once on the scene's owning thread after build.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _shape_support(collision: dict, directions: np.ndarray) -> np.ndarray:
    """Authored support in link coordinates, independent of importer metadata."""
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_euler("xyz", collision["origin"]["rpy"])
    local_directions = rotation.inv().apply(directions)
    kind = collision["type"]
    if kind == "mesh":
        import trimesh

        mesh = trimesh.load(collision["filename"], force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("expected triangle mesh for spatial audit")
        vertices = _array(mesh.vertices) * collision["scale"]
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
            raise ValueError("invalid source mesh for spatial audit")
        support = np.max(vertices @ local_directions.T, axis=0)
    elif kind == "box":
        support = np.abs(local_directions) @ (np.asarray(collision["size"]) / 2)
    elif kind == "sphere":
        support = np.linalg.norm(local_directions, axis=1) * collision["radius"]
    elif kind == "cylinder":
        support = (
            np.linalg.norm(local_directions[:, :2], axis=1) * collision["radius"]
            + np.abs(local_directions[:, 2]) * collision["length"] / 2
        )
    else:
        raise ValueError(f"unsupported collision shape: {kind}")
    return np.asarray(support + directions @ np.asarray(collision["origin"]["xyz"]))


def _processed_expectations(robot: Any, directions: np.ndarray) -> dict:
    """Reparse source with pinned SDK; no entity, scene, solver or step is created.

    MuJoCo recentres/reorders and quantizes mesh vertices before simplification.
    Skipping that step changes QEM results, even for identical source meshes.
    This checks loading consistency, not independent correctness of the importer.
    """
    import genesis as gs
    from genesis.utils import mesh as mesh_utils
    from genesis.utils import mjcf
    from scipy.spatial.transform import Rotation

    morph = robot.morph
    if (not morph.convexify and morph.watertighten is not None) or np.isfinite(morph.decompose_robot_error_threshold):
        raise ValueError("audit requires explicit prepared parts, without implicit wrapping/decomposition")
    links, _, geometries, _ = mjcf.parse_xml(morph.model_copy(update={"visualization": False}), gs.surfaces.Default())
    result = {}
    for link, geoms in zip(links, geometries, strict=True):
        collisions = [g for g in geoms if g["contype"] or g["conaffinity"]]
        processed = mesh_utils.postprocess_collision_geoms(
            collisions,
            morph.decimate,
            morph.decimate_face_num,
            morph.decimate_aggressiveness,
            morph.convexify,
            morph.decompose_robot_error_threshold,
            morph.coacd_options,
            morph.watertighten,
        )
        supports = []
        for geom in processed:
            vertices = Rotation.from_quat(geom["quat"], scalar_first=True).apply(_array(geom["mesh"].verts))
            vertices += _array(geom["pos"])
            supports.append(np.max(vertices @ directions.T, axis=0))
        result[link["name"]] = supports
    return result


def _spatial_comparison(robot: Any, manifest: dict, processing: dict) -> list[dict]:
    from itertools import product

    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.transform import Rotation

    directions = np.asarray([d for d in product((-1, 0, 1), repeat=3) if any(d)], dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    processed = _processed_expectations(robot, directions) if processing["decimate"] else None
    actual = {link.name: link for link in robot.links}
    results = []
    for link in manifest["links"]:
        expected = link["collisions"]
        runtime = actual.get(link["name"])
        if runtime is None or len(runtime.geoms) != len(expected):
            results.append({"link": link["name"], "status": "error", "reason": "count_or_link_mismatch"})
            continue
        if not expected:
            continue
        authored = np.asarray(
            processed[link["name"]] if processed is not None else [_shape_support(c, directions) for c in expected]
        )
        if len(authored) != len(expected):
            raise ValueError("source reprocessing changed collision instance count")
        loaded = []
        for geom in runtime.geoms:
            vertices = Rotation.from_quat(_array(geom.init_quat), scalar_first=True).apply(_array(geom.init_verts))
            vertices += _array(geom.init_pos)
            loaded.append(np.max(vertices @ directions.T, axis=0))
        cost = np.max(np.abs(authored[:, None, :] - np.asarray(loaded)[None, :, :]), axis=2)
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols, strict=True):
            # Primitive tessellation is approximate in Genesis. Mesh hulls must
            # preserve the declared processing pipeline's support.
            tolerance = 1e-5
            if processed is None and expected[row]["type"] in {"sphere", "cylinder"}:
                tolerance += expected[row]["radius"] * 0.025
            error = float(cost[row, col])
            results.append(
                {
                    "link": link["name"],
                    ("collision_index" if processed is None else "processed_index"): int(row),
                    "geom_idx": int(runtime.geoms[col].idx),
                    "support_error_m": error,
                    "tolerance_m": tolerance,
                    "status": "pass" if error <= tolerance else "error",
                }
            )
    return results


def audit_loaded_model(robot: Any, manifest: dict, destination: Path) -> dict:
    """Persist evidence and fail closed before a controller starts issuing motion.

    Supports unmerged URDF links and one environment. A shape check compares 26
    directional supports in link space; it detects misplaced/scaled/wrong meshes
    but does not prove concave surface equality or acceptable contact dynamics.
    """
    from importlib import import_module

    model_module = import_module(f"{__package__}.robot_model" if __package__ else "robot_model")
    report: dict = {"schema_version": 1, "status": "error", "errors": [], "warnings": [], "unverifiable": []}
    failure = None
    try:
        model_module.verify_model_files(manifest)
        if any(getattr(link, "aligned", False) for link in robot.links):
            raise ValueError("model admission requires authored link frames; build with align=False")
        report["geometry"] = collision_geometry_audit(robot, manifest)
        geometry = report["geometry"]
        if not any(link.geoms for link in robot.links):
            report["errors"].append({"check": "no_collision_geometry"})
        for key in ("missing_runtime_links", "count_mismatch_links", "source_mismatch_links"):
            if geometry[key]:
                report["errors"].append({"check": key, "links": geometry[key]})
        morph = getattr(robot, "morph", None)
        processing = {
            "convexify": bool(getattr(morph, "convexify", True)),
            "decimate": bool(getattr(morph, "decimate", False)),
            "decimate_face_num": int(getattr(morph, "decimate_face_num", 500)),
            "decimate_aggressiveness": int(getattr(morph, "decimate_aggressiveness", 2)),
        }
        report["mesh_processing"] = processing
        report["import_options"] = {
            "fixed": getattr(morph, "fixed", None),
            "align": getattr(morph, "align", None),
            "merge_fixed_links": getattr(morph, "merge_fixed_links", None),
            "watertighten": getattr(morph, "watertighten", None),
            "decompose_robot_error_threshold": str(getattr(morph, "decompose_robot_error_threshold", None)),
        }
        report["tool_versions"] = model_module.tool_versions()
        report["spatial"] = _spatial_comparison(robot, manifest, processing)
        if any(item["status"] != "pass" for item in report["spatial"]):
            report["errors"].append({"check": "collision_shape_or_transform"})
        flag_names = ("collision", "self_collision", "adjacent_collision", "neutral_collision")
        report["solver_flags"] = {key: bool(getattr(robot.solver, f"_enable_{key}")) for key in flag_names}
        if not report["solver_flags"]["collision"]:
            report["errors"].append({"check": "collision_disabled"})
        # Pinned Genesis 1.4.4: this is the collider's built pair table, not an
        # inference from its options. Includes adjacent/neutral/mask exclusions.
        pairs = _array(getattr(robot.solver.collider, "_valid_collision_pairs")).astype(int)  # noqa: B009 -- SDK ABI
        own = {g.idx: link.name for link in robot.links for g in link.geoms}
        report["pair_filter"] = {
            "source": "Genesis 1.4.4 Collider._valid_collision_pairs",
            "robot_geometry_links": own,
            "eligible_robot_pairs": [p.tolist() for p in pairs if p[0] in own and p[1] in own],
            "eligible_environment_pairs": [p.tolist() for p in pairs if (p[0] in own) != (p[1] in own)],
            "interpretation": "Absent geometry pairs are filtered; eligibility is not proof of contact.",
        }
        if any(
            not (g["contype"] or g["conaffinity"]) for link in geometry["links"].values() for g in link["geometries"]
        ):
            report["errors"].append({"check": "collision_masks_disabled"})
        report["limitations"] = ["Physical contact stability and exact concave surface equivalence are not tested."]
        report["warnings"].append("Directional supports do not establish exact concave surface equivalence.")
        model_module.verify_model_files(manifest)
        report["status"] = "pass" if not report["errors"] else "error"
    except Exception as exc:
        failure = exc
        if isinstance(exc, (ImportError, AttributeError, TypeError)):
            report["status"] = "unverifiable"
            report["unverifiable"].append(f"{type(exc).__name__}: {exc}")
        report["errors"].append({"check": "readback", "error": f"{type(exc).__name__}: {exc}"})
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["status"] != "pass":
        raise ValueError(f"robot model admission failed; inspect {destination}") from failure
    return report


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result: np.ndarray = np.asarray(value)
    if not np.isfinite(result).all():
        raise ValueError("collision audit requires finite geometry")
    return result


def _bounds(vertices: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in point] for point in (vertices.min(axis=0), vertices.max(axis=0))]


def _topology_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256(json.dumps([vertices.shape, faces.shape]).encode())
    digest.update(np.ascontiguousarray(vertices, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(faces, dtype="<i8").tobytes())
    return digest.hexdigest()


def _file_reference(uri: str, directory: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"uri": uri, "sha256": None}
    if "://" in uri:
        return {**result, "status": "unresolved_uri"}
    path = (directory / uri).resolve()
    result["resolved_path"] = str(path)
    try:
        with path.open("rb") as stream:
            result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        return {**result, "status": "unavailable", "error": str(exc)}
    return {**result, "status": "available"}


def _geometry(geom: Any, directory: Path, references: dict[str, dict]) -> dict:
    local = _array(geom.init_verts)
    faces = _array(geom.init_faces)
    world = _array(geom.get_verts())
    if world.ndim == 3 and world.shape[0] == 1:
        world = world[0]
    if local.ndim != 2 or local.shape[1:] != (3,) or len(local) == 0:
        raise ValueError("invalid collision mesh local vertices")
    if world.ndim != 2 or world.shape[1:] != (3,):
        raise ValueError("collision audit requires non-parallel world vertices")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("invalid collision mesh triangle faces")
    if world.shape != local.shape or len(local) != geom.n_verts or len(faces) != geom.n_faces:
        raise ValueError("collision mesh topology and runtime vertex counts disagree")
    if faces.size and (faces.min() < 0 or faces.max() >= len(local)):
        raise ValueError("collision mesh faces reference invalid vertices")

    # Canonical numeric encoding, preserving vertex/face order. This fingerprints
    # the processed shape, not the source file, its metadata, or the current pose.
    reference: dict[str, Any] = {"status": "not_provided", "uri": None, "sha256": None}
    metadata = geom.metadata
    uri = metadata.get("mesh_path") if isinstance(metadata, Mapping) else None
    if not isinstance(metadata, Mapping):
        reference["status"] = "invalid_metadata"
    elif uri is not None and (not isinstance(uri, (str, Path)) or not str(uri)):
        # A list of possible sources cannot identify one loaded geometry.
        reference["status"] = "invalid_uri"
        reference["value_type"] = type(uri).__name__
    elif uri is not None:
        uri = str(uri)
        if uri not in references:
            references[uri] = _file_reference(uri, directory)
        reference = references[uri]
    return {
        "geom_idx": int(geom.idx),
        "origin_xyz": _array(geom.init_pos).tolist(),
        "origin_quat_wxyz": _array(geom.init_quat).tolist(),
        "geom_type": str(geom.type),
        "is_convex": bool(geom.is_convex),
        "contype": int(geom.contype),
        "conaffinity": int(geom.conaffinity),
        "vertex_count": len(local),
        "face_count": len(faces),
        "bounds_geom_local_m": _bounds(local),
        "bounds_world_m": _bounds(world),
        "processed_geometry_sha256": _topology_hash(local, faces),
        "source_reference": dict(reference),
    }


def _source_topology(reference: dict) -> dict:
    """Read an unprocessed source mesh only when a path alias needs evidence."""
    path = reference["resolved_path"]
    result = {"resolved_path": path, "sha256": reference["sha256"]}
    try:
        import trimesh

        mesh = trimesh.load(path, process=False, maintain_order=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("source alias requires one mesh; scenes are not flattened")
        vertices, faces = _array(mesh.vertices), _array(mesh.faces)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
            raise ValueError("invalid source mesh vertices")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
            raise ValueError("invalid source mesh triangle faces")
        if not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("invalid source mesh triangle faces")
        with Path(path).open("rb") as stream:
            current_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if current_hash != reference["sha256"]:
            raise ValueError("source file changed during collision audit")
        return {
            **result,
            "status": "available",
            "source_topology_sha256": _topology_hash(vertices, faces),
            "vertex_count": len(vertices),
            "face_count": len(faces),
        }
    except Exception as exc:
        return {**result, "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def _resolve_source_aliases(missing: Counter, extra: Counter, references: dict[str, dict], cache: dict) -> list[dict]:
    aliases = []
    for expected_path in sorted(missing):
        if references[expected_path]["status"] != "available":
            continue
        for actual_path in sorted(extra):
            if not missing[expected_path] or not extra[actual_path] or references[actual_path]["status"] != "available":
                continue
            for path in (expected_path, actual_path):
                if path not in cache:
                    cache[path] = _source_topology(references[path])
            authored, runtime = cache[expected_path], cache[actual_path]
            if authored["status"] != "available" or runtime["status"] != "available":
                continue
            if authored["source_topology_sha256"] != runtime["source_topology_sha256"]:
                continue
            count = min(missing[expected_path], extra[actual_path])
            missing[expected_path] -= count
            extra[actual_path] -= count
            aliases.append(
                {
                    "expected_path": expected_path,
                    "runtime_path": actual_path,
                    "expected_file_sha256": authored["sha256"],
                    "runtime_file_sha256": runtime["sha256"],
                    "source_topology_sha256": authored["source_topology_sha256"],
                    "vertex_count": authored["vertex_count"],
                    "face_count": authored["face_count"],
                    "instance_count": count,
                    "evidence": "identical ordered source vertices/faces with trimesh "
                    "process=False, maintain_order=True; "
                    "no transforms, scaling, convexification, or tolerance applied",
                }
            )
    return aliases


def _source_comparison(
    expected: dict | None, geoms: list[dict], *, link_available: bool, topology_cache: dict | None = None
) -> dict:
    """Compare source instances by full path; filenames alone are not identities."""
    expected_counts: Counter[str] = Counter()
    actual_counts: Counter[str] = Counter()
    unavailable_expected = []
    unverifiable_actual = []
    no_source_indices = []
    references = {}
    for reference in () if expected is None else expected["mesh_references"]:
        if reference.get("resolved_path") is not None:
            expected_counts[reference["resolved_path"]] += 1
            references[reference["resolved_path"]] = reference
        if reference["status"] != "available":
            unavailable_expected.append(reference["collision_index"])
    for geom in geoms:
        reference = geom["source_reference"]
        if reference.get("resolved_path") is not None:
            actual_counts[reference["resolved_path"]] += 1
            references[reference["resolved_path"]] = reference
        if reference["status"] == "not_provided":
            no_source_indices.append(geom["geom_idx"])
        elif reference["status"] != "available":
            unverifiable_actual.append(geom["geom_idx"])
    non_mesh_count = None if expected is None else expected["non_mesh_collision_count"]
    missing, extra = expected_counts - actual_counts, actual_counts - expected_counts
    if topology_cache is None:
        topology_cache = {}
    aliases = _resolve_source_aliases(missing, extra, references, topology_cache)
    reasons = []
    if expected is None:
        reasons.append("missing_urdf_link")
    if not link_available:
        reasons.append("missing_runtime_link")
    if expected is not None and len(geoms) != expected["collision_count"]:
        reasons.append("collision_instance_count_mismatch")
    if +missing or +extra:
        reasons.append("mesh_source_path_or_instance_count_mismatch")
    if unavailable_expected:
        reasons.append("unavailable_urdf_mesh_source")
    if unverifiable_actual:
        reasons.append("unverifiable_runtime_mesh_source")
    if non_mesh_count is not None and len(no_source_indices) != non_mesh_count:
        reasons.append("missing_or_unexpected_runtime_mesh_source")
    return {
        "matches_urdf": not reasons,
        "paths_match_urdf": expected_counts == actual_counts,
        "method": "full-path multiset with exact ordered source-topology evidence for aliases; "
        "instance counts consumed",
        "reasons": reasons,
        "expected_mesh_path_counts": dict(sorted(expected_counts.items())),
        "runtime_mesh_path_counts": dict(sorted(actual_counts.items())),
        "missing_mesh_path_counts": dict(sorted((expected_counts - actual_counts).items())),
        "unexpected_mesh_path_counts": dict(sorted((actual_counts - expected_counts).items())),
        "unresolved_missing_mesh_path_counts": dict(sorted((+missing).items())),
        "unresolved_unexpected_mesh_path_counts": dict(sorted((+extra).items())),
        "resolved_source_aliases": aliases,
        "source_topology_read_errors": {
            path: record
            for path, record in sorted(topology_cache.items())
            if path in references and record["status"] != "available"
        },
        "unavailable_urdf_collision_indices": unavailable_expected,
        "unverifiable_runtime_geom_indices": unverifiable_actual,
        "runtime_without_mesh_source_geom_indices": no_source_indices,
        "expected_non_mesh_collision_count": non_mesh_count,
    }


def collision_geometry_audit(robot: Any, model: Mapping[str, Any], *, link_names: Sequence[str] | None = None) -> dict:
    """Return JSON collision-shape evidence for one built, non-parallel robot.

    Default selection is the union of runtime and URDF link names, including
    links with no collisions. Missing runtime links are unknown, never zero.
    A URDF/runtime count match is only a count check, not shape equivalence or
    proof of physical accuracy. Readback failures propagate, without substituting
    source meshes or metadata for unavailable runtime geometry.

    Source references are provenance only: their hashes describe files currently
    on disk, not proof those bytes were used at build time. The independent
    processed-geometry hash and bounds always come from runtime collision data.
    Source matching requires the full resolved path or exactly equal ordered
    source vertices/faces, and the number of instances per link. It fails closed
    for missing or ambiguous mesh source references.
    It does not assume the importer preserves collision order.
    """
    if not robot.is_built:
        raise ValueError("collision audit requires a built robot")
    if robot.solver.n_envs not in (0, 1):
        raise ValueError("collision audit supports one scene (unbatched or n_envs=1)")
    path = Path(model["urdf"]).resolve(strict=True)
    urdf_bytes = path.read_bytes()
    root = ET.fromstring(urdf_bytes)
    authored = {}
    references: dict[str, dict] = {}
    for element in root.findall("link"):
        name = element.attrib["name"]
        if name in authored:
            raise ValueError(f"duplicate URDF link: {name}")
        collisions = element.findall("collision")
        mesh_references = []
        for index, collision in enumerate(collisions):
            for mesh in collision.findall("geometry/mesh"):
                uri = mesh.attrib["filename"]
                if uri not in references:
                    references[uri] = _file_reference(uri, path.parent)
                mesh_references.append({**references[uri], "collision_index": index})
        authored[name] = {
            "collision_count": len(collisions),
            "mesh_uris": [reference["uri"] for reference in mesh_references],
            "mesh_references": mesh_references,
            "non_mesh_collision_count": sum(collision.find("geometry/mesh") is None for collision in collisions),
        }
    actual = {link.name: link for link in robot.links}
    if len(actual) != len(robot.links):
        raise ValueError("duplicate runtime link names")
    selected = sorted(actual.keys() | authored.keys()) if link_names is None else list(link_names)
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate requested link names")
    records: dict[str, dict] = {}
    indices: set[int] = set()
    topology_cache: dict[str, dict] = {}
    for name in selected:
        expected = authored.get(name)
        link = actual.get(name)
        geoms = []
        if link is not None:
            for geom in link.geoms:
                record = _geometry(geom, path.parent, references)
                if record["geom_idx"] in indices:
                    raise ValueError("duplicate runtime collision geometry index")
                indices.add(record["geom_idx"])
                geoms.append(record)
        count = None if link is None else len(geoms)
        sources = _source_comparison(expected, geoms, link_available=link is not None, topology_cache=topology_cache)
        records[name] = {
            "status": "available" if link is not None else "missing_runtime_link",
            "link_idx": None if link is None else int(link.idx),
            "collision_geom_count": count,
            "collision_masks_nonzero_count": None
            if link is None
            else sum(bool(g["contype"] or g["conaffinity"]) for g in geoms),
            "urdf_expectation": expected,
            "count_matches_urdf": None if expected is None or link is None else count == expected["collision_count"],
            "source_matches_urdf": sources["matches_urdf"],
            "source_comparison": sources,
            "geometries": geoms,
        }
    file_hash = hashlib.sha256(urdf_bytes).hexdigest()
    recorded_hash = model.get("model_sha256")
    return {
        "schema_version": 1,
        "status": "available",
        "units": {"length": "m"},
        "scope": "selected robot collision geometries at call time; no visual geometry",
        "sources": {
            "local_topology": "RigidGeom.init_verts/init_faces: processed geometry-local collision mesh",
            "world_vertices": "RigidGeom.get_verts(): refreshed world collision vertices, no extra transform",
            "source_reference": "RigidGeom.metadata.mesh_path: provenance only; file hash is current disk content",
            "shape_hash_encoding": "JSON [vertex_shape,face_shape] + "
            "C-order little-endian float64 vertices/int64 faces",
        },
        "model": {
            "urdf": str(path),
            "current_file_sha256": file_hash,
            "manifest_sha256": recorded_hash,
            "matches_manifest_sha256": None if recorded_hash is None else recorded_hash == file_hash,
        },
        "collision_geom_count": len(indices),
        "missing_runtime_links": [name for name, item in records.items() if item["status"] == "missing_runtime_link"],
        "count_mismatch_links": [name for name, item in records.items() if item["count_matches_urdf"] is False],
        "source_mismatch_links": [name for name, item in records.items() if not item["source_matches_urdf"]],
        "source_path_mismatch_links": [
            name for name, item in records.items() if not item["source_comparison"]["paths_match_urdf"]
        ],
        "resolved_source_aliases": [
            {"link": name, **alias}
            for name, item in records.items()
            for alias in item["source_comparison"]["resolved_source_aliases"]
        ],
        "links": records,
        "limitations": [
            "Counts and bounds do not prove exact surface equivalence or acceptable contact physics.",
            "Fixed-link merging and importer processing may intentionally change per-link collision counts.",
            "Matching source paths do not prove build-time file bytes or processed topology match the source mesh.",
            "Resolved aliases prove current source topology equality, "
            "not runtime processing or collision-origin equality.",
            "Repeated instances of one source are checked by count, not by collision origin or scale correspondence.",
            "A nonzero collision mask alone does not prove a particular pair can collide.",
        ],
    }
