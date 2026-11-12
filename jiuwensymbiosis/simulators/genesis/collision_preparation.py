"""Content-addressed convex decomposition for explicit URDF collision policies.

No robot-specific names, default exclusions, or fallback convex hulls.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import io
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# Explicit screening candidate shared with the retained geometry diagnostic.
# These parameters have no implicit claim of contact-error or grasp acceptance.
DEFAULT_COACD_PARAMETERS: dict[str, Any] = {
    "threshold": 0.03,
    "max_convex_hull": 32,
    "preprocess_mode": "auto",
    "preprocess_resolution": 50,
    "resolution": 4000,
    "mcts_nodes": 20,
    "mcts_iterations": 150,
    "mcts_max_depth": 3,
    "pca": False,
    "merge": True,
    "decimate": False,
    "max_ch_vertex": 256,
    "extrude": False,
    "extrude_margin": 0.01,
    "apx_mode": "ch",
    "seed": 0,
    "real_metric": False,
}


def _validated_part(vertices, faces):
    import numpy as np
    import trimesh

    vertices, faces = np.asarray(vertices), np.asarray(faces)
    vertex_shape_ok = vertices.ndim == 2 and vertices.shape[1:] == (3,) and len(vertices) >= 4
    face_shape_ok = faces.ndim == 2 and faces.shape[1:] == (3,) and len(faces) >= 4
    if not vertex_shape_ok or not face_shape_ok:
        raise ValueError("invalid decomposition vertices or triangle indices")
    if not np.isfinite(vertices).all() or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("invalid decomposition vertices or triangle indices")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("invalid decomposition vertices or triangle indices")
    part = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not part.is_watertight or not part.is_winding_consistent:
        raise ValueError("decomposition part is not closed with consistent winding")
    if not np.isfinite(part.volume) or part.volume <= 0:
        raise ValueError("decomposition part must have positive finite volume")
    if not part.is_convex:
        raise ValueError("decomposition part is not convex")
    return part


def _validate_frame(parts, source_mesh, parameters) -> dict:
    import numpy as np

    if not parts or len(parts) > parameters["max_convex_hull"]:
        raise ValueError("decomposition returned an empty or excessive part set")
    source_bounds = source_mesh.bounds
    bounds = np.array(
        [np.min([part.bounds[0] for part in parts], axis=0), np.max([part.bounds[1] for part in parts], axis=0)]
    )
    # CoACD's automatic remeshing may move surface vertices by voxel-sized
    # amounts. This guards lost units/translations, not surface-error acceptance.
    tolerance = max(1e-8, float(np.max(source_mesh.extents)) * 2 / parameters["preprocess_resolution"])
    drift = float(np.max(np.abs(bounds - source_bounds)))
    if not np.isfinite(drift) or drift > tolerance:
        raise ValueError(f"decomposition changed mesh frame or bounds ({drift:g} > {tolerance:g})")
    return {
        "source_bounds": source_bounds.tolist(),
        "parts_bounds": bounds.tolist(),
        "max_bounds_difference": drift,
        "bounds_tolerance": tolerance,
    }


def _part_record(part, filename: str, digest: str) -> dict:
    return {
        "file": filename,
        "sha256": digest,
        "vertices": len(part.vertices),
        "triangles": len(part.faces),
        "volume_mesh_units_cubed": float(part.volume),
        "watertight": True,
        "winding_consistent": True,
        "convex": True,
    }


def _cache_key(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_cache(path: Path, identity: dict, source_mesh) -> dict:
    import trimesh

    try:
        metadata = json.loads((path / "manifest.json").read_text())
        if not isinstance(metadata, dict):
            raise ValueError("manifest must be a JSON object")
        if metadata["identity"] != identity:
            raise ValueError("identity mismatch")
        parts = []
        key = _cache_key(identity)
        for index, record in enumerate(metadata["parts"]):
            filename = f"{key}_part_{index:03d}.obj"
            if record["file"] != filename:
                raise ValueError("invalid part filename")
            part_path = path / filename
            if part_path.is_symlink() or hashlib.sha256(part_path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("part checksum mismatch")
            mesh = trimesh.load(part_path, force="mesh", process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError("cached decomposition part is not a triangle mesh")
            part = _validated_part(mesh.vertices, mesh.faces)
            if _part_record(part, filename, record["sha256"]) != record:
                raise ValueError("part validation metadata mismatch")
            parts.append(part)
        if _validate_frame(parts, source_mesh, identity["parameters"]) != metadata["frame_validation"]:
            raise ValueError("frame validation metadata mismatch")
        return metadata
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"invalid cached decomposition at {path}: {exc}") from exc


def _cached_parts(source: Path, directory: Path, version: str, parameters: dict) -> dict:
    import fcntl

    import numpy as np
    import trimesh

    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    identity = {
        "schema": 2,
        "preparer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": source_sha256,
        "source_file_type": source.suffix.lower(),
        "coacd_version": version,
        "trimesh_version": importlib.metadata.version("trimesh"),
        "source_load_process": True,
        "parameters": dict(parameters),
    }
    key = _cache_key(identity)
    path = directory / key
    source_mesh = trimesh.load(
        io.BytesIO(source_bytes), file_type=source.suffix[1:].lower(), force="mesh", process=True
    )
    if not isinstance(source_mesh, trimesh.Trimesh):
        raise ValueError(f"invalid decomposition source mesh: {source}")
    if len(source_mesh.vertices) < 4 or len(source_mesh.faces) == 0:
        raise ValueError(f"invalid decomposition source mesh: {source}")
    finite = np.isfinite(source_mesh.vertices).all() and np.isfinite(source_mesh.bounds).all()
    if not finite or float(np.max(source_mesh.extents)) <= 0:
        raise ValueError(f"invalid decomposition source mesh: {source}")
    directory.mkdir(parents=True, exist_ok=True)
    # A per-key process lock prevents publishing a partial/racing cache. All
    # generated files are validated before the directory is atomically renamed.
    with (directory / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cache_hit = path.exists()
        if not cache_hit:
            import coacd

            raw_parts = coacd.run_coacd(coacd.Mesh(source_mesh.vertices, source_mesh.faces), **parameters)
            parts = [_validated_part(vertices, faces) for vertices, faces in raw_parts]
            _validate_frame(parts, source_mesh, identity["parameters"])
            with tempfile.TemporaryDirectory(prefix=f".{key}.", dir=directory) as temporary:
                pending = Path(temporary) / "parts"
                pending.mkdir()
                records = []
                persisted_parts = []
                for index, part in enumerate(parts):
                    # MuJoCo's URDF importer identifies assets by basename even
                    # with strippath=false. Separate cache directories alone do
                    # not prevent different jaws from sharing the wrong mesh.
                    filename = f"{key}_part_{index:03d}.obj"
                    part_path = pending / filename
                    part_path.write_text(trimesh.exchange.obj.export_obj(part, digits=17))
                    loaded = trimesh.load(part_path, force="mesh", process=False)
                    if not isinstance(loaded, trimesh.Trimesh):
                        raise ValueError("persisted decomposition part is not a triangle mesh")
                    persisted = _validated_part(loaded.vertices, loaded.faces)
                    persisted_parts.append(persisted)
                    records.append(
                        _part_record(persisted, filename, hashlib.sha256(part_path.read_bytes()).hexdigest())
                    )
                metadata: dict[str, Any] = {
                    "identity": identity,
                    "parts": records,
                    "frame_validation": _validate_frame(persisted_parts, source_mesh, identity["parameters"]),
                }
                (pending / "manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
                _read_cache(pending, identity, source_mesh)
                pending.rename(path)
        metadata = _read_cache(path, identity, source_mesh)
    return {
        "source_mesh": str(source),
        "source_sha256": source_sha256,
        "cache_key": key,
        "cache_hit": cache_hit,
        "cache_manifest": str(path / "manifest.json"),
        "frame_validation": metadata["frame_validation"],
        "parts": [{**record, "path": str(path / record["file"])} for record in metadata["parts"]],
    }


def apply_collision_policy(root: ET.Element, directory: Path, policy: dict) -> dict:
    """Replace selected collision meshes in-place; return JSON audit data.

    Input mesh filenames must already be absolute. Parts remain in the source
    mesh coordinate frame; each collision's origin and scale stay unchanged.
    Source assets, visuals, inertial elements and joint definitions are retained.
    A corrupt cache raises ValueError; it is never silently used or replaced by
    an undecomposed hull. The input root is only changed after all parts pass.
    """
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise ValueError("collision policy requires schema_version=1 and a name")
    if not isinstance(policy.get("name"), str) or not policy["name"].strip():
        raise ValueError("collision policy requires schema_version=1 and a name")
    if set(policy) - {"schema_version", "name", "disabled_links", "decomposed_links", "parameters"}:
        raise ValueError("unknown collision policy field")
    disabled = policy.get("disabled_links", {})
    decomposed_links = policy.get("decomposed_links", [])
    if not isinstance(disabled, dict) or any(not isinstance(v, str) or not v.strip() for v in disabled.values()):
        raise ValueError("every disabled link requires an explicit reason")
    if (
        not isinstance(decomposed_links, list)
        or any(not isinstance(v, str) for v in decomposed_links)
        or len(set(decomposed_links)) != len(decomposed_links)
    ):
        raise ValueError("decomposed_links must be a unique list")
    if set(disabled) & set(decomposed_links):
        raise ValueError("a collision cannot be disabled and decomposed")
    parameters = dict(DEFAULT_COACD_PARAMETERS)
    overrides = policy.get("parameters", {})
    if set(overrides) - parameters.keys():
        raise ValueError("unknown CoACD parameter")
    parameters.update(overrides)
    for key, value in parameters.items():
        default = DEFAULT_COACD_PARAMETERS[key]
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, int):
            valid = type(value) is int and value >= (0 if key == "seed" else 1)
        elif isinstance(default, float):
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        else:
            valid = isinstance(value, str) and bool(value)
        if not valid:
            raise ValueError(f"invalid CoACD parameter: {key}")
    directory = Path(directory).resolve() / "collision_parts"
    links = {link.get("name"): link for link in root.findall("link")}
    missing = {*disabled, *decomposed_links} - links.keys()
    if missing:
        raise ValueError(f"collision policy missing links: {sorted(missing)}")
    for name in decomposed_links:
        if not links[name].findall("collision"):
            raise ValueError(f"missing collision on {name}")
    version = importlib.metadata.version("coacd") if decomposed_links else "not_used"
    replacements = []
    decomposed = {}
    for name in decomposed_links:
        link = links[name]
        records = []
        for index, collision in enumerate(link.findall("collision")):
            mesh_element = collision.find("geometry/mesh")
            if mesh_element is None:
                raise ValueError(f"expected mesh collision on {name}")
            source = Path(mesh_element.attrib["filename"])
            if not source.is_absolute():
                raise ValueError(f"collision mesh filename must be absolute: {source}")
            source = source.resolve(strict=True)
            record = _cached_parts(source, directory, version, parameters)
            generated = []
            for part_index, part in enumerate(record["parts"]):
                replacement = copy.deepcopy(collision)
                replacement.set("name", f"{name}_collision_{index}_part_{part_index}")
                replacement_mesh = replacement.find("geometry/mesh")
                if replacement_mesh is None:
                    raise ValueError(f"missing replacement collision mesh: {name}")
                replacement_mesh.set("filename", part["path"])
                generated.append(replacement)
            replacements.append((link, collision, generated))
            origin = collision.find("origin")
            records.append(
                {
                    **record,
                    "original_collision_index": index,
                    "origin": None if origin is None else dict(origin.attrib),
                    "mesh_scale": mesh_element.get("scale", "1 1 1"),
                }
            )
        decomposed[name] = records
    removed = {name: len(links[name].findall("collision")) for name in disabled}
    for link, collision, generated in replacements:
        index = list(link).index(collision)
        link.remove(collision)
        for offset, replacement in enumerate(generated):
            link.insert(index + offset, replacement)
    for name in disabled:
        for collision in links[name].findall("collision"):
            links[name].remove(collision)
    return {
        "profile": policy["name"],
        "disabled_link_reasons": dict(disabled),
        "algorithm": "coacd",
        "algorithm_version": version,
        "validation_scope": "convex mesh integrity and coordinate bounds; not surface-error or physical acceptance",
        "parameters": dict(parameters),
        "removed_collisions": removed,
        "decomposed_links": decomposed,
    }
