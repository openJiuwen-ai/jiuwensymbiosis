# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Self-describing waypoint / station archives for calibration replay.

Two NPZ archive families:

* **waypoint** (for ``--auto``): the operator-confirmed sparse trajectory as
  joint values or cartesian flange transforms, plus joint metadata and a
  sanitised config digest. ``kind``/``schema_version`` guards refuse the wrong
  archive family at load time.
* **station** (for ``--replay``): per-station flange + board-cam SE(3) (mm),
  intrinsics, optional distortion, capture metadata and joint metadata.

Both use atomic writes (write-to-temp + ``os.replace``) and
``allow_pickle=False`` loads. The archive never stores raw RGB frames — those
live alongside as separate files when trace diagnostics are enabled.

Adapter identity is an opaque integration-owned string; this module stores it
without interpreting a framework-specific package layout. Secrets are stripped
by :func:`_sanitize_config_digest` before the digest is archived.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np

from jiuwensymbiosis.calibration._mount import MOUNT_VALUES, Mount, canonical_mount
from jiuwensymbiosis.calibration.domain.models import Station, ViewDetection, require_se3_mm
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger("jiuwensymbiosis.calibration.archives")

# Archive kinds — loaders reject the wrong kind for a mode.
KIND_WAYPOINTS = "eye_to_hand_waypoints"
KIND_STATIONS = "eye_to_hand_stations"
ARCHIVE_SCHEMA_VERSION = "1.0"


def _validate_mount(mount: Any, *, source: str) -> Mount:
    """Validate and canonicalise a mount value at the archive boundary."""
    try:
        return canonical_mount(mount)
    except ValueError as exc:
        raise ValueError(
            f"{source} mount={mount!r} is not a valid camera mount "
            f"(expected one of {MOUNT_VALUES}); refusing to write/load the archive."
        ) from exc


def _adapter_identity(value: Any) -> str:
    """Return a non-empty opaque identity supplied by the integration layer."""
    identity = str(value) if not hasattr(value, "__module__") else str(getattr(value, "__module__", "") or "")
    if not identity:
        raise ValueError("adapter identity must be a non-empty string")
    return identity


def _require_finite_array(value: Any, *, field: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Return a finite float64 array and optionally enforce its exact shape."""
    arr = np.asarray(value, dtype=np.float64)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{field}: expected shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field}: non-finite values")
    return arr


def _require_se3_stack(value: Any, *, field: str) -> np.ndarray:
    """Validate an ``(N,4,4)`` transform stack through the canonical SE(3) gate."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"{field}: expected (N,4,4), got {arr.shape}")
    if len(arr) == 0:
        raise ValueError(f"{field}: at least one transform is required")
    return np.stack([require_se3_mm(tf, field=f"{field}[{i}]") for i, tf in enumerate(arr)])


def _require_count(value: Any, *, field: str) -> int:
    """Require an actual non-negative integer, rejecting bools/floats/arrays."""
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"{field} must be a scalar integer")
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    count = int(value)
    if count < 0:
        raise ValueError(f"{field} must be non-negative, got {count}")
    return count


# =============================================================================
# Sanitised config digest (strip secrets before archiving)
# =============================================================================
_SECRET_KEYS = frozenset(
    {
        "port",
        "can_port",
        "camera_serial",
        "api_key",
        "token",
        "password",
        "serial",
        "host",
        "url",
    }
)


def _sanitize_config_digest(cfg: Any) -> dict[str, Any]:
    """Return a calibration-relevant, secret-stripped digest of an adapter config.

    Only calibration-related scalar fields are kept; anything that smells like a
    credential or device address (serial/port/host/url/api_key/...) is dropped.
    """
    try:
        import dataclasses as _dc

        if _dc.is_dataclass(cfg):
            raw = {f.name: getattr(cfg, f.name) for f in _dc.fields(cfg)}
        elif isinstance(cfg, dict):
            raw = dict(cfg)
        else:
            return {"_type": type(cfg).__name__}
    except Exception:
        return {"_type": type(cfg).__name__}

    clean: dict[str, Any] = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if k.lower() in _SECRET_KEYS or any(s in k.lower() for s in ("key", "token", "secret", "password")):
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, (list, tuple)):
            clean[k] = [float(x) if isinstance(x, (int, float)) else str(x) for x in v][:32]
        elif isinstance(v, dict):
            # keep nested mount / frame / trajectory fields, strip nested secrets
            nested = {
                kk: vv
                for kk, vv in v.items()
                if kk.lower() not in _SECRET_KEYS and not isinstance(vv, (dict, list, tuple))
            }
            if nested:
                clean[k] = nested
    clean["_type"] = type(cfg).__name__
    return clean


# =============================================================================
# Atomic NPZ / JSON write
# =============================================================================
def _atomic_savez(path: str | Path, **arrays: Any) -> None:
    """Write an NPZ atomically: write to a temp file in the same dir then ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        tmp_name = tmp.name
        np.savez(tmp_name, **arrays)
    os.replace(tmp_name, str(path))


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically (UTF-8, sorted keys)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".json", encoding="utf-8", delete=False) as tmp:
        tmp.write(text)
        tmp_name = tmp.name
    os.replace(tmp_name, str(path))


def _load_archive(path: str | Path, *, expected_kind: str) -> tuple[Any, dict[str, Any]]:
    """Load an NPZ archive, validating kind + major version + metadata shape.

    Returns ``(data, metadata)``. Refuses unknown major schema versions and the
    wrong ``kind`` for the caller's mode (``--auto`` rejects stations, vice versa).
    """
    data = np.load(str(path), allow_pickle=False)
    meta_raw = str(data["metadata"]) if "metadata" in data.files else "{}"
    meta = json.loads(meta_raw)
    kind = meta.get("kind")
    if kind != expected_kind:
        raise ValueError(f"{path}: archive kind={kind!r} but this mode expects {expected_kind!r}.")
    version = str(meta.get("schema_version", ""))
    major = version.split(".", 1)[0]
    if major != ARCHIVE_SCHEMA_VERSION.split(".", 1)[0]:
        raise ValueError(
            f"{path}: unknown archive schema_version {version!r}; expected major "
            f"{ARCHIVE_SCHEMA_VERSION.split('.', 1)[0]}."
        )
    # Validate the mount field on read too — a hand-edited, incomplete or
    # legacy archive without an explicit frame is rejected rather than being
    # silently interpreted as eye-to-hand by a replay caller.
    if "mount" not in meta or meta["mount"] is None:
        raise ValueError(f"{path}:metadata.mount is required; archive camera frame is ambiguous.")
    meta["mount"] = _validate_mount(meta["mount"], source=f"{path}:metadata.mount")
    return data, meta


# =============================================================================
# Waypoint archive (for --auto): trajectory space + joint/cartesian values
# =============================================================================
def dump_waypoint_archive(
    path: str | Path,
    *,
    space: str,  # "joint" or "cartesian"
    joint_values: np.ndarray | None = None,
    joint_unit: str | None = None,
    joint_order: tuple[str, ...] | None = None,
    joint_periodic: tuple[bool, ...] | None = None,
    cartesian_tfs: np.ndarray | None = None,
    adapter_module: str,
    mount: str,
    config: Any,
) -> Path:
    """Write a self-describing waypoint archive for ``--auto`` consumption.

    Either joint values (``space='joint'``) or cartesian flange transforms
    (``space='cartesian'``, mm) must be supplied; the other must be None.
    """
    path = Path(path)
    if space not in ("joint", "cartesian"):
        raise ValueError(f"space must be 'joint' or 'cartesian', got {space!r}")
    if space == "joint":
        if joint_values is None:
            raise ValueError("space='joint' requires joint_values")
        if joint_order is None or joint_unit is None:
            raise ValueError("space='joint' requires joint_order and joint_unit")
        if joint_periodic is None:
            joint_periodic = tuple(False for _ in joint_order)
        if cartesian_tfs is not None:
            raise ValueError("space='joint' must not also set cartesian_tfs")
    else:
        if cartesian_tfs is None:
            raise ValueError("space='cartesian' requires cartesian_tfs")
        if joint_values is not None:
            raise ValueError("space='cartesian' must not also set joint_values")

    arrays: dict[str, Any] = {"space": np.array(space)}
    if space == "joint":
        # The validation branch above already rejected every None; re-narrow for the
        # checker without an assert, which -O would strip out of a production run.
        joint_order = cast(tuple[str, ...], joint_order)
        joint_unit = cast(str, joint_unit)
        joint_periodic = cast(tuple[bool, ...], joint_periodic)
        arrays["joint_values"] = _require_finite_array(joint_values, field="joint_values")
        if arrays["joint_values"].ndim != 2:
            raise ValueError(f"joint_values: expected 2-D, got {arrays['joint_values'].shape}")
        if arrays["joint_values"].shape[1] != len(joint_order):
            raise ValueError(
                f"joint_values width {arrays['joint_values'].shape[1]} != joint_order length {len(joint_order)}"
            )
        if len(joint_periodic) != len(joint_order):
            raise ValueError(f"joint_periodic length {len(joint_periodic)} != joint_order length {len(joint_order)}")
        arrays["joint_unit"] = np.array(joint_unit)
        arrays["joint_order"] = np.array(list(joint_order))
        arrays["joint_periodic"] = np.array(list(joint_periodic))
    else:
        arrays["cartesian_tfs"] = _require_se3_stack(cartesian_tfs, field="cartesian_tfs")

    metadata = {
        "kind": KIND_WAYPOINTS,
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "space": space,
        "adapter_module": _adapter_identity(adapter_module),
        "mount": _validate_mount(mount, source="dump_waypoint_archive"),
        "config_digest": _sanitize_config_digest(config),
        "n": int(len(arrays["joint_values"] if space == "joint" else arrays["cartesian_tfs"])),
    }
    arrays["metadata"] = np.array(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    _atomic_savez(path, **arrays)
    logger.info("dumped %d waypoints (space=%s) -> %s", metadata["n"], space, path)
    return path


def load_waypoint_archive(path: str | Path) -> dict[str, Any]:
    """Load a waypoint archive, validating kind + shapes; raise on station archive."""
    path = Path(path)
    data, meta = _load_archive(path, expected_kind=KIND_WAYPOINTS)
    space = str(data["space"])
    if space not in ("joint", "cartesian"):
        raise ValueError(f"{path}: unknown waypoint space {space!r}")
    if meta.get("space") != space:
        raise ValueError(f"{path}: metadata.space={meta.get('space')!r} != archive space={space!r}")
    if "n" not in meta:
        raise ValueError(f"{path}: metadata.n is required")
    metadata_n = _require_count(meta["n"], field=f"{path}:metadata.n")
    out: dict[str, Any] = {"space": space, "metadata": meta, "n": metadata_n}
    if space == "joint":
        jv = _require_finite_array(data["joint_values"], field=f"{path}:joint_values")
        if jv.ndim != 2:
            raise ValueError(f"waypoint joint_values must be 2-D, got {jv.shape}")
        order = tuple(str(x) for x in data["joint_order"])
        periodic = tuple(bool(x) for x in data["joint_periodic"])
        if jv.shape[1] != len(order):
            raise ValueError(f"waypoint joint_values width {jv.shape[1]} != order length {len(order)}")
        if len(periodic) != len(order):
            raise ValueError(f"waypoint joint_periodic length {len(periodic)} != order length {len(order)}")
        if len(jv) != metadata_n:
            raise ValueError(f"{path}: metadata.n={metadata_n} != joint_values length {len(jv)}")
        out.update(
            {
                "joint_values": jv,
                "joint_unit": str(data["joint_unit"]),
                "joint_order": order,
                "joint_periodic": periodic,
            }
        )
    else:
        tfs = _require_se3_stack(data["cartesian_tfs"], field=f"{path}:cartesian_tfs")
        if len(tfs) != metadata_n:
            raise ValueError(f"{path}: metadata.n={metadata_n} != cartesian_tfs length {len(tfs)}")
        out["cartesian_tfs"] = tfs
    return out


# =============================================================================
# Station archive (for --replay): flange/board_cam/intrinsics + capture meta
# =============================================================================
def dump_station_archive(
    path: str | Path,
    stations: list[Station],
    intrinsics: np.ndarray,
    *,
    adapter_module: str,
    mount: str,
    capture_meta: list[dict[str, Any]] | None = None,
    joint_meta: dict[str, Any] | None = None,
    distortion: np.ndarray | None = None,
    board_spec: dict[str, Any] | None = None,
    trajectory_space: str | None = None,
    config: Any = None,
) -> Path:
    """Write a self-describing station archive for ``--replay``."""
    path = Path(path)
    n = len(stations)
    if n < 1:
        raise ValueError("dump_station_archive requires at least one station")
    flange = np.stack(
        [require_se3_mm(s.tf_base_flange, field=f"stations[{i}].tf_base_flange") for i, s in enumerate(stations)]
    )
    board_cam_values: list[np.ndarray] = []
    for i, station in enumerate(stations):
        tf_cam_target = station.detection.tf_cam_target
        if tf_cam_target is None:
            raise ValueError(f"stations[{i}].tf_cam_target is missing")
        board_cam_values.append(require_se3_mm(tf_cam_target, field=f"stations[{i}].tf_cam_target"))
    board_cam = np.stack(board_cam_values)
    reproj = np.array([float(s.detection.reproj_rms_px or 0.0) for s in stations], dtype=np.float64)
    if not np.all(np.isfinite(reproj)) or np.any(reproj < 0):
        raise ValueError("station reproj values must be finite and non-negative")
    intrinsics_array = _require_finite_array(intrinsics, field="intrinsics", shape=(3, 3))
    arrays: dict[str, Any] = {
        "n": np.array(n),
        "intrinsics": intrinsics_array,
        "flange": flange,
        "board_cam": board_cam,
        "reproj": reproj,
    }
    if distortion is not None:
        arrays["distortion"] = _require_finite_array(distortion, field="distortion")
    if capture_meta:
        # Capture metadata records every attempted target (including rejected
        # attempts), so it may legitimately be longer than the accepted
        # station arrays.
        arrays["capture_meta_json"] = np.array(json.dumps(capture_meta, ensure_ascii=False, sort_keys=True))
    metadata = {
        "kind": KIND_STATIONS,
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "adapter_module": _adapter_identity(adapter_module),
        "mount": _validate_mount(mount, source="dump_station_archive"),
        "n": n,
        "joint_meta": joint_meta,
        "board_spec": board_spec,
        "trajectory_space": trajectory_space,
        "config_digest": _sanitize_config_digest(config) if config is not None else None,
    }
    arrays["metadata"] = np.array(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    _atomic_savez(path, **arrays)
    logger.info("dumped %d stations -> %s", n, path)
    return path


def load_station_archive(path: str | Path) -> tuple[list[Station], np.ndarray, dict[str, Any]]:
    """Load a station archive; raise on a waypoint archive or unknown version."""
    path = Path(path)
    data, meta = _load_archive(path, expected_kind=KIND_STATIONS)
    if "n" not in data.files or "n" not in meta:
        raise ValueError(f"{path}: archive n and metadata.n are required")
    n = _require_count(data["n"], field=f"{path}:n")
    metadata_n = _require_count(meta["n"], field=f"{path}:metadata.n")
    if n < 1:
        raise ValueError(f"{path}: station archive has n={n}")
    if metadata_n != n:
        raise ValueError(f"{path}: metadata.n={metadata_n} != archive n={n}")
    flange = _require_se3_stack(data["flange"], field=f"{path}:flange")
    board_cam = _require_se3_stack(data["board_cam"], field=f"{path}:board_cam")
    if len(flange) != n or len(board_cam) != n:
        raise ValueError(f"{path}: n={n}, flange length={len(flange)}, board_cam length={len(board_cam)}")
    intrinsics = _require_finite_array(data["intrinsics"], field=f"{path}:intrinsics", shape=(3, 3))
    reproj = _require_finite_array(data["reproj"], field=f"{path}:reproj") if "reproj" in data else np.zeros(n)
    if reproj.shape != (n,):
        raise ValueError(f"{path}: reproj shape {reproj.shape} != ({n},)")
    if np.any(reproj < 0):
        raise ValueError(f"{path}: reproj values must be non-negative")
    stations = [
        Station(
            tf_base_flange=flange[i],
            detection=ViewDetection(
                ok=True,
                tf_cam_target=board_cam[i],
                reproj_rms_px=float(reproj[i]),
            ),
        )
        for i in range(n)
    ]
    return stations, intrinsics, meta


__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "KIND_STATIONS",
    "KIND_WAYPOINTS",
    "_adapter_identity",
    "_atomic_savez",
    "_atomic_write_json",
    "_load_archive",
    "_sanitize_config_digest",
    "dump_station_archive",
    "dump_waypoint_archive",
    "load_station_archive",
    "load_waypoint_archive",
]
