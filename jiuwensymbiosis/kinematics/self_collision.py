# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""pin+coal self-collision check for any URDF body (mesh-based).

Builds the URDF collision geometry once (cached), excludes the always-touching
adjacent-link pairs (those in collision at the neutral pose), and reports whether
a full configuration self-collides. Degrades to "unavailable" (no crash, loud
warning) when pinocchio/coal or the meshes are missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover
    pin = None

# (urdf_path, package_dir) -> (model, data, geom_model, geom_data, excluded_set) or None
_GEOM: dict[tuple[str, str], tuple | None] = {}


def _derive_pkg(urdf_path: str) -> str:
    """Fallback mesh package dir for the Cruzr URDF layout (.../<pkg>/<pkg>/urdf/<robot>/x.urdf)."""
    p = Path(urdf_path)
    return str(p.parents[3]) if len(p.parents) > 3 else str(p.parent)


def _geom(urdf_path: str, package_dir: str | None):
    key = (str(urdf_path), str(package_dir or ""))
    if key in _GEOM:
        return _GEOM[key]
    result = None
    if pin is not None:
        pkg = package_dir or _derive_pkg(urdf_path)
        try:
            model = pin.buildModelFromUrdf(urdf_path)
            data = model.createData()
            gm = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION, package_dirs=[pkg])
            gm.addAllCollisionPairs()
            gd = gm.createData()
            # Exclude pairs that can never be a REAL self-collision:
            #  (a) ADJACENT links — on the same joint or directly parent/child-connected —
            #      whose meshes naturally overlap at the shared joint as it rotates (URDF/SRDF
            #      convention). Missing these flags false collisions (e.g. elbow_roll<->elbow_yaw)
            #      that wrongly blocked the home path;
            #  (b) pairs already overlapping at the neutral pose.
            adjacent = set()
            for k, cp in enumerate(gm.collisionPairs):
                j1 = gm.geometryObjects[cp.first].parentJoint
                j2 = gm.geometryObjects[cp.second].parentJoint
                if j1 == j2 or model.parents[j1] == j2 or model.parents[j2] == j1:
                    adjacent.add(k)
            pin.computeCollisions(model, data, gm, gd, pin.neutral(model), False)
            excluded = adjacent | {k for k, r in enumerate(gd.collisionResults) if r.isCollision()}
            result = (model, data, gm, gd, excluded)
            logger.info("self-collision model: %d pairs, %d excluded (%d adjacent + neutral)",
                        len(gm.collisionPairs), len(excluded), len(adjacent))
        except Exception as exc:  # noqa: BLE001  # missing meshes / build failure -> degrade, never crash
            logger.warning("self-collision geometry UNAVAILABLE (%s); collision checks DISABLED", exc)
    else:
        logger.warning("pinocchio not installed; self-collision checks DISABLED (no body protection)")
    _GEOM[key] = result
    return result


def available(urdf_path: str, package_dir: str | None = None) -> bool:
    """True when the collision model built (pinocchio + coal + meshes present)."""
    return _geom(urdf_path, package_dir) is not None


def full_q(urdf_path: str, package_dir: str | None, joint_values: dict[str, float]):
    """Build a pinocchio config vector from a name->angle dict (missing joints stay neutral); None if unavailable."""
    g = _geom(urdf_path, package_dir)
    if g is None:
        return None
    model = g[0]
    q = pin.neutral(model)
    for name, val in joint_values.items():
        if model.existJointName(name):
            q[model.joints[model.getJointId(name)].idx_q] = float(val)
    return q


def in_self_collision(urdf_path: str, package_dir: str | None, q_full) -> bool:
    """True if the config self-collides (excluding always-touching adjacent pairs). False if unavailable."""
    g = _geom(urdf_path, package_dir)
    if g is None:
        return False
    model, data, gm, gd, excluded = g
    pin.computeCollisions(model, data, gm, gd, np.asarray(q_full, dtype=float), False)
    hits = {k for k, r in enumerate(gd.collisionResults) if r.isCollision()}
    return bool(hits - excluded)
