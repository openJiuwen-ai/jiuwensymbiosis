"""Independent XL330 mouth servo on Pollen's otherwise rigid training head."""

from __future__ import annotations

import math

import numpy as np

# Pollen duck-control/src/model.rs, mouth_target (alpha). The ONNX wire omits this servo.
CLOSED = math.radians(-5)
OPEN = math.radians(30)


def add_mouth(spec):
    """Articulate the two existing jaw meshes at the CAD bearing, preserving neutral head inertia."""
    import mujoco as mj

    head = spec.body("jaw_soft")
    for name in ("jaw", "jaw_soft"):
        # The solid CAD meshes are watertight. Convex/legacy inertia would fill
        # the air between the mandible's sides and overestimate the moving mass.
        spec.mesh(name).inertia = mj.mjtMeshInertia.mjMESH_INERTIA_EXACT
    bearing = next(g for g in head.geoms if g.meshname == "seeed_bearing__configuration_default")
    jaw = head.add_body(name="lower_beak")
    # CAD local +X is up and -Z is forward. Opening rotates the lower beak down about +Y.
    jaw.add_joint(
        name="mouth", type=mj.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], pos=bearing.pos, ref=CLOSED, range=[CLOSED, OPEN]
    )
    for geom in list(head.geoms):
        if geom.meshname in {"jaw", "jaw_soft"}:
            jaw.add_geom(
                geom.classname,
                type=geom.type,
                meshname=geom.meshname,
                material=geom.material,
                pos=geom.pos,
                quat=geom.quat,
                density=0 if geom.contype else geom.density,
            )
            spec.delete(geom)

    # The official MJCF aggregates the entire head into a single CAD inertia.
    # Use its mesh density to estimate the moving share, then subtract that exact
    # contribution from the parent so the assembled neutral model is conserved.
    model = spec.compile()
    total = model.body(head.name)
    moving = model.body(jaw.name)
    mass = total.mass[0] - moving.mass[0]
    center = (total.mass[0] * total.ipos - moving.mass[0] * moving.ipos) / mass

    def inertia_at(body, origin):
        rotation = np.empty(9)
        mj.mju_quat2Mat(rotation, body.iquat)
        axes = rotation.reshape(3, 3)
        offset = body.ipos - origin
        return axes @ np.diag(body.inertia) @ axes.T + body.mass[0] * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )

    inertia = inertia_at(total, center) - inertia_at(moving, center)
    head.mass = mass
    head.ipos = center
    head.fullinertia = inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    spec.add_actuator(name="mouth", trntype=mj.mjtTrn.mjTRN_JOINT, target="mouth")
