# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Cross-vendor adapter building blocks — used *by* adapters, nothing else.

What's left here is only consumed inside ``adapters/``: the generic session
builder (``builder``), cartesian workspace bounds (``safety``), and the reusable
motion core for joint-level arms — ``geometry`` (pose/unit/SE(3) conversions),
``joint_transport`` (the vendor SDK seam), ``kinematics`` (FK/IK backend seam +
waypoint planning/rejection), and ``kinematic_driver`` (``KinematicArmDriver``,
a ``CartesianDriver`` built from a transport + an FK/IK backend). Adapters import
these submodules directly, so importing this package stays side-effect free.

Things consumed outside ``adapters/`` moved to where their consumer lives:
* the driver Protocols → ``jiuwensymbiosis.env.protocol`` (Env delegates
  to the driver, so the contract lives with the env layer — no more TYPE_CHECKING
  dance in ``env/base.py``);
* sensing (camera, detector client/sidecar, vision, calibration) →
  ``jiuwensymbiosis.perception``;
* SE(3)/pinhole math → ``jiuwensymbiosis.utils.geometry``.

Per-vendor adapters under ``adapters/<vendor>/`` import from here.
"""
