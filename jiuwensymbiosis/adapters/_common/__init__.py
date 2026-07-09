# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Cross-vendor adapter building blocks — used *by* adapters, nothing else.

What's left here is only consumed inside ``adapters/``: the generic session
builder (``builder``) and cartesian workspace bounds (``safety``).

Things consumed outside ``adapters/`` moved to where their consumer lives:
* the ``RobotDriver`` Protocol → ``jiuwensymbiosis.env.protocol`` (Env delegates
  to the driver, so the contract lives with the env layer — no more TYPE_CHECKING
  dance in ``env/base.py``);
* sensing (camera, detector client/sidecar, vision, calibration) →
  ``jiuwensymbiosis.perception``;
* SE(3)/pinhole math → ``jiuwensymbiosis.utils.geometry``.

Per-vendor adapters under ``adapters/<vendor>/`` import from here.
"""
