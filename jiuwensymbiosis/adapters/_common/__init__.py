# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Cross-vendor adapter building blocks.

Things that are NOT specific to any one robot family — RealSense pipeline,
hand-eye calibration loading, SE(3)/pinhole math, workspace bounds,
detector client + sidecar, vision projection helpers, pick/place skill
choreography, the generic session builder, and the ``RobotDriver``
Protocol that new vendors implement.

Per-vendor adapters under ``adapters/<vendor>/`` import from here.
"""
