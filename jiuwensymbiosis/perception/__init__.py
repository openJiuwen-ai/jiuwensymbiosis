# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Perception — robot-agnostic sensing (sibling of ``rails`` / ``tools`` / …).

In-process sensing and the vision→base-frame pipeline any robot can reuse:

* ``camera`` — Intel RealSense wrapper (color+depth, aligned).
* ``detector_client`` / ``detector_sidecar`` — open-vocabulary detection: the
  HTTP client + the subprocess that runs ``serving.grounding_dino_sam2_server``.
* ``vision`` — cross-vendor projection helpers (detect → centroid → base XYZ,
  grasp/place-z), consumed by an adapter's ``get_grasp_info_simple``.
* ``calibration`` — hand-eye calibration JSON loading.

Heavy/optional deps (pyrealsense2, cv2, requests) are imported lazily by the
submodules; importing this package pulls in none of them. Consumers import the
submodule they need directly (this ``__init__`` deliberately re-exports nothing,
to keep imports light). Shared SE(3)/pinhole math lives in
``jiuwensymbiosis.utils.geometry``; the detection model server lives in
``jiuwensymbiosis.serving``.
"""
