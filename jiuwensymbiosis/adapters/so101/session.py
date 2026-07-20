# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``build_so101_session`` — one call from YAML to a ready-to-connect session.

Milestone B: desktop eye-to-hand vision (RealSense D405 + open-vocab detection).
``So101Api`` now takes vision kwargs (detector_service_url + grasp geometry);
``make_detector_sidecar()`` spawns the GroundingDINO+SAM2 server when
``cfg.detector.spawn`` is set. Connection, kinematics, gripper and camera config
are all consumed by ``So101Env``/``So101Driver`` from the cfg.
"""

from __future__ import annotations

from jiuwensymbiosis.adapters._common.builder import make_builder, make_detector_sidecar
from jiuwensymbiosis.adapters.so101.api import So101Api
from jiuwensymbiosis.adapters.so101.config import So101Config
from jiuwensymbiosis.adapters.so101.env import So101Env


def _attach_so101_cfg(session, cfg: So101Config) -> None:
    """Expose the so101 config to InProcessCodeTool-executed code."""
    session.extra_globals["so101_cfg"] = cfg


build_so101_session = make_builder(
    So101Config,
    So101Env,
    So101Api,
    api_kwargs_from_cfg=[
        "detector.url:detector_service_url",
        "z_correction_mm",
        "grasp_z_offset_mm",
        "chip_thickness_mm",
    ],
    sidecar_builders=[make_detector_sidecar()],
    decorate=_attach_so101_cfg,
)
