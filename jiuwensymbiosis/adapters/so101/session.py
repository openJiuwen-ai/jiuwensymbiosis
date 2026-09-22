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

import os

from jiuwensymbiosis.adapters._common.builder import make_builder, make_detector_sidecar
from jiuwensymbiosis.adapters.so101.api import So101Api
from jiuwensymbiosis.adapters.so101.config import So101Config
from jiuwensymbiosis.adapters.so101.env import So101Env


def _attach_so101_cfg(session, cfg: So101Config) -> None:
    """Expose the so101 config to InProcessCodeTool-executed code."""
    session.extra_globals["so101_cfg"] = cfg


def _so101_resource_keys(cfg: So101Config) -> tuple[str, ...]:
    """Reserve the canonical serial-device path used by the SO-101 driver."""
    if not cfg.port or not str(cfg.port).strip():
        return ()
    return (f"serial:{os.path.realpath(os.path.expanduser(str(cfg.port).strip()))}",)


build_so101_session = make_builder(
    So101Config,
    So101Env,
    So101Api,
    resource_keys=_so101_resource_keys,
    api_kwargs_from_cfg=[
        "detector.url:detector_service_url",
        "z_correction_mm",
        "grasp_z_offset_mm",
        "place_z_offset_mm",
        "grasp_top_surface_enabled",
        "grasp_top_band_mm",
        "grasp_top_percentile",
        "grasp_min_points",
        "grasp_mask_erode_px",
    ],
    sidecar_builders=[make_detector_sidecar()],
    decorate=_attach_so101_cfg,
)
