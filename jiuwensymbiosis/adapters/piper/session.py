# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``build_piper_session`` — one call from YAML to a ready-to-connect session.

handles the detection-server (GroundingDINO + SAM2) sidecar lifecycle and passes
``detector_service_url`` into the Api so the user just does::

    session = build_piper_session.from_yaml("configs/piper/<task>.yaml")
    with session:
        agent = build_robot_agent(session)
        ...
"""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters._common.detector_sidecar import detector_subprocess
from jiuwensymbiosis.adapters.piper.api import PiperApi
from jiuwensymbiosis.adapters.piper.config import PiperConfig
from jiuwensymbiosis.adapters.piper.env import PiperEnv


def _detector_sidecar_from_cfg(cfg: PiperConfig):
    """Return a zero-arg context-manager factory if the detector should be spawned;
    None means "no sidecar to start"."""
    if not cfg.detector.spawn:
        return None
    detector_kwargs: dict[str, Any] = dict(
        host=cfg.detector.host,
        port=cfg.detector.port,
        device=cfg.detector.device,
        startup_timeout_s=cfg.detector.startup_timeout_s,
        gdino_model_id=cfg.detector.gdino_model_id,
        sam2_model_id=cfg.detector.sam2_model_id,
        box_threshold=cfg.detector.box_threshold,
        text_threshold=cfg.detector.text_threshold,
        use_sam2=cfg.detector.use_sam2,
    )
    return lambda: detector_subprocess(**detector_kwargs)


def _api_kwargs_from_cfg(cfg: PiperConfig) -> dict:
    """Extract PiperApi constructor kwargs from a PiperConfig."""
    return {
        "detector_service_url": cfg.detector.url,
        "z_correction_mm": cfg.z_correction_mm,
        "grasp_z_offset_mm": cfg.grasp_z_offset_mm,
        "chip_thickness_mm": cfg.chip_thickness_mm,
    }


def _decorate(session, cfg: PiperConfig) -> None:
    """Attach piper config to session extra globals."""
    session.extra_globals["piper_cfg"] = cfg


build_piper_session = make_builder(
    PiperConfig,
    PiperEnv,
    PiperApi,
    api_kwargs_from_cfg=_api_kwargs_from_cfg,
    sidecar_builders=[_detector_sidecar_from_cfg],
    decorate=_decorate,
)
