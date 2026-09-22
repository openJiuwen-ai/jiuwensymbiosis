# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``build_cruzr_session`` — one call from YAML to a ready Cruzr session.

Wires the detection-server (GroundingDINO+SAM2) sidecar and passes
``detector_service_url`` / ``camera_calib_path`` into the Api.
"""

from __future__ import annotations

from jiuwensymbiosis.adapters._common.builder import make_builder, make_detector_sidecar
from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv


def _api_kwargs_from_cfg(cfg: CruzrConfig) -> dict:
    """CruzrApi 构造 kwargs。"""
    return {
        "detector_service_url": cfg.detector.url,
        "camera_calib_path": cfg.camera_calib_path,
    }


def _cruzr_resource_keys(cfg: CruzrConfig) -> tuple[str, ...]:
    """Reserve Cruzr's command endpoint in the domain captured by its config."""
    domain_id = cfg.ros_domain_id
    if cfg.command_topic is None:
        return ()
    command_topic = str(cfg.command_topic).strip()
    return (f"ros:{domain_id}:{command_topic}",) if command_topic else ()


def _decorate(session, cfg: CruzrConfig) -> None:
    """Attach Cruzr config to session globals."""
    session.extra_globals["cruzr_cfg"] = cfg


build_cruzr_session = make_builder(
    CruzrConfig,
    CruzrEnv,
    CruzrApi,
    api_kwargs_from_cfg=_api_kwargs_from_cfg,
    resource_keys=_cruzr_resource_keys,
    sidecar_builders=[make_detector_sidecar()],
    decorate=_decorate,
)
