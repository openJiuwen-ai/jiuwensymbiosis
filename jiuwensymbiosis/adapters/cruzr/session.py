# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``build_cruzr_session`` — one call from YAML to a ready Cruzr session.

Wires the detection-server (GroundingDINO+SAM2) sidecar and passes
``detector_service_url`` / ``camera_calib_path`` into the Api.
"""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.perception.detector_sidecar import detector_subprocess
from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv


def _detector_sidecar_from_cfg(cfg: CruzrConfig):
    """按 cfg.detector.spawn 返回检测服务 sidecar 的零参工厂；None 表示不拉起。"""
    if not cfg.detector.spawn:
        return None
    kwargs: dict[str, Any] = dict(
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
    return lambda: detector_subprocess(**kwargs)


def _api_kwargs_from_cfg(cfg: CruzrConfig) -> dict:
    """CruzrApi 构造 kwargs。"""
    return {
        "detector_service_url": cfg.detector.url,
        "camera_calib_path": cfg.camera_calib_path,
    }


def _decorate(session, cfg: CruzrConfig) -> None:
    """Attach Cruzr config to session globals."""
    session.extra_globals["cruzr_cfg"] = cfg


build_cruzr_session = make_builder(
    CruzrConfig,
    CruzrEnv,
    CruzrApi,
    api_kwargs_from_cfg=_api_kwargs_from_cfg,
    sidecar_builders=[_detector_sidecar_from_cfg],
    decorate=_decorate,
)
