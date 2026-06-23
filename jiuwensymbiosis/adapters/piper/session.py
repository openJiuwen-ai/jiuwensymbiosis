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

from jiuwensymbiosis.adapters._common.builder import make_builder, make_detector_sidecar
from jiuwensymbiosis.adapters.piper.api import PiperApi
from jiuwensymbiosis.adapters.piper.config import PiperConfig
from jiuwensymbiosis.adapters.piper.env import PiperEnv


def _attach_piper_cfg(session, cfg: PiperConfig) -> None:
    """Expose the piper config to InProcessCodeTool-executed code."""
    session.extra_globals["piper_cfg"] = cfg


build_piper_session = make_builder(
    PiperConfig,
    PiperEnv,
    PiperApi,
    # Declarative field mapping: cfg attr → PiperApi __init__ kwarg.
    # "detector.url:detector_service_url" reaches into the nested detector
    # sub-config and renames it for the Api; the rest pass through unchanged.
    api_kwargs_from_cfg=[
        "detector.url:detector_service_url",
        "z_correction_mm",
        "grasp_z_offset_mm",
        "chip_thickness_mm",
    ],
    sidecar_builders=[make_detector_sidecar()],
    decorate=_attach_piper_cfg,
)
