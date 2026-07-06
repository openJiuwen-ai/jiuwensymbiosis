# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``build_ubetech_cruzr_s2_session`` — one call from YAML to a ready-to-connect session.

Wires the detector-server (GroundingDINO + SAM2) sidecar lifecycle and passes
the detector URL + geometry constants into the Api, so the user just does::

    session = build_ubetech_cruzr_s2_session.from_yaml("configs/ubetech_cruzr_s2/default.yaml")
    with session:
        agent = build_robot_agent(session)
        ...
"""

from __future__ import annotations

from jiuwensymbiosis.adapters._common.builder import make_builder, make_detector_sidecar
from jiuwensymbiosis.adapters.ubetech_cruzr_s2.api import UbetechCruzrS2Api
from jiuwensymbiosis.adapters.ubetech_cruzr_s2.config import UbetechCruzrS2Config
from jiuwensymbiosis.adapters.ubetech_cruzr_s2.env import UbetechCruzrS2Env


def _attach_ubetech_cruzr_s2_cfg(session, cfg: UbetechCruzrS2Config) -> None:
    """Expose the config to InProcessCodeTool-executed code."""
    session.extra_globals["ubetech_cruzr_s2_cfg"] = cfg


build_ubetech_cruzr_s2_session = make_builder(
    UbetechCruzrS2Config,
    UbetechCruzrS2Env,
    UbetechCruzrS2Api,
    # Declarative field mapping: cfg attr → UbetechCruzrS2Api __init__ kwarg.
    # ``detector_url:detector_service_url`` renames on the way to the Api;
    # the rest pass through unchanged.
    api_kwargs_from_cfg=[
        "detector_url:detector_service_url",
        "z_correction_mm",
        "grasp_z_offset_mm",
        "chip_thickness_mm",
    ],
    sidecar_builders=[make_detector_sidecar()],
    decorate=_attach_ubetech_cruzr_s2_cfg,
)
