# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""build_xxx_session — one call from YAML to a ready-to-connect session.

Usage::

    session = build_xxx_session.from_yaml("configs/xxx/default.yaml")
    with session:
        agent = build_robot_agent(session)
        ...

See docs/hardware-porting-guide.md Step 6 for wiring details.
"""

from __future__ import annotations

from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters.xxx.config import XxxConfig
from jiuwensymbiosis.adapters.xxx.env import XxxEnv
from jiuwensymbiosis.adapters.xxx.api import XxxApi


# ============================================================================
# Basic wiring — use this if your Api and Env don't need extra setup.
# Uncomment the advanced version below if you need sidecars or api kwargs.
# ============================================================================

build_xxx_session = make_builder(XxxConfig, XxxEnv, XxxApi)


# ============================================================================
# Advanced wiring — uncomment and customize if you need:
#   1. api_kwargs_from_cfg — pass extra __init__ params to Api
#   2. sidecar_builders     — start/stop subprocesses (e.g. detector)
#   3. decorate             — inject objects into session.extra_globals
# ============================================================================

# def _api_kwargs_from_cfg(cfg: XxxConfig) -> dict:
#     """Extract Api.__init__ kwargs from config."""
#     return {
#         # "detector_service_url": cfg.detector_url,
#         # "z_correction_mm": cfg.z_correction_mm,
#     }
#
#
# def _decorate(session, cfg: XxxConfig) -> None:
#     """Inject config into session for InProcessCodeTool access."""
#     session.extra_globals["xxx_cfg"] = cfg
#
#
# build_xxx_session = make_builder(
#     XxxConfig,
#     XxxEnv,
#     XxxApi,
#     api_kwargs_from_cfg=_api_kwargs_from_cfg,
#     # sidecar_builders=[_detector_sidecar],
#     decorate=_decorate,
# )
