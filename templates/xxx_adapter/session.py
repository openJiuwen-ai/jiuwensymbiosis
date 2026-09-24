# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""build_xxx_session — one call from YAML to a ready-to-connect session.

Usage::

    session = build_xxx_session.from_yaml("configs/xxx/default.yaml")
    with session:
        agent = build_robot_agent(session)
        ...

See docs/zh/how-to/port-hardware-adapter.md for wiring details.
"""

from __future__ import annotations

from jiuwensymbiosis.adapters.xxx.api import XxxApi
from jiuwensymbiosis.adapters.xxx.config import XxxConfig
from jiuwensymbiosis.adapters.xxx.env import XxxEnv

from jiuwensymbiosis.adapters._common.builder import make_builder

# ============================================================================
# Basic wiring — use this if your Api and Env don't need extra setup.
# Uncomment the advanced version below if you need sidecars or api kwargs.
# ============================================================================


def _resource_keys(cfg: XxxConfig) -> tuple[str, ...]:
    """Identify the actual command endpoint without connecting hardware.

    Replace this CAN example when changing the template's connection type:
    serial devices use Path(cfg.port).expanduser().resolve(); ROS devices use
    the captured domain and command topic. Configuration aliases for the same
    device must yield identical keys. Camera/detector keys are added centrally.
    """
    return (f"can:{cfg.can_port}",)


build_xxx_session = make_builder(XxxConfig, XxxEnv, XxxApi, resource_keys=_resource_keys)


# ============================================================================
# Advanced wiring — uncomment and customize if you need:
#   1. api_kwargs_from_cfg — pass extra __init__ params to Api
#   2. sidecar_builders     — start/stop subprocesses (e.g. detector)
#   3. decorate             — inject objects into session.extra_globals
# ============================================================================

# def _api_kwargs_from_cfg(cfg: XxxConfig) -> dict:
#     """Extract Api.__init__ kwargs from config."""
#     return {
#         # "detector_service_url": cfg.detector.url,
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
#     # managed_detector=True,  # API __init__ 接收 detector_client；由 session 管理关闭
#     decorate=_decorate,
#     resource_keys=_resource_keys,  # required command-endpoint admission wiring
# )
