# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""build_ubetech_cruzr_s2_session — one call from YAML to a ready-to-connect session.

Usage::

    session = build_ubetech_cruzr_s2_session.from_yaml("configs/ubetech_cruzr_s2/default.yaml")
    with session:
        agent = build_robot_agent(session)
        ...
"""

from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters.ubetech_cruzr_s2.api import UbetechCruzrS2Api
from jiuwensymbiosis.adapters.ubetech_cruzr_s2.config import UbetechCruzrS2Config
from jiuwensymbiosis.adapters.ubetech_cruzr_s2.env import UbetechCruzrS2Env

build_ubetech_cruzr_s2_session = make_builder(UbetechCruzrS2Config, UbetechCruzrS2Env, UbetechCruzrS2Api)
