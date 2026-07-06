# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""build_unitree_go2_session — one call from YAML to a ready-to-connect session.

Usage::

    session = build_unitree_go2_session.from_yaml("configs/unitree_go2/default.yaml")
    with session:
        agent = build_robot_agent(session)
        ...
"""

from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters.unitree_go2.api import UnitreeGo2Api
from jiuwensymbiosis.adapters.unitree_go2.config import UnitreeGo2Config
from jiuwensymbiosis.adapters.unitree_go2.env import UnitreeGo2Env

build_unitree_go2_session = make_builder(UnitreeGo2Config, UnitreeGo2Env, UnitreeGo2Api)
