# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

# --COPY AND RENAME THIS DIRECTORY--
# 1. cp -r templates/xxx_adapter/ jiuwensymbiosis/adapters/your_robot/
# 2. Rename all Xxx / xxx placeholders to your robot name
# 3. Follow docs/hardware-porting-guide.md for step-by-step instructions

"""build_xxx_session — one call from YAML to a ready-to-connect session.

Usage::

    session = build_xxx_session.from_yaml("configs/xxx/default.yaml")
    with session:
        print(session.describe())
"""

from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters.xxx.config import XxxConfig
from jiuwensymbiosis.adapters.xxx.env import XxxEnv
from jiuwensymbiosis.adapters.xxx.api import XxxApi


build_xxx_session = make_builder(XxxConfig, XxxEnv, XxxApi)
