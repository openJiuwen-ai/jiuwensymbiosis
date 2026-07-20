# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""SO-101 adapter (5-DoF underactuated arm + parallel gripper via LeRobot 0.6.x).

Public surface mirrors the other adapters::

    from jiuwensymbiosis.adapters.so101 import (
        So101Config,
        So101Env,
        So101Api,
        build_so101_session,
    )

    session = build_so101_session.from_yaml("configs/so101/so101.yaml")
    with session:
        agent = build_robot_agent(session)
        ...
"""

from __future__ import annotations

from jiuwensymbiosis.adapters.so101.api import So101Api
from jiuwensymbiosis.adapters.so101.config import So101Config
from jiuwensymbiosis.adapters.so101.env import So101Env
from jiuwensymbiosis.adapters.so101.geometry import So101Pose
from jiuwensymbiosis.adapters.so101.lowlevel import So101PoseConvergenceError
from jiuwensymbiosis.adapters.so101.session import build_so101_session

__all__ = [
    "So101Config",
    "So101Env",
    "So101Api",
    "So101Pose",
    "So101PoseConvergenceError",
    "build_so101_session",
]
