# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from jiuwensymbiosis.api import defaults
from jiuwensymbiosis.api.actions import ACTIONS, ActionSpec, implements, planner_vocabulary
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import ToolMeta, robot_tool
from jiuwensymbiosis.api.components import Approach, Reachability, Scene3D

__all__ = [
    "BaseRobotApi",
    # The shared action vocabulary — what an action IS.
    "ACTIONS",
    "ActionSpec",
    "implements",
    "planner_vocabulary",
    "ToolMeta",
    # Reusable implementations an adapter CALLS. Not a base class: a body takes one
    # action without taking its neighbours.
    "defaults",
    # For a tool only one body has (bring-up, calibration, vendor demo). Planner-invisible.
    "robot_tool",
    # Components an adapter HOLDS, never inherits — so a body takes one component's
    # actions without taking its neighbours' (see api/components.py).
    "Scene3D",
    "Approach",
    "Reachability",
]
