# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""jiuwensymbiosis — robotics agent framework on top of openjiuwen."""

# Proxy hygiene: clear proxy env BEFORE the agent/openjiuwen imports below pull
# openjiuwen's httpx layer (which would route local vLLM/detection through a
# SOCKS proxy and require socksio). Must run at the very top of package init so
# even `from jiuwensymbiosis.utils.proxy import clear_proxy_env` (which triggers
# this __init__) finds the env already clean. Logic mirrors utils/proxy.py
# (single source of truth); kept inline here to avoid recursing through
# utils.__init__ (which imports agent.config → openjiuwen). See AGENTS.md.
import os as _os

_popped_proxy_env: dict[str, str] = {}
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    _v = _os.environ.pop(_k, None)
    if _v is not None:
        _popped_proxy_env[_k] = _v
_os.environ.setdefault("NO_PROXY", "*")
_os.environ.setdefault("no_proxy", "*")
_proxy = (
    _popped_proxy_env.get("https_proxy")
    or _popped_proxy_env.get("HTTPS_PROXY")
    or _popped_proxy_env.get("http_proxy")
    or _popped_proxy_env.get("HTTP_PROXY")
    or _popped_proxy_env.get("all_proxy")
    or _popped_proxy_env.get("ALL_PROXY")
)
if _proxy:
    _os.environ.setdefault("JIUWEN_LLM_PROXY", _proxy)
del _os, _popped_proxy_env, _proxy

from jiuwensymbiosis.agent import (
    AgentRail,
    LocalFunction,
    ModelSpec,
    Tool,
    ToolCard,
    ToolOutput,
    build_model,
    build_robot_agent_config,
    clear_proxy_env,
    run_fast_task,
    run_robot_task,
)
from jiuwensymbiosis.agent.builder import build_robot_agent
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.api.mixins import (
    JointMotionMixin,
    MotionMixin,
    ParallelGripperMixin,
    SuctionMixin,
    VisionMixin,
)
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation
from jiuwensymbiosis.tools.builder import build_robot_tools
from jiuwensymbiosis.tools.inproc_code import InProcessCodeTool

__all__ = [
    "BaseRobotEnv",
    "RobotObservation",
    "BaseRobotApi",
    "robot_tool",
    "MotionMixin",
    "JointMotionMixin",
    "SuctionMixin",
    "ParallelGripperMixin",
    "VisionMixin",
    "build_robot_tools",
    "InProcessCodeTool",
    "RobotSession",
    "build_robot_agent",
    "build_robot_agent_config",
    "run_robot_task",
    "run_fast_task",
    "AgentRail",
    "Tool",
    "ToolCard",
    "LocalFunction",
    "ToolOutput",
    "ModelSpec",
    "build_model",
    "clear_proxy_env",
]
