# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unified public API for the jiuwensymbiosis agent layer.

All symbols that wrap or re-export openjiuwen primitives are accessible from
this single module. Internal consumers (tools / rails / utils) should import
from ``jiuwensymbiosis.agent.abstractions`` rather than from ``openjiuwen``
directly to avoid circular imports during package initialisation.
"""

from jiuwensymbiosis.agent.abstractions import (
    AgentRail,
    Tool,
    ToolCard,
    LocalFunction,
    ToolOutput,
    Model,
    ModelClientConfig,
    ModelRequestConfig,
    create_deep_agent,
    SubAgentConfig,
    AgentCard,
    SkillUseRail,
)
from jiuwensymbiosis.agent.config import (
    ModelSpec,
    RobotAgentConfig,
    ROBOT_PROMPT_TEMPLATE,
    build_model,
)
from jiuwensymbiosis.agent.builder import (
    build_robot_agent,
    build_robot_agent_config,
)
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.utils.proxy import clear_proxy_env

__all__ = [
    "AgentRail",
    "Tool",
    "ToolCard",
    "LocalFunction",
    "ToolOutput",
    "Model",
    "ModelClientConfig",
    "ModelRequestConfig",
    "create_deep_agent",
    "SubAgentConfig",
    "AgentCard",
    "SkillUseRail",
    "ModelSpec",
    "build_model",
    "build_robot_agent",
    "build_robot_agent_config",
    "RobotAgentConfig",
    "RobotSession",
    "ROBOT_PROMPT_TEMPLATE",
    "clear_proxy_env",
]
