# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unified public API for the jiuwensymbiosis agent layer.

All symbols that wrap or re-export openjiuwen primitives are accessible from
this single module. Internal consumers (tools / rails / utils) should import
from ``jiuwensymbiosis.agent.abstractions`` rather than from ``openjiuwen``
directly to avoid circular imports during package initialisation.
"""

from jiuwensymbiosis.agent.abstractions import (
    AgentCard,
    AgentRail,
    LocalFunction,
    Model,
    ModelClientConfig,
    ModelRequestConfig,
    SkillUseRail,
    SubAgentConfig,
    Tool,
    ToolCard,
    ToolOutput,
    create_deep_agent,
)
from jiuwensymbiosis.agent.builder import (
    build_robot_agent,
    build_robot_agent_config,
)
from jiuwensymbiosis.agent.config import (
    ROBOT_PROMPT_TEMPLATE,
    ModelSpec,
    RobotAgentConfig,
    build_model,
)
from jiuwensymbiosis.agent.run import run_fast_task, run_robot_task
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.agent.trace import (
    ExecutionTrace,
    StepAwareTraceEventSink,
    TraceEntry,
    TraceEventSink,
    TraceRail,
)
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
    "run_robot_task",
    "run_fast_task",
    "RobotAgentConfig",
    "RobotSession",
    "ROBOT_PROMPT_TEMPLATE",
    "clear_proxy_env",
    "ExecutionTrace",
    "TraceEntry",
    "TraceEventSink",
    "StepAwareTraceEventSink",
    "TraceRail",
]
