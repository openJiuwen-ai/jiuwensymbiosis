# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Agent configuration types — model, rail, prompt, and agent-level settings.

All configuration dataclasses and constants that control agent behaviour
are defined here, keeping ``builder.py`` focused on pure construction logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

from jiuwensymbiosis.agent.abstractions import (
    Model,
    ModelClientConfig,
    ModelRequestConfig,
)

Mode = Literal["tool", "code", "hybrid"]

__all__ = [
    "Mode",
    "ModelSpec",
    "RailConfig",
    "RobotAgentConfig",
    "ROBOT_PROMPT_TEMPLATE",
    "build_model",
]


ROBOT_PROMPT_TEMPLATE = (
    # Aligned with openjiuwen sub-agent prompt convention (see
    # ``harness/subagents/browser_agent.py`` / ``research_agent.py``):
    # role + which kind of tools to use, no tool-list duplication, no
    # pseudocode procedure. Tool names / parameters / semantics reach the
    # LLM through the OpenAI ``tools`` field; repeating them in prose only
    # confuses the model and biases it toward writing code in ``content``.
    "你是机器人控制代理，负责操作 {robot_name} 完成物理任务。"
    "请使用提供的工具完成感知、运动、抓取和释放，从工具描述中读取每个工具的参数和返回值。"
    "**要抓取/放置的物体由用户的自然语言任务决定**：你自己从用户的话里识别出"
    "「要抓的物体」和「放置目标」，把对应的物体名作为检测工具(get_grasp_info_simple / "
    "analyze_scene 的 object_name, 或 slot_pick 的 chip_object_name/slot_object_name)的参数；"
    "用户不会、也不需要再单独传物体参数。"
    "开始任何动作前先回到 home 位姿；视觉检测应在 home 高度进行以获得清晰深度。"
    "完成或失败时简洁汇报最终结果。"
)


@dataclass
class ModelSpec:
    """Backend-agnostic model description.

    Attributes:
        provider: Provider name (e.g. ``OpenAI``, ``SiliconFlow``).
        api_base: Base URL, must NOT include ``/chat/completions``.
        api_key: API key; pass empty string for endpoints that don't auth.
        model_name: Model id understood by the endpoint.
        temperature: Sampling temperature.
        max_tokens: Output cap.
        verify_ssl: Set ``False`` for self-signed dev endpoints.
        extra_request_kwargs: Forwarded into ``ModelRequestConfig``.
    """

    provider: str = "OpenAI"
    api_base: str = "http://127.0.0.1:8110/v1"
    api_key: str = "EMPTY"
    model_name: str = "Qwen/Qwen3-VL-32B-Instruct"
    temperature: float = 0.3
    max_tokens: int = 2048
    verify_ssl: bool = False
    extra_request_kwargs: dict[str, Any] = field(default_factory=dict)


def build_model(spec: Optional[ModelSpec] = None) -> Any:
    """Construct an openjiuwen ``Model`` from a ``ModelSpec``.

    Args:
        spec: Model specification; defaults to
            ``ModelSpec()`` (local vLLM with Qwen3-VL-32B).

    Returns:
        An openjiuwen ``Model`` instance ready for use in ``create_deep_agent``.
    """
    spec = spec or ModelSpec()
    return Model(
        model_client_config=ModelClientConfig(
            client_provider=spec.provider,
            api_key=spec.api_key,
            api_base=spec.api_base,
            verify_ssl=spec.verify_ssl,
        ),
        model_config=ModelRequestConfig(
            model_name=spec.model_name,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            **spec.extra_request_kwargs,
        ),
    )


@dataclass
class RailConfig:
    """Configuration for a rail's activation conditions.

    Attributes:
        rail_class_path: Fully-qualified import path (``module.ClassName``).
        required_flags: Flag names that must all be ``True``.
        required_capabilities: All of these capabilities must be present.
        any_capabilities: At least one of these capabilities must be present.
    """

    rail_class_path: str
    required_flags: List[str]
    required_capabilities: Optional[List[str]] = None
    any_capabilities: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Normalize empty capability lists to ``None``."""
        if self.required_capabilities == []:
            self.required_capabilities = None
        if self.any_capabilities == []:
            self.any_capabilities = None


@dataclass
class RobotAgentConfig:
    """Configuration for building a robot agent.

    Attributes:
        mode: Operating mode — ``"tool"``, ``"code"``, or ``"hybrid"``.
        model: Pre-built model instance; takes precedence over ``model_spec``.
        model_spec: ``ModelSpec`` for automatic model construction.
        system_prompt: Custom system prompt; defaults to ``ROBOT_PROMPT_TEMPLATE``.
        enable_visual_feedback: Attach ``VisualFeedbackRail`` if capabilities allow.
        enable_safety: Attach ``SafetyRail`` if capabilities allow.
        enable_recovery: Attach ``RecoveryRail`` if capabilities allow.
        enable_skill: Enable ``SkillUseRail`` and ``RobotControlTool``.
        extra_tools: Additional tools appended to the tool list.
        extra_rails: Additional rails appended to the rail list.
        max_iterations: Maximum agent loop iterations.
        workspace: Agent workspace directory.
    """

    mode: Mode = "hybrid"
    model: Any = None
    model_spec: Optional[ModelSpec] = None
    system_prompt: Optional[str] = None
    enable_visual_feedback: bool = True
    enable_safety: bool = True
    enable_recovery: bool = True
    enable_skill: bool = False
    extra_tools: Optional[list[Any]] = None
    extra_rails: Optional[list[Any]] = None
    max_iterations: int = 15
    workspace: Optional[str] = None
