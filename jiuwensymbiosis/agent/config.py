# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Agent configuration types — model, rail, prompt, and agent-level settings.

All configuration dataclasses and constants that control agent behaviour
are defined here, keeping ``builder.py`` focused on pure construction logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from jiuwensymbiosis.agent.abstractions import (
    Model,
    ModelClientConfig,
    ModelRequestConfig,
)

Mode = Literal["tool", "code", "hybrid"]

# Execution mechanism (the speed switch):
#   "agent" — per-step LLM orchestration (current default; many LLM round-trips).
#   "fast"  — plan once (one LLM inference selecting skills), then run an
#             in-process Perceive+Act real-time loop with NO LLM in the loop.
ExecMode = Literal["agent", "fast"]

__all__ = [
    "Mode",
    "ExecMode",
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
    "**要操作的目标由用户的自然语言任务决定**：你自己从用户的话里识别出要检测/操作的目标"
    "（用它的自然语言描述——颜色/形状/大小/类别/材质等任意特征），作为检测工具"
    "(get_grasp_info_simple / analyze_scene)的 object_name 参数；用户不会、也不需要再单独传物体参数。"
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


def build_model(spec: ModelSpec | None = None) -> Any:
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
    required_flags: list[str]
    required_capabilities: list[str] | None = None
    any_capabilities: list[str] | None = None

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
        strict_capabilities: When True, raise on connect if the api declares
            capabilities the env do not provide (a config error). Defaults to
            False (warn only) for backward compatibility.
        enable_tracing: Attach ``TraceRail`` to record / persist / replay each
            invoke. Defaults False (zero overhead when off).
        trace_max_entries / trace_max_frames: Caps on recorded steps / frames.
        trace_save_frames: Save JPEG frames to ``<workspace>/traces/frames/{run_token}/``.
        trace_console: Print a one-line per-step dashboard to stdout.
        trace_dir: Override trace output dir (default ``<workspace>/traces``).
        trace_capture_loggers: Logger-name prefixes whose WARNING+ records are
            captured into the trace.
        enable_diagnosis: Attach ``DiagnosisRail`` to feed a compact diagnosis
            of a failed step back into the next LLM turn — current params,
            relevant recent history, and system state (recovery result / pose).
            Requires ``enable_tracing=True``; auto-disables (with a warning)
            when tracing is off. Defaults False.
        diagnosis_max_chars: Soft cap on the rendered diagnosis message; when
            exceeded the causal-chain history is dropped first, keeping the
            current step and system state.
        diagnosis_history_steps: How many recent related entries to include in
            the causal chain (same tool or matching rail-event kind).
        diagnosis_history_kinds: Rail-event ``kind`` values that mark a history
            entry as relevant to the current failure (default: safety reject
            + recovery).
        log_level / log_dir: Centralised logging level and file dir
            (see ``jiuwensymbiosis.utils.logging.configure_logging``).
            ``log_dir`` defaults to ``"./logs"`` so framework logs land at
            ``logs/jiuwensymbiosis.log``. openjiuwen's own log backend lands
            under ``logs/logs/`` due to its implementation (a double-join of
            the relative ``log_path``); the two are independent — jiuwensymbiosis
            does not configure openjiuwen's log path. Set ``None`` for
            console-only.
        parallel_tool_calls: Whether the agent loop may dispatch multiple tool
            calls concurrently. Defaults **False** (sequential) — robot motion
            is inherently sequential, and openjiuwen's per-tool ``ctx.extra``
            is a shared dict, so parallel dispatch races every rail that
            locates the "current step" via ``ctx.extra`` or
            ``trace.entries[-1]`` (TraceRail / VisualFeedbackRail). Set True
            only with non-motion parallel tools after auditing the rail stack.
    """

    mode: Mode = "hybrid"
    model: Any = None
    model_spec: ModelSpec | None = None
    system_prompt: str | None = None
    enable_visual_feedback: bool = True
    enable_safety: bool = True
    enable_recovery: bool = True
    enable_skill: bool = False
    extra_tools: list[Any] | None = None
    extra_rails: list[Any] | None = None
    max_iterations: int = 15
    workspace: str | None = None
    strict_capabilities: bool = False
    # -- Execution trace — all default OFF for zero overhead. --
    enable_tracing: bool = False
    trace_max_entries: int = 200
    trace_max_frames: int = 50
    trace_save_frames: bool = False
    trace_console: bool = False
    trace_dir: str | None = None  # default <workspace>/traces
    trace_capture_loggers: list[str] = field(default_factory=lambda: ["jiuwensymbiosis"])
    # -- Online diagnosis (DiagnosisRail) — requires enable_tracing. --
    enable_diagnosis: bool = False
    diagnosis_max_chars: int = 1500
    diagnosis_history_steps: int = 3
    diagnosis_history_kinds: tuple[str, ...] = ("reject", "recover")
    # -- Centralised logging --
    log_level: str = "INFO"
    # Default "./logs" so framework logs land at ``logs/jiuwensymbiosis.log``.
    # openjiuwen's own log backend lands under ``logs/logs/`` due to its
    # implementation (double-join of relative log_path) — independent of this
    # setting; jiuwensymbiosis does not touch openjiuwen's log path. Set None
    # for console-only; override via env or YAML ``agent.log_dir``.
    log_dir: str | None = "./logs"
    parallel_tool_calls: bool = False

    # --- speed switch (fast path) ---
    # exec_mode: "agent" (per-step LLM, current) or "fast" agent (plan-once + real-time
    #   Perceive+Act loop with no LLM in the loop). See ExecMode.
    exec_mode: ExecMode = "agent"
    exec_config: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RobotAgentConfig:
        """Build a ``RobotAgentConfig`` from a YAML ``agent:`` mapping.

        Mirrors how ``model:`` maps to ``ModelSpec(**data)`` and ``env.cfg``
        maps to ``PiperConfig.from_dict`` — the three top-level YAML blocks
        (``env`` / ``model`` / ``agent``) each drive their own dataclass.

        ``model`` / ``model_spec`` are deliberately popped: they describe a
        pre-built model instance / spec, owned by the separate ``model:``
        block (and the demo's ``_build_model_spec``), not declarable here.
        The caller assigns ``config.model_spec = spec`` after this call.

        Any unknown key raises ``TypeError`` (catches YAML typos at load
        time rather than silently ignoring them).
        """
        data = dict(data or {})
        data.pop("model", None)
        data.pop("model_spec", None)
        return cls(**data)
