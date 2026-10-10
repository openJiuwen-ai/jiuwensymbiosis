# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Agent configuration types — model, rail, prompt, and agent-level settings.

All configuration dataclasses and constants that control agent behaviour
are defined here, keeping ``builder.py`` focused on pure construction logic.
"""

from __future__ import annotations

import copy
import math
import types
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from jiuwensymbiosis.agent.abstractions import (
    Model,
    ModelClientConfig,
    ModelRequestConfig,
)
from jiuwensymbiosis.agent.execution_config import TrackingConfig

Mode = Literal["tool", "code", "hybrid"]

# Execution mechanism (the speed switch):
#   "fastagent" — plan once (one LLM inference selecting skills), then run the
#                 action sequence with NO per-step LLM. The task-running default.
#   "stepagent" — per-step LLM orchestration (many round-trips); for single-step
#                 debugging / verification.
ExecMode = Literal["fastagent", "stepagent"]

__all__ = [
    "Mode",
    "ExecMode",
    "ModelSpec",
    "RailConfig",
    "RobotAgentConfig",
    "ExecutionConfig",
    "FastAgentConfig",
    "StepAgentConfig",
    "TrackingConfig",
    "ModulesConfig",
    "SwitchConfig",
    "DiagnosisConfig",
    "TracingConfig",
    "LoggingConfig",
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
class SwitchConfig:
    enabled: bool = False


@dataclass
class DiagnosisConfig(SwitchConfig):
    max_chars: int = 1500
    history_steps: int = 3
    history_kinds: tuple[str, ...] = ("reject", "recover")


@dataclass
class TracingConfig(SwitchConfig):
    max_entries: int = 200
    max_frames: int = 50
    save_frames: bool = False
    console: bool = False
    dir: str | None = None
    capture_loggers: list[str] = field(default_factory=lambda: ["jiuwensymbiosis"])


@dataclass
class ModulesConfig:
    skills: SwitchConfig = field(default_factory=lambda: SwitchConfig(True))
    safety: SwitchConfig = field(default_factory=lambda: SwitchConfig(True))
    recovery: SwitchConfig = field(default_factory=lambda: SwitchConfig(True))
    visual_feedback: SwitchConfig = field(default_factory=SwitchConfig)
    diagnosis: DiagnosisConfig = field(default_factory=DiagnosisConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    dir: str | None = "./logs"
    motion_dir: str | None = None


@dataclass
class StepAgentConfig:
    mode: Mode = "hybrid"
    max_iterations: int = 15
    system_prompt: str | None = None
    parallel_tool_calls: bool = False


@dataclass
class FastAgentConfig:
    max_replans: int = 2
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


@dataclass
class ExecutionConfig:
    mode: ExecMode = "fastagent"
    stepagent: StepAgentConfig = field(default_factory=StepAgentConfig)
    fastagent: FastAgentConfig = field(default_factory=FastAgentConfig)


def _merge(base: Mapping, override: Mapping) -> dict:
    """Deep-merge explicit settings; defaults are resolved only by the parser."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _typed(value: Any, annotation: Any, path: str) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        for choice in args:
            if choice is type(None):
                continue
            try:
                return _typed(value, choice, path)
            except (TypeError, ValueError):
                pass
        raise ValueError(f"{path}: invalid value {value!r}")
    if origin is Literal:
        if value not in args or not isinstance(value, str):
            raise ValueError(f"{path} must be one of {args}")
        return value
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _parse_settings(annotation, value, path)
    # Only homogeneous collections are part of the configuration schema.
    # A fixed-length tuple must not be validated using its first item type.
    is_homogeneous_list = origin is list and len(args) == 1
    is_variadic_tuple = origin is tuple and len(args) == 2 and args[1] is Ellipsis
    if is_homogeneous_list or is_variadic_tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path} must be a list")
        items = [_typed(item, args[0], path) for item in value]
        return tuple(items) if origin is tuple else items
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{path} must be a finite number")
        return float(value)
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{path} must be bool")
        return value
    if annotation is int:
        # bool is an int subclass and must not silently pass as int.
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{path} must be int")
        return int(value)
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(f"{path} must be str")
        # Preserve the underlying text as plain str, bypassing subclass
        # overrides of __str__ and __getitem__.
        return str.__str__(value)
    raise TypeError(f"{path}: unsupported configuration annotation {annotation!r}")


def _parse_settings(cls: type, data: Any, path: str, *, defaults: Any = None) -> Any:
    if isinstance(data, cls) and is_dataclass(data) and not isinstance(data, type):
        data = asdict(data)
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must be a mapping or {cls.__name__}")
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise TypeError(f"unknown fields in {path}: {sorted(unknown)}")
    values = _merge(asdict(defaults or cls()), data)
    hints = get_type_hints(cls)
    return cls(**{key: _typed(value, hints[key], f"{path}.{key}") for key, value in values.items()})


@dataclass(init=False, slots=True)
class RobotAgentConfig:
    """Agent settings shared by Python, YAML, CLI and the workbench.

    YAML fields live in execution/modules/logging; model and extension objects
    are Python construction dependencies. See docs/zh/reference/agent-config.md.
    """

    execution: ExecutionConfig
    modules: ModulesConfig
    logging: LoggingConfig
    workspace: str | None
    strict_capabilities: bool
    model: Any = field(repr=False)
    model_spec: ModelSpec | None
    extra_tools: list[Any] | None = field(repr=False)
    extra_rails: list[Any] | None = field(repr=False)

    def __init__(
        self,
        *,
        execution: ExecutionConfig | Mapping | None = None,
        modules: ModulesConfig | Mapping | None = None,
        logging: LoggingConfig | Mapping | None = None,
        workspace: str | None = None,
        strict_capabilities: bool = False,
        model: Any = None,
        model_spec: ModelSpec | None = None,
        extra_tools: list[Any] | None = None,
        extra_rails: list[Any] | None = None,
    ) -> None:
        self.execution = _parse_settings(ExecutionConfig, execution if execution is not None else {}, "agent.execution")
        fast = self.execution.mode == "fastagent"
        defaults = ModulesConfig(skills=SwitchConfig(fast), visual_feedback=SwitchConfig(not fast))
        self.modules = _parse_settings(
            ModulesConfig, modules if modules is not None else {}, "agent.modules", defaults=defaults
        )
        self.logging = _parse_settings(LoggingConfig, logging if logging is not None else {}, "agent.logging")
        self.workspace = _typed(workspace, str | None, "agent.workspace")
        self.strict_capabilities = _typed(strict_capabilities, bool, "agent.strict_capabilities")
        self.model, self.model_spec = model, model_spec
        self.extra_tools, self.extra_rails = extra_tools, extra_rails
        self.validate()

    def validate(self, *, for_execution: bool = False) -> None:
        """Recheck mutable Python settings before use; mode rules at task dispatch."""
        self.strict_capabilities = _typed(self.strict_capabilities, bool, "agent.strict_capabilities")
        self.workspace = _typed(self.workspace, str | None, "agent.workspace")
        self.execution = _parse_settings(ExecutionConfig, self.execution, "agent.execution")
        self.modules = _parse_settings(ModulesConfig, self.modules, "agent.modules")
        self.logging = _parse_settings(LoggingConfig, self.logging, "agent.logging")
        limits = {
            "execution.stepagent.max_iterations": (self.execution.stepagent.max_iterations, 1),
            "execution.fastagent.max_replans": (self.execution.fastagent.max_replans, 0),
            "modules.tracing.max_entries": (self.modules.tracing.max_entries, 1),
            "modules.tracing.max_frames": (self.modules.tracing.max_frames, 0),
            "modules.diagnosis.max_chars": (self.modules.diagnosis.max_chars, 1),
            "modules.diagnosis.history_steps": (self.modules.diagnosis.history_steps, 0),
        }
        for path, (value, minimum) in limits.items():
            if value < minimum:
                raise ValueError(f"agent.{path} must be >= {minimum}")
        if self.logging.level.upper() not in {
            "DEBUG",
            "INFO",
            "WARNING",
            "WARN",
            "ERROR",
            "CRITICAL",
            "FATAL",
            "NOTSET",
        }:
            raise ValueError("agent.logging.level is not a logging level")
        if self.modules.diagnosis.enabled and not self.modules.tracing.enabled:
            raise ValueError("agent.modules.diagnosis.enabled requires modules.tracing.enabled")
        if for_execution and self.execution.mode == "fastagent":
            for name in ("visual_feedback", "diagnosis"):
                if getattr(self.modules, name).enabled:
                    raise ValueError(
                        f"agent.modules.{name}.enabled is only supported by stepagent; disable it for fastagent"
                    )

    def prepare_session(self, session: Any) -> None:
        """Apply connection-time settings before connect, shared by official hosts."""
        session.strict_capabilities = self.strict_capabilities
        session.motion_log_dir = self.logging.motion_dir

    def to_dict(self) -> dict[str, Any]:
        """Serializable effective settings; never serialize injected Python objects."""
        return {
            "execution": asdict(self.execution),
            "modules": asdict(self.modules),
            "logging": asdict(self.logging),
            "workspace": self.workspace,
            "strict_capabilities": self.strict_capabilities,
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, *, overrides: Mapping[str, Any] | None = None
    ) -> RobotAgentConfig:
        for source in (data, overrides):
            if source is not None and not isinstance(source, Mapping):
                raise TypeError("agent must be a mapping")
            for name in ("model", "model_spec", "extra_tools", "extra_rails"):
                if source is not None and name in source:
                    raise TypeError(f"agent.{name} is Python-only, not a YAML setting")
        return cls(**_merge(data or {}, overrides or {}))
