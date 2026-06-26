# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Robotics SubAgent builder — the equivalent of openjiuwen's
``code_agent`` / ``browser_agent`` factories, but for robots.

``build_robot_agent(session)`` returns a ready-to-invoke ``DeepAgent``.
``build_robot_agent_config(...)`` returns a ``SubAgentConfig`` that can be
plugged into a higher-level agent's ``subagents=[...]`` for multi-robot setups.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from jiuwensymbiosis.agent.abstractions import (
    AgentCard,
    SkillUseRail,
    SubAgentConfig,
    create_deep_agent,
)
from jiuwensymbiosis.agent.config import (
    ROBOT_PROMPT_TEMPLATE,
    Mode,
    RailConfig,
    RobotAgentConfig,
    build_model,
)
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.agent.trace import TraceRail
from jiuwensymbiosis.skills import SKILLS_DIR
from jiuwensymbiosis.utils.logging import TraceLogHandler, configure_logging

_JIUWENSYMBIOSIS_SETTINGS = Path.home() / ".jiuwensymbiosis" / "settings.json"

logger = logging.getLogger(__name__)

__all__ = [
    "build_robot_agent",
    "build_robot_agent_config",
]


def _read_settings_workspace() -> str | None:
    """Read workspace path from ``~/.jiuwensymbiosis/settings.json``."""
    try:
        if _JIUWENSYMBIOSIS_SETTINGS.exists():
            data = json.loads(_JIUWENSYMBIOSIS_SETTINGS.read_text(encoding="utf-8"))
            return data.get("workspace")
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _resolve_workspace(session: RobotSession, workspace: str | None) -> str:
    """Resolve the agent workspace path.

    Priority: explicit ``workspace`` argument > ``$JIUWENSYMBIOSIS_WORKSPACE`` >
    ``~/.jiuwensymbiosis/settings.json`` > ``~/.jiuwensymbiosis/{session.name}_workspace/``.

    The directory is created on first use so DeepAgent's ``AGENT.md`` /
    ``SOUL.md`` / ``skills/`` / ``sessions/`` persist across runs.
    """
    if workspace:
        path = Path(workspace).expanduser().resolve()
    else:
        env_ws = os.environ.get("JIUWENSYMBIOSIS_WORKSPACE")
        if env_ws:
            path = Path(env_ws).expanduser().resolve()
        else:
            settings_ws = _read_settings_workspace()
            if settings_ws:
                path = Path(settings_ws).expanduser().resolve()
            else:
                path = (Path.home() / ".jiuwensymbiosis" / f"{session.name}_workspace").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _format_tool_list(api: Any) -> str:
    """Format available tools as a bullet list for diagnostics."""
    from jiuwensymbiosis.tools.builder import list_tool_meta

    metas = list_tool_meta(api)
    if not metas:
        return "(no tools)"
    return "\n".join(f"- {m['name']}: {m['description']}" for m in metas)


class _RailRegistry:
    """Registry managing rail activation conditions and dynamic imports.

    Centralises all rail configuration and provides methods to check whether
    a rail should be enabled based on current session flags and capabilities.
    """

    _rails: list[RailConfig] = [
        RailConfig(
            rail_class_path="jiuwensymbiosis.rails.visual_feedback.VisualFeedbackRail",
            required_flags=["enable_visual_feedback"],
            required_capabilities=["vision.camera"],
        ),
        RailConfig(
            rail_class_path="jiuwensymbiosis.rails.safety.SafetyRail",
            required_flags=["enable_safety"],
            required_capabilities=["motion.cartesian"],
        ),
        RailConfig(
            rail_class_path="jiuwensymbiosis.rails.recovery.RecoveryRail",
            required_flags=["enable_recovery"],
            any_capabilities=["motion.cartesian", "grasp.suction", "grasp.parallel"],
        ),
    ]

    @classmethod
    def get_enabled_rails(
        cls,
        flag_states: dict[str, bool],
        session_capabilities: set[str],
        session: Any,
    ) -> list[Any]:
        """Return all rail instances whose activation conditions are met.

        Args:
            flag_states: Mapping of flag names to boolean values.
            session_capabilities: Available session capabilities.
            session: ``RobotSession`` passed to each rail constructor.

        Returns:
            List of instantiated rail objects.
        """
        enabled_rails: list[Any] = []
        for config in cls._rails:
            if cls._should_enable(flag_states, session_capabilities, config):
                try:
                    rail_class = cls._import_rail_class(config.rail_class_path)
                    enabled_rails.append(rail_class(session))
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Warning: Failed to import {config.rail_class_path}: {e}")
                    continue
        return enabled_rails

    @classmethod
    def _should_enable(
        cls,
        flag_states: dict[str, bool],
        session_capabilities: set[str],
        config: RailConfig,
    ) -> bool:
        """Check whether a rail's activation conditions are satisfied."""
        if not all(flag_states.get(flag, False) for flag in config.required_flags):
            return False
        if config.required_capabilities:
            if not all(cap in session_capabilities for cap in config.required_capabilities):
                return False
        if config.any_capabilities:
            if not any(cap in session_capabilities for cap in config.any_capabilities):
                return False
        return True

    @classmethod
    def _import_rail_class(cls, rail_class_path: str) -> Any:
        """Dynamically import a rail class from its fully-qualified path."""
        module_path, class_name = rail_class_path.rsplit(".", 1)
        module: ModuleType = importlib.import_module(module_path)
        return getattr(module, class_name)


def _inject_trace_sinks(rails: list[Any], trace_rail: TraceRail | None) -> None:
    """Wire the TraceRail as a ``trace_sink`` on rails that accept one.

    SafetyRail / RecoveryRail / VisualFeedbackRail each accept an optional
    ``trace_sink`` constructor arg, but the registry builds them with just
    ``session``. We set the attribute post-hoc when tracing is on. Missing the
    attribute or a None TraceRail → no-op (fully backward compatible).

    VisualFeedbackRail additionally takes a ``frame_sink`` so the JPEG it
    injects into the agent context is *also* saved to the trace frames dir.
    """
    if trace_rail is None:
        return
    for rail in rails:
        if rail is trace_rail:
            continue
        try:
            if hasattr(rail, "trace_sink"):
                rail.trace_sink = trace_rail
            if hasattr(rail, "frame_sink"):
                rail.frame_sink = _make_frame_sink(trace_rail)
        except (AttributeError, TypeError) as exc:
            logger.warning("Failed to inject trace_sink into %r: %s", rail, exc)


def _make_frame_sink(trace_rail: TraceRail):
    """Return a ``(rgb, tool_name) -> path`` callable that saves a trace frame."""

    def _sink(rgb: Any, _tool_name: str):
        return trace_rail.save_frame_for_sink(rgb)

    return _sink


def _attach_trace_log_handlers(trace_rail: TraceRail, loggers: list[str], level: int) -> TraceLogHandler:
    """Attach a TraceLogHandler to each named logger, bound to the TraceRail.

    Removes any previously-attached ``TraceLogHandler`` instances first so
    repeated ``build_robot_agent`` calls don't accumulate no-op handlers.
    """
    import logging as _logging

    handler = TraceLogHandler(sink=trace_rail, level=level)
    for name in loggers:
        lg = _logging.getLogger(name)
        # Purge any stale TraceLogHandler instances (from a prior build).
        for h in list(lg.handlers):
            if isinstance(h, TraceLogHandler):
                lg.removeHandler(h)
        lg.addHandler(handler)
    return handler


def _resolve_rails(
    session: RobotSession,
    enable_visual_feedback: bool,
    enable_safety: bool,
    enable_recovery: bool,
    extra_rails: list[Any] | None,
) -> list[Any]:
    """Resolve enabled rails based on session capabilities and flags.

    Args:
        session: RobotSession instance.
        enable_visual_feedback: Enable visual feedback rail.
        enable_safety: Enable safety rail.
        enable_recovery: Enable recovery rail.
        extra_rails: Optional additional rails to append.

    Returns:
        List of enabled rail instances.
    """
    flag_states: dict[str, bool] = {
        "enable_visual_feedback": enable_visual_feedback,
        "enable_safety": enable_safety,
        "enable_recovery": enable_recovery,
    }
    session_capabilities: set[str] = set(session.env.capabilities)
    rails: list[Any] = _RailRegistry.get_enabled_rails(
        flag_states,
        session_capabilities,
        session,
    )
    if extra_rails:
        rails.extend(extra_rails)
    return rails


def _build_tools(
    session: RobotSession,
    mode: Mode,
    extra_tools: list[Any] | None,
    enable_skill: bool = False,
) -> list[Any]:
    """Build tool list for the agent based on operating mode and skill flag.

    Args:
        session: RobotSession providing api and globals.
        mode: ``"tool"`` / ``"code"`` / ``"hybrid"``.
        extra_tools: Additional tools to include.
        enable_skill: Append ``RobotControlTool`` when ``True``.

    Returns:
        List of openjiuwen ``Tool`` / ``LocalFunction`` instances.
    """
    from jiuwensymbiosis.tools.builder import build_robot_tools
    from jiuwensymbiosis.tools.inproc_code import make_inproc_code_tool
    from jiuwensymbiosis.tools.robot_control_tool import RobotControlTool

    tools: list[Any] = []
    if mode in ("tool", "hybrid"):
        tools.extend(build_robot_tools(session.api, env=session.env))
    if mode in ("code", "hybrid"):
        tools.append(make_inproc_code_tool(session.globals_provider))
    if enable_skill:
        tools.append(RobotControlTool(session.api))
    if extra_tools:
        tools.extend(extra_tools)
    return tools


def _maybe_append_skill_rail(rails: list[Any], enable_skill: bool) -> list[Any]:
    """Append ``SkillUseRail`` when ``enable_skill`` is set.

    Loads skills from the built-in ``jiuwensymbiosis/skills/`` directory
    without exposing generic tools (bash / code / read_file).
    """
    if not enable_skill:
        return rails
    rails.append(
        SkillUseRail(
            skills_dir=str(SKILLS_DIR),
            skill_mode="auto_list",
            include_tools=False,
        )
    )
    return rails


def _build_system_prompt(session: RobotSession, custom_prompt: str | None, mode: Mode = "hybrid") -> str:
    """Render ``ROBOT_PROMPT_TEMPLATE`` for this session.

    Only the robot name is interpolated; tool descriptions reach the LLM
    through the OpenAI ``tools`` field, not through prose duplication.

    When the agent runs an in-process code tool (``mode`` in ``code``/``hybrid``),
    the names available to that code (``env``, ``api``, ``np`` + any
    ``extra_globals``) are appended so the model knows what it can reference —
    otherwise ``extra_globals`` helpers stay invisible to the LLM.
    """
    if custom_prompt is not None:
        return custom_prompt
    desc = session.describe()
    base = ROBOT_PROMPT_TEMPLATE.format(robot_name=desc["name"])
    if mode not in ("code", "hybrid"):
        return base
    globals_section = _render_globals_section(session)
    if globals_section:
        return base + "\n\n" + globals_section
    return base


def _render_globals_section(session: RobotSession) -> str:
    """One prose line listing the names ``InProcessCodeTool`` injects.

    Reflected from ``session.globals_provider()`` so ``extra_globals`` additions
    are auto-documented without the adapter author editing the prompt.
    """
    keys = list(session.globals_provider().keys())
    if not keys:
        return ""
    names = ", ".join(f"`{k}`" for k in keys)
    return (
        "在代码模式（run_python）中，以下全局变量可直接使用："
        f"{names}。多步控制流可写进 run_python，把最终值赋给 RESULT。"
    )


def build_robot_agent(
    session: RobotSession,
    config: RobotAgentConfig | None = None,
) -> Any:
    """Build a ready-to-invoke ``DeepAgent`` bound to one robot session.

    Args:
        session: ``RobotSession`` with connected env and api.
        config: Agent configuration; defaults to ``RobotAgentConfig()``.

    Returns:
        An openjiuwen ``DeepAgent`` instance. Invoke with
        ``asyncio.run(agent.invoke({"query": "...", "conversation_id": "..."}))``.

    The session's ``connect()`` / ``disconnect()`` is the caller's
    responsibility (use ``with session:`` for clean teardown).
    """
    config = config or RobotAgentConfig()
    # Propagate the strictness flag onto the session BEFORE the caller connects
    # it (typical: ``agent = build_robot_agent(session); with session: ...``).
    # If the session is already connected this is a no-op for this connect cycle.
    session.strict_capabilities = config.strict_capabilities
    # Centralised logging: one uniform format across all modules,
    # optional file output. Idempotent.
    configure_logging(level=config.log_level, log_dir=config.log_dir)
    model = config.model or build_model(config.model_spec)
    tools = _build_tools(session, config.mode, config.extra_tools, enable_skill=config.enable_skill)
    rails = _resolve_rails(
        session, config.enable_visual_feedback, config.enable_safety, config.enable_recovery, config.extra_rails
    )
    rails = _maybe_append_skill_rail(rails, config.enable_skill)
    sys_prompt = _build_system_prompt(session, config.system_prompt, mode=config.mode)
    workspace = _resolve_workspace(session, config.workspace)

    # Execution trace: prepend TraceRail so it observes every step.
    trace_rail: TraceRail | None = None
    if config.enable_tracing:
        import logging as _logging

        trace_dir = config.trace_dir or str(Path(workspace) / "traces")
        trace_rail = TraceRail(
            session,
            workspace=workspace,
            max_entries=config.trace_max_entries,
            max_frames=config.trace_max_frames,
            save_frames=config.trace_save_frames,
            console=config.trace_console,
            capture_loggers=tuple(config.trace_capture_loggers),
            capture_log_level=_logging.WARNING,
            traces_dir=Path(trace_dir),
        )
        _inject_trace_sinks(rails, trace_rail)
        log_handler = _attach_trace_log_handlers(trace_rail, list(config.trace_capture_loggers), _logging.WARNING)
        trace_rail.attach_log_handler(log_handler, tuple(config.trace_capture_loggers))
        session.attach_trace_rail(trace_rail)
        rails.insert(0, trace_rail)

    return create_deep_agent(
        model=model,
        system_prompt=sys_prompt,
        tools=tools,
        rails=rails,
        max_iterations=config.max_iterations,
        workspace=workspace,
        # Built-in skills (visual_pick / visual_place SKILL.md) live in the
        # package source tree, outside the agent workspace. With the default
        # sandbox (restrict_to_work_dir=True) SkillUseRail's read_file is denied
        # ("outside sandbox"). We expose no read_file/bash/code tools to the LLM
        # (include_tools=False), so widening fs access to load trusted bundled
        # skills is safe.
        restrict_to_work_dir=False,
    )


def build_robot_agent_config(
    session: RobotSession,
    *,
    config: RobotAgentConfig | None = None,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Return a ``SubAgentConfig`` for multi-robot top-level agents.

    This produces a config suitable for use as ``subagents=[cfg, ...]``
    in another ``create_deep_agent`` call. For the single-robot case,
    prefer ``build_robot_agent``.

    Args:
        session: ``RobotSession`` for this sub-agent.
        config: Agent configuration.
        name: Override sub-agent name (defaults to ``"robot_{session.name}"``).
        description: Override sub-agent description.

    Returns:
        An openjiuwen ``SubAgentConfig`` instance.
    """
    config = config or RobotAgentConfig()
    model = config.model or build_model(config.model_spec)
    tools = _build_tools(session, config.mode, config.extra_tools, enable_skill=config.enable_skill)
    rails = _resolve_rails(
        session, config.enable_visual_feedback, config.enable_safety, config.enable_recovery, config.extra_rails
    )

    agent_name = name or f"robot_{session.name}"
    return SubAgentConfig(
        agent_card=AgentCard(
            name=agent_name,
            description=description or f"Robot control agent for {session.name}.",
        ),
        system_prompt=config.system_prompt,
        tools=tools,
        rails=rails,
        model=model,
        max_iterations=config.max_iterations,
    )
