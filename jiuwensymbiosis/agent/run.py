# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unified task entry — the speed switch between the two execution mechanisms.

``run_robot_task(session, query, config)`` dispatches on ``config.exec_mode``:

* ``"agent"`` (default, unchanged): build the ``DeepAgent`` and ``invoke`` it —
  per-step LLM orchestration, many round-trips. Identical to calling
  ``build_robot_agent`` + ``agent.invoke`` directly.

* ``"fast"``: the C1 single-source path (see ``fast_path_single_source_design.md``).
    1. **Compile once** — a single LLM inference reads the candidate skills'
       SKILL.md (the same files the agent reads) and emits, in that one call, the
       flat **action sequence** for the task (skill selection + workflow
       transcription together — no separate compile round-trip).
    2. **Run** — the generic ``run_sequence`` executes that sequence in order with
       NO per-step LLM, passing detection results between steps and real-time-
       tracking targets at ``track_detect`` steps.

Single source of truth is each skill's SKILL.md; there is no per-skill Python
executor, so fast and agent can never drift, and a new skill is just a new
SKILL.md.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from jiuwensymbiosis.agent.builder import build_robot_agent
from jiuwensymbiosis.agent.config import RobotAgentConfig
from jiuwensymbiosis.agent.session import RobotSession

logger = logging.getLogger(__name__)

__all__ = ["run_robot_task", "run_fast_task"]


def run_robot_task(
    session: RobotSession,
    query: str,
    config: RobotAgentConfig | None = None,
    *,
    conversation_id: str | None = None,
) -> Any:
    """Run a task on ``session`` using the mechanism selected by ``config.exec_mode``.

    The session's ``connect()``/``disconnect()`` is the caller's responsibility
    (use ``with session:``).
    """
    config = config or RobotAgentConfig()
    if config.exec_mode == "fast":
        return run_fast_task(session, query, config)

    # --- slow path: per-step LLM orchestration (unchanged behaviour) ---
    agent = build_robot_agent(session, config)
    conv_id = conversation_id or f"task-{uuid.uuid4().hex[:8]}"
    return asyncio.run(agent.invoke({"query": query, "conversation_id": conv_id}))


def run_fast_task(
    session: RobotSession,
    query: str,
    config: RobotAgentConfig,
) -> dict:
    """Fast path: compile the task to an action sequence (1 LLM call), then run it
    through the SAME agent + rails the slow path uses — no per-step LLM.

    Fast and agent now share one execution engine: we build the agent exactly as
    agent mode does (``build_robot_agent`` → all rails), then drive its
    ``ability_manager`` with the precompiled sequence instead of looping the LLM.
    SafetyRail / VisualFeedbackRail / RecoveryRail therefore all apply.
    """
    # Imported lazily so the slow path never pulls in the realtime stack.
    from jiuwensymbiosis.agent.fast import (
        DEFAULT_REGISTRY,
        SkillExecConfig,
        compile_sequence,
        parse_sequence,
        run_sequence,
    )
    from jiuwensymbiosis.agent.fast.ability_exec import build_ability_executor
    from jiuwensymbiosis.tools.robot_control_tool import _build_action_index

    spec = config.model_spec
    if spec is None:
        return {"ok": False, "reason": "no_model_spec", "query": query}

    exec_cfg = config.exec_config or SkillExecConfig()
    action_index = _build_action_index(session.api)
    vocab = sorted(action_index)
    skills_md = DEFAULT_REGISTRY.skills_markdown()

    try:
        raw = compile_sequence(
            query,
            skills_md=skills_md,
            action_vocab=vocab,
            allowed_ops=set(action_index),
            api_base=spec.api_base,
            api_key=spec.api_key,
            model_name=spec.model_name,
            temperature=spec.temperature,
        )
    except RuntimeError as exc:
        logger.error("[fast] sequence compiler unavailable/failed: %s", exc)
        return {"ok": False, "reason": f"compile_failed: {exc}", "query": query}

    steps = parse_sequence(raw, allowed_ops=set(action_index))
    logger.info("[fast] compiled %d-step sequence for task=%r", len(steps), query)

    # Build the agent (same rails as agent mode) and run the sequence through its
    # ability_manager so every op passes the rail stack — no LLM in the loop.
    agent = build_robot_agent(session, config)
    executor = build_ability_executor(agent)
    result = run_sequence(session, steps, config=exec_cfg, executor=executor)
    result["sequence"] = raw
    return result
