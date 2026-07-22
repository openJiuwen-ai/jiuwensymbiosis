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
from jiuwensymbiosis.agent.fast.sequence import TRACK_DETECT, TRACK_GRASP
from jiuwensymbiosis.agent.session import RobotSession

logger = logging.getLogger(__name__)

__all__ = ["run_robot_task", "run_fast_task"]


def _prime_fast_agent(agent: Any) -> None:
    """Run the agent's lazy async rail registration (its only ``invoke()``-time
    init the fast path skips).

    ``build_robot_agent`` only *queues* rails (``_pending_rails``); they are
    registered onto ``Runner.callback_framework`` lazily inside
    ``DeepAgent.invoke()`` → ``_ensure_initialized()``. The fast path never
    calls ``invoke()``, so without this the rails (SafetyRail / RecoveryRail /
    TraceRail) are never wired up and ``BEFORE_TOOL_CALL``/``AFTER_TOOL_CALL``
    fire to nothing. ``callback_framework`` is a process-wide class singleton
    whose registered callbacks survive across event loops, so running init in
    its own loop here is fine — the later per-op ``asyncio.run`` in
    ``ability_exec`` sees the same registered callbacks.
    """
    asyncio.run(agent.ensure_initialized())


def _fire_invoke_event(agent: Any, event: Any, *, conversation_id: str, query: str) -> None:
    """Fire one invoke-lifecycle event (BEFORE/AFTER_INVOKE) on the outer agent.

    These are ``_OUTER_ONLY_EVENTS`` in openjiuwen, so they route to the outer
    DeepAgent's callback manager (not ``react_agent``, which ``ability_exec``
    uses for the per-op tool-call events). BEFORE_INVOKE primes TraceRail's
    ``ExecutionTrace``; AFTER_INVOKE flushes the trace JSON to disk. Each runs
    in its own short-lived loop — no per-step cost, and the real-time servo
    ticks (which bypass ``ability_manager`` entirely) are never traced.
    """
    from openjiuwen.core.single_agent.rail.base import (
        AgentCallbackContext,
        InvokeInputs,
    )

    async def _fire() -> None:
        ctx = AgentCallbackContext(
            agent=agent,
            inputs=InvokeInputs(query=query, conversation_id=conversation_id),
        )
        await ctx.fire(event)

    asyncio.run(_fire())


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
        conv_id = conversation_id or f"task-{uuid.uuid4().hex[:8]}"
        return run_fast_task(session, query, config, conversation_id=conv_id)

    # --- slow path: per-step LLM orchestration (unchanged behaviour) ---
    agent = build_robot_agent(session, config)
    conv_id = conversation_id or f"task-{uuid.uuid4().hex[:8]}"
    return asyncio.run(agent.invoke({"query": query, "conversation_id": conv_id}))


def _resolve_fast_special_ops(
    caps: frozenset[str] | set[str],
    api: Any,
    env: Any,
) -> frozenset[str]:
    """Derive the authorized fast-path special ops from session capabilities.

    ``ServoBinding`` needs ``api.get_pose`` plus at least one dispatch sink —
    ``api.servo_to_tip`` OR ``env.servo_to_flange`` (it falls back to the env
    verb). Requiring both would wrongly disable tracking for an adapter that
    only implements the env sink.
    """
    has_grasp = bool(caps & {"grasp.parallel", "grasp.suction"})
    binding_available = callable(getattr(api, "get_pose", None)) and (
        callable(getattr(api, "servo_to_tip", None)) or callable(getattr(env, "servo_to_flange", None))
    )
    if "vision.eye_to_hand" in caps:
        if {"motion.servo", "vision.detection"} <= caps and has_grasp and binding_available:
            return frozenset({TRACK_GRASP})
        return frozenset()
    if {"motion.servo", "vision.detection"} <= caps and binding_available:
        # Eye-in-hand adapters: relative tracking via track_detect.
        return frozenset({TRACK_DETECT})
    return frozenset()


def run_fast_task(
    session: RobotSession,
    query: str,
    config: RobotAgentConfig,
    *,
    conversation_id: str | None = None,
) -> dict:
    """Fast path: compile the task to an action sequence (1 LLM call), then run it
    through the SAME agent + rails the slow path uses — no per-step LLM.

    Fast and agent now share one execution engine: we build the agent exactly as
    agent mode does (``build_robot_agent`` → all rails), then drive its
    ``ability_manager`` with the precompiled sequence instead of looping the LLM.
    SafetyRail / VisualFeedbackRail / RecoveryRail therefore all apply.

    ``conversation_id`` seeds the trace's run token (its JSON filename + frames
    subdir) the same way the agent path's ``invoke`` does. The fast path skips
    ``agent.invoke()`` (no per-step LLM), so it manually primes the rails
    (``_prime_fast_agent``) and fires the invoke lifecycle
    (``_fire_invoke_event`` BEFORE/AFTER) so TraceRail records each discrete
    sequence step and persists a trace JSON — exactly the trace the agent path
    produces, with zero overhead when tracing is off.
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

    caps = set(getattr(session.env, "capabilities", frozenset()))
    special_ops = _resolve_fast_special_ops(caps, session.api, session.env)

    try:
        raw = compile_sequence(
            query,
            skills_md=skills_md,
            action_vocab=vocab,
            allowed_ops=set(action_index),
            special_ops=special_ops,
            api_base=spec.api_base,
            api_key=spec.api_key,
            model_name=spec.model_name,
            temperature=spec.temperature,
        )
    except RuntimeError as exc:
        logger.error("[fast] sequence compiler unavailable/failed: %s", exc)
        return {"ok": False, "reason": f"compile_failed: {exc}", "query": query}

    steps = parse_sequence(raw, allowed_ops=set(action_index), special_ops=special_ops)
    logger.info("[fast] compiled %d-step sequence for task=%r", len(steps), query)

    # The trace run token (JSON filename + frames subdir) derives from this; the
    # dispatch site in run_robot_task always supplies one, default here if called
    # directly so the trace is never written under a "noinv" placeholder.
    conv_id = conversation_id or f"task-{uuid.uuid4().hex[:8]}"

    # Build the agent (same rails as agent mode) and run the sequence through its
    # ability_manager so every op passes the rail stack — no LLM in the loop.
    agent = build_robot_agent(session, config)
    # The fast path never calls agent.invoke(), so do its two invoke-time side
    # effects by hand: lazy rail registration, then the BEFORE/AFTER_INVOKE
    # lifecycle that primes + flushes the TraceRail (a no-op when tracing is off).
    _prime_fast_agent(agent)
    from openjiuwen.core.single_agent.rail.base import AgentCallbackEvent

    if config.enable_tracing:
        _fire_invoke_event(
            agent,
            AgentCallbackEvent.BEFORE_INVOKE,
            conversation_id=conv_id,
            query=query,
        )
    executor = build_ability_executor(agent)
    result = run_sequence(session, steps, config=exec_cfg, executor=executor)
    if config.enable_tracing:
        _fire_invoke_event(
            agent,
            AgentCallbackEvent.AFTER_INVOKE,
            conversation_id=conv_id,
            query=query,
        )
    result["sequence"] = raw
    return result
