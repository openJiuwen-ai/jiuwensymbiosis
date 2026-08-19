# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unified task entry — the speed switch between the two execution mechanisms.

``run_robot_task(session, query, config)`` dispatches on ``config.exec_mode``:

* ``"stepagent"``: build the ``DeepAgent`` and ``invoke`` it — per-step LLM
  orchestration, many round-trips. Identical to calling ``build_robot_agent`` +
  ``agent.invoke`` directly. For single-step debugging / verification.

* ``"fastagent"`` (the task-running default): the C1 single-source path (see
  ``fast_path_single_source_design.md``).
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
import inspect
import logging
import uuid
from collections.abc import Mapping
from typing import Any

from jiuwensymbiosis.agent.builder import build_robot_agent
from jiuwensymbiosis.agent.config import RobotAgentConfig
from jiuwensymbiosis.agent.fast.sequence import TRACK_DETECT, TRACK_GRASP, qualifiers_for
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.api.state import CONTAINMENT_RELATIONS

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
    cancel_token: Any = None,
) -> Any:
    """Run a task on ``session`` using the mechanism selected by ``config.exec_mode``.

    The session's ``connect()``/``disconnect()`` is the caller's responsibility
    (use ``with session:``).

    ``cancel_token`` (GUI-only ``CancelToken``) makes the fast path's compile LLM
    call abandonable; ``run_sequence`` reads it off ``session.cancel_token``. The
    slow agent path does not use it (its LLM calls live inside openjiuwen). ``None``
    → unchanged behaviour for CLI / tests.
    """
    config = config or RobotAgentConfig()
    if config.exec_mode == "fastagent":
        conv_id = conversation_id or f"task-{uuid.uuid4().hex[:8]}"
        return run_fast_task(session, query, config, conversation_id=conv_id, cancel_token=cancel_token)

    # --- slow path: per-step LLM orchestration (unchanged behaviour) ---
    agent = build_robot_agent(session, config)
    conv_id = conversation_id or f"task-{uuid.uuid4().hex[:8]}"
    return asyncio.run(agent.invoke({"query": query, "conversation_id": conv_id}))


def _action_param_sig(fn: Any) -> str:
    """Render a bound action's call params as ``(a, b?)`` — ``?`` = optional (has a
    default). Feeds the compiler exact param names for every action so it never
    invents them (e.g. ``target`` instead of ``box``). Object-agnostic: params are
    the same regardless of which object the task names.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return ""
    parts = [
        name + ("?" if p.default is not inspect.Parameter.empty else "")
        for name, p in params.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    return "(" + ", ".join(parts) + ")"


def _scan_for(api: Any, names: list[str]) -> tuple[list[dict], list[str]]:
    """Detect each name via the standing detector.

    Returns ``(found, scanned)`` — ``scanned`` lists only the names the detector actually
    answered for. A name whose detector call RAISED is absent from both: a broken detector
    is not evidence of absence, and reporting "looked for it, not there" off a crash would
    hand the planner a fact nobody established.
    """
    found: list[dict] = []
    scanned: list[str] = []
    for n in names:
        try:
            res = api.analyze_scene(n)
        except Exception as exc:  # noqa: BLE001 - detector is best-effort
            logger.debug("[fast] analyze_scene(%r) failed: %s", n, exc)
            continue
        if isinstance(res, dict) and res.get("ok"):
            found.extend(res.get("objects") or [])
            scanned.append(n)
    return found, scanned


def _split_by_qualifier(objs: list[dict], refs: list[dict], reference: str,
                        relation: str) -> tuple[list[dict], list[dict]]:
    """Split ``objs`` into (satisfies ``<relation> reference``, seen but does not).

    Same predicate the grasp path uses, so the pre-plan prompt and execution agree on which
    apple the task means. When the reference wasn't found or either side lacks measured
    bounds nothing is judged — everything stays in the first list, because a qualifier we
    could not evaluate must not silently disqualify the only candidate.
    """
    from jiuwensymbiosis.perception import scene3d

    ref = next((r for r in refs if r.get("object") == reference and scene3d.has_extent(r)), None)
    if ref is None:
        return objs, []
    ok, rejected = [], []
    for obj in objs:
        if not scene3d.has_extent(obj):
            ok.append(obj)
        elif scene3d.relation_holds(obj, ref, relation):
            ok.append(obj)
        else:
            rejected.append(obj)
    return ok, rejected


def _perceive_scene(
    session: Any,
    targets: list[str],
    references: list[str] | None = None,
    grounding: dict[str, dict] | None = None,
) -> dict | None:
    """Pre-plan scene perception (NO LLM): detect the parsed targets AND the references
    they are located by, aggregated into an ``analyze_scene``-shaped summary for LLM②.

    References are scanned too because a reference the planner can SEE is what lets it
    reason about the target it CANNOT: told "the apple in the drawer", a prompt that
    reports only "apple 未见" is scene-blind, while one that also reports "drawer 在
    (0.6,0,0.7)" hands the model something its own commonsense can act on.

    Best-effort: returns ``None`` if the api has no ``analyze_scene`` / no
    ``vision.detection`` capability / nothing at all is detected — the compiler then
    plans scene-blind (backward compatible).
    """
    api = getattr(session, "api", None)
    if api is None or not hasattr(api, "analyze_scene"):
        return None
    if "vision.detection" not in getattr(api, "capabilities", frozenset()):
        return None
    objects, scanned_t = _scan_for(api, targets)
    refs, scanned_r = _scan_for(api, [r for r in (references or []) if r not in targets])
    # Apply the task's own qualifier ("the apple IN the drawer") before anything is reported.
    # Skipping this is how a scene with two same-label objects gets the wrong one advertised at a
    # concrete coordinate — and the planner then has every reason to go grasp it.
    unqualified: list[dict] = []
    for name in list(grounding or {}):
        same = [obj for obj in objects if obj.get("object") == name]
        quals = qualifiers_for(grounding, name)
        if not same or not quals:
            continue
        # Several qualifiers on one name mean several DIFFERENT objects the task wants
        # ("the box beside the banana" and "the box in the cabinet"). An instance is kept
        # if it satisfies any of them; only one satisfying none is out.
        keep: list[dict] = []
        for g in quals:
            kept, _ = _split_by_qualifier(same, refs, g["reference"], g["relation"])
            keep.extend(obj for obj in kept if obj not in keep)
        rejected = [obj for obj in same if obj not in keep]
        if rejected:
            objects = [obj for obj in objects if obj not in rejected]
            unqualified.append({
                "object": name,
                "reference": " / ".join(g["reference"] for g in quals),
                "relation": " / ".join(g["relation"] for g in quals),
                "count": len(rejected),
                "nearest_mm": min((obj.get("distance_mm") for obj in rejected
                                   if isinstance(obj.get("distance_mm"), (int, float))), default=None),
            })
    has_reach = "planning.reachability" in getattr(api, "capabilities", frozenset())
    looked_for = [*scanned_t, *scanned_r]
    if not objects and not refs:
        # Nothing found. Still report WHAT WAS LOOKED FOR: "we scanned for apple and drawer and saw
        # neither" tells the planner to search before reaching, whereas returning None tells it
        # nothing happened at all. If the body has a URDF reach model, add its reach envelope as a
        # prior so it can reason "reachable only after moving" instead of planning scene-blind.
        blind: dict[str, Any] = {"count": 0, "objects": []}
        if looked_for:
            blind["missing"] = looked_for
        if unqualified:
            blind["unqualified"] = unqualified
        if has_reach:
            try:
                prior = api.describe_reach()
            except Exception as exc:  # noqa: BLE001 - best-effort
                logger.debug("[fast] describe_reach failed: %s", exc)
                prior = None
            if prior:
                blind["reach_prior"] = prior
        return blind if (looked_for or "reach_prior" in blind) else None
    # Annotate each detected object with the framework reachability capability (URDF IK at the current
    # body pose). Capability-gated: bodies without planning.reachability get no field (piper unchanged).
    if has_reach:
        # References get it too: whether the drawer is within reach is exactly what decides
        # "open it from here" vs "drive up to it first".
        for obj in (*objects, *refs):
            try:
                obj["reachable"] = bool(api.check_reachable(obj))
            except Exception as exc:  # noqa: BLE001 - precheck is best-effort
                logger.debug("[fast] reachability precheck failed: %s", exc)
    objects.sort(key=lambda obj: obj.get("distance_mm", float("inf")))
    refs.sort(key=lambda obj: obj.get("distance_mm", float("inf")))
    scene: dict[str, Any] = {"count": len(objects), "objects": objects}
    if refs:
        scene["references"] = refs
    if unqualified:
        scene["unqualified"] = unqualified
    # What was LOOKED FOR but not found is as informative as what was: it is the difference
    # between "the apple isn't here" and "nobody ever looked". Only stated when a scan ran.
    seen = {obj.get("object") for obj in (*objects, *refs)}
    missing = [n for n in looked_for if n not in seen]
    if missing:
        scene["missing"] = missing
    return scene


def _blocked_access(scene: Any, grounding: Mapping[str, Mapping[str, str]] | None) -> dict[str, str]:
    """``{thing: what the pre-plan look found in its way}`` — the evidence for check 5.

    Two routes, because a barrier reaches the planner two different ways:

    * **The task named it** — "the box IN the cabinet", and the look found the cabinet but
      not the box. Something is closing it; the box cannot be taken until that is dealt with.
    * **Only the camera knows** — a crate measured sitting ON the box. Nobody mentioned it,
      because the person giving the task neither knew nor cared. This is the common case for
      obstruction, and it is why the check cannot be driven off the task text alone.

    Absence of evidence stays absence: a thing nobody looked for, or a reference nobody
    found, produces no entry. What is returned is only ever "we saw this in the way".
    """
    if not isinstance(scene, dict):
        return {}
    from jiuwensymbiosis.perception import scene3d

    objects = scene.get("objects") or []
    refs = scene.get("references") or []
    missing = set(scene.get("missing") or [])
    seen = {obj.get("object"): obj for obj in (*objects, *refs) if obj.get("object")}
    out: dict[str, str] = {}
    for target in list(grounding or {}):
        for g in qualifiers_for(grounding, target):
            ref = g["reference"]
            if g["relation"] in CONTAINMENT_RELATIONS and target in missing and ref in seen:
                out[target] = str(ref)
                break
    for obj in objects:
        name = obj.get("object")
        if not name or name in out or not scene3d.has_extent(obj):
            continue
        for other in (*objects, *refs):
            if other is obj or not scene3d.has_extent(other) or other.get("object") == name:
                continue
            try:
                on_top = scene3d.relation_holds(obj, other, "under")
            except Exception as exc:  # noqa: BLE001 - a predicate that cannot judge blocks nothing
                logger.debug("[fast] relation_holds(%r under %r) failed: %s", name, other.get("object"), exc)
                continue
            if on_top:
                out[name] = str(other.get("object"))
                break
    return out


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
    cancel_token: Any = None,
) -> dict:
    """Fast path: plan the task into an action sequence, then run it through the
    SAME agent + rails the slow path uses — no per-step LLM.

    Planning is ``plan_task``: compose registered skills when the library covers the
    task, derive a sequence from the action contracts when it does not. Which tier
    ran is reported back as ``plan_tier``. Both tiers stay available mid-run: if the
    measured state contradicts a step's pre-conditions, the runner re-plans the
    remainder from what the world actually is rather than failing there.

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
        parse_sequence,
        plan_task,
        run_sequence,
    )
    from jiuwensymbiosis.agent.fast.ability_exec import build_ability_executor
    from jiuwensymbiosis.agent.fast.planner import parse_task
    from jiuwensymbiosis.api.world_state import WorldState
    from jiuwensymbiosis.tools.robot_control_tool import _build_action_index

    spec = config.model_spec
    if spec is None:
        return {"ok": False, "reason": "no_model_spec", "query": query}

    exec_cfg = config.exec_config or SkillExecConfig()
    # planner_only: this index is BOTH the vocabulary the planner is shown and the
    # allow-list ``parse_sequence`` validates against, so narrowing it here is what
    # actually keeps a bring-up tool out of a plan (a prompt instruction would not).
    action_index = _build_action_index(session.api, planner_only=True)
    action_sigs = {name: _action_param_sig(fn) for name, fn in action_index.items()}
    skills_md = DEFAULT_REGISTRY.skills_markdown()

    caps = set(getattr(session.env, "capabilities", frozenset()))
    special_ops = _resolve_fast_special_ops(caps, session.api, session.env)

    # ① LLM① parse the task → structured intent (targets); ② perceive the scene
    # (no LLM, reuses the standing detector) so ③ the compiler (LLM②) plans from
    # perception (far → move first; multiple targets → a loop). Both steps are
    # best-effort: if the parser/detector is unavailable, we fall back to a
    # scene-blind compile (backward compatible).
    try:
        intent = parse_task(
            query, api_base=spec.api_base, api_key=spec.api_key,
            model_name=spec.model_name, temperature=spec.temperature,
        )
    except Exception as exc:  # noqa: BLE001 - parser is best-effort
        logger.warning("[fast] task parse failed (scene-blind compile): %s", exc)
        intent = {"targets": []}
    scene = _perceive_scene(
        session, intent.get("targets") or [], intent.get("references") or [], intent.get("grounding") or {}
    )

    plan_kwargs = {
        "skills_md": skills_md,
        "action_index": action_index,
        # The index, not just its keys: parse_sequence reads each op's ToolMeta off it,
        # and a bare set of names silently no-ops the binding/freshness/state checks.
        "allowed_ops": action_index,
        "special_ops": special_ops,
        "api_base": spec.api_base,
        "api_key": spec.api_key,
        "model_name": spec.model_name,
        "temperature": spec.temperature,
        "api_capabilities": sorted(session.api.capabilities),
        "action_sigs": action_sigs,
        "scene": scene,
        # The task's own spatial qualifiers. parse_sequence rejects a step that could carry
        # one and drops it, so "the apple in the drawer" cannot compile into a call that
        # would grasp whichever apple the detector happens to rank first.
        "grounding": intent.get("grounding") or {},
        # What the look found in the way of what (shut container, crate stacked on top), so a
        # plan that reaches through it is rejected with the barrier named instead of running.
        "blocked_access": _blocked_access(scene, intent.get("grounding") or {}),
        # GUI-only: makes the compile/re-plan LLM call abandonable. None → unchanged
        # synchronous behaviour for CLI / tests. Kept in plan_kwargs so the initial
        # plan AND every re-plan are cancellable, not just the first one.
        "cancel_token": cancel_token,
    }

    world = WorldState.snapshot(session)
    try:
        planned = plan_task(
            query,
            world_block=world.as_prompt_block(),
            world_tokens=sorted(world.tokens) or None,
            **plan_kwargs,
        )
    except RuntimeError as exc:
        logger.error("[fast] planning failed: %s", exc)
        return {"ok": False, "reason": f"compile_failed: {exc}", "query": query}

    raw = planned.sequence
    steps = parse_sequence(raw, allowed_ops=action_index, special_ops=special_ops,
                           blocked_access=plan_kwargs["blocked_access"])
    logger.info("[fast] %s-tier plan → %d steps for task=%r", planned.tier, len(steps), query)

    def replan(measured: WorldState, why: str) -> list | None:
        """Re-plan the remainder from what the world actually is, both tiers again.

        The interruption reason goes into the task text so the planner works around
        what went wrong instead of re-deriving the sequence that just stopped fitting.
        """
        again = plan_task(
            f"{query}\n\n（执行中断：{why}。请依据【当前状态】重新规划剩余动作。）",
            world_block=measured.as_prompt_block(),
            world_tokens=sorted(measured.tokens) or None,
            **plan_kwargs,
        )
        logger.info("[fast] re-planned at %s-tier → %d steps", again.tier, len(again.sequence))
        return parse_sequence(again.sequence, allowed_ops=action_index, special_ops=special_ops,
                              blocked_access=plan_kwargs["blocked_access"])

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
    result = run_sequence(session, steps, config=exec_cfg, executor=executor, replan=replan)
    if config.enable_tracing:
        _fire_invoke_event(
            agent,
            AgentCallbackEvent.AFTER_INVOKE,
            conversation_id=conv_id,
            query=query,
        )
    result["sequence"] = raw
    # How the sequence was planned — a successful "action"-tier run is a candidate
    # for distillation into a new skill, and there is no other way to tell.
    result["plan_tier"] = planned.tier
    if planned.skills:
        result["plan_skills"] = list(planned.skills)
    if planned.reason:
        result["plan_reason"] = planned.reason
    return result
