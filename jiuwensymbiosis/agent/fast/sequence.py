# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Action-sequence schema + safe expression evaluator for the C1 fast path.

The fast path (design: ``fast_path_single_source_design.md``) has the skill-
selection LLM emit, in the *same* call, an ordered **action sequence** — the
deterministic transcription of the selected skills' SKILL.md workflows. A
generic runner then executes that sequence with NO per-step LLM, passing
detection results between steps internally.

This module is the contract between the LLM (producer) and the runner
(consumer). It defines:

  * ``ActionStep`` — one step: an ``op`` (a ``@robot_tool`` action name, or the
    compound real-time op ``track_detect``) + ``params`` (literals or symbolic
    expressions) + optional ``bind`` for detection steps.
  * ``parse_sequence`` — validate a raw ``list[dict]`` (the LLM output) into
    ``list[ActionStep]``, rejecting unknown ops / malformed detection steps.
  * ``evaluate_expr`` / ``resolve_params`` — a **whitelisted-AST** evaluator so a
    param like ``"obj.z"`` or ``"obj.z + 30"`` resolves against the
    runtime variable environment (the detection bindings). It never executes
    arbitrary Python: only numbers, ``+ - * /``, unary ``-``, name lookup,
    ``var.field``, and ``var.field[idx]`` are allowed.
  * ``normalize_detection`` — the **task-agnostic** shape a detection binds: the
    raw perception fields passed through, plus geometric conveniences
    ``x/y/z = position[0]/[1]/[2]``. It bakes in NO task semantics (no
    pick/place); which field an expression reads (``grasp_z``, ``place_z``,
    ``position[0]``, …) is decided by the skill's SKILL.md, so the same
    machinery serves pick-place, carry, push, wipe, … equally.

Why string-or-number params: numeric targets (``goto_xyzr`` x/y/z) are
expressions; string args (``object_name``) are literals. ``resolve_params``
distinguishes them by trying to evaluate; a value that does not parse as a
numeric expression (e.g. ``"红杯子"``) is left as the literal string.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from jiuwensymbiosis.api.decorators import ToolMeta
from jiuwensymbiosis.api.state import (
    CONTAINMENT_RELATIONS,
    PAYLOAD_TOKENS,
    apply_effects,
    missing_requirements,
)

# Compound real-time ops implemented by the runner (not raw @robot_tool actions).
TRACK_DETECT = "track_detect"
TRACK_GRASP = "track_grasp"
KNOWN_SPECIAL_OPS = frozenset({TRACK_DETECT, TRACK_GRASP})

# Fields ``normalize_detection`` synthesises on top of whatever the action returned.
# They are always addressable, so field checking must not reject them.
_SYNTHETIC_BINDING_FIELDS = frozenset({"x", "y", "z"})

# Freshness slot for a sensing step that carries no ``bind``. Such an action leaves
# its reading in the api's own cache for the next step to consume implicitly (the
# common shape for drive-and-re-sense actions). Not a valid identifier, so it can
# never collide with a real bind name.
_CACHE_SLOT = "<cache>"


# --------------------------------------------------------------------------- #
# Access — "is the thing reachable, or is something in the way?" (api/state.py §3)
# --------------------------------------------------------------------------- #
def _referent_key(name: Any) -> str:
    """Normalised form used to decide whether two names mean the same thing."""
    return str(name).strip().lower() if isinstance(name, str) else ""


def _same_referent(a: str, b: str) -> bool:
    """Whether two names denote the same physical thing, for access purposes.

    A barrier is routinely named at a different grain than the thing it blocks:
    ``pull("cabinet door")`` clears ``"cabinet"``, ``pull("drawer handle")`` clears
    ``"drawer"``, ``"purple table"`` and ``"table"`` are one surface. Substring either
    way covers all of those. It is a shallow test on purpose — the consequence of a
    wrong match is one superfluous ordering constraint, never a motion.
    """
    ka, kb = _referent_key(a), _referent_key(b)
    return bool(ka) and bool(kb) and (ka in kb or kb in ka)


def _step_focus(params: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """``(subject, enclosure)`` a step names, or ``(None, None)`` if it names neither.

    ``enclosure`` is the reference the task put the subject *inside* — the only
    relations that mean enclosure are in :data:`CONTAINMENT_RELATIONS`; ``on`` /
    ``beside`` / ``near`` locate without blocking.
    """
    obj = params.get("object_name")
    if not isinstance(obj, str) or not obj.strip():
        return None, None
    ref, rel = params.get("reference"), params.get("relation")
    enclosed_by = ref if (isinstance(ref, str) and ref.strip() and rel in CONTAINMENT_RELATIONS) else None
    return obj, enclosed_by


def _focus_scan(raw: list[Any]) -> list[tuple[str | None, str | None]]:
    """Per-step ``(subject, enclosure)``, carrying the last named one forward.

    ``dual_arm_grasp`` / ``dual_arm_place`` take no object name — by contract they act
    on the most recent detection (their spec says so). Modelling that here is what lets
    the access check see *what* a parameterless manipulation is about; without it the
    step is anonymous and nothing can be said. A loop resets the carry: its body is
    validated on its own and re-senses each pass.
    """
    out: list[tuple[str | None, str | None]] = []
    subject: str | None = None
    enclosure: str | None = None
    for item in raw:
        params = item.get("params") if isinstance(item, dict) else None
        # Runs BEFORE per-step validation, so a malformed step must yield "no focus"
        # rather than raise — the real error belongs to the check that names it.
        if not isinstance(item, dict) or "loop" in item or not isinstance(params, (dict, type(None))):
            subject, enclosure = None, None
            out.append((None, None))
            continue
        named, enclosed_by = _step_focus(params or {})
        if named is not None:
            subject, enclosure = named, enclosed_by
        out.append((subject, enclosure))
    return out


def _clears_access(meta: ToolMeta | None) -> bool:
    """Whether this action makes its subject stop blocking things.

    Two ways, and a body needs neither annotated beyond the first: it **opens** the
    thing (``opens_access`` — a door, a lid, a lock, which stay where they are), or it
    **picks the thing up** (``payload.held``), which is how an obstacle stacked on top
    of the target gets dealt with when there is nothing to open. Leaving the second one
    out would reject the correct plan "move the crate off, then take the box".
    """
    if meta is None:
        return False
    return meta.opens_access or "payload.held" in meta.provides


def _manipulates(meta: ToolMeta | None) -> bool:
    """Whether the step changes what the end effector holds — i.e. it REACHES.

    Looking at a shut cabinet is how you discover it is shut, and driving up to it is
    how you get in range; neither is an error. Reaching into it is. ``provides`` over
    :data:`PAYLOAD_TOKENS` is exactly that line, read off the vocabulary rather than a
    hard-coded list of grasp/place names.
    """
    return meta is not None and bool(set(meta.provides) & PAYLOAD_TOKENS)


def _clearing_steps(raw: list[Any], focus: list[tuple[str | None, str | None]],
                    metas: Mapping[str, ToolMeta]) -> dict[str, int]:
    """``{referent_key: first index that clears it}`` — the plan's own testimony.

    A plan that opens X at step k is *stating* X was shut before k. That makes every
    earlier reach through X contradictory on the plan's own terms, with no world
    knowledge needed at all.
    """
    out: dict[str, int] = {}
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "loop" in item:
            continue
        meta = metas.get(item.get("op") or "")
        subject = focus[i][0]
        if subject and _clears_access(meta):
            out.setdefault(_referent_key(subject), i)
    return out


def _check_access(
    index: int,
    op: str,
    subject: str | None,
    enclosure: str | None,
    *,
    cleared: set[str],
    clears_at: Mapping[str, int],
    blocked: Mapping[str, str],
    shut_again: set[str],
    pending_clear: set[str],
    metas: Mapping[str, ToolMeta],
) -> None:
    """Reject a reach whose way is still blocked.

    Three independent grounds, all decidable:

    * **The plan contradicts itself** — it clears the barrier at a later step, so it has
      already said the barrier was there. Needs no world knowledge, always checked.
    * **The plan already put the barrier back** — a ``closes_access`` step, then a reach
      through it. Same evidence, other direction; also always checked.
    * **The pre-plan look contradicts it** — ``blocked`` maps a thing to whatever
      perception found in its way (shut container, crate stacked on top). Only checked
      when the body owns an ``opens_access`` action, because "not seen" is weak evidence:
      a body that can open something loses one step by acting on a false positive, while
      a body that cannot would lose the only plan it had.
    """
    # Barriers this step reaches THROUGH — the task put the subject inside one, or the
    # look found one on top of it. Acting *on* a barrier is a different thing entirely
    # (it is how you clear it), so the subject itself is only ever judged against the
    # plan's own testimony below, never against what perception saw.
    through: dict[str, str] = {}
    if enclosure:
        through[_referent_key(enclosure)] = enclosure
    if subject:
        by = blocked.get(_referent_key(subject))
        if by:
            through[_referent_key(by)] = by
    for key, name in through.items():
        if not key or any(_same_referent(key, c) for c in cleared):
            continue
        if any(_same_referent(key, s) for s in shut_again):
            raise SequenceError(
                f"step {index}: {op} reaches {subject!r} through {name!r}, which an earlier step in "
                f"this plan closed again. Re-open it, or do this before closing it."
            )
        if any(_same_referent(key, s) for s in pending_clear):
            raise SequenceError(
                f"step {index}: {op} reaches {subject!r} through {name!r}, which the plan only clears "
                f"AFTER this loop. Moving the reach inside a loop does not open it any earlier — "
                f"clear {name!r} before the loop."
            )
        if any(_same_referent(key, b) for b in blocked.values()) and _has_opener(metas):
            raise SequenceError(
                f"step {index}: {op} reaches {subject!r}, but {name!r} is in the way — the pre-plan "
                f"look found it blocking. Clear it first{_hint(_openers(metas))}, or move it aside "
                f"(picking it up counts)."
            )
    for key, name in {**through, **({_referent_key(subject): subject} if subject else {})}.items():
        if not key or any(_same_referent(key, c) for c in cleared):
            continue
        at = next((v for k, v in clears_at.items() if _same_referent(key, k)), None)
        if at is not None and at > index:
            raise SequenceError(
                f"step {index}: {op} reaches {subject!r} through {name!r}, but this plan only clears "
                f"{name!r} at step {at} — so it is still in the way here. Move that step earlier."
            )


def _has_opener(metas: Mapping[str, ToolMeta]) -> bool:
    return any(m.opens_access for m in metas.values())


def _openers(metas: Mapping[str, ToolMeta]) -> list[str]:
    return sorted(name for name, m in metas.items() if m.opens_access)


def _op_metas(allowed_ops: Any) -> dict[str, ToolMeta]:
    """``{op: ToolMeta}`` when ``allowed_ops`` is an action index, else empty.

    Callers that pass a bare set of names (tests, legacy call sites) get no contract
    data, so every contract check degrades to a no-op rather than a false rejection.
    """
    if not isinstance(allowed_ops, Mapping):
        return {}
    metas: dict[str, ToolMeta] = {}
    for name, fn in allowed_ops.items():
        meta = getattr(fn, "__robot_tool__", None)
        if isinstance(meta, ToolMeta):
            metas[name] = meta
    return metas


def _producers_of(metas: Mapping[str, ToolMeta], token: str) -> list[str]:
    """Op names whose ``provides`` establishes ``token`` — the actionable half of an error."""
    return sorted(name for name, m in metas.items() if token in m.provides)


def _location_producers(metas: Mapping[str, ToolMeta]) -> list[str]:
    """Op names that sense a position."""
    return sorted(name for name, m in metas.items() if m.produces_location)


def _hint(candidates: list[str]) -> str:
    """Render a 'what could fix this' clause, or nothing when we have no candidate."""
    return f"; actions that establish it: {candidates}" if candidates else ""


def _declared_params(meta: ToolMeta | None) -> frozenset[str] | None:
    """Param names an action advertises, or None when it advertises no schema.

    ``None`` means "cannot tell" (no contract at all) and disables the check —
    rejecting against an unknown schema would fail valid plans. An **empty** frozenset
    is different: the action declares it takes nothing, so any param is wrong. A body's
    *unadvertised* extras are deliberately absent — ``@implements`` publishes only the
    spec's params, and a plan may only use what the contract states.
    """
    if meta is None:
        return None
    props = (meta.input_params or {}).get("properties")
    return frozenset(props) if isinstance(props, dict) else None


class SequenceError(ValueError):
    """A raw action sequence failed schema validation."""


class ExprError(ValueError):
    """An expression could not be evaluated to a number under the safe grammar."""


# --------------------------------------------------------------------------- #
# Action step schema
# --------------------------------------------------------------------------- #
@dataclass
class ActionStep:
    """One step of an action sequence.

    Attributes:
        op: action name — a ``@robot_tool`` action (``home``, ``goto_xyzr``,
            ``open_gripper``, ``close_gripper``, ``get_grasp_info_simple``, …) or
            the compound ``track_detect``.
        params: keyword args for the action. Values are literals (number/str) or
            symbolic expression strings resolved at run time against the env.
        bind: for a detection op, the variable name its (normalized) result is
            bound to in the env (e.g. ``"box"``). ``None`` for non-detection ops.
            The binding carries ALL raw perception fields plus ``x/y/z``; which
            one an expression reads is the skill's choice (no task semantics here).
    """

    op: str
    params: dict[str, Any] = field(default_factory=dict)
    bind: str | None = None

    def is_detection(self) -> bool:
        """True if this step produces a binding (a ``track_detect`` / detect op)."""
        return self.bind is not None


@dataclass
class LoopStep:
    """A perception-terminated loop for a continuous multi-object task.

    Iteration count is decided **entirely by whether a target still exists** — not
    a pre-counted N. Each iteration:

      1. run ``detect_op`` (a **single-target** detector, e.g. ``locate_for_grasp`` /
         ``get_grasp_info_simple``) — it returns ``ok=True`` for one target, or
         ``ok=False`` when none is found;
      2. if no target → **stop** (the scene is clear);
      3. else bind that one target to ``bind`` and run ``body`` once (e.g. pick →
         place, referencing ``<bind>.field``);
      4. re-detect and repeat.

    So 2 targets → 2 passes, 100 → 100, 1 → 1. The initial detection needs only to
    find **one** target to start; re-detecting each pass makes it robust to objects
    moving. ``max_iters`` is just a safety cap; real termination is "detect returns
    no target". No per-step LLM. This is how "把所有箱子依次搬走" runs.
    """

    detect_op: str
    detect_params: dict[str, Any] = field(default_factory=dict)
    bind: str = "target"
    body: list[ActionStep] = field(default_factory=list)
    max_iters: int | None = None


def referenced_fields(value: Any) -> set[tuple[str, str]]:
    """``(binding, field)`` pairs a param value reads directly off a binding.

    Only first-level access counts — ``obj.position`` in ``obj.position[0]`` yields
    ``("obj", "position")``. Detection bindings are flat dicts, so a deeper chain is
    not a field of the binding and is left for runtime to reject.
    """
    if not isinstance(value, str):
        return set()
    try:
        tree = ast.parse(value, mode="eval")
    except SyntaxError:
        return set()
    return {
        (node.value.id, node.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }


def referenced_binding_names(value: Any) -> set[str]:
    """Root binding names a param value reads via ``.field`` / ``[idx]`` access.

    Returns the set of ``ast.Name`` roots reached through an attribute or
    subscript — e.g. ``{"obj"}`` for ``"obj.z + 30"`` or
    ``"obj.position[0]"``. A plain literal yields an empty set: an
    ``object_name`` like ``"red cup"`` is a syntax error, and a bare word like
    ``"红杯子"`` is a lone ``Name`` (not an access), so literals are never
    mistaken for binding references. Only attribute/subscript access means "read
    a detection field" — exactly the shape that must resolve against a prior
    ``track_detect`` bind.
    """
    if not isinstance(value, str):
        return set()
    try:
        tree = ast.parse(value, mode="eval")
    except SyntaxError:
        return set()
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            base = node.value
            while isinstance(base, (ast.Attribute, ast.Subscript)):
                base = base.value
            if isinstance(base, ast.Name):
                roots.add(base.id)
    return roots


def _validate_track_detect(params: Mapping[str, Any], bind: str | None, *, index: int) -> None:
    obj = params.get("object_name")
    if not isinstance(obj, str) or not obj:
        raise SequenceError(f"step {index}: track_detect requires params.object_name")
    if not bind:
        raise SequenceError(f"step {index}: track_detect requires a bind name")


def _validate_track_grasp(params: Mapping[str, Any], bind: str | None, *, index: int) -> None:
    obj = params.get("object_name")
    if not isinstance(obj, str) or not obj:
        raise SequenceError(f"step {index}: track_grasp requires params.object_name")
    if not bind:
        raise SequenceError(f"step {index}: track_grasp requires a bind name")
    approach = params.get("approach_mm")
    if isinstance(approach, bool) or not isinstance(approach, (int, float)):
        raise SequenceError(f"step {index}: track_grasp requires numeric approach_mm")
    if not math.isfinite(float(approach)) or not 30.0 <= float(approach) <= 100.0:
        raise SequenceError(f"step {index}: track_grasp approach_mm must be finite and in [30, 100]")


_SPECIAL_OP_VALIDATORS = {
    TRACK_DETECT: _validate_track_detect,
    TRACK_GRASP: _validate_track_grasp,
}


def _parse_loop(
    loop: Any,
    step_i: int,
    *,
    allowed_ops: Mapping[str, Any] | frozenset | set,
    special_ops: frozenset[str] | set[str] = frozenset(),
    bound: set[str] | None = None,
    fresh: set[str] | None = None,
    producer: Mapping[str, str] | None = None,
    grounding: Mapping[str, Mapping[str, str]] | None = None,
    blocked_access: Mapping[str, str] | None = None,
    cleared: set[str] | None = None,
    not_yet_clear: set[str] | None = None,
) -> LoopStep:
    """Validate a ``{"detect":{op,params}, "bind", "body":[...], "max_iters"?}`` loop.

    ``detect_op`` may be a normal action (``allowed_ops``) or an authorized
    special op (``special_ops``); the body is parsed recursively with the loop
    ``bind`` seeded as bound **and fresh** — it is re-sensed at the top of every
    iteration, so a body step reading it is always reading a current position.

    Access state crosses the boundary in both directions: what the enclosing plan has
    already cleared, and what it only clears later. Without that, wrapping a reach in a
    loop while leaving the barrier's opening outside it would hide the contradiction
    from both halves.
    """
    if not isinstance(loop, dict):
        raise SequenceError(f"step {step_i}: 'loop' must be an object")
    detect = loop.get("detect") or {}
    detect_op = detect.get("op")
    if not isinstance(detect_op, str) or (detect_op not in special_ops and detect_op not in allowed_ops):
        raise SequenceError(f"step {step_i}: loop.detect.op invalid/unknown: {detect_op!r}")
    detect_params = detect.get("params") or {}
    if not isinstance(detect_params, dict):
        raise SequenceError(f"step {step_i}: loop.detect.params must be an object")
    detect_meta = _op_metas(allowed_ops).get(detect_op)
    _check_param_names(step_i, f"loop.detect.{detect_op}", detect_params, detect_meta)
    _check_qualifier(step_i, f"loop.detect.{detect_op}", detect_params, detect_meta, grounding or {})
    bind = loop.get("bind", "target")
    if not isinstance(bind, str) or not bind.isidentifier():
        raise SequenceError(f"step {step_i}: loop.bind must be a valid identifier, got {bind!r}")
    body_raw = loop.get("body")
    if not isinstance(body_raw, list) or not body_raw:
        raise SequenceError(f"step {step_i}: loop.body must be a non-empty list")
    # Body may reference the loop's <bind>.field (bound fresh each iteration), plus
    # anything already bound before the loop.
    body = parse_sequence(
        body_raw,
        allowed_ops=allowed_ops,
        special_ops=special_ops,
        blocked_access=blocked_access,
        _precleared=set(cleared or ()),
        _not_yet_clear=set(not_yet_clear or ()),
        _prebound=set(bound or ()) | {bind},
        _prefresh=set(fresh or ()) | {bind},
        _preproducer={**dict(producer or {}), bind: detect_op},
    )
    if any(isinstance(s, LoopStep) for s in body):
        raise SequenceError(f"step {step_i}: nested loops are not supported")
    max_iters = loop.get("max_iters")
    if max_iters is not None and not (isinstance(max_iters, int) and max_iters > 0):
        raise SequenceError(f"step {step_i}: loop.max_iters must be a positive int or null")
    return LoopStep(
        detect_op=detect_op,
        detect_params=dict(detect_params),
        bind=bind,
        body=body,
        max_iters=max_iters,  # type: ignore[arg-type]
    )


def parse_sequence(
    raw: Any,
    *,
    allowed_ops: Mapping[str, Any] | frozenset | set,
    special_ops: frozenset[str] | set[str] = frozenset(),
    initial_state: Iterable[str] | None = None,
    grounding: Mapping[str, Mapping[str, str]] | None = None,
    blocked_access: Mapping[str, str] | None = None,
    _prebound: set[str] | None = None,
    _prefresh: set[str] | None = None,
    _preproducer: dict[str, str] | None = None,
    _precleared: set[str] | None = None,
    _not_yet_clear: set[str] | None = None,
) -> list[ActionStep | LoopStep]:
    """Validate a raw action sequence (the LLM output) into ``list[ActionStep]``.

    Five independent checks, each rejecting only what it can prove wrong so a plan
    the planner derived on its own is not second-guessed. **None of them imposes an
    order** — they reject unmet pre-conditions, so any permutation that satisfies
    them is accepted.

    1. *Vocabulary* — the op exists and, for a compound op, is authorized.
    2. *Bindings* — ``<bind>.field`` reads a name an earlier step bound, and (when
       the producing action publishes a ``returns`` schema) a field it really emits.
    3. *Location freshness* — a bind read for coordinates is still valid: an action
       that moved the body since it was sensed makes it stale.
    4. *Robot self-state* — each action's ``requires`` holds, given ``initial_state``
       advanced by every preceding action's effects.
    5. *Access* — nothing is in the way of what a step reaches for: no reaching into a
       container this plan only opens later, and none into one perception found shut.

    Args:
        raw: the LLM-produced sequence — must be a ``list`` of ``dict`` steps.
        allowed_ops: the action names the runner can execute. Pass the action
            **index** (``{name: bound_method}``) to enable checks 2–4; a bare set of
            names still works but carries no contract, so those checks no-op. A name
            in ``KNOWN_SPECIAL_OPS`` is **never** authorized via ``allowed_ops`` — it
            must be explicitly listed in ``special_ops``.
        special_ops: runner-owned compound operations authorized for this
            session. Defaults to empty — so callers that use no special ops
            stay compatible (``parse_sequence(raw, allowed_ops=...)`` still
            works), and ``track_detect`` / ``track_grasp`` are never implicitly
            authorized: an empty set rejects any special op with a
            ``SequenceError`` rather than silently letting it through.
        initial_state: robot self-state tokens true before the first step (from
            ``WorldState``). ``None`` means "unknown", which **disables** check 4 —
            asserting pre-conditions against a guessed state would reject valid plans.
        grounding: ``{object: {reference, relation}}`` — the spatial qualifiers the task
            itself stated. A step that could carry one and drops it is rejected: with two
            same-label objects in view, the bare call silently grasps the wrong one.
        blocked_access: ``{thing: what is in its way}`` as the pre-plan look found it —
            a shut container, a crate stacked on top. ``None``/empty leaves check 5 with
            only the plan's own testimony to go on, which is the safe default: a barrier
            nobody observed must not be invented.

    Returns:
        Validated steps, in order.

    Raises:
        SequenceError: on a non-list payload, a malformed step, an unknown op, a
            special op missing ``object_name`` / ``bind``, a param expression that
            reads an unbound / stale / non-existent field, or an unmet requirement.
    """
    if not isinstance(raw, list):
        raise SequenceError(f"sequence must be a list, got {type(raw).__name__}")
    active_special_ops = frozenset(special_ops)
    unknown_special = active_special_ops - KNOWN_SPECIAL_OPS
    if unknown_special:
        raise SequenceError(f"unknown special ops: {sorted(unknown_special)}")
    metas = _op_metas(allowed_ops)
    steps: list[ActionStep | LoopStep] = []
    bound: set[str] = set(_prebound or ())  # binding names available (seeded for loop bodies)
    fresh: set[str] = set(_prefresh if _prefresh is not None else bound)  # bindings still valid
    producer: dict[str, str] = dict(_preproducer or {})  # bind -> op that produced it
    track_state = initial_state is not None
    state: frozenset[str] = frozenset(initial_state or ())
    focus = _focus_scan(raw)
    clears_at = _clearing_steps(raw, focus, metas)
    blocked = dict(blocked_access or {})
    cleared: set[str] = set(_precleared or ())
    shut_again: set[str] = set()
    # Barriers the ENCLOSING plan does not clear until after this point. A loop body is
    # validated on its own, so without this a plan could put the reach inside a loop and
    # the barrier's opening outside it, and neither half would see the contradiction.
    pending_clear: set[str] = set(_not_yet_clear or ())
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SequenceError(f"step {i}: must be an object, got {type(item).__name__}")
        if "loop" in item:
            loop = _parse_loop(
                item["loop"],
                i,
                allowed_ops=allowed_ops,
                special_ops=active_special_ops,
                bound=bound,
                fresh=fresh,
                producer=producer,
                grounding=grounding,
                blocked_access=blocked,
                cleared=cleared,
                not_yet_clear=shut_again | {k for k, at in clears_at.items() if at > i},
            )
            steps.append(loop)
            continue
        op = item.get("op")
        if not isinstance(op, str) or not op:
            raise SequenceError(f"step {i}: missing/invalid 'op'")
        if op in KNOWN_SPECIAL_OPS:
            # Known special ops need explicit special_ops authorization — they
            # must not sneak in via allowed_ops (would skip their schema check).
            if op not in active_special_ops:
                raise SequenceError(
                    f"step {i}: special op {op!r} not authorized in special_ops"
                    f" (known special ops require explicit authorization)"
                )
        elif op not in allowed_ops:
            raise SequenceError(f"step {i}: unknown op {op!r} (not a known action)")
        params = item.get("params") or {}
        if not isinstance(params, dict):
            raise SequenceError(f"step {i}: 'params' must be an object")
        bind = item.get("bind")
        if bind is not None and (not isinstance(bind, str) or not bind.isidentifier()):
            raise SequenceError(f"step {i}: 'bind' must be a valid identifier, got {bind!r}")
        validator = _SPECIAL_OP_VALIDATORS.get(op) if op in active_special_ops else None
        if validator is not None:
            validator(params, bind, index=i)

        meta = metas.get(op)
        _check_param_names(i, op, params, meta)
        _check_qualifier(i, op, params, meta, grounding or {})
        _check_bindings(i, params, bound=bound, fresh=fresh, producer=producer, metas=metas)
        _check_location_need(i, op, params, meta, fresh=fresh, bound=bound,
                             producer=producer, metas=metas)
        if track_state and meta is not None:
            unmet = missing_requirements(state, meta.requires)
            if unmet:
                raise SequenceError(
                    f"step {i}: {op} requires {list(unmet)} but the state here is "
                    f"{sorted(state) or ['—']}" + "".join(_hint(_producers_of(metas, t)) for t in unmet)
                )
        subject, enclosure = focus[i]
        if _manipulates(meta):
            _check_access(i, op, subject, enclosure, cleared=cleared, clears_at=clears_at,
                          blocked=blocked, shut_again=shut_again, pending_clear=pending_clear,
                          metas=metas)

        steps.append(ActionStep(op=op, params=dict(params), bind=bind))
        # Effects: staleness first, then this step's own production, so a search-and-
        # sense action (drives, then senses) leaves its own reading fresh.
        if meta is not None and meta.invalidates_locations:
            fresh.clear()
        produces = meta is None or meta.produces_location
        if bind:
            bound.add(bind)
            producer[bind] = op
            if produces:
                # An unannotated op (a special op, or a bare-name vocabulary) that
                # binds is treated as producing — never invent staleness we can't prove.
                fresh.add(bind)
        elif meta is not None and meta.produces_location:
            fresh.add(_CACHE_SLOT)  # sensed, but left in the api cache rather than a bind
        if track_state and meta is not None:
            state = apply_effects(state, provides=meta.provides, invalidates=meta.invalidates)
        if subject:
            key = _referent_key(subject)
            if _clears_access(meta):
                cleared.add(key)
                shut_again = {s for s in shut_again if not _same_referent(key, s)}
                pending_clear = {s for s in pending_clear if not _same_referent(key, s)}
            elif meta is not None and meta.closes_access:
                cleared = {c for c in cleared if not _same_referent(key, c)}
                shut_again.add(key)
    return steps


def _check_param_names(index: int, op: str, params: Mapping[str, Any], meta: ToolMeta | None) -> None:
    """Reject a param the action does not declare, naming the ones it does.

    Without this, an invented param (``beside=`` on an action that only takes ``on=``)
    survives every other check and only fails at dispatch, as a ``TypeError`` that aborts
    the run. Caught here it becomes a compile-time message the planner can correct from —
    the same reason the op name and the bind fields are checked rather than trusted.
    """
    declared = _declared_params(meta)
    if declared is None:
        return
    unknown = sorted(set(params) - declared)
    if unknown:
        takes = f"it takes {sorted(declared)}" if declared else "it takes no params"
        raise SequenceError(f"step {index}: {op} got unknown param(s) {unknown}; {takes}")


def qualifiers_for(grounding: Mapping[str, Any] | None, name: Any) -> tuple[dict[str, str], ...]:
    """The task's spatial qualifiers for one object name, as a tuple.

    A name can carry MORE THAN ONE — "把香蕉旁的箱子放到紫桌上，把柜子里的箱子放到白桌上"
    is two boxes, both called box, told apart only by where each one is. Keying a single
    qualifier per name silently drops one of them and then rejects the step that handles
    it, so the value is a list. A bare dict is accepted too: one qualifier is the common
    case and callers (and tests) write it that way.
    """
    if not grounding or not isinstance(name, str):
        return ()
    raw = grounding.get(name)
    items = raw if isinstance(raw, (list, tuple)) else ([raw] if isinstance(raw, Mapping) else [])
    return tuple(
        {"reference": str(g["reference"]), "relation": str(g["relation"])}
        for g in items
        if isinstance(g, Mapping) and g.get("reference") and g.get("relation")
    )


def _check_qualifier(
    index: int, op: str, params: Mapping[str, Any], meta: ToolMeta | None,
    grounding: Mapping[str, Any],
) -> None:
    """Reject a step that drops or contradicts the spatial qualifier the task stated.

    When the task says "the apple IN the drawer" and the scene holds another apple on the
    table, a bare ``locate_for_grasp("apple")`` type-checks, dispatches, and grasps the wrong
    one — a success the run has no way to notice. The qualifier is not decoration; it is the
    only thing distinguishing two same-label objects, so an action able to carry it must.

    With several qualified instances of one name, a step is right if it names **any** of
    them: the task said which boxes exist, not which order to take them in. What stays
    rejected is the step that names none, or one the task never stated.

    Applies to any action whose contract declares both ``reference`` and ``relation``, so it
    follows the vocabulary rather than a hard-coded list of action names.
    """
    declared = _declared_params(meta)
    if declared is None or not {"reference", "relation"} <= declared:
        return
    obj = params.get("object_name")
    wanted = qualifiers_for(grounding, obj)
    if not wanted:
        return
    got_ref, got_rel = params.get("reference"), params.get("relation")
    if any(got_ref == w["reference"] and got_rel == w["relation"] for w in wanted):
        return
    options = " 或 ".join(f"reference={w['reference']!r}, relation={w['relation']!r}" for w in wanted)
    as_stated = " / ".join(f"{w['relation']} {w['reference']!r}" for w in wanted)
    stated = (f"but it passes reference={got_ref!r}, relation={got_rel!r}"
              if got_ref is not None or got_rel is not None else "but it passes neither")
    raise SequenceError(
        f"step {index}: the task locates {obj!r} as {as_stated}, {stated}. "
        f"There may be more than one {obj} in view and this is what tells them apart — "
        f"pass {options}."
    )


def _check_bindings(
    index: int,
    params: Mapping[str, Any],
    *,
    bound: set[str],
    fresh: set[str],
    producer: Mapping[str, str],
    metas: Mapping[str, ToolMeta],
) -> None:
    """Validate every ``<bind>.field`` a step reads: bound, still fresh, and real.

    Catches at compile time what would otherwise be a cryptic ``str + float`` crash
    deep inside a motion tool, or — worse — a silent move to coordinates measured
    from a standpoint the robot has since left.

    Staleness applies **only to sensed positions**. That is what ``invalidates_locations``
    is about — a coordinate measured from a standpoint the body has left. Everything else
    a step returns (a bearing, a pose reading, a joint vector) is plain data: treating it
    as permanently stale would mean it could never be read at all.
    """
    for key, val in params.items():
        referenced = referenced_binding_names(val)
        missing = referenced - bound
        if missing:
            raise SequenceError(
                f"step {index}: param {key!r}={val!r} reads unbound {sorted(missing)}; "
                f"an earlier step must `bind` it under that exact name "
                f"(bound so far: {sorted(bound) or ['—']})"
            )
        # An unannotated producer (special op / bare-name vocabulary) counts as sensing,
        # mirroring how ``parse_sequence`` marks its bind fresh — never invent leniency
        # for a step whose contract we cannot see.
        stale = set()
        for name in referenced - fresh:
            meta = metas.get(producer.get(name, ""))
            if meta is None or meta.produces_location:
                stale.add(name)
        if stale:
            raise SequenceError(
                f"step {index}: param {key!r}={val!r} reads {sorted(stale)}, sensed before the "
                f"body moved — those coordinates are stale. Re-sense after the move "
                f"(producing action: {[producer.get(b, '?') for b in sorted(stale)]})"
            )
        for root, fname in referenced_fields(val):
            src = producer.get(root)
            meta = metas.get(src) if src else None
            known = meta.result_fields() if meta is not None else frozenset()
            if known and fname not in known and fname not in _SYNTHETIC_BINDING_FIELDS:
                raise SequenceError(
                    f"step {index}: param {key!r}={val!r} reads {root}.{fname}, which {src} "
                    f"does not return; it returns {sorted(known | _SYNTHETIC_BINDING_FIELDS)}"
                )


def _check_location_need(
    index: int,
    op: str,
    params: Mapping[str, Any],
    meta: ToolMeta | None,
    *,
    fresh: set[str],
    bound: set[str],
    producer: Mapping[str, str],
    metas: Mapping[str, ToolMeta],
) -> None:
    """Reject an action that acts on a sensed position when none is current.

    Only for actions declaring ``consumes_location``. Two ways a step can name its
    source, and both must be freshness-checked:

    * ``<bind>.field`` — already checked by :func:`_check_bindings`.
    * a param whose whole value IS a bind name (``target="crate"``) — **not** checked
      there, because ``referenced_binding_names`` deliberately ignores a bare name so
      an ``object_name="box"`` is never mistaken for a binding. So it is checked here;
      otherwise "drive somewhere, then grasp the coordinate measured before the move"
      compiles cleanly and only fails on the robot.
    """
    if meta is None or not meta.consumes_location:
        return
    named = {b for v in params.values() for b in referenced_binding_names(v)}
    named |= {v for v in params.values() if isinstance(v, str) and v in bound}
    if named:
        stale = sorted(named - fresh)
        if stale:
            raise SequenceError(
                f"step {index}: {op} acts on {stale}, sensed before the body moved — those "
                f"coordinates are stale. Re-sense after the move "
                f"(producing action: {[producer.get(b, '?') for b in stale]})"
            )
        return
    if fresh:
        return
    raise SequenceError(
        f"step {index}: {op} acts on a sensed position but none is current here{_hint(_location_producers(metas))}"
    )


# --------------------------------------------------------------------------- #
# Safe expression evaluation
# --------------------------------------------------------------------------- #
_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARYOPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _attr_or_item(base: Any, name: str) -> Any:
    """Resolve ``base.name`` — dict key first (detection bindings are dicts)."""
    if isinstance(base, Mapping):
        if name in base:
            return base[name]
        raise ExprError(f"no field {name!r}")
    try:
        return getattr(base, name)
    except AttributeError as exc:  # noqa: TRY003
        raise ExprError(f"no attribute {name!r}") from exc


def _slice_index(node: ast.AST, env: Mapping[str, Any]) -> int:
    """Evaluate a subscript index to an int (handles py<3.9 ast.Index)."""
    inner = node.value if isinstance(node, ast.Index) else node  # type: ignore[attr-defined]  # py<3.9 ast.Index compat
    val = _eval_node(inner, env)
    return int(val)


def _eval_node(node: ast.AST, env: Mapping[str, Any]) -> Any:
    """Recursively evaluate a whitelisted AST node."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExprError(f"non-numeric constant {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left, env), _eval_node(node.right, env))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand, env))
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ExprError(f"unknown name {node.id!r}")
    if isinstance(node, ast.Attribute):
        return _attr_or_item(_eval_node(node.value, env), node.attr)
    if isinstance(node, ast.Subscript):
        return _eval_node(node.value, env)[_slice_index(node.slice, env)]
    raise ExprError(f"unsupported expression element: {type(node).__name__}")


def evaluate_expr(expr: str, env: Mapping[str, Any]) -> float:
    """Evaluate a symbolic param expression to a number under the safe grammar.

    Allowed: numeric literals, ``+ - * /``, unary ``+``/``-``, name lookup,
    ``var.field`` (dict key or attribute), ``var.field[idx]``. Nothing else —
    no function calls, no arbitrary Python.

    Args:
        expr: the expression text, e.g. ``"obj.z"`` or ``"obj.z + 30"``.
        env: variable environment (config constants + detection bindings).

    Returns:
        The numeric result as ``float``.

    Raises:
        ExprError: on a parse error, an unknown name, a non-numeric result, or
            any disallowed construct.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"cannot parse expression {expr!r}: {exc}") from exc
    try:
        val = _eval_node(tree, env)
    except (TypeError, KeyError, IndexError, AttributeError, ZeroDivisionError) as exc:
        raise ExprError(f"error evaluating {expr!r}: {exc}") from exc
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ExprError(f"expression {expr!r} did not evaluate to a number (got {type(val).__name__})")
    return float(val)


def resolve_value(value: Any, env: Mapping[str, Any]) -> Any:
    """Resolve one param value: evaluate numeric expressions, keep literals.

    A number passes through. A string is tried as an expression; if it does not
    parse/evaluate to a number (e.g. an ``object_name`` like ``"红杯子"``), the
    original string is returned as a literal.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return evaluate_expr(value, env)
        except ExprError:
            return value
    return value


def resolve_params(params: Mapping[str, Any], env: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve every param value against ``env`` (see ``resolve_value``)."""
    return {k: resolve_value(v, env) for k, v in params.items()}


# --------------------------------------------------------------------------- #
# Detection binding shape
# --------------------------------------------------------------------------- #
def normalize_detection(gi: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a detection result into a **task-agnostic** binding for the env.

    The binding is the raw perception dict passed through (so any field the
    detector emits — ``grasp_z``, ``place_z``, ``score``, ``depth_m``, … — is
    addressable), plus purely geometric conveniences ``x/y/z`` taken from
    ``position[0]/[1]/[2]``. NO task semantics are baked in: a pick skill's
    expression reads ``obj.grasp_z``, a place skill reads ``obj.place_z``, a
    carry/push skill reads ``obj.x`` / ``obj.position[0]`` — the choice lives in
    the skill's SKILL.md, not here.

    Args:
        gi: a detection dict with at least ``position`` (``[x, y, z]`` mm). Other
            fields are copied through verbatim.

    Returns:
        The binding dict (raw fields + ``x/y/z``).
    """
    pos = list(gi.get("position") or gi.get("grasp_position") or gi.get("center_mm") or [0.0, 0.0, 0.0])
    binding: dict[str, Any] = dict(gi)  # pass every raw field through
    binding["x"] = float(pos[0]) if len(pos) > 0 else 0.0
    binding["y"] = float(pos[1]) if len(pos) > 1 else 0.0
    binding["z"] = float(pos[2]) if len(pos) > 2 else 0.0
    return binding
