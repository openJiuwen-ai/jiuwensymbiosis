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
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

# The compound real-time op the runner implements (not a raw @robot_tool action).
TRACK_DETECT = "track_detect"


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


def parse_sequence(raw: Any, *, allowed_ops: Mapping[str, Any] | frozenset | set) -> list[ActionStep]:
    """Validate a raw action sequence (the LLM output) into ``list[ActionStep]``.

    Args:
        raw: the LLM-produced sequence — must be a ``list`` of ``dict`` steps.
        allowed_ops: the set/collection of action names the runner can execute
            (the api action vocabulary). ``track_detect`` is always allowed in
            addition. Membership is tested with ``in``.

    Returns:
        Validated steps, in order.

    Raises:
        SequenceError: on a non-list payload, a malformed step, an unknown op, a
            ``track_detect`` missing ``object_name`` / ``bind``, or a param
            expression that reads a binding no earlier step produced.
    """
    if not isinstance(raw, list):
        raise SequenceError(f"sequence must be a list, got {type(raw).__name__}")
    steps: list[ActionStep] = []
    bound: set[str] = set()  # binding names produced by earlier steps' `bind`
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SequenceError(f"step {i}: must be an object, got {type(item).__name__}")
        op = item.get("op")
        if not isinstance(op, str) or not op:
            raise SequenceError(f"step {i}: missing/invalid 'op'")
        if op != TRACK_DETECT and op not in allowed_ops:
            raise SequenceError(f"step {i}: unknown op {op!r} (not a known action)")
        params = item.get("params") or {}
        if not isinstance(params, dict):
            raise SequenceError(f"step {i}: 'params' must be an object")
        bind = item.get("bind")
        if bind is not None and (not isinstance(bind, str) or not bind.isidentifier()):
            raise SequenceError(f"step {i}: 'bind' must be a valid identifier, got {bind!r}")
        if op == TRACK_DETECT:
            obj = params.get("object_name")
            if not isinstance(obj, str) or not obj:
                raise SequenceError(f"step {i}: track_detect requires params.object_name")
            if not bind:
                raise SequenceError(
                    f"step {i}: track_detect must have a 'bind' name — its detection is read "
                    f"later as <bind>.field; without it the reference resolves to nothing"
                )
        # Every binding a param reads via <name>.field / <name>[i] must already
        # be produced by an earlier step's `bind`. Catches the LLM naming its
        # bind one thing and referencing another (e.g. object_name 'red cup'
        # but no matching bind for a later 'red_cup.position[0]') at compile
        # time, instead of a cryptic 'str < float' crash deep in a motion tool.
        for key, val in params.items():
            missing = referenced_binding_names(val) - bound
            if missing:
                raise SequenceError(
                    f"step {i}: param {key!r}={val!r} reads unbound {sorted(missing)}; "
                    f"an earlier track_detect must `bind` it under that exact name "
                    f"(bound so far: {sorted(bound) or ['—']})"
                )
        steps.append(ActionStep(op=op, params=dict(params), bind=bind))
        if bind:
            bound.add(bind)
    return steps


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
    pos = list(gi.get("position") or gi.get("grasp_position") or [0.0, 0.0, 0.0])
    binding: dict[str, Any] = dict(gi)  # pass every raw field through
    binding["x"] = float(pos[0]) if len(pos) > 0 else 0.0
    binding["y"] = float(pos[1]) if len(pos) > 1 else 0.0
    binding["z"] = float(pos[2]) if len(pos) > 2 else 0.0
    return binding
