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
    param like ``"box.grasp_z"`` or ``"box.grasp_z + 30"`` resolves against the
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
numeric expression (e.g. ``"黑盒子"``) is left as the literal string.
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
        SequenceError: on a non-list payload, a malformed step, an unknown op, or
            a detection step missing ``object_name`` / a bad ``role``.
    """
    if not isinstance(raw, list):
        raise SequenceError(f"sequence must be a list, got {type(raw).__name__}")
    steps: list[ActionStep] = []
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
        steps.append(ActionStep(op=op, params=dict(params), bind=bind))
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
        expr: the expression text, e.g. ``"pick.grasp_z"`` or ``"pick.grasp_z + 30"``.
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
    parse/evaluate to a number (e.g. an ``object_name`` like ``"黑盒子"``), the
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
    expression reads ``box.grasp_z``, a place skill reads ``box.place_z``, a
    carry/push skill reads ``box.x`` / ``box.position[0]`` — the choice lives in
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
