# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""@robot_tool decorator + JSON-Schema generation from type hints.

`@robot_tool` annotates an *unbound* method on an api class. When
`build_robot_tools(api)` later walks the api instance, it picks up these
methods, generates a `ToolCard.input_params` from the function signature,
and binds the now-bound method into a `LocalFunction`. Compared to writing
ToolCards by hand, this saves ~70% boilerplate and keeps schema and
signature in sync.

Type → JSON-Schema mapping is intentionally minimal; if you need richer
schemas (enums, regex, nested objects), pass `input_params=` explicitly
to `@robot_tool`.

Beyond the call schema, a tool carries its **planning contract** — pre-conditions
and effects, never an order, so a planner can *derive* a legal order instead of
following one written down in prose:

* `returns` — what fields the result exposes, so a later step can read
  `<bind>.field`. Derived from the return annotation when it is a `TypedDict`
  (or a union of them); pass `returns=` for anything else.
* `requires` / `provides` / `invalidates` — robot self-state, over the closed
  vocabulary in `jiuwensymbiosis.api.state`.
* `produces_location` / `consumes_location` / `invalidates_locations` — location
  freshness, scoped to the step's `bind` (see `api.state`). An action that senses
  where something is *produces*; one that moves the base *invalidates* every prior
  location, since they were measured from the old standpoint.

  `consumes_location` is only for actions that read a location **implicitly**,
  off a cache, without naming it in their params. A step that passes
  `<bind>.field` already declares its dependency through that reference, and the
  sequence validator checks its freshness from the reference alone — annotating
  it would wrongly force a location on the same action when it is called with
  plain literal coordinates.
"""

from __future__ import annotations

import inspect
import logging
import types
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin, get_type_hints

from jiuwensymbiosis.api.state import validate_tokens

logger = logging.getLogger(__name__)


@dataclass
class ToolMeta:
    """Metadata attached to @robot_tool-decorated methods.

    The planning contract is everything past ``tags`` (see the module docstring).
    An empty ``returns`` means "output shape unknown" — consumers must degrade to
    no field checking rather than reject.
    """

    name: str
    description: str
    input_params: dict[str, Any]
    # Implementation detail true of THIS body only (an ``ActionSpec`` implementation may
    # supply one). Kept separate from ``description`` so a reader — human or coding agent
    # — can tell which half stays true when the plan moves to another robot.
    capability: str | None = None
    tags: list[str] = field(default_factory=list)
    returns: dict[str, Any] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    produces_location: bool = False
    consumes_location: bool = False
    invalidates_locations: bool = False
    # Access, scoped to the thing this action acts on (see api/state.py). Only the
    # barrier side is declared: which steps reach THROUGH a barrier is derivable from
    # the sequence, so no grasp/place action ever annotates this.
    opens_access: bool = False
    closes_access: bool = False
    # Whether the action enters the planner's vocabulary. False for bring-up /
    # calibration / diagnostic actions, which stay dispatchable but must not compete
    # for the planner's attention. Body-specific tools (plain ``@robot_tool``, no
    # ``ActionSpec``) default to False: only the shared vocabulary is plannable.
    planner_visible: bool = True

    def result_fields(self) -> frozenset[str]:
        """Field names ``returns`` advertises; empty when the shape is unknown."""
        props = self.returns.get("properties")
        return frozenset(props) if isinstance(props, dict) else frozenset()

    def full_description(self) -> str:
        """What a planner reads. Every word of it comes from the ActionSpec: a body has no
        channel for adding prose of its own, because a fact that changes a plan and is true
        of only one robot means the action is not the same action there."""
        return self.description


_BASIC_TYPES = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _annotation_to_schema(ann: Any) -> dict[str, Any]:
    """Best-effort conversion of a Python type annotation to a JSON Schema fragment."""
    if ann is inspect.Parameter.empty or ann is Any:
        return {}
    if ann in _BASIC_TYPES:
        return {"type": _BASIC_TYPES[ann]}

    origin = get_origin(ann)
    args = get_args(ann)

    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0])
        return {"oneOf": [_annotation_to_schema(a) for a in non_none]}

    if origin in (list, tuple):
        if args:
            return {"type": "array", "items": _annotation_to_schema(args[0])}
        return {"type": "array"}

    if origin is dict:
        return {"type": "object"}

    return {}


def _resolve_hints(func: Callable) -> dict[str, Any]:
    """Resolve ``func``'s annotations to real type objects.

    Handles ``from __future__ import annotations`` (which makes annotations
    string literals at runtime) by calling ``typing.get_type_hints`` with
    progressively wider namespaces. Falls back to whatever
    ``func.__annotations__`` already holds (may be strings) if everything
    else fails — the schema for those params will degrade to ``{}``.
    """
    # First try the standard call — works for fully-qualified annotations.
    try:
        return get_type_hints(func)
    except Exception as e:
        logger.debug("get_type_hints(%s) failed: %s; trying fallback.", func.__name__, e)
    # Try with the function's own module globals + builtins (helps for
    # locally defined functions whose forward refs reference globals).
    try:
        mod = inspect.getmodule(func)
        ns = getattr(mod, "__dict__", {}) if mod is not None else {}
        return get_type_hints(func, globalns=ns)
    except Exception as e:
        logger.debug(
            "get_type_hints(%s, globalns=...) failed: %s; falling back to raw __annotations__.",
            func.__name__,
            e,
        )
    # Last resort: raw __annotations__ (may still be strings).
    return getattr(func, "__annotations__", {}) or {}


def schema_from_signature(func: Callable) -> dict[str, Any]:
    """Public alias of :func:`_schema_from_signature` for the ``ActionSpec`` layer."""
    return _schema_from_signature(func)


def schema_from_typeddict(annotation: Any) -> dict[str, Any]:
    """JSON Schema for a ``TypedDict`` (or a union of them); ``{}`` when the shape is unknown.

    ``{}`` means "output shape unknown", which consumers must treat as "skip field
    checking" rather than "declares no fields".
    """
    if annotation is None:
        return {}
    members = (
        [a for a in get_args(annotation) if a is not type(None)]
        if get_origin(annotation) in (typing.Union, types.UnionType)
        else [annotation]
    )
    schemas = [_typeddict_schema(m) for m in members if typing.is_typeddict(m)]
    if not schemas:
        return {}
    return schemas[0] if len(schemas) == 1 else _merge_schemas(schemas)


def _merge_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    """Union of several result shapes: properties merge, ``required`` intersects.

    A field is only *guaranteed* when every branch declares it — the success and the
    failure shape of one action rarely agree on more than ``ok``.
    """
    properties: dict[str, Any] = {}
    for s in schemas:
        properties.update(s.get("properties", {}))
    merged: dict[str, Any] = {"type": "object", "properties": properties}
    required = set(schemas[0].get("required", []))
    for s in schemas[1:]:
        required &= set(s.get("required", []))
    if required:
        merged["required"] = sorted(required)
    return merged


def _schema_from_signature(func: Callable) -> dict[str, Any]:
    """Build a JSON-Schema-style ``input_params`` from a function signature.

    Returns ``{"type":"object","properties":{...},"required":[...]}``.
    """
    sig = inspect.signature(func)
    hints = _resolve_hints(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = hints.get(name, param.annotation)
        prop = _annotation_to_schema(ann)
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)
        # Fall back to string only when we truly couldn't infer anything;
        # never overwrite a populated schema (e.g. {"default": None}).
        if not prop:
            prop = {"type": "string"}
        properties[name] = prop
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _typeddict_schema(td: Any) -> dict[str, Any]:
    """JSON-Schema fragment for one ``TypedDict`` class."""
    hints = get_type_hints(td)
    properties = {key: _annotation_to_schema(ann) or {"type": "string"} for key, ann in hints.items()}
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    required = sorted(getattr(td, "__required_keys__", frozenset()))
    if required:
        schema["required"] = required
    return schema


def _schema_from_return_annotation(func: Callable) -> dict[str, Any]:
    """Derive a ``returns`` schema from the return annotation, or ``{}`` if not inferable.

    Only ``TypedDict`` (and unions of them, e.g. ``GraspResult | GraspFailure``) carry
    field names — a bare ``dict`` says nothing useful, so it yields ``{}`` ("unknown"),
    which consumers treat as "skip field checking" rather than "no fields".

    For a union the properties are merged and ``required`` is the intersection: a field
    is only guaranteed when every branch declares it.
    """
    return schema_from_typeddict(_resolve_hints(func).get("return"))


def robot_tool(
    _func: Callable | None = None,
    *,
    name: str | None = None,
    desc: str | None = None,
    capability: str | None = None,
    input_params: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    returns: dict[str, Any] | None = None,
    requires: Iterable[str] = (),
    provides: Iterable[str] = (),
    invalidates: Iterable[str] = (),
    produces_location: bool = False,
    consumes_location: bool = False,
    invalidates_locations: bool = False,
    opens_access: bool = False,
    closes_access: bool = False,
    planner_visible: bool = False,
):
    """Decorate a **body-specific** api method to make it discoverable by ``build_robot_tools``.

    For an action in the shared vocabulary use ``@implements(SPEC)``
    (``jiuwensymbiosis.api.actions``) instead — it takes the contract from the one
    place that owns it, so two bodies cannot drift apart on what the action means.
    This decorator is for tools only one body has: bring-up, calibration, vendor
    demos. Hence ``planner_visible`` defaults to **False** — a plan written against
    a tool that exists on a single robot would not survive being moved.

    Args:
        name: Override tool name (default: function name).
        desc: Override description (default: docstring's first line).
        capability: A capability string this tool requires. If set, the tool
            is only emitted when the api advertises it. Useful when one mixin
            covers several capabilities and a single method is conditional.
        input_params: Override the auto-generated JSON-Schema. Use when the
            signature is too poor to infer (e.g. ``**kwargs``).
        tags: Free-form tags; rails may use them to gate behavior
            (e.g. ``["motion"]`` triggers visual feedback).
        returns: Override the auto-derived output schema. Needed whenever the
            return annotation is not a ``TypedDict``, which is most methods.
        requires: Robot self-state tokens that must hold before this action runs.
        provides: Robot self-state tokens this action establishes.
        invalidates: Robot self-state tokens this action makes stale.
        produces_location: This action senses where something is; the step's
            ``bind`` becomes a fresh location later steps may consume.
        consumes_location: This action acts on a previously sensed position, so a
            fresh location must exist when it runs.
        invalidates_locations: This action moves the body, making every location
            sensed from the old standpoint stale.
        opens_access: This action clears whatever was blocking the thing it acts on —
            pulls a door, lifts a lid, unlocks, moves an obstacle off. What was
            unreachable becomes reachable.
        closes_access: The reverse: it puts the barrier back.
        planner_visible: opt a body-specific tool INTO the planner vocabulary.
            Rarely correct — prefer promoting the action to an ``ActionSpec``.

    Raises:
        UnknownStateToken: if any token is outside ``api.state.KNOWN_STATE_TOKENS``.
    """

    def _wrap(f: Callable) -> Callable:
        """Attach ``ToolMeta`` metadata to the decorated function."""
        first_doc_line = (f.__doc__ or "").strip().split("\n", 1)[0]
        tool_name = name or f.__name__
        # decorator attaches attr at runtime; mypy can't model fn.__dict__
        f.__robot_tool__ = ToolMeta(  # type: ignore[attr-defined]
            name=tool_name,
            description=desc or first_doc_line or f.__name__,
            input_params=input_params or _schema_from_signature(f),
            capability=capability,
            tags=list(tags) if tags else [],
            returns=returns if returns is not None else _schema_from_return_annotation(f),
            requires=validate_tokens(requires, field="requires", owner=tool_name),
            provides=validate_tokens(provides, field="provides", owner=tool_name),
            invalidates=validate_tokens(invalidates, field="invalidates", owner=tool_name),
            produces_location=produces_location,
            consumes_location=consumes_location,
            invalidates_locations=invalidates_locations,
            opens_access=opens_access,
            closes_access=closes_access,
            planner_visible=planner_visible,
        )
        return f

    if _func is not None and callable(_func):
        return _wrap(_func)
    return _wrap
