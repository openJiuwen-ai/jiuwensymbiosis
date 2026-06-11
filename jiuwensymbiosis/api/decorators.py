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
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union, get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)


@dataclass
class ToolMeta:
    """Metadata attached to @robot_tool-decorated methods."""

    name: str
    description: str
    input_params: dict[str, Any]
    capability: Optional[str] = None
    tags: list[str] = field(default_factory=list)


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

    if origin is Union:
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


def robot_tool(
    _func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    desc: Optional[str] = None,
    capability: Optional[str] = None,
    input_params: Optional[dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
):
    """Decorate an api method to make it discoverable by ``build_robot_tools``.

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
    """

    def _wrap(f: Callable) -> Callable:
        """Attach ``ToolMeta`` metadata to the decorated function."""
        first_doc_line = (f.__doc__ or "").strip().split("\n", 1)[0]
        f.__robot_tool__ = ToolMeta(
            name=name or f.__name__,
            description=desc or first_doc_line or f.__name__,
            input_params=input_params or _schema_from_signature(f),
            capability=capability,
            tags=list(tags) if tags else [],
        )
        return f

    if _func is not None and callable(_func):
        return _wrap(_func)
    return _wrap
