# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Walk a `BaseRobotApi` instance, find @robot_tool methods, build openjiuwen Tools.

Capability gating: a tool whose `capability` is set is emitted only if the
api advertises it. This lets us share Mixins across robots that implement
only a subset (e.g. a mixin defining 6 vision methods, but a robot has only
detection + pixel projection).
"""

from __future__ import annotations

from typing import Any, Optional

from jiuwensymbiosis.agent.abstractions import LocalFunction, ToolCard


def build_robot_tools(
    api: Any,
    *,
    allow: Optional[set[str]] = None,
    deny: Optional[set[str]] = None,
) -> list[Any]:
    """Return a list of `openjiuwen.LocalFunction` Tools bound to the api.

    Args:
        api: An instance of a class that mixes ``BaseRobotApi`` with capability
            mixins. Must have ``capabilities`` (frozenset[str]).
        allow: If given, only tool *names* in this set are emitted.
        deny: If given, tool names in this set are skipped (applied after ``allow``).

    Returns:
        A list of openjiuwen ``Tool`` instances (specifically ``LocalFunction``).

    Raises:
        ImportError: if openjiuwen is not installed.
    """
    api_caps = getattr(api, "capabilities", None)
    if api_caps is None:
        # Fall back to method-level discovery only.
        api_caps = frozenset()

    tools: list[Any] = []
    seen: set[str] = set()
    # Walk MRO so subclass overrides are preferred but base-class decorators are still picked up.
    for cls in type(api).__mro__:
        for attr_name, attr_value in cls.__dict__.items():
            if attr_name in seen:
                continue
            if not callable(attr_value):
                continue
            meta = getattr(attr_value, "__robot_tool__", None)
            if meta is None:
                continue
            seen.add(attr_name)

            if allow is not None and meta.name not in allow:
                continue
            if deny is not None and meta.name in deny:
                continue
            if meta.capability and meta.capability not in api_caps:
                continue

            bound = getattr(api, attr_name)  # bound method on the api instance
            card = ToolCard(
                name=meta.name,
                description=meta.description,
                input_params=meta.input_params,
            )
            tools.append(LocalFunction(card=card, func=bound))

    return tools


def list_tool_meta(api: Any) -> list[dict]:
    """Diagnostics: enumerate the tools `build_robot_tools` would emit, without
    actually instantiating openjiuwen objects. Useful in tests and for logging.
    """
    api_caps = getattr(api, "capabilities", frozenset())
    out: list[dict] = []
    seen: set[str] = set()
    for cls in type(api).__mro__:
        for attr_name, attr_value in cls.__dict__.items():
            if attr_name in seen or not callable(attr_value):
                continue
            meta = getattr(attr_value, "__robot_tool__", None)
            if meta is None:
                continue
            seen.add(attr_name)
            if meta.capability and meta.capability not in api_caps:
                continue
            out.append(
                {
                    "name": meta.name,
                    "description": meta.description,
                    "capability": meta.capability,
                    "tags": list(meta.tags),
                    "input_params": meta.input_params,
                }
            )
    return out
