# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Walk a `BaseRobotApi` instance, find @robot_tool methods, build openjiuwen Tools.

Capability gating: a tool is emitted only if its owning capability is in the
gate set. The owning capability is the tool's explicit ``meta.capability``, else
the ``capability`` of the mixin that declares the method. The gate set is
``api.capabilities & env.capabilities`` when ``env`` is given, else
``api.capabilities``.
"""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.agent.abstractions import LocalFunction, ToolCard
from jiuwensymbiosis.api.decorators import ToolMeta


def _effective_capabilities(api: Any, env: Any) -> frozenset[str]:
    """Capabilities a tool may be gated against: api ∩ env (or api alone)."""
    api_caps = getattr(api, "capabilities", None) or frozenset()
    if env is None:
        return frozenset(api_caps)
    env_caps = getattr(env, "capabilities", None) or frozenset()
    return frozenset(api_caps) & frozenset(env_caps)


def _owning_capability(api_type: type, attr_name: str, meta: ToolMeta) -> str | None:
    """Resolve the capability a tool belongs to.

    Explicit ``meta.capability`` wins; otherwise find the mixin in the MRO that
    declares ``attr_name`` and carries a ``capability`` class attribute. Returns
    None for body-specific tools owned by no capability mixin (never gated).
    """
    if meta.capability:
        return meta.capability
    for cls in api_type.__mro__:
        cap = cls.__dict__.get("capability")
        if isinstance(cap, str) and attr_name in cls.__dict__:
            return cap
    return None


def build_robot_tools(
    api: Any,
    *,
    env: Any = None,
    allow: set[str] | None = None,
    deny: set[str] | None = None,
) -> list[Any]:
    """Return a list of `openjiuwen.LocalFunction` Tools bound to the api.

    Args:
        api: An instance of a class that mixes ``BaseRobotApi`` with capability
            mixins. Must have ``capabilities`` (frozenset[str]).
        env: Optional ``BaseRobotEnv``. When given, tools are gated by
            ``api.capabilities & env.capabilities`` so the hardware's actual
            capabilities are respected. When None, only ``api.capabilities``.
        allow: If given, only tool *names* in this set are emitted.
        deny: If given, tool names in this set are skipped (applied after ``allow``).

    Returns:
        A list of openjiuwen ``Tool`` instances (specifically ``LocalFunction``).

    Raises:
        ImportError: if openjiuwen is not installed.
    """
    effective_caps = _effective_capabilities(api, env)
    api_type = type(api)

    tools: list[Any] = []
    seen: set[str] = set()
    # Walk MRO so subclass overrides are preferred but base-class decorators are still picked up.
    for cls in api_type.__mro__:
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
            owning_cap = _owning_capability(api_type, attr_name, meta)
            if owning_cap and owning_cap not in effective_caps:
                continue

            bound = getattr(api, attr_name)  # bound method on the api instance
            card = ToolCard(
                name=meta.name,
                description=meta.description,
                input_params=meta.input_params,
            )
            tools.append(LocalFunction(card=card, func=bound))

    return tools


def list_tool_meta(api: Any, *, env: Any = None) -> list[dict]:
    """Diagnostics: enumerate the tools `build_robot_tools` would emit, without
    actually instantiating openjiuwen objects. Useful in tests and for logging.
    """
    effective_caps = _effective_capabilities(api, env)
    api_type = type(api)
    out: list[dict] = []
    seen: set[str] = set()
    for cls in api_type.__mro__:
        for attr_name, attr_value in cls.__dict__.items():
            if attr_name in seen or not callable(attr_value):
                continue
            meta = getattr(attr_value, "__robot_tool__", None)
            if meta is None:
                continue
            seen.add(attr_name)
            owning_cap = _owning_capability(api_type, attr_name, meta)
            if owning_cap and owning_cap not in effective_caps:
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
