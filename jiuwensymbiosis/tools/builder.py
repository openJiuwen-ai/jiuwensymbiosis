# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Walk a `BaseRobotApi` instance, find its actions, build openjiuwen Tools.

Capability gating: a tool is emitted only if its capability is in the gate set.
That capability comes from the action's own contract (``ToolMeta.capability``,
which an ``ActionSpec`` supplies), never from whichever class happens to declare
the method — resolving it by MRO used to gate every tool an adapter declared
alongside its vision tools. The gate set is ``api.capabilities & env.capabilities``
when ``env`` is given, else ``api.capabilities``.
"""

from __future__ import annotations

import copy
import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jiuwensymbiosis.api.decorators import ToolMeta
from jiuwensymbiosis.tools.bindings import ActionBinding, build_action_tools
from jiuwensymbiosis.tools.robot_control_tool import record_action


def _recording(api: Any, bound: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a bound action so its outcome reaches the api's ``ExecutionMemory``.

    ``RobotControlTool`` records at its own dispatch point; this covers the
    separate-tool strategy, so world state is tracked identically whichever
    strategy an agent was built with. The wrapper preserves ``__tool_meta__``
    (via ``functools.wraps``) because the tool builder and rails read it back off
    the callable.
    """

    if inspect.iscoroutinefunction(bound):

        @functools.wraps(bound)
        async def run_async(*args: Any, **kwargs: Any) -> Any:
            result = await bound(*args, **kwargs)
            record_action(api, bound, kwargs, result)
            return result

        return run_async

    @functools.wraps(bound)
    def run(*args: Any, **kwargs: Any) -> Any:
        result = bound(*args, **kwargs)
        record_action(api, bound, kwargs, result)
        return result

    return run


def _effective_capabilities(api: Any, env: Any) -> frozenset[str]:
    """Capabilities a tool may be gated against: api ∩ env (or api alone)."""
    api_caps = getattr(api, "capabilities", None) or frozenset()
    if env is None:
        return frozenset(api_caps)
    # ``effective_capabilities`` when the env offers it: it adds what the body SHIPS
    # (a URDF → planning.reachability) on top of what it declares.
    env_caps = getattr(env, "effective_capabilities", None) or getattr(env, "capabilities", None) or frozenset()
    return frozenset(api_caps) & frozenset(env_caps)


@dataclass(frozen=True)
class RobotContractView:
    """Projection of the existing ActionSpec/ToolMeta; no copied action truth."""

    meta: ToolMeta

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def description(self) -> str:
        return self.meta.full_description()

    @property
    def input_schema(self) -> dict[str, Any]:
        return copy.deepcopy(self.meta.input_params)

    @property
    def output_schema(self) -> dict[str, Any]:
        return copy.deepcopy(self.meta.returns)

    @property
    def required_capabilities(self) -> frozenset[str]:
        return frozenset({self.meta.capability}) if self.meta.capability else frozenset()

    @property
    def planner_visible(self) -> bool:
        return self.meta.planner_visible


def robot_action_bindings(
    api: Any,
    *,
    env: Any = None,
    device_id: str = "robot",
    resource_keys: tuple[str, ...] = (),
) -> list[ActionBinding]:
    """Adapt existing decorated methods to the common tool-consumption interface.

    Legacy standalone builders have no Runtime resource identity; an admitted
    Runtime session supplies its frozen physical identity and resource keys.
    """
    bindings: list[ActionBinding] = []
    seen: set[str] = set()
    for cls in type(api).__mro__:
        for attr_name, attr_value in cls.__dict__.items():
            if attr_name in seen or not callable(attr_value):
                continue
            meta = getattr(attr_value, "__tool_meta__", None)
            if not isinstance(meta, ToolMeta):
                continue
            seen.add(attr_name)
            recorded = _recording(api, getattr(api, attr_name))

            def bind_call(fn: Callable[..., Any]) -> Callable[..., Any]:
                if inspect.iscoroutinefunction(fn):

                    async def call_async(params: dict, _key: str | None, _principal: str) -> Any:
                        return await fn(**params)

                    return call_async

                def call(params: dict, _key: str | None, _principal: str) -> Any:
                    return fn(**params)

                return call

            def authorized(_principal: str, *, required: str | None = meta.capability) -> bool:
                return not required or required in _effective_capabilities(api, env)

            bindings.append(
                ActionBinding(
                    RobotContractView(meta), device_id, resource_keys, "robot", bind_call(recorded), authorized
                )
            )
    return bindings


def build_robot_tools(
    api: Any,
    *,
    env: Any = None,
    allow: set[str] | None = None,
    deny: set[str] | None = None,
    planner_only: bool = False,
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
        planner_only: Emit only the shared-vocabulary actions a planner may use
            (``ActionSpec`` + ``planner_visible``). Default False keeps every
            dispatchable tool; the agent builder passes True for the LLM's tool list.

    Returns:
        A list of openjiuwen ``Tool`` instances (specifically ``LocalFunction``).

    Raises:
        ImportError: if openjiuwen is not installed.
    """
    return build_action_tools(
        robot_action_bindings(api, env=env),
        _effective_capabilities(api, env),
        allow=allow,
        deny=deny,
        planner_only=planner_only,
    )


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
            meta = getattr(attr_value, "__tool_meta__", None)
            if meta is None:
                continue
            seen.add(attr_name)
            if meta.capability and meta.capability not in effective_caps:
                continue
            out.append(
                {
                    "name": meta.name,
                    "description": meta.full_description(),
                    "capability": meta.capability,
                    "tags": list(meta.tags),
                    "input_params": meta.input_params,
                }
            )
    return out
