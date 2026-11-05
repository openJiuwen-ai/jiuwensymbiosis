"""Read-only action consumption shared by domain adapters and tool frontends.

The contract remains owned by its domain. A binding freezes the device identity
and the managed call; it neither describes a workflow nor owns device execution.
"""

from __future__ import annotations

import copy
import inspect
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol


class ActionContract(Protocol):
    """Minimal view a tool builder needs, projected from a domain declaration."""

    @property
    def name(self) -> str:
        """Return the domain action name."""
        ...

    @property
    def description(self) -> str:
        """Return the planner-facing description."""
        ...

    @property
    def input_schema(self) -> dict[str, Any]:
        """Return the domain-owned parameter schema."""
        ...

    @property
    def output_schema(self) -> dict[str, Any]:
        """Return the domain-owned result schema."""
        ...

    @property
    def required_capabilities(self) -> frozenset[str]:
        """Return the capabilities required to expose the action."""
        ...

    @property
    def planner_visible(self) -> bool:
        """Report whether the action is available for planning."""
        ...


@dataclass(frozen=True)
class ActionBinding:
    """Frozen physical binding plus an authorization-checked managed call.

    ``call`` receives parameters, an optional durable action request key, and the
    trusted caller identity. Identity comes from the host, never tool arguments.
    Domain entry points recheck policy against fresh state before dispatch.
    """

    contract: ActionContract
    device_id: str
    resource_keys: tuple[str, ...]
    domain: str
    call: Callable[[dict[str, Any], str | None, str], Any]
    authorize: Callable[[str], bool]

    def __post_init__(self) -> None:
        if not self.device_id or not self.domain or not self.contract.name:
            raise ValueError("action, domain and device identity are required")
        if not isinstance(self.resource_keys, tuple) or any(not key for key in self.resource_keys):
            raise ValueError("resource_keys must be a frozen tuple of non-empty identities")

    def invoke(self, params: dict[str, Any], *, request_key: str | None = None, principal: str = "local") -> Any:
        """Recheck permission on every call, including tools cached before revocation."""
        if not isinstance(principal, str) or not principal or not self.authorize(principal):
            raise PermissionError(f"caller is not authorized for {self.contract.name}")
        if not isinstance(params, dict):
            raise TypeError("action parameters must be an object")
        if request_key is not None and (not isinstance(request_key, str) or not request_key.strip()):
            raise ValueError("request_key must be non-empty")
        # Robot signatures historically accept tuples and numpy values, so their
        # existing Python/rail validation remains the compatibility boundary.
        # Instrument declarations validate full JSON at their domain entry point.
        return self.call(copy.deepcopy(params), request_key, principal)


def select_bindings(
    bindings: Iterable[ActionBinding],
    capabilities: Iterable[str],
    *,
    principal: str = "local",
    planner_only: bool = False,
    allow: set[str] | frozenset[str] | None = None,
    deny: set[str] | frozenset[str] | None = None,
) -> dict[str, ActionBinding]:
    """One gated catalog for stepagent, fastagent and protocol projections."""
    available = frozenset(capabilities)
    selected: dict[str, ActionBinding] = {}
    for binding in bindings:
        contract = binding.contract
        if not contract.required_capabilities <= available:
            continue
        if planner_only and not contract.planner_visible:
            continue
        if allow is not None and contract.name not in allow:
            continue
        if deny is not None and contract.name in deny:
            continue
        if not binding.authorize(principal):
            continue
        if contract.name in selected:
            raise ValueError(f"duplicate action name: {contract.name}")
        selected[contract.name] = binding
    return selected


def build_action_tools(
    bindings: Iterable[ActionBinding],
    capabilities: Iterable[str],
    *,
    principal: str = "local",
    planner_only: bool = False,
    allow: set[str] | frozenset[str] | None = None,
    deny: set[str] | frozenset[str] | None = None,
    request_key_factory: Callable[[], str] | None = None,
) -> list[Any]:
    """Project the admitted catalog into openjiuwen tools without domain policy."""
    from jiuwensymbiosis.agent.abstractions import LocalFunction, ToolCard

    selected = select_bindings(
        bindings, capabilities, principal=principal, planner_only=planner_only, allow=allow, deny=deny
    )
    tools = []
    next_request_key = request_key_factory or (lambda: uuid.uuid4().hex)
    for binding in selected.values():

        def make_call(bound: ActionBinding) -> Callable[..., Any]:
            def invoke(**params: Any) -> Any:
                return bound.invoke(params, request_key=next_request_key(), principal=principal)

            wrapped: Callable[..., Any] = invoke
            # LocalFunction detects async callables to await them correctly.
            if inspect.iscoroutinefunction(bound.call):

                async def invoke_async(**params: Any) -> Any:
                    return await bound.invoke(params, request_key=next_request_key(), principal=principal)

                wrapped = invoke_async
            # Existing robot memory/rails inspect the carrier on LocalFunction's
            # callable. Preserve the original ToolMeta, never a copied contract.
            if bound.domain == "robot":
                meta = getattr(bound.contract, "meta", None)
                if meta is not None:
                    wrapped.__dict__["__tool_meta__"] = meta
            return wrapped

        tools.append(
            LocalFunction(
                card=ToolCard(
                    name=binding.contract.name,
                    description=binding.contract.description,
                    input_params=binding.contract.input_schema,
                ),
                func=make_call(binding),
            )
        )
    return tools
