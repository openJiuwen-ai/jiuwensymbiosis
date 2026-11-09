"""Instrument policy and adapter lifecycle inside the existing Runtime owner."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

from jiuwensymbiosis.adapters.instrument.common.naming import tool_name
from jiuwensymbiosis.adapters.instrument.common.vocabulary import CAPABILITIES
from jiuwensymbiosis.adapters.instrument.direct.action_registry import ActionRegistry, InstrumentOperation
from jiuwensymbiosis.adapters.instrument.direct.config import action_ids, action_ready, validate_config
from jiuwensymbiosis.adapters.instrument.direct.contracts import ensure_json_value
from jiuwensymbiosis.adapters.instrument.direct.policy import ExecutionPolicy
from jiuwensymbiosis.adapters.instrument.direct.vocabulary import CONDITIONS, validate_state
from jiuwensymbiosis.agent.lifecycle import CleanupReport, HardwareCleanupError
from jiuwensymbiosis.tools.bindings import ActionBinding

if TYPE_CHECKING:
    from jiuwensymbiosis.runtime.invocations import InvocationContext, InvocationOutcome


@dataclass(frozen=True)
class InstrumentContractView:
    """Projection of one declaration, not a second writable action contract."""

    operation: InstrumentOperation
    name: str

    @property
    def description(self) -> str:
        return self.operation.description

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.operation.request_schema

    @property
    def output_schema(self) -> dict[str, Any]:
        return self.operation.output_schema

    @property
    def required_capabilities(self) -> frozenset[str]:
        return frozenset({"instrument.observe" if self.operation.definition["read_only"] else "instrument.control"})

    @property
    def planner_visible(self) -> bool:
        return True


class InstrumentSession:
    """Own connections and domain checks; Runtime owns jobs, leases and intents.

    Construction and discovery are offline. Invocations require a Runtime
    context; there is no unmanaged fallback that can silently open hardware.
    """

    capabilities = CAPABILITIES

    def __init__(
        self,
        installation: dict[str, Any],
        *,
        permissions: Mapping[str, list[str]],
        name: str = "instruments",
        providers: Mapping | None = None,
        policy: ExecutionPolicy | None = None,
    ) -> None:
        from jiuwensymbiosis.adapters.instrument.direct.providers import default_providers

        self.name = name
        self._providers = dict(default_providers() if providers is None else providers)
        self._config = validate_config(installation, providers=self._providers)
        self._registry = ActionRegistry(self._config["extensions"])
        self._profiles = {p["device_id"]: p for p in self._config["devices"]}
        self._permissions = {principal: frozenset(names) for principal, names in permissions.items()}
        self._policy = policy or ExecutionPolicy()
        self._adapters: dict[str, Any] = {}
        self._context: InvocationContext | None = None
        self._uncertain: set[str] = set()
        self._cleanup_errors: tuple[str, ...] = ()
        self._observations: dict[str, dict[str, Any]] = {}
        self._connected = False
        self.cancel_token = None
        physical = [p.get("physical_device_id") for p in self._profiles.values()]
        if any(not item for item in physical) or len(set(physical)) != len(physical):
            raise ValueError("each instrument needs a unique explicit physical_device_id")
        for profile in self._profiles.values():
            if set(profile["required_confirmations"]) - CONDITIONS:
                raise ValueError("unregistered instrument precondition")

    def bind_execution(self, context: InvocationContext) -> None:
        if self._connected or self._context is not None:
            raise RuntimeError("execution binding is frozen before connection")
        self._context = context

    @property
    def execution_request_prefix(self) -> str:
        if self._context is None:
            raise RuntimeError("request identity requires Runtime admission")
        return self._context.job_id

    def action_bindings(self) -> tuple[ActionBinding, ...]:
        bindings = []
        for device_id, profile in self._profiles.items():
            for action in action_ids(profile):
                if not action_ready(profile, action):
                    continue
                name = tool_name(device_id, action)
                bindings.append(
                    ActionBinding(
                        contract=InstrumentContractView(self._registry.operations[action], name),
                        device_id=profile["physical_device_id"],
                        resource_keys=(f"physical:{profile['physical_device_id']}",),
                        domain="instrument",
                        call=lambda params, key, principal, d=device_id, a=action: self._invoke(
                            d, a, params, key, principal
                        ),
                        authorize=lambda principal, n=name: n in self._permissions.get(principal, ()),
                    )
                )
        return tuple(bindings)

    def connect(self) -> None:
        if self._connected:
            return
        if self._context is None:
            raise RuntimeError("InstrumentSession must be bound to Runtime before connect")
        if self._adapters:
            raise RuntimeError("prior connection cleanup is incomplete")
        try:
            for device_id, profile in self._profiles.items():
                self._adapters[device_id] = self._providers[profile["adapter"]].create_adapter(profile, self._registry)
        except Exception as exc:
            self.disconnect()
            if not self.cleanup_report().released:
                raise HardwareCleanupError("instrument connection", exc) from exc
            raise
        self._connected = True

    def disconnect(self) -> None:
        errors = []
        for device_id, adapter in tuple(self._adapters.items()):
            try:
                adapter.close()
            except Exception as exc:
                errors.append(f"{device_id}: {type(exc).__name__}: {exc}")
            else:
                self._adapters.pop(device_id)
        self._connected = bool(self._adapters)
        self._cleanup_errors = tuple(errors)

    def cleanup_report(self) -> CleanupReport:
        return CleanupReport(
            pending_work=tuple(sorted(self._uncertain)),
            connected=self._connected,
            errors=self._cleanup_errors,
        )

    def observations(self) -> dict[str, dict[str, Any]]:
        """Read the last sourced, device-scoped observations without hardware I/O."""
        return deepcopy(self._observations)

    def _validate(self, device_id: str, action: str, parameters: dict[str, Any], principal: str) -> dict[str, Any]:
        name = tool_name(device_id, action)
        if name not in self._permissions.get(principal, ()):
            raise PermissionError("caller is not authorized for this instrument action")
        if not self._connected or self._context is None:
            raise RuntimeError("instrument action requires a connected Runtime session")
        if principal != self._context.principal:
            raise PermissionError("caller cannot replace the Runtime principal")
        profile = self._profiles[device_id]
        if not action_ready(profile, action):
            raise ValueError("instrument action is not reviewed/configured")
        self._registry.validate_parameters(action, parameters)
        for name, value in parameters.items():
            limit = profile["limits"].get(name)
            if limit is None:
                continue
            if type(value) not in (int, float) or not limit["min"] <= value <= limit["max"]:
                raise ValueError(f"{name} exceeds reviewed deployment limits")
            steps = (value - limit["min"]) / limit["step"]
            if not math.isclose(steps, round(steps), abs_tol=1e-7):
                raise ValueError(f"{name} violates documented step/unit encoding")
        observation = {}
        if action not in self._registry.read_only:
            observation = self._adapters[device_id].snapshot()
            ensure_json_value(observation)
            state = validate_state(observation.get("device_state", "unknown"))
            if (
                observation.get("status_available") is not True
                or state not in self._registry.operations[action].required_conditions
            ):
                raise ValueError("fresh device observation does not admit this action")
            self._observations[profile["physical_device_id"]] = {
                **deepcopy(observation),
                "physical_device_id": profile["physical_device_id"],
            }
            for condition in profile["required_confirmations"]:
                if observation.get(condition) is not True:
                    raise ValueError(f"instrument precondition has no current evidence: {condition}")
        context = {
            "profile": deepcopy(profile),
            "parameters": deepcopy(parameters),
            "observation": deepcopy(observation),
        }
        self._policy.check_request(context)
        self._policy.before_dispatch(context)
        return context

    def _invoke(
        self, device_id: str, action: str, parameters: dict[str, Any], request_key: str | None, principal: str
    ) -> Any:
        from jiuwensymbiosis.runtime.invocations import InvocationOutcome

        if self._context is None or not request_key:
            raise ValueError("managed instrument actions require Runtime and a stable request key")
        profile = self._profiles[device_id]
        adapter = self._adapters.get(device_id)
        policy_context = {}
        verified_response: dict[str, Any] | None = None

        def validate() -> None:
            policy_context.update(self._validate(device_id, action, parameters, principal))

        def outcome(update: dict[str, Any]) -> InvocationOutcome:
            nonlocal verified_response
            ensure_json_value(update)
            evidence = update.get("evidence", [])
            verified = update.get("executed") is True and any(
                item.get("field") == "protocol_response_verified"
                and item.get("value") is True
                and item.get("source") == "hardware"
                for item in evidence
            )
            accepted = self._policy.completion(deepcopy(policy_context), deepcopy(update), verified)
            status = update.get("status")
            data = deepcopy(update.get("data", {}))
            data.pop("completion_scope", None)
            data.pop("physical_completion_verified", None)
            self._observations[profile["physical_device_id"]] = {
                "physical_device_id": profile["physical_device_id"],
                "data": data,
                "evidence": deepcopy(evidence),
                "physical_completion_verified": False,
            }
            if status == "succeeded" and verified and accepted:
                Draft202012Validator(self._registry.operations[action].output_schema).validate(data)
                completion = {"observations": evidence, "physical_completion_verified": False}
                if action not in self._registry.read_only:
                    # An ACK can precede continuing device work. Retain the
                    # Runtime owner until the reviewed status protocol reports
                    # idle; this is not a physical completion assertion.
                    verified_response = deepcopy(update)
                    observation = adapter.snapshot()
                    ensure_json_value(observation)
                    state = validate_state(observation.get("device_state", "unknown"))
                    completion["post_write_observation"] = observation
                    if observation.get("status_available") is not True or state not in {"idle", "busy"}:
                        return InvocationOutcome("unknown", data, "response", completion)
                    if state == "busy":
                        return InvocationOutcome("pending", data, "response", completion)
                self._uncertain.discard(request_key)
                return InvocationOutcome(
                    "succeeded",
                    result=data,
                    scope="response",
                    completion_evidence=completion,
                )
            if status in {"running", "accepted"}:
                return InvocationOutcome(
                    "pending", result=data, scope="response", completion_evidence={"observations": evidence}
                )
            return InvocationOutcome(
                "unknown", result=data, scope="response", completion_evidence={"observations": evidence}
            )

        def dispatch(invocation_id: str, stable_key: str) -> InvocationOutcome:
            from jiuwensymbiosis.adapters.instrument.direct.errors import CommandNotSent

            self._uncertain.add(request_key)
            try:
                update = adapter.start(action, parameters)
            except CommandNotSent as exc:
                self._uncertain.discard(request_key)
                return InvocationOutcome(
                    "failed",
                    scope="response",
                    completion_evidence={"dispatch_started": False, "reason_code": exc.code},
                )
            return outcome(update)

        def poll(record: dict[str, Any]) -> InvocationOutcome:
            return outcome(verified_response if verified_response is not None else adapter.poll())

        def stop(record: dict[str, Any]) -> None:
            adapter.cancel()

        return self._context.invoke(
            resource_key=f"physical:{profile['physical_device_id']}",
            action_name=action,
            contract_version=self._registry.operations[action].version,
            parameters=parameters,
            request_key=request_key,
            validate=validate,
            dispatch=dispatch,
            poll=poll,
            stop=stop if adapter is not None and adapter.can_cancel(action) else None,
            device_binding={
                "physical_device_id": profile["physical_device_id"],
                "declaration_digest": hashlib.sha256(
                    json.dumps(self._registry.operations[action].definition, sort_keys=True).encode()
                ).hexdigest(),
                "profile_digest": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
            },
        )
