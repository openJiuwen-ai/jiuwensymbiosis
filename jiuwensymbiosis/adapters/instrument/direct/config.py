# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Derived from the instrument integration; licensed under Apache-2.0.

"""Strict installation configuration; validation never opens a hardware channel."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, model_validator

from .contracts import ConditionId, Identifier, PositiveNumber, StrictModel, ensure_json_value


class NumericLimit(StrictModel):
    min: float
    max: float
    step: PositiveNumber
    default: int | float | None = None
    unit: str | None = None
    basis: Literal["software_policy", "hardware_verified", "protocol_encoding"] | None = None
    evidence: str | None = None

    @model_validator(mode="after")
    def ordered_limits(self) -> NumericLimit:
        if self.max < self.min:
            raise ValueError("limit max cannot be less than min")
        if self.default is not None:
            if not self.min <= self.default <= self.max:
                raise ValueError("limit default must be within min and max")
            steps = (self.default - self.min) / self.step
            if not math.isclose(steps, round(steps), abs_tol=1e-7):
                raise ValueError("limit default must respect step")
        return self


class ActionReadiness(StrictModel):
    configured: bool = False
    evidence: str = ""

    @model_validator(mode="after")
    def evidence_for_readiness(self) -> ActionReadiness:
        if self.configured and not self.evidence.strip():
            raise ValueError("configured action_readiness entries require evidence")
        return self


def action_ids(profile: dict[str, Any]) -> tuple[str, ...]:
    return tuple(profile.get("operation_ids") or [profile["operation_id"]])


def action_ready(profile: dict[str, Any], action: str) -> bool:
    if action not in action_ids(profile):
        return False
    readiness = profile.get("action_readiness", {}).get(action)
    return readiness["configured"] if readiness is not None else profile.get("configured", False)


class DeviceProfile(StrictModel):
    device_id: Identifier
    physical_device_id: Identifier | None = None
    adapter: Identifier
    operation_id: Identifier
    operation_ids: Annotated[list[Identifier], Field(min_length=1)] | None = None
    model: Identifier
    connection: dict[str, JsonValue] = Field(default_factory=dict)
    protocol: dict[str, JsonValue] = Field(default_factory=dict)
    limits: dict[Identifier, NumericLimit] = Field(default_factory=dict)
    slots: Annotated[list[JsonValue], Field(max_length=0)] = Field(default_factory=list)
    liquid_paths: dict[str, JsonValue] = Field(default_factory=dict)
    configured: bool = False
    configuration_evidence: str = ""
    action_readiness: dict[Identifier, ActionReadiness] = Field(default_factory=dict)
    required_confirmations: list[ConditionId] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_profile(self) -> DeviceProfile:
        self.operation_ids = list(self.operation_ids) if self.operation_ids is not None else [self.operation_id]
        actions = set(self.operation_ids)
        if len(actions) != len(self.operation_ids) or self.operation_id not in actions:
            raise ValueError("operation_ids must be unique and include operation_id")
        if self.action_readiness.keys() - actions:
            raise ValueError("action_readiness can reference only declared operation_ids")
        if self.liquid_paths:
            raise ValueError("generic profiles do not implement physical object or material ownership")
        if len(set(self.required_confirmations)) != len(self.required_confirmations):
            raise ValueError("required_confirmations must be unique")
        for action in self.operation_ids:
            readiness = self.action_readiness.get(action)
            configured = readiness.configured if readiness is not None else self.configured
            evidence = readiness.evidence if readiness is not None else self.configuration_evidence
            if configured and not evidence.strip():
                raise ValueError(f"configured profiles require configuration_evidence for {action}")
        return self


class InstrumentConfig(StrictModel):
    config_version: Literal["1"]
    devices: Annotated[list[DeviceProfile], Field(min_length=1)]
    object_types: Annotated[list[JsonValue], Field(max_length=0)] = Field(default_factory=list)
    extensions: list[dict[str, JsonValue]] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_devices(self) -> InstrumentConfig:
        identifiers = [profile.device_id for profile in self.devices]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate device_id in configuration")
        return self


def validate_config(config: dict[str, Any], *, providers: Mapping | None = None) -> dict[str, Any]:
    """Normalize only data and call explicitly supplied provider validation hooks."""
    ensure_json_value(config)
    result = InstrumentConfig.model_validate(config).model_dump(mode="json")
    from jiuwensymbiosis.adapters.instrument.direct.providers import default_providers

    from .action_registry import ActionRegistry

    registry = ActionRegistry(result["extensions"])
    selected = default_providers() if providers is None else providers
    for profile in result["devices"]:
        unknown = set(action_ids(profile)) - registry.operations.keys()
        if unknown:
            raise ValueError(f"Unregistered actions: {sorted(unknown)}")
        provider = selected.get(profile["adapter"])
        if provider is None:
            raise ValueError(f"No explicitly installed provider for {profile['adapter']!r}")
        for name, limit in profile["limits"].items():
            schemas = [
                registry.definitions[action]["parameters_schema"].get("properties", {}).get(name)
                for action in action_ids(profile)
            ]
            schemas = [schema for schema in schemas if schema is not None]
            if not schemas or any(schema["type"] not in {"integer", "number"} for schema in schemas):
                raise ValueError(f"limit {name} requires a matching numeric action parameter")
            for schema in schemas:
                if max(limit["min"], schema.get("minimum", -math.inf)) > min(
                    limit["max"], schema.get("maximum", math.inf)
                ):
                    raise ValueError(f"limit {name} has no overlap with its documented parameter range")
                if schema["type"] == "integer" and limit["default"] is not None and type(limit["default"]) is not int:
                    raise ValueError(f"integer parameter {name} requires an integer default")
        provider.validate_profile(profile, registry)
    return result


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON configuration key: {key}")
        result[key] = value
    return result


def load_config(path: str | Path, *, providers: Mapping | None = None) -> dict[str, Any]:
    return validate_config(
        json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys), providers=providers
    )
