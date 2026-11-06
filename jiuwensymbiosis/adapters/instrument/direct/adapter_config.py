"""Instrument deployment assembly, sharing the adapter/config convention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jiuwensymbiosis.adapters._common.config import load_yaml_config
from jiuwensymbiosis.adapters.instrument.direct.config import action_ids, validate_config
from jiuwensymbiosis.adapters.instrument.direct.vocabulary import CONDITIONS


@dataclass
class DirectConfig:
    installation: dict[str, Any]
    permissions: dict[str, list[str]]
    name: str = "instruments"

    path_fields = ("port",)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DirectConfig:
        values = data["instrument"]
        installation = validate_config(values["installation"])
        physical = []
        names = set()
        from jiuwensymbiosis.adapters.instrument.common.naming import tool_name

        for profile in installation["devices"]:
            if not profile.get("physical_device_id"):
                raise ValueError("Runtime requires an explicit physical_device_id for every instrument")
            physical.append(profile["physical_device_id"])
            if set(profile["required_confirmations"]) - CONDITIONS:
                raise ValueError("unregistered instrument precondition")
            for action in action_ids(profile):
                name = tool_name(profile["device_id"], action)
                if name in names:
                    raise ValueError("instrument tool name collision")
                names.add(name)
        if len(physical) != len(set(physical)):
            raise ValueError("one physical instrument must have one profile")
        permissions = values.get("permissions", {})
        if not isinstance(permissions, dict):
            raise ValueError("instrument_permissions must map trusted principals to tool names")
        for principal, allowed in permissions.items():
            if not isinstance(principal, str) or not principal.strip() or not isinstance(allowed, list):
                raise ValueError("invalid instrument permission entry")
            if any(not isinstance(name, str) or name not in names for name in allowed):
                raise ValueError("permissions may grant only declared device-qualified tools")
        return cls(
            installation, {key: list(value) for key, value in permissions.items()}, data.get("name", "instruments")
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> DirectConfig:
        return load_yaml_config(cls, path)
