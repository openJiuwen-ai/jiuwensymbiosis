"""Offline selection of exactly one instrument access implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from jiuwensymbiosis.adapters._common.config import load_yaml_config, normalize_config_source


def _access(data: dict[str, Any]) -> str:
    if data.get("adapter", "instrument") != "instrument" or "adapters" in data:
        raise ValueError("instrument requires one adapter; combination configuration is not supported")
    if any(key in data for key in ("instruments", "instrument_permissions", "instrument_mcp")):
        raise ValueError("legacy instrument fields must be migrated to instrument.access")
    values = data.get("instrument")
    if not isinstance(values, dict):
        raise ValueError("instrument must be a mapping")
    access = values.get("access")
    if not isinstance(access, str) or access not in {"direct", "mcp"}:
        raise ValueError("instrument.access must explicitly select direct or mcp")
    if access == "direct":
        if set(values) - {"access", "installation", "permissions"} or "installation" not in values:
            raise ValueError("direct access requires installation and accepts only direct fields")
    elif any(key in values for key in ("installation", "permissions")):
        raise ValueError("mcp access does not accept direct installation or permissions")
    if access == "mcp" and "physical_device_id" in data:
        raise ValueError("remote physical_device_id belongs under instrument, not the local root")
    return access


def _factory(access: str) -> Any:
    # Only the selected branch participates in parsing and construction.
    if access == "direct":
        from jiuwensymbiosis.adapters.instrument.direct.adapter_config import DirectConfig

        return DirectConfig
    from jiuwensymbiosis.adapters.instrument.mcp.config import InstrumentMCPConfig

    return InstrumentMCPConfig


@dataclass(frozen=True)
class InstrumentConfig:
    access: str
    selected: Any

    @classmethod
    def normalize_source(cls, data: dict[str, Any], source_dir: str | Path | None) -> dict[str, Any]:
        factory = _factory(_access(data))
        normalize = getattr(factory, "normalize_source", None)
        if callable(normalize):
            return cast(dict[str, Any], normalize(data, source_dir))
        return normalize_config_source(factory, data, source_dir)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstrumentConfig:
        access = _access(data)
        return cls(access, _factory(access).from_dict(data))

    @classmethod
    def from_yaml(cls, path: str | Path) -> InstrumentConfig:
        return load_yaml_config(cls, path)
