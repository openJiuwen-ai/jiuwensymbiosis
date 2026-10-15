# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Explicit, disjoint discovery for hardware adapters and simulation backends."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["SessionTarget", "session_target", "backend_module_name", "adapter_module_name"]

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class SessionTarget:
    """One selected session provider; its selector determines the import namespace."""

    selector: Literal["adapter", "backend"]
    name: str

    def __post_init__(self) -> None:
        """Validate selectors before any module lookup or import can occur."""
        if self.selector not in {"adapter", "backend"}:
            raise ValueError("session selector must be adapter or backend")
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError(f"configuration must declare a valid top-level {self.selector} name")

    @property
    def module_name(self) -> str:
        """Resolve only the namespace selected by the configuration, without importing it."""
        namespace = "adapters" if self.selector == "adapter" else "simulators"
        return f"jiuwensymbiosis.{namespace}.{self.name}"

    def as_config(self) -> dict[str, str]:
        """Preserve hardware/simulation identity in config snapshots and runtime events."""
        return {self.selector: self.name}

    def bind_config(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Supply a missing selector, but reject a conflicting GUI body/config selection."""
        if "adapter" in data or "backend" in data:
            if session_target(data) != self:
                raise ValueError(f"配置中的 {self.selector} 与当前选择的本体不一致，请重新选择本体和配置。")
        return {**data, **self.as_config()}


def session_target(data: Mapping[str, Any]) -> SessionTarget:
    """Require exactly one explicit selector; never infer a namespace from a name."""
    if ("adapter" in data) == ("backend" in data):
        raise ValueError("configuration must declare exactly one of adapter (hardware) or backend (simulation)")
    selector: Literal["adapter", "backend"] = "backend" if "backend" in data else "adapter"
    value = data[selector]
    if not isinstance(value, str):
        raise ValueError(f"configuration must declare a valid top-level {selector} name")
    return SessionTarget(selector, value.strip())


def backend_module_name(name: str) -> str:
    """Resolve a simulation backend exclusively under simulators/."""
    return SessionTarget("backend", name).module_name


def adapter_module_name(name: str) -> str:
    """Resolve a hardware adapter exclusively under adapters/."""
    return SessionTarget("adapter", name).module_name
