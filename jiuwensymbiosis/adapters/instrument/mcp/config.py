"""Deployment-owned MCP endpoint and reviewed remote execution semantics."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jiuwensymbiosis.adapters._common.config import load_yaml_config

_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


def input_schema_digest(schema: dict[str, Any]) -> str:
    """Hash the complete discovered schema for a deployer's review and pin.

    This helper does not approve a tool. The deployer reviews ranges, units and
    authority before putting this digest in deployment configuration.
    """
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RemoteSemantics:
    """Reviewed server contract, never inferred from tool descriptions/annotations.

    Paths address structuredContent only. Without this declaration a successful
    tools/call proves a protocol response, not completion of physical work.
    """

    request_key_parameter: str | None = None
    request_key_meta: str | None = None
    handle_field: str | None = None
    status_field: str | None = None
    query_tool: str | None = None
    stop_tool: str | None = None
    handle_parameter: str = "job_id"
    success_values: tuple[str, ...] = ("succeeded",)
    failure_values: tuple[str, ...] = ("failed", "cancelled")
    completion_scope: str = "response"
    evidence_field: str | None = None
    evidence_equals: Mapping[str, Any] | tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.request_key_parameter and self.request_key_meta:
            raise ValueError("choose either a remote request-key parameter or metadata field")
        if self.completion_scope not in {"response", "physical"}:
            raise ValueError("remote completion_scope must be response or physical")
        expected = dict(self.evidence_equals)
        if any(
            not isinstance(key, str) or not key or type(value) not in (str, bool, int, float)
            for key, value in expected.items()
        ):
            raise ValueError("evidence_equals must map nonempty fields to reviewed scalar values")
        json.dumps(expected, allow_nan=False)
        object.__setattr__(self, "evidence_equals", tuple(sorted(expected.items())))
        if self.completion_scope == "physical" and (not self.evidence_field or not expected):
            raise ValueError("physical completion requires a reviewed evidence field and evidence_equals predicate")
        if self.handle_field and (not self.status_field or not self.query_tool):
            raise ValueError("remote handles require status_field and query_tool")
        if self.stop_tool and not self.handle_field:
            raise ValueError("stop_tool requires an authoritative remote handle")
        for name in ("success_values", "failure_values"):
            values = getattr(self, name)
            if not values or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain nonempty status values")
            object.__setattr__(self, name, tuple(values))
        if set(self.success_values) & set(self.failure_values):
            raise ValueError("remote success and failure statuses must be disjoint")


@dataclass(frozen=True)
class RemoteToolPolicy:
    name: str
    read_only: bool = False
    principals: tuple[str, ...] = ("local",)
    semantics: RemoteSemantics | None = None
    input_schema_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", self.name):
            raise ValueError("invalid remote tool name")
        if not isinstance(self.read_only, bool):
            raise ValueError("read_only must be explicitly boolean")
        if not self.read_only and self.input_schema_sha256 is None:
            raise ValueError("write tools require a deployer-reviewed input_schema_sha256")
        if self.input_schema_sha256 is not None and not re.fullmatch(r"[a-f0-9]{64}", self.input_schema_sha256):
            raise ValueError("input_schema_sha256 must be a lowercase SHA-256 digest")
        if not self.principals or any(not isinstance(value, str) or not value for value in self.principals):
            raise ValueError("tool principals must be a nonempty allowlist")
        object.__setattr__(self, "principals", tuple(self.principals))


@dataclass(frozen=True)
class InstrumentMCPConfig:
    physical_device_id: str
    authority: str
    transport: str
    tools: tuple[RemoteToolPolicy, ...]
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    headers_env: tuple[tuple[str, str], ...] = ()
    timeout_s: float = 30.0

    @classmethod
    def normalize_source(cls, data: dict[str, Any], source_dir: str | Path | None) -> dict[str, Any]:
        value = deepcopy(data)
        command = value.get("instrument", {}).get("command")
        if source_dir is not None and isinstance(command, str) and command.strip():
            # Keep executable symlinks: a venv's Python loses its environment
            # when dereferenced to the base interpreter.
            expanded = os.path.expandvars(os.path.expanduser(command))
            value["instrument"]["command"] = os.path.abspath(os.path.join(source_dir, expanded))
        return value

    def __post_init__(self) -> None:
        for name in ("physical_device_id", "authority"):
            if not isinstance(getattr(self, name), str) or not _IDENTITY.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be an explicit stable identity, not a URL")
        if not self.tools or len({item.name for item in self.tools}) != len(self.tools):
            raise ValueError("tools must be a nonempty unique deployment allowlist")
        if self.transport == "stdio":
            if not self.command or not Path(self.command).is_absolute() or self.url:
                raise ValueError("stdio requires an absolute deployer-selected command and no URL")
        elif self.transport == "streamable_http":
            address = urlsplit(self.url or "")
            invalid_origin = address.scheme not in {"http", "https"} or not address.hostname
            if invalid_origin or address.username or address.password:
                raise ValueError("streamable_http requires an HTTP(S) URL without credentials")
            if self.command or self.args:
                raise ValueError("HTTP transport does not accept subprocess commands")
        else:
            raise ValueError("supported MCP transports are stdio and streamable_http")
        if isinstance(self.timeout_s, bool) or not 0 < self.timeout_s <= 600:
            raise ValueError("timeout_s must be within (0, 600]")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstrumentMCPConfig:
        values = dict(data.get("instrument", {}))
        values.pop("access", None)
        if not values:
            raise ValueError("configuration requires instrument mapping")
        if data.get("physical_device_id") is not None:
            raise ValueError("remote physical_device_id belongs under instrument; the remote owner holds its lease")
        tools = []
        for raw in values.pop("tools", []):
            item = dict(raw)
            if item.get("semantics") is not None:
                item["semantics"] = RemoteSemantics(**item["semantics"])
            tools.append(RemoteToolPolicy(**item))
        values["tools"] = tuple(tools)
        values["args"] = tuple(values.get("args", ()))
        values["headers_env"] = tuple(dict(values.get("headers_env", {})).items())
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> InstrumentMCPConfig:
        return load_yaml_config(cls, path)


def resource_keys(cfg: InstrumentMCPConfig) -> tuple[str, ...]:
    """Lease local call records; only the remote owner leases physical hardware.

    Acquiring physical:<id> on a same-host client before contacting its owner
    would deadlock admission. This delegated identity is independent of URL.
    """
    return (f"remote-owner:{cfg.authority}:physical:{cfg.physical_device_id}",)
