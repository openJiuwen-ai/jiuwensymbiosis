# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Derived from the instrument integration; licensed under Apache-2.0.

"""Explicit adapter providers: pure validation and identity, lazy connection creation."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from .mapped import MappedProtocolAdapter
from .mapping_rules import validate_profile as validate_mapping_profile


class AdapterProvider(Protocol):
    def validate_profile(self, profile: dict, registry: Any) -> None:
        """Reject unsupported configuration without opening connections."""
        ...

    def binding(self, profile: dict, registry: Any) -> dict:
        """Return a stable physical/configuration identity for durable admission."""
        ...

    def create_adapter(self, profile: dict, registry: Any) -> Any:
        """Construct an adapter; the InstrumentSession owns and closes the returned object."""
        ...


class MappedAdapterProvider:
    @staticmethod
    def validate_profile(profile: dict, registry: Any) -> None:
        actions = profile.get("operation_ids") or [profile["operation_id"]]
        if not set(actions) <= registry.definitions.keys():
            raise ValueError("mapped_protocol requires registered extension actions")
        transports = {registry.definitions[name]["mapping"]["transport"] for name in actions}
        if len(transports) != 1 or not transports <= {"http_json", "serial_line"}:
            raise ValueError("one mapped device must use one supported transport")
        validate_mapping_profile(profile, registry)
        connection = profile["connection"]
        transport = next(iter(transports))
        allowed = {"port", "baudrate", "timeout_s"} if transport == "serial_line" else {"base_url", "timeout_s"}
        if set(connection) - allowed:
            raise ValueError("unknown mapped transport connection fields")
        timeout = connection.get("timeout_s", 3.0)
        if type(timeout) not in (int, float) or not 0.01 <= timeout <= 60:
            raise ValueError("mapped timeout_s must be between 0.01 and 60 seconds")
        if transport == "serial_line":
            if not isinstance(connection.get("port"), str) or not connection["port"].strip():
                raise ValueError("mapped serial device requires an explicit port")
            if type(connection.get("baudrate")) is not int or not 1 <= connection["baudrate"] <= 4000000:
                raise ValueError("mapped serial device requires a documented baudrate")
        else:
            value = connection.get("base_url")
            try:
                parsed = urlsplit(value) if isinstance(value, str) else None
                valid_port = parsed is not None and (parsed.port is None or 1 <= parsed.port <= 65535)
            except ValueError as exc:
                raise ValueError("mapped HTTP device requires a valid HTTP(S) origin and port") from exc
            invalid_origin = "mapped HTTP device requires a credential-free HTTP(S) origin"
            if parsed is None or not valid_port:
                raise ValueError(invalid_origin)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None:
                raise ValueError(invalid_origin)
            if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
                raise ValueError(invalid_origin)
            if any(character.isspace() or ord(character) < 32 for character in value):
                raise ValueError(invalid_origin)

    def binding(self, profile: dict, registry: Any) -> dict:
        self.validate_profile(profile, registry)
        actions = profile.get("operation_ids") or [profile["operation_id"]]
        mappings = {name: registry.definitions[name] for name in sorted(actions)}
        connection = deepcopy(profile["connection"])
        if "base_url" in connection:
            parsed = urlsplit(connection["base_url"])
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            host = parsed.hostname.casefold().rstrip(".")
            try:
                address = ipaddress.ip_address(host)
                host = "loopback" if address.is_loopback else str(address)
            except ValueError:
                if host == "localhost":
                    host = "loopback"
            endpoint = {"transport": "tcp", "host": host, "port": port}
        else:
            port = connection["port"]
            plain_port = port.removeprefix("\\\\.\\")
            normalized = (
                plain_port.upper()
                if re.fullmatch(r"COM[0-9]+", plain_port, re.IGNORECASE)
                else str(Path(port).expanduser().resolve())
            )
            endpoint = {"transport": "serial", "port": normalized}
        return {
            "adapter": "mapped_protocol",
            "endpoint": endpoint,
            "connection": connection,
            "protocol": deepcopy(profile.get("protocol", {})),
            "mapping_digest": hashlib.sha256(
                json.dumps(mappings, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
        }

    def create_adapter(self, profile: dict, registry: Any) -> MappedProtocolAdapter:
        self.validate_profile(profile, registry)
        return MappedProtocolAdapter(profile, registry)


def default_providers() -> dict[str, AdapterProvider]:
    return {"mapped_protocol": MappedAdapterProvider()}
