# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Resource identities shared by adapter session factories.

Adapter sessions contribute the resource that identifies their command endpoint
(CAN interface, serial device, ROS command topic, ...).  This module adds the
optional camera and detector-sidecar resources every adapter can share, plus an
explicit physical-device identity when the adapter cannot derive one.

The keys are local admission identities, not configuration summaries.  In
particular, detector URLs are shared dependencies unless this process is asked
to spawn the detector itself.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "common_resource_keys",
    "complete_resource_keys",
    "physical_device_resource_key",
]


def _read(value: Any, name: str, default: Any = None) -> Any:
    """Read one config field from either a dataclass-like object or mapping."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _validate_detector_host(host: Any) -> None:
    """Validate a detector bind host without treating it as a remote service."""
    value = str(host or "").strip().lower()
    if value.startswith("[") or value.endswith("]"):
        if not (value.startswith("[") and value.endswith("]")):
            raise ValueError(f"invalid detector bind host: {host!r}")
        value = value[1:-1]
    value = value.rstrip(".")
    if not value:
        raise ValueError("detector host must be configured when spawn=true")
    try:
        ipaddress.ip_address(value)
    except ValueError as exc:
        # The host is a local bind address (even when it names a LAN interface),
        # so do not resolve DNS here. Accept ordinary hostnames and reject values
        # that cannot plausibly be passed to a socket bind call.
        labels = value.split(".")
        if value != "localhost" and any(
            not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label) for label in labels
        ):
            raise ValueError(f"invalid detector bind host: {host!r}") from exc


def common_resource_keys(cfg: Any, *, include_sidecars: bool = True) -> tuple[str, ...]:
    """Return camera and locally spawned detector resources declared by ``cfg``.

    ``camera_serial`` is an exclusive physical camera identity. A remote detector
    is a shared service, so it gets no lock. When the config starts a detector,
    its local bind port is exclusive. Host is validated but omitted from the
    key: binding to a LAN address still conflicts with a wildcard bind on the
    same machine and port.
    """
    keys: list[str] = []
    serial = _read(cfg, "camera_serial")
    if serial is not None and str(serial).strip():
        keys.append(f"camera:{str(serial).strip()}")

    detector = _read(cfg, "detector")
    if include_sidecars and detector is not None and bool(_read(detector, "spawn", False)):
        _validate_detector_host(_read(detector, "host"))
        raw_port = _read(detector, "port")
        if isinstance(raw_port, bool) or not isinstance(raw_port, (int, str)):
            raise ValueError("detector port must be an integer between 1 and 65535")
        try:
            if isinstance(raw_port, str) and not raw_port.strip().isdigit():
                raise ValueError("port is not an integer")
            port = int(raw_port)
        except (TypeError, ValueError) as exc:
            raise ValueError("detector port must be an integer between 1 and 65535") from exc
        if not 1 <= port <= 65535:
            raise ValueError("detector port must be an integer between 1 and 65535")
        keys.append(f"detector:local:{port}")

    return tuple(keys)


def physical_device_resource_key(physical_device_id: Any) -> str | None:
    """Make a supplemental lock key for an explicit top-level device identity."""
    if physical_device_id is None:
        return None
    if not isinstance(physical_device_id, str) or not physical_device_id.strip():
        raise ValueError("physical_device_id must be a non-empty string when supplied")
    return f"physical:{physical_device_id.strip()}"


def complete_resource_keys(
    device_keys: Iterable[str],
    cfg: Any,
    *,
    physical_device_id: Any = None,
    include_sidecars: bool = True,
) -> tuple[str, ...]:
    """Combine adapter, camera, sidecar, and optional physical-device keys.

    An adapter must identify its command endpoint or supply a top-level
    ``physical_device_id``. Camera and detector locks alone are not enough to
    admit robot motion safely; the explicit identity supplements all derived
    resources instead of replacing them.
    """
    if isinstance(device_keys, (str, bytes)):
        raise TypeError("adapter resource_keys(cfg) must return an iterable of resource-key strings")
    device = tuple(device_keys)
    if any(not isinstance(key, str) or not key.strip() for key in device):
        raise ValueError("adapter resource keys must be non-empty strings")

    physical_key = physical_device_resource_key(physical_device_id)
    if not device and physical_key is None:
        raise ValueError("adapter must derive a device resource key or config must set top-level physical_device_id")

    return tuple(
        sorted(
            {
                *(key.strip() for key in device),
                *common_resource_keys(cfg, include_sidecars=include_sidecars),
                *(() if physical_key is None else (physical_key,)),
            }
        )
    )
