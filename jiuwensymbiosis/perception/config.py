# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Detection deployment configuration, shared by every hardware adapter."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from jiuwensymbiosis.utils.logging import get_logger
from jiuwensymbiosis.utils.service_http import HttpEndpointConfig
from jiuwensymbiosis.utils.validation import require_positive_number, require_unit_interval

logger = get_logger(__name__)


def _positive(value: Any, name: str) -> float:
    require_positive_number(value, message=f"detector.{name} must be finite and positive")
    return float(value)


@dataclass
class LocalDetectorConfig:
    """Explicitly owned local model process; remote clients never read these fields."""

    host: str = "127.0.0.1"
    port: int = 8114
    device: str = "cuda"
    startup_timeout_s: float = 300.0
    gdino_model_id: str = "IDEA-Research/grounding-dino-base"
    sam2_model_id: str = "facebook/sam2.1-hiera-large"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    use_sam2: bool = True

    def __post_init__(self) -> None:
        if self.host not in {"127.0.0.1", "localhost"}:
            raise ValueError("detector.local.host must be loopback; use remote mode for an external server")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("detector.local.port must be an integer between 1 and 65535")
        self.startup_timeout_s = _positive(self.startup_timeout_s, "local.startup_timeout_s")
        for name in ("box_threshold", "text_threshold"):
            require_unit_interval(getattr(self, name), message=f"detector.local.{name} must be between 0 and 1")
        if not isinstance(self.use_sam2, bool):
            raise ValueError("detector.local.use_sam2 must be bool")
        for name in ("device", "gdino_model_id", "sam2_model_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"detector.local.{name} must be a nonempty string")


@dataclass
class DetectorConfig:
    """A disabled, externally managed HTTP, or explicitly local detector."""

    mode: str = "disabled"
    endpoint: HttpEndpointConfig | None = None
    local: LocalDetectorConfig | None = None
    max_frame_age_s: float | None = None

    def __post_init__(self) -> None:
        if self.max_frame_age_s is not None:
            self.max_frame_age_s = _positive(self.max_frame_age_s, "max_frame_age_s")
        if self.mode not in {"disabled", "remote", "local"}:
            raise ValueError("detector.mode must be disabled, remote or local")
        if self.mode == "disabled" and (self.endpoint is not None or self.local is not None):
            raise ValueError("disabled detector cannot configure endpoint or local")
        if self.mode == "remote" and (self.endpoint is None or self.local is not None):
            raise ValueError("remote detector requires endpoint and cannot configure local")
        if self.mode == "local":
            if self.endpoint is not None:
                raise ValueError("local detector cannot configure endpoint")
            self.local = self.local or LocalDetectorConfig()

    @property
    def url(self) -> str | None:
        if self.local is not None:
            return f"http://{self.local.host}:{self.local.port}"
        return self.endpoint.url if self.endpoint is not None else None

    @property
    def spawn(self) -> bool:
        return self.mode == "local"

    def __getattr__(self, name: str) -> Any:
        # Transitional read-only field view used by existing lifecycle/resource code.
        if name in {f.name for f in dataclasses.fields(LocalDetectorConfig)}:
            return getattr(self.local or LocalDetectorConfig(), name)
        raise AttributeError(name)


class DetectorServerConfig(DetectorConfig):
    """One-version Python compatibility constructor for the former per-body config."""

    def __init__(
        self, url: str | None = None, spawn: bool = True, *, max_frame_age_s: float | None = None, **kwargs: Any
    ) -> None:
        logger.warning("DetectorServerConfig is deprecated; use DetectorConfig(mode=..., ...)")
        if not isinstance(spawn, bool):
            raise ValueError("detector.spawn must be bool")
        unknown = set(kwargs) - {f.name for f in dataclasses.fields(LocalDetectorConfig)}
        if unknown:
            raise ValueError(f"unknown detector fields: {sorted(unknown)}")
        if not spawn:
            remote_url = url or f"http://{kwargs.get('host', '127.0.0.1')}:{kwargs.get('port', 8114)}"
            super().__init__(
                mode="remote",
                endpoint=HttpEndpointConfig(url=remote_url),
                max_frame_age_s=max_frame_age_s,
            )
            return
        if spawn and url is not None:
            parsed = urlsplit(url)
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"127.0.0.1", "localhost"}
                or parsed.path not in {"", "/"}
            ):
                raise ValueError("local detector URL must be an http loopback origin")
            if parsed.username or parsed.query or parsed.fragment:
                raise ValueError("local detector URL must be an http loopback origin")
            kwargs["host"] = parsed.hostname
            kwargs["port"] = parsed.port or 80
        local = LocalDetectorConfig(**kwargs)
        super().__init__(mode="local", local=local, max_frame_age_s=max_frame_age_s)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectorServerConfig:
        return cls(**data)


def _is_legacy_detector(entry: Any) -> bool:
    return isinstance(entry, Mapping) and any(
        name in str(entry.get("_target_", "")).lower() for name in ("grounding_dino", "gdino")
    )


def parse_detector_config(data: Mapping[str, Any]) -> DetectorConfig:
    """Normalize the full runtime mapping, rejecting mixed old/new detector settings."""
    nested = data.get("env", {})
    nested = nested.get("cfg", {}) if isinstance(nested, Mapping) else {}
    entries = list(data.get("api_servers") or []) + list(nested.get("api_servers") or [])
    legacy = [s for s in entries if _is_legacy_detector(s)]
    if len(legacy) > 1:
        raise ValueError("multiple legacy api_servers detectors are ambiguous")
    if "detector" in data and legacy:
        raise ValueError("detector conflicts with legacy api_servers detector")
    if "detector" in data:
        raw = data["detector"]
        if isinstance(raw, DetectorConfig):
            return raw
        if not isinstance(raw, Mapping):
            raise ValueError("detector must be a mapping")
        unknown = set(raw) - {"mode", "endpoint", "local", "max_frame_age_s"}
        if unknown:
            raise ValueError(f"unknown detector fields: {sorted(unknown)}")
        endpoint = raw.get("endpoint")
        local = raw.get("local")
        if endpoint is not None and not isinstance(endpoint, Mapping):
            raise ValueError("detector.endpoint must be a mapping")
        if local is not None and not isinstance(local, Mapping):
            raise ValueError("detector.local must be a mapping")
        if local is not None and set(local) - {f.name for f in dataclasses.fields(LocalDetectorConfig)}:
            raise ValueError("unknown detector.local fields")
        return DetectorConfig(
            mode=raw.get("mode", "disabled"),
            endpoint=HttpEndpointConfig.from_dict(endpoint) if endpoint is not None else None,
            local=LocalDetectorConfig(**local) if local is not None else None,
            max_frame_age_s=raw.get("max_frame_age_s"),
        )
    if not legacy:
        return DetectorConfig()
    logger.warning("api_servers detector is deprecated; migrate to detector.mode and detector.local/endpoint")
    raw = legacy[0]
    spawn = raw.get("spawn", True)
    if not isinstance(spawn, bool):
        raise ValueError("api_servers detector.spawn must be bool")
    defaults = LocalDetectorConfig()
    fields: dict[str, Any] = {
        f.name: raw.get(f.name) if raw.get(f.name) is not None else getattr(defaults, f.name)
        for f in dataclasses.fields(LocalDetectorConfig)
    }
    if not spawn:
        url = raw.get("url") or f"http://{fields['host']}:{fields['port']}"
        return DetectorConfig(
            mode="remote",
            endpoint=HttpEndpointConfig(url=url),
            max_frame_age_s=raw.get("max_frame_age_s"),
        )
    fields["gdino_model_id"] = os.environ.get("GDINO_MODEL_ID") or fields["gdino_model_id"]
    fields["sam2_model_id"] = os.environ.get("SAM2_MODEL_ID") or fields["sam2_model_id"]
    return DetectorConfig(mode="local", local=LocalDetectorConfig(**fields), max_frame_age_s=raw.get("max_frame_age_s"))
