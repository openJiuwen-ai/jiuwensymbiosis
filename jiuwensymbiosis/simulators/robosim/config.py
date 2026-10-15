"""Configuration for the separate RoboSim service (MOZ1, one selected arm)."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml


@dataclass
class RobosimConfig:
    """RoboSim connection and readback tolerances, independent of the service runtime."""

    name: str = "robosim_moz1"
    service_url: str = "http://127.0.0.1:8001"
    robot_type: str = "moz1"
    side: str = "left"
    load_robot: bool = True
    request_timeout_s: float = 180.0
    joint_timeout_s: float = 20.0
    joint_tolerance_rad: float = 0.01
    position_tolerance_mm: float = 20.0
    orientation_tolerance_deg: float = 5.0

    def __post_init__(self) -> None:
        """Reject unsupported bodies and ambiguous connection/measurement settings."""
        url = urlsplit(self.service_url)
        if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("service_url must be a loopback HTTP origin")
        if url.username or url.password or url.port == 0:
            raise ValueError("service_url must be a loopback HTTP origin")
        if url.query or url.fragment or url.path not in {"", "/"}:
            raise ValueError("service_url must be a loopback HTTP origin")
        if self.robot_type != "moz1" or self.side not in {"left", "right"}:
            raise ValueError("This adapter currently supports MOZ1 left/right arms")
        if not isinstance(self.load_robot, bool):
            raise ValueError("load_robot must be a boolean")
        for name in (
            "request_timeout_s",
            "joint_timeout_s",
            "joint_tolerance_rad",
            "position_tolerance_mm",
            "orientation_tolerance_deg",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite and positive")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobosimConfig:
        """Read adapter fields, leaving runner/LLM configuration to its existing owner."""
        names = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobosimConfig:
        """Load a standalone runtime YAML."""
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
