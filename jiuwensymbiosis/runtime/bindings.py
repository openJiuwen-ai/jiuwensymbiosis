"""Immutable, source-aware bindings between config and robot sessions.

Preparing a binding validates one adapter config, resolves source-relative paths,
captures environment-derived defaults, and derives its admission resources. It
does not connect a driver or start a sidecar. A binding keeps the typed config
private and only hands a fresh deep copy to each session builder invocation.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import enum
import hashlib
import importlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jiuwensymbiosis.adapters._common.config import parse_config
from jiuwensymbiosis.adapters._common.resources import complete_resource_keys
from jiuwensymbiosis.agent.session import RobotSession

__all__ = ["BindingSnapshot", "prepare_binding"]

_ADAPTER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class BindingSnapshot:
    """A validated, immutable reference to the effective robot configuration.

    ``config_data()`` returns a deep copy of the complete, normalized source
    mapping (including non-adapter fields such as agent settings). Session
    construction uses the private typed adapter config parsed during
    ``prepare_binding``; it never re-reads files or environment defaults.
    """

    binding_id: str
    fingerprint: str
    source: Path
    adapter: str
    workspace: Path
    resources: tuple[str, ...]
    _config_data: dict[str, Any] = field(repr=False, compare=False)
    _effective_config: Any = field(repr=False, compare=False)
    _session_builder: Callable[..., RobotSession] = field(repr=False, compare=False)
    _device_resource_keys: tuple[str, ...] = field(repr=False, compare=False)
    _physical_device_id: str | None = field(repr=False, compare=False)
    _include_sidecars: bool = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_config_data", copy.deepcopy(self._config_data))
        object.__setattr__(self, "_effective_config", copy.deepcopy(self._effective_config))

    def config_data(self) -> dict[str, Any]:
        """Return an isolated copy of the normalized source mapping."""
        return copy.deepcopy(self._config_data)

    def build_session(self) -> RobotSession:
        """Build a new, disconnected RobotSession from the frozen effective config."""
        cfg = copy.deepcopy(self._effective_config)
        return self._session_builder(cfg, include_sidecars=self._include_sidecars)

    def without_sidecars(self) -> BindingSnapshot:
        """Return this frozen binding without locally spawned sidecar admission.

        The effective config and adapter resource identities are reused from the
        original preparation. Environment defaults are not re-read, so a
        maintenance session cannot silently target another device.
        """
        if not self._include_sidecars:
            return self
        resources = complete_resource_keys(
            self._device_resource_keys,
            self._effective_config,
            physical_device_id=self._physical_device_id,
            include_sidecars=False,
        )
        fingerprint = _fingerprint(
            source=self.source,
            adapter=self.adapter,
            config_data=self._config_data,
            effective_config=self._effective_config,
            workspace=self.workspace,
            resources=resources,
        )
        return BindingSnapshot(
            binding_id=fingerprint,
            fingerprint=fingerprint,
            source=self.source,
            adapter=self.adapter,
            workspace=self.workspace,
            resources=resources,
            _config_data=self._config_data,
            _effective_config=self._effective_config,
            _session_builder=self._session_builder,
            _device_resource_keys=self._device_resource_keys,
            _physical_device_id=self._physical_device_id,
            _include_sidecars=False,
        )


def _resolve_source(config_source: str | Path) -> Path:
    """Resolve and require the source file even when an edit snapshot is supplied."""
    source = Path(config_source).expanduser()
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"configuration source does not exist: {config_source}") from exc
    if not source.is_file():
        raise FileNotFoundError(f"configuration source is not a file: {config_source}")
    return source


def default_workspace() -> Path:
    """Default workspace shared by Runtime bindings and GUI hosts."""
    return Path.home() / ".jiuwensymbiosis" / "gui_workspace"


def _load_source_data(source: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML configuration source: {source}") from exc
    if not isinstance(loaded, Mapping) or not loaded:
        raise ValueError("configuration source must contain a non-empty mapping")
    return copy.deepcopy(dict(loaded))


def _snapshot_data(config_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config_snapshot, Mapping) or not config_snapshot:
        raise ValueError("config_snapshot must be a non-empty configuration mapping")
    return copy.deepcopy(dict(config_snapshot))


def _adapter_from_data(data: Mapping[str, Any]) -> str:
    value = data.get("adapter")
    if not isinstance(value, str) or not _ADAPTER_NAME.fullmatch(value.strip()):
        raise ValueError("configuration must declare a valid top-level adapter name")
    return value.strip()


def _discover_adapter(adapter: str) -> tuple[Any, Callable[..., RobotSession], Callable[[Any], Any]]:
    """Resolve an adapter by the existing package/session naming convention."""
    module = importlib.import_module(f"jiuwensymbiosis.adapters.{adapter}")
    builder = getattr(module, f"build_{adapter}_session", None)
    if not callable(builder):
        raise ValueError(f"adapter {adapter!r} must export build_{adapter}_session")
    config_factory = getattr(builder, "config_factory", None)
    resource_keys = getattr(builder, "resource_keys", None)
    if config_factory is None or not callable(getattr(config_factory, "from_dict", None)):
        raise ValueError(f"adapter {adapter!r} session builder must expose config_factory.from_dict")
    if not callable(resource_keys):
        raise ValueError(f"adapter {adapter!r} session builder must expose resource_keys(cfg)")
    return config_factory, builder, resource_keys


def _json_value(value: Any) -> Any:
    """Convert effective config values into deterministic JSON-compatible data."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("configuration mapping keys must be strings")
            normalized[key] = _json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [_json_value(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        )
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return _json_value(value.value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported config value for binding fingerprint: {type(value).__name__}")


def _fingerprint(
    *,
    source: Path,
    adapter: str,
    config_data: Mapping[str, Any],
    effective_config: Any,
    workspace: Path,
    resources: tuple[str, ...],
) -> str:
    payload = {
        "source": str(source),
        "adapter": adapter,
        "config_data": _json_value(config_data),
        "effective_config": _json_value(effective_config),
        "workspace": str(workspace),
        "resources": list(resources),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prepare_binding(
    config_source: str | Path,
    *,
    config_snapshot: Mapping[str, Any] | None = None,
    workspace: str | Path | None = None,
    include_sidecars: bool = True,
) -> BindingSnapshot:
    """Validate and freeze an adapter configuration without connecting hardware.

    ``config_snapshot`` is the complete unsaved config mapping, such as a GUI
    edit or a CLI adapter override. ``config_source`` remains required and must
    identify an existing YAML file because it supplies provenance and the base
    directory for relative paths.
    """
    source = _resolve_source(config_source)
    source_data = _load_source_data(source) if config_snapshot is None else _snapshot_data(config_snapshot)
    adapter = _adapter_from_data(source_data)
    factory, builder, resource_key_factory = _discover_adapter(adapter)

    # Parse once. The returned typed config captures adapter defaults derived
    # from process environment (for example CAMERA_SERIAL); build_session() only
    # deep-copies this result and never resolves the environment a second time.
    normalized_data, effective_config = parse_config(factory, source_data, source_dir=source.parent)
    device_keys = resource_key_factory(effective_config)
    physical_device_id = normalized_data.get("physical_device_id")
    resources = complete_resource_keys(
        device_keys,
        effective_config,
        physical_device_id=physical_device_id,
        include_sidecars=include_sidecars,
    )

    selected_workspace = Path(workspace).expanduser() if workspace is not None else default_workspace()
    selected_workspace = selected_workspace.resolve(strict=False)
    fingerprint = _fingerprint(
        source=source,
        adapter=adapter,
        config_data=normalized_data,
        effective_config=effective_config,
        workspace=selected_workspace,
        resources=resources,
    )
    return BindingSnapshot(
        binding_id=fingerprint,
        fingerprint=fingerprint,
        source=source,
        adapter=adapter,
        workspace=selected_workspace,
        resources=resources,
        _config_data=normalized_data,
        _effective_config=effective_config,
        _session_builder=builder,
        _device_resource_keys=device_keys,
        _physical_device_id=physical_device_id,
        _include_sidecars=include_sidecars,
    )
