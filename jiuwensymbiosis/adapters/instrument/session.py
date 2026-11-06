"""Single offline instrument builder; Runtime owns admission and execution."""

from jiuwensymbiosis.adapters._common.builder import make_session_builder
from jiuwensymbiosis.adapters.instrument.config import InstrumentConfig
from jiuwensymbiosis.runtime.session import ManagedSession


def resource_keys(cfg: InstrumentConfig) -> tuple[str, ...]:
    if cfg.access == "direct":
        from jiuwensymbiosis.adapters._common.resources import physical_device_resource_key

        keys = []
        for profile in cfg.selected.installation["devices"]:
            key = physical_device_resource_key(profile["physical_device_id"])
            if key is None:
                raise ValueError("direct profiles require physical_device_id")
            keys.append(key)
        return tuple(keys)
    from jiuwensymbiosis.adapters.instrument.mcp.config import resource_keys as remote_resources

    return remote_resources(cfg.selected)


def _build(cfg: InstrumentConfig, *, include_sidecars: bool = True) -> ManagedSession:
    if cfg.access == "direct":
        from jiuwensymbiosis.adapters.instrument.direct.session import InstrumentSession

        return InstrumentSession(
            cfg.selected.installation, permissions=cfg.selected.permissions, name=cfg.selected.name
        )
    from jiuwensymbiosis.adapters.instrument.mcp.session import InstrumentMCPSession

    return InstrumentMCPSession(cfg.selected)


build_instrument_session = make_session_builder(_build, InstrumentConfig, resource_keys=resource_keys)
