"""Instrument-owned vocabulary; robot capability/state validation stays strict."""

STATES = frozenset({"idle", "busy", "running", "fault", "offline", "unknown"})
CONDITIONS = frozenset({"interlock_closed", "emergency_stop_clear", "device_ready"})


def validate_state(state: str) -> str:
    if state not in STATES:
        raise ValueError(f"unregistered instrument state: {state!r}")
    return "busy" if state == "running" else state
