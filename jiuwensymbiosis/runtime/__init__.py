"""Headless task submission and local resource admission.

Importing this package does not load a GUI, calibration or an adapter. Bindings
load the selected adapter only when prepare_binding() is called.
"""

from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.runtime.artifacts import decode_frame
from jiuwensymbiosis.runtime.jobs import Runtime, RuntimeClosedError
from jiuwensymbiosis.runtime.resources import ResourceBlockedError, ResourceBusyError, ResourceLease, ResourceManager
from jiuwensymbiosis.runtime.results import normalize_result
from jiuwensymbiosis.runtime.store import RequestConflictError

__all__ = [
    "Runtime",
    "RuntimeClosedError",
    "CleanupReport",
    "ResourceManager",
    "ResourceLease",
    "ResourceBusyError",
    "ResourceBlockedError",
    "RequestConflictError",
    "prepare_binding",
    "default_workspace",
    "BindingSnapshot",
    "decode_frame",
    "normalize_result",
]


def __getattr__(name):
    if name in {"prepare_binding", "BindingSnapshot", "default_workspace"}:
        from jiuwensymbiosis.runtime import bindings

        return getattr(bindings, name)
    raise AttributeError(name)
