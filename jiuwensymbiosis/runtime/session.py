"""Lifecycle consumed by the headless Runtime for every execution domain."""

from __future__ import annotations

from typing import Protocol

from jiuwensymbiosis.agent.lifecycle import CleanupReport


class ManagedSession(Protocol):
    """A connected resource owner; no robot motion or observation assumptions."""

    def connect(self) -> None:
        """Acquire the configured resources."""
        ...

    def disconnect(self) -> None:
        """Release resources or report unconfirmed cleanup."""
        ...

    def cleanup_report(self) -> CleanupReport:
        """Describe the evidence for the latest cleanup attempt."""
        ...
