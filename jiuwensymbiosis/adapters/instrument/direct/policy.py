"""Trusted application policy extension; requests cannot select or replace it.

Protocol success requires verified response evidence; observations cannot manufacture it.
Extensions receive detached context data and must not issue hardware commands.
"""

from __future__ import annotations

from typing import Any


class PolicyError(ValueError):
    def __init__(self, code: str, message: str, field: str | None = None):
        super().__init__(message)
        self.code = code
        self.field = field


class ExecutionPolicy:
    @staticmethod
    def check_request(context: dict[str, Any]) -> None:
        """Apply additional server-owned scope restrictions before accepting work."""

    @staticmethod
    def precondition(context: dict[str, Any], field: str) -> dict[str, Any] | None:
        """Return sourced evidence, or None when an external observation is required."""
        return None

    @staticmethod
    def before_dispatch(context: dict[str, Any]) -> None:
        """Recheck additional restrictions against the latest device observation."""

    @staticmethod
    def completion(context: dict[str, Any], update: dict[str, Any], verified: bool) -> bool:
        """Apply an additional completion restriction; missing protocol evidence cannot be promoted."""
        return verified

    @staticmethod
    def accepts_external_observation(context: dict[str, Any], phase: str) -> bool:
        """Whether unresolved evidence may be supplied through the observation API."""
        return True

    @staticmethod
    def on_completion_confirmed(context: dict[str, Any], observation: dict[str, Any]) -> None:
        """Compatibility hook for host applications; generic protocol completion requires a response."""
