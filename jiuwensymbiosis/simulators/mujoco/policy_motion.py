"""Shared lease and result ownership for native policy actions and upstream relative navigation."""

from __future__ import annotations

from typing import Any, cast


class PolicyMotion:
    def __init__(self, robot: Any, base: Any) -> None:
        self.robot, self.base = robot, base
        self.task: dict | None = None
        self._result: dict = {}
        self._navigation = False

    @property
    def result(self) -> dict:
        return cast(dict, self.base.result) if self._navigation else self._result

    @property
    def active(self) -> bool:
        return self.task is not None or bool(self.base.active)

    def accept_request(self, request: dict | None, *, scene_id: str, revision: int, now: float, step: int) -> None:
        bound = (
            request is not None and request["scene_instance_id"] == scene_id and request["config_revision"] == revision
        )
        live = bound and request is not None and not request["cancelled"] and request["lease_until"] > now
        same = request is not None and request["id"] == self.result.get("id")
        if self.active:
            if not live or not same:
                self.cancel("cancelled_or_lease_expired")
        elif bound and not same and request is not None:
            if live:
                self.start(request, step)
            else:
                self._navigation = False
                self._result = {
                    "id": request["id"],
                    "status": "completed",
                    "ok": False,
                    "reason": "cancelled_or_lease_expired",
                    "settled": False,
                    "end_step": step,
                }

    def start(self, request: dict, step: int) -> None:
        raise NotImplementedError

    def cancel(self, reason: str = "cancelled") -> None:
        raise NotImplementedError
