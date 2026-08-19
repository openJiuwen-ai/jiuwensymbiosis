# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``ExecutionMemory`` — what running actions has established, per session.

Two things a planner needs to know at plan time, both produced by *executing*
actions rather than by reading a sensor:

* **where things are** — the most recent sensing per referent, and whether it is
  still valid (moving the body invalidates every reading taken from the old
  standpoint);
* **robot self-state** — believed tokens (``payload.held``, ``body.home``, …),
  advanced by each action's declared effects.

They live in one object because one hook maintains both: every dispatch calls
:meth:`ExecutionMemory.observe` with the action's ``ToolMeta``, params and result.
Belief is corrected by observation wherever the env can actually measure the
thing (see ``WorldState``); it is a starting point, not the last word.

This replaces per-adapter private caches (a ``_last_detection`` /
``_last_surface`` pair and friends): the bookkeeping is identical for every body,
and only the framework knows the action contracts that drive it.

**Only successful actions count.** A detection that ran and returned
``ok=False`` establishes nothing, so a later step that needs a location finds
none and the caller can re-plan — which is exactly how "the target was not in
view, go search for it" recovers without any special-casing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from jiuwensymbiosis.api.decorators import ToolMeta
from jiuwensymbiosis.api.state import apply_effects

logger = logging.getLogger(__name__)


def action_succeeded(result: Any) -> bool:
    """Whether an action's return value reports success.

    The convention across the framework is ``{"ok": bool}``; anything else (a
    ``None`` return, a bare value) counts as success, since only an explicit
    ``ok=False`` is evidence of failure.
    """
    return not (isinstance(result, Mapping) and result.get("ok") is False)


@dataclass(frozen=True)
class LocationRecord:
    """One sensing: what was sensed, by which action, and when."""

    referent: str
    op: str
    result: dict[str, Any]
    at: float

    def age_s(self, now: float | None = None) -> float:
        """Seconds since this reading was taken."""
        return (time.monotonic() if now is None else now) - self.at


@dataclass
class ExecutionMemory:
    """Locations sensed and self-state established by the actions run so far."""

    locations: dict[str, LocationRecord] = field(default_factory=dict)
    self_state: frozenset[str] = frozenset()
    _order: list[str] = field(default_factory=list, repr=False)  # referents, oldest first

    # ------------------------------------------------------------------ update
    def observe(self, meta: ToolMeta | None, params: Mapping[str, Any], result: Any) -> None:
        """Fold one executed action's outcome into the memory.

        No-op for an action with no contract (``meta is None``) or one that
        reported failure — neither tells us anything new about the world.
        """
        if meta is None or not action_succeeded(result):
            return
        if meta.invalidates_locations:
            self.invalidate_locations(f"{meta.name} moved the body")
        if meta.produces_location and isinstance(result, Mapping):
            self.record_location(self._referent_of(meta, params), meta.name, dict(result))
        self.self_state = apply_effects(self.self_state, provides=meta.provides, invalidates=meta.invalidates)

    def record_location(self, referent: str, op: str, result: dict[str, Any]) -> None:
        """Store (or refresh) the sensing for ``referent``."""
        if referent in self.locations:
            self._order.remove(referent)
        self.locations[referent] = LocationRecord(referent, op, result, time.monotonic())
        self._order.append(referent)

    def invalidate_locations(self, reason: str) -> None:
        """Drop every sensing — they were measured from a standpoint we have left."""
        if self.locations:
            logger.debug("[memory] invalidating %d location(s): %s", len(self.locations), reason)
        self.locations.clear()
        self._order.clear()

    def clear(self) -> None:
        """Forget everything (a fresh session, or an explicit reset)."""
        self.invalidate_locations("reset")
        self.self_state = frozenset()

    # ------------------------------------------------------------------- query
    def latest(self) -> LocationRecord | None:
        """The most recent valid sensing — what a cache-consuming action acts on."""
        return self.locations[self._order[-1]] if self._order else None

    def get(self, referent: str) -> LocationRecord | None:
        """The sensing for one referent, or ``None`` if absent / invalidated."""
        return self.locations.get(referent)

    def describe(self) -> dict[str, Any]:
        """JSON-able summary for the planner prompt and the ``state`` CLI."""
        now = time.monotonic()
        return {
            "self_state": sorted(self.self_state),
            "locations": [
                {
                    "referent": r.referent,
                    "sensed_by": r.op,
                    "age_s": round(r.age_s(now), 1),
                    "position_mm": r.result.get("position") or r.result.get("center_mm"),
                }
                for r in (self.locations[k] for k in self._order)
            ],
        }

    @staticmethod
    def _referent_of(meta: ToolMeta, params: Mapping[str, Any]) -> str:
        """Name the thing that was sensed.

        ``object_name`` is the framework-wide way a caller says *what* to sense, so
        it identifies the referent whenever present; otherwise the action itself is
        the key, which keeps one entry per sensing action rather than merging them.
        """
        name = params.get("object_name")
        return name if isinstance(name, str) and name else meta.name
