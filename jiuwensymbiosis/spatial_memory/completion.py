"""Separate stopping, observed geometric scope and semantic inventory evidence."""

from collections import deque


def observed_scope(center: tuple[int, int], free: set, occupied: set) -> dict:
    """Infer only the observed connected component; never accept an oracle boundary."""
    if center not in free:
        return {"kind": "unknown", "boundary_unknown_cells": None, "cells": 0}
    component = {center}
    queue = deque([center])
    unknown = set()
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cell = x + dx, y + dy
            if cell in occupied:
                continue
            if cell in free:
                if cell not in component:
                    component.add(cell)
                    queue.append(cell)
            else:
                unknown.add(cell)
    return {
        "kind": "unknown" if unknown else "observed_enclosed_component",
        "boundary_unknown_cells": len(unknown),
        "cells": len(component),
    }


def assess_completion(
    *, reason: str, scope: dict, frontier_cells: int, gains: list[int], tasks: list[dict], stopped: bool
) -> dict:
    """Low gain is corroboration, never a substitute for scope/stop evidence."""
    exhausted = reason == "no_candidates"
    geometry_complete = (
        exhausted
        and stopped
        and scope["kind"] == "observed_enclosed_component"
        and frontier_cells == 0
        and len(gains) >= 3
        and all(g == 0 for g in gains[-3:])
    )
    gaps = {state: sum(t["state"] == state for t in tasks) for state in ("open", "deferred", "resolved")}
    return {
        "stop_reason": reason,
        "stopped": stopped,
        "scope": scope,
        "geometry_complete_in_observed_scope": geometry_complete,
        "scan_complete": True if geometry_complete and not (gaps["open"] or gaps["deferred"]) else None,
        "scan_complete_semantics": "observed_component_only_not_whole_room",
        "memory_quality": "known_gaps" if gaps["open"] or gaps["deferred"] else "no_known_gaps",
        "view_task_counts": gaps,
        "inventory_complete": False,
    }
