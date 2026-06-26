# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Generic, task-agnostic helpers for the slot-pick loop.

Detection is delegated to the api's generic ``get_grasp_info_simple(object_name)``
(detect the named object + project to base XYZ). There are **no per-task
candidate filters** here (no chip/slot/blue/metal special-casing): the loop works
for ANY pick target and ANY place target the user names. This module only holds
the small shared helpers the loop needs — config coercion, the generic detect
wrapper, position extraction, and the structured-stop builder.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Coercion helpers (shared with SlotPickConfig.from_mapping)
# =============================================================================
def _coerce_float(value: Any, field: str) -> float:
    """Coerce a value to float; raise ValueError on failure."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number, got {value!r}") from exc


def _coerce_int(value: Any, field: str) -> int:
    """Coerce a value to int (>=1); raise ValueError on failure."""
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if out < 1:
        raise ValueError(f"{field} must be >= 1, got {out}")
    return out


def _coerce_bool(value: Any, field: str) -> bool:
    """Coerce a value to bool (supports string, int, float); raise ValueError on failure."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"{field} must be a boolean, got {value!r}")


def _coerce_pose(value: Any, field: str = "slot_observe_pose_xyzr") -> tuple[float, float, float, float]:
    """Coerce a 4-element [x, y, z, r] sequence; raise ValueError on failure."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"slot_pick.{field} must be [x, y, z, r]")
    return tuple(_coerce_float(v, field) for v in value)  # type: ignore[return-value]


def _coerce_optional_pose(
    value: Any,
    field: str,
) -> tuple[float, float, float, float] | None:
    """Coerce an optional 4-element pose; return None for empty/None values."""
    if value is None or value == "":
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"slot_pick.{field} must be [x, y, z, r]")
    return tuple(_coerce_float(v, field) for v in value)  # type: ignore[return-value]


# =============================================================================
# Result helpers
# =============================================================================
def _stop(
    stage: str,
    reason: str,
    *,
    fallback_recommended: bool,
    **extra: Any,
) -> dict[str, Any]:
    """Build a structured ``{ok: False, stage, reason, fallback_recommended, ...}`` dict."""
    result = {
        "ok": False,
        "stage": stage,
        "reason": reason,
        "fallback_recommended": bool(fallback_recommended),
    }
    result.update(extra)
    return result


def _call_ok(result: Any) -> bool:
    """Check whether a result dict has ``ok=True``; treat non-dict as success."""
    return not isinstance(result, Mapping) or result.get("ok", True) is not False


def _position_from_detection(
    detection: Mapping[str, Any],
    *,
    stage: str,
) -> tuple[float, float, float] | dict[str, Any]:
    """Extract ``(x, y, z)`` from detection's ``position``; return ``_stop`` dict on failure."""
    position = detection.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return _stop(
            stage,
            "missing_position",
            fallback_recommended=True,
            detection=dict(detection),
        )
    return (
        _coerce_float(position[0], f"{stage}.position.x"),
        _coerce_float(position[1], f"{stage}.position.y"),
        _coerce_float(position[2], f"{stage}.position.z"),
    )


# =============================================================================
# Generic detection (task-agnostic)
# =============================================================================
def _detect_object(api: Any, object_name: str) -> dict[str, Any]:
    """Detect ``object_name`` generically and project to base XYZ.

    Delegates to ``api.get_grasp_info_simple(object_name)`` — the same generic
    detect + reproject path the single-shot pick/place uses. Returns a dict
    with at least ``{ok, position:[x,y,z], ...}``; on failure ``{ok: False,
    reason: ...}``. No object-specific filtering is applied.
    """
    det = dict(api.get_grasp_info_simple(object_name))
    det.setdefault("selection_method", "get_grasp_info_simple")
    det.setdefault("object", object_name)
    return det
