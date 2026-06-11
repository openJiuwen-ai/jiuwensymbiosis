# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Vision-driven pick / place choreography — hardware-agnostic.

Operates purely through a small mixin-style protocol on the api object:
  * ``api.home() -> None``
  * ``api.goto_xyzr(x, y, z, r=None) -> None``         (TIP frame, mm/deg)
  * ``api.get_home_pose() -> {"x","y","z","r"}``
  * ``api.get_grasp_info_simple(name) -> {"ok": bool, "position": [x,y,z], ...}``
  * ``api.activate_suction() / api.deactivate_suction() -> {"ok": bool, ...}``

Any adapter whose ``Api`` provides those methods can call these skills
verbatim.

Safety idiom: every approach goes (home → horizontal XY at home_z →
descend to target_z), and every release goes back through home. This is
the SCARA-friendly motion order; for a 6-DoF arm we may need a different
pre-grasp pose but the same choreography shape — extend by adding a
``approach_pose`` arg later if needed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _stop_result(stage: str, reason: str, **extra: Any) -> dict:
    result = {"ok": False, "stage": stage, "reason": reason}
    result.update(extra)
    return result


def move_tip_xy_then_z(api: Any, position: list[float], *, stage: str) -> None:
    """Approach target safely: horizontal XY at home height, then drop in Z."""
    if len(position) < 3:
        raise ValueError(f"{stage}: expected [x, y, z], got {position!r}")
    x, y, z = [float(v) for v in position[:3]]
    home_pose = api.get_home_pose()
    home_z = float(home_pose["z"])
    logger.info(
        "[skills] %s: move via home_z %.2f to target=(%.2f, %.2f, %.2f)",
        stage, home_z, x, y, z,
    )
    api.goto_xyzr(x=x, y=y, z=home_z)
    api.goto_xyzr(x=x, y=y, z=z)


def pick_object_to_suction(api: Any, object_name: str) -> dict:
    """Choreographed pick: home → detect → approach → descend → suck → home.

    Stops immediately and returns ``ok=False`` if any step fails. Errors are
    surfaced to the agent via the result dict (never re-raised) so the agent
    can decide whether to retry / abort.
    """
    logger.info("[skills] pick_object_to_suction(object=%r)", object_name)
    try:
        api.home()
        detection = api.get_grasp_info_simple(object_name)
        if not detection.get("ok"):
            return _stop_result(
                "detect_object",
                str(detection.get("reason", "detection_failed")),
                detection=detection,
                object=object_name,
            )

        position = detection.get("position")
        if not isinstance(position, list):
            return _stop_result(
                "detect_object",
                "missing_position",
                detection=detection,
                object=object_name,
            )

        move_tip_xy_then_z(api, position, stage="pick_object")
        suction = api.activate_suction()
        if not suction.get("ok"):
            return _stop_result(
                "activate_suction",
                str(suction.get("reason", "suction_failed")),
                detection=detection,
                suction=suction,
                object=object_name,
            )

        api.home()
        return {
            "ok": True,
            "object": object_name,
            "picked_position": detection["position"],
            "detection": detection,
            "suction": suction,
        }
    except Exception as exc:  # noqa: BLE001 - surface hardware/runtime failures to the agent
        logger.exception("[skills] pick_object_to_suction failed")
        return _stop_result(
            "pick_object_to_suction",
            f"{type(exc).__name__}: {exc}",
            object=object_name,
        )


def place_suction_to_target(api: Any, target_name: str) -> dict:
    """Choreographed place: home → detect → approach → descend → release → home.

    Mirror of ``pick_object_to_suction`` — same step ordering, just
    deactivate_suction instead of activate. Failure handling identical.
    """
    logger.info("[skills] place_suction_to_target(target=%r)", target_name)
    try:
        api.home()
        detection = api.get_grasp_info_simple(target_name)
        if not detection.get("ok"):
            return _stop_result(
                "detect_target",
                str(detection.get("reason", "detection_failed")),
                detection=detection,
                target=target_name,
            )

        position = detection.get("position")
        if not isinstance(position, list):
            return _stop_result(
                "detect_target",
                "missing_position",
                detection=detection,
                target=target_name,
            )

        move_tip_xy_then_z(api, position, stage="place_target")
        suction = api.deactivate_suction()
        if not suction.get("ok"):
            return _stop_result(
                "deactivate_suction",
                str(suction.get("reason", "suction_failed")),
                detection=detection,
                suction=suction,
                target=target_name,
            )

        api.home()
        return {
            "ok": True,
            "target": target_name,
            "placed_position": detection["position"],
            "detection": detection,
            "suction": suction,
        }
    except Exception as exc:  # noqa: BLE001 - surface hardware/runtime failures to the agent
        logger.exception("[skills] place_suction_to_target failed")
        return _stop_result(
            "place_suction_to_target",
            f"{type(exc).__name__}: {exc}",
            target=target_name,
        )
