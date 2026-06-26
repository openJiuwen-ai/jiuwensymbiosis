# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Body-agnostic, task-agnostic executable slot-pick loop.

One tool call drives a repeated pick→place loop in-process: each cycle detects
the place target and the pick object (by the object names the user named),
grasps the object, places it at the target, and repeats until the pick object is
no longer detected (or ``max_pick_place_cycles`` is reached).

Layering:
  * ``detect.py``   — generic helpers + ``_detect_object`` (delegates to the api's
                      task-agnostic ``get_grasp_info_simple``). No per-task filters.
  * ``strategy.py`` — per-body motion guards + grasp/release (``SlotPickStrategy``).
  * ``skill.py``    — *this* module: ``SlotPickConfig`` + ``run_slot_pick`` loop +
                      the openjiuwen Tool wrapper.

What to detect (``chip_object_name`` = what to pick, ``slot_object_name`` = where
to place) comes from the user's task — set in the YAML or passed by the agent at
invocation; there is nothing object-specific baked into the detection.
"""

from __future__ import annotations

import inspect
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import MISSING, dataclass, fields
from typing import Any, NamedTuple

from jiuwensymbiosis.agent import Tool, ToolCard, ToolOutput
from jiuwensymbiosis.tools.slot_pick.detect import (
    _call_ok,
    _coerce_bool,
    _coerce_float,
    _coerce_int,
    _coerce_optional_pose,
    _detect_object,
    _position_from_detection,
    _stop,
)
from jiuwensymbiosis.tools.slot_pick.strategy import GripperStrategy, SlotPickStrategy

logger = logging.getLogger(__name__)

_DEBUG_SOURCE = "slot_pick_skill"


class PoseXYZR(NamedTuple):
    x: float
    y: float
    z: float
    r: float


class PoseXYZ(NamedTuple):
    x: float
    y: float
    z: float


@dataclass
class CycleState:
    idx: int
    cycles: list[dict[str, Any]]
    home_pose: list[float]
    stage_setter: Callable[[str], None]
    slot_observe: PoseXYZR | None = None
    chip_observe: PoseXYZR | None = None
    pick_r: float = 0.0
    place_r: float = 0.0
    slot_detection: dict[str, Any] | None = None
    chip_detection: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class GraspApproachParams:
    chip_x: float
    chip_y: float
    chip_pick_z: float
    chip_above_z: float
    pick_r: float


# =============================================================================
# Config
# =============================================================================
@dataclass(frozen=True)
class SlotPickConfig:
    chip_object_name: str = ""
    slot_object_name: str = ""
    slot_observe_pose_xyzr: tuple[float, float, float, float] | None = None
    chip_observe_pose_xyzr: tuple[float, float, float, float] | None = None
    place_done_radius_mm: float = 60.0
    place_r_delta_deg: float = -90.0
    chip_pick_r_offset_deg: float = 0.0
    chip_thickness_mm: float = 5.0
    chip_pick_z_offset_mm: float = 0.0
    chip_approach_hover_mm: float = 100.0
    slot_place_x_offset_mm: float = 0.0
    slot_place_y_offset_mm: float = 0.0
    slot_place_z_offset_mm: float = 0.0
    place_approach_hover_mm: float = 100.0
    max_reach_radius_mm: float = 0.0
    safe_travel_z_min_mm: float = 0.0
    pick_hold_s: float = 0.0
    enable_contact_settle: bool = False
    contact_settle_radius_mm: float = 2.0
    contact_settle_push_mm: float = 0.0
    contact_settle_passes: int = 1
    contact_settle_sleep_s: float = 0.2
    enable_slot_refine: bool = False
    slot_refine_lift_mm: float = 200.0
    slot_refine_min_z_margin_mm: float = 5.0
    slot_refine_max_shift_mm: float = 25.0
    slot_refine_min_score: float = 0.001
    slot_refine_max_z_drop_mm: float = 15.0
    max_pick_place_cycles: int = 8

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SlotPickConfig:
        """Build a SlotPickConfig from a mapping (e.g. a YAML block or dict)."""
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            raw = data.get(f.name, f.default)
            if f.name in ("chip_object_name", "slot_object_name"):
                val = str(data.get(f.name) or "").strip()
                if not val:
                    raise ValueError(f"slot_pick.{f.name} is required")
                kwargs[f.name] = val
            elif "xyzr" in f.name:
                kwargs[f.name] = _coerce_optional_pose(raw, f.name)
            elif isinstance(f.default, bool):
                kwargs[f.name] = _coerce_bool(raw, f.name)
            elif isinstance(f.default, int):
                kwargs[f.name] = _coerce_int(raw, f.name)
            elif isinstance(f.default, float):
                kwargs[f.name] = _coerce_float(raw, f.name)
            else:
                kwargs[f.name] = raw
        return cls(**kwargs)

    def merged(self, overrides: Mapping[str, Any]) -> SlotPickConfig:
        """Return a new SlotPickConfig with non-None overrides applied."""
        if not overrides:
            return self
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return type(self).from_mapping(data)


# =============================================================================
# JSON schema auto-generation
# =============================================================================
_INPUT_PARAMS_EXCLUDE = {"place_done_radius_mm"}

_FIELD_DESCRIPTIONS: dict[str, str] = {
    "chip_object_name": (
        "What to pick — the object words from the user's task "
        "(e.g. 'blue chip', 'black box'). Overrides the config default."
    ),
    "slot_object_name": (
        "Where to place it — the target words from the user's task "
        "(e.g. 'metal slot', 'white box'). Overrides the config default."
    ),
    "slot_observe_pose_xyzr": "Optional [x, y, z, r] TIP-frame place-target observation pose.",
    "chip_observe_pose_xyzr": "Optional [x, y, z, r] TIP-frame pick-object observation pose.",
}


def _json_type_for_field(f: Any) -> dict[str, Any]:
    if "xyzr" in f.name:
        return {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        }
    if f.default is not MISSING:
        if isinstance(f.default, bool):
            return {"type": "boolean"}
        if isinstance(f.default, int):
            return {"type": "integer"}
        if isinstance(f.default, float):
            return {"type": "number"}
    return {"type": "string"}


def _input_params() -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for f in fields(SlotPickConfig):
        if f.name in _INPUT_PARAMS_EXCLUDE:
            continue
        schema = _json_type_for_field(f)
        desc = _FIELD_DESCRIPTIONS.get(f.name)
        if desc:
            schema["description"] = desc
        properties[f.name] = schema
    return {"type": "object", "properties": properties, "required": []}


# =============================================================================
# Choreography
# =============================================================================
def _execute_grasp_sequence(
    strategy: SlotPickStrategy,
    config: SlotPickConfig,
    approach: GraspApproachParams,
    *,
    stage_prefix: str,
    slot_detection: dict[str, Any],
    chip_detection: dict[str, Any],
    extra_stop_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Open gripper, approach, descend, grasp, hold, lift.
    Returns None on success, or a stop-dict on failure."""
    extra = extra_stop_kwargs or {}

    opened = strategy.release()
    if not _call_ok(opened):
        return _stop(
            f"{stage_prefix}_pre_grasp_open",
            str(opened.get("reason", "open_failed")),
            fallback_recommended=False,
            open=opened,
            slot_detection=slot_detection,
            chip_detection=chip_detection,
            **extra,
        )

    strategy.goto_process(approach.chip_x, approach.chip_y, approach.chip_above_z, approach.pick_r)
    strategy.goto_critical(approach.chip_x, approach.chip_y, approach.chip_pick_z, approach.pick_r)

    closed = strategy.grasp()
    if not _call_ok(closed):
        return _stop(
            f"{stage_prefix}_close_gripper",
            str(closed.get("reason", "grasp_failed")),
            fallback_recommended=False,
            grasp=closed,
            slot_detection=slot_detection,
            chip_detection=chip_detection,
            **extra,
        )

    if config.pick_hold_s > 0.0:
        time.sleep(config.pick_hold_s)
    strategy.goto_process(approach.chip_x, approach.chip_y, approach.chip_above_z, approach.pick_r)
    return None


def _place_held_object(
    *,
    strategy: SlotPickStrategy,
    config: SlotPickConfig,
    place_r: float,
    slot_xyz: PoseXYZ,
    slot_detection: dict[str, Any],
    chip_detection: dict[str, Any],
    stage_prefix: str,
) -> dict[str, Any]:
    """Place the held object: travel above the target already at place_r, descend
    to target.z + thickness, (optional contact self-centering), release, retract."""
    place_x = slot_xyz.x + config.slot_place_x_offset_mm
    place_y = slot_xyz.y + config.slot_place_y_offset_mm
    place_tip_z = slot_xyz.z + config.chip_thickness_mm + config.slot_place_z_offset_mm
    place_hover_z = place_tip_z + config.place_approach_hover_mm

    strategy.goto_process(place_x, place_y, place_hover_z, place_r)
    strategy.goto_critical(place_x, place_y, place_tip_z, place_r)

    contact_settle_done = 0
    if config.enable_contact_settle:
        settle_z = place_tip_z - max(0.0, config.contact_settle_push_mm)
        r = config.contact_settle_radius_mm
        cross = [(r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)]
        strategy.goto_exact(place_x, place_y, settle_z, place_r)
        for _pass in range(max(1, config.contact_settle_passes)):
            for dx, dy in cross:
                strategy.goto_exact(place_x + dx, place_y + dy, settle_z, place_r)
                time.sleep(max(0.0, config.contact_settle_sleep_s))
                strategy.goto_exact(place_x, place_y, settle_z, place_r)
                contact_settle_done += 1
        strategy.goto_exact(place_x, place_y, place_tip_z, place_r)

    release = strategy.release()
    if not _call_ok(release):
        return _stop(
            f"{stage_prefix}_final_release",
            str(release.get("reason", "release_failed")),
            fallback_recommended=False,
            release=release,
            slot_detection=slot_detection,
            chip_detection=chip_detection,
        )
    strategy.goto_process(place_x, place_y, place_hover_z, place_r)
    return {
        "ok": True,
        "place_xy": [place_x, place_y],
        "slot_place_x_offset_mm": config.slot_place_x_offset_mm,
        "slot_place_y_offset_mm": config.slot_place_y_offset_mm,
        "slot_place_z_offset_mm": config.slot_place_z_offset_mm,
        "place_tip_z": place_tip_z,
        "enable_contact_settle": config.enable_contact_settle,
        "contact_settle_done": contact_settle_done,
        "final_release": release,
    }


def _maybe_refine_slot(
    *,
    api: Any,
    strategy: SlotPickStrategy,
    config: SlotPickConfig,
    slot_xyz: PoseXYZ,
    slot_observe: PoseXYZR,
    state: CycleState,
) -> tuple[PoseXYZ, dict[str, Any] | None, dict[str, Any] | None]:
    """Optional second-pass place-target refine (OFF by default). Moves straight
    above the first-pass point, re-detects generically, and overrides the target
    xyz only when the refine passes the shift / score / z-drop gates."""
    if not config.enable_slot_refine:
        return slot_xyz, None, None

    requested_z = slot_xyz.z + config.slot_refine_lift_mm
    refine_z = min(slot_observe.z, requested_z)
    z_min = strategy.z_min_safe()
    if z_min is not None:
        refine_z = max(refine_z, z_min + max(0.0, config.slot_refine_min_z_margin_mm))

    state.stage_setter(f"cycle_{state.idx}_slot_refine_move")
    strategy.goto_process(slot_xyz.x, slot_xyz.y, refine_z, slot_observe.r)
    state.stage_setter(f"cycle_{state.idx}_slot_refine_detect")
    refined = _detect_object(api, config.slot_object_name)
    meta: dict[str, Any] = {
        "refine_pose": [slot_xyz.x, slot_xyz.y, refine_z, slot_observe.r],
        "ok": bool(refined.get("ok")),
        "reason": refined.get("reason"),
        "applied": False,
    }
    if refined.get("ok"):
        rpos = _position_from_detection(refined, stage="slot_refine")
        if not isinstance(rpos, dict):
            rx, ry, rz = rpos
            shift = math.hypot(rx - slot_xyz.x, ry - slot_xyz.y)
            refined_score = float(refined.get("score", 0.0))
            z_drop = slot_xyz.z - rz
            reject_reason: str | None = None
            if refined_score < config.slot_refine_min_score:
                reject_reason = "refine_score_too_low"
            elif shift > config.slot_refine_max_shift_mm:
                reject_reason = "refine_shift_too_large"
            elif z_drop > config.slot_refine_max_z_drop_mm:
                reject_reason = "refine_z_too_deep"
            meta.update(
                {
                    "refined_xyz": [rx, ry, rz],
                    "first_xyz": list(slot_xyz),
                    "shift_mm": shift,
                    "refined_score": refined_score,
                    "z_drop_mm": z_drop,
                    "reject_reason": reject_reason,
                }
            )
            if reject_reason is None:
                meta["applied"] = True
                return PoseXYZ(rx, ry, rz), meta, refined
    return slot_xyz, meta, None


# =============================================================================
# Split helpers for run_slot_pick
# =============================================================================
def _init_cycle_poses(
    api: Any,
    config: SlotPickConfig,
) -> dict[str, Any]:
    """Fetch home pose and derive observe/pick/place parameters."""
    home = api.get_home_pose()
    home_x = _coerce_float(home["x"], "home.x")
    home_y = _coerce_float(home["y"], "home.y")
    home_z = _coerce_float(home["z"], "home.z")
    home_r = _coerce_float(home.get("r", home.get("rz", 0.0)), "home.r")

    slot_observe = config.slot_observe_pose_xyzr or (home_x, home_y, home_z, home_r)
    chip_observe = config.chip_observe_pose_xyzr or (home_x, home_y, home_z, home_r)
    sx, sy, sz, sr = slot_observe
    cx_obs, cy_obs, cz_obs, cr_obs = chip_observe
    pick_r = cr_obs + config.chip_pick_r_offset_deg
    place_r = pick_r + config.place_r_delta_deg

    return {
        "home": PoseXYZR(home_x, home_y, home_z, home_r),
        "slot_observe": PoseXYZR(sx, sy, sz, sr),
        "chip_observe": PoseXYZR(cx_obs, cy_obs, cz_obs, cr_obs),
        "pick_r": pick_r,
        "place_r": place_r,
    }


def _do_slot_detection(
    api: Any,
    config: SlotPickConfig,
    strategy: SlotPickStrategy,
    state: CycleState,
) -> dict[str, Any]:
    """Move to slot observe pose, detect slot, optionally refine.
    Returns ``{ok: True, slot_detection, slot_xyz, slot_refine, refined_detection}``
    on success, or a stop-dict on failure."""
    slot_observe = state.slot_observe
    state.stage_setter(f"cycle_{state.idx}_slot_observe")
    strategy.goto_transit(slot_observe.x, slot_observe.y, slot_observe.z, slot_observe.r)
    state.stage_setter(f"cycle_{state.idx}_detect_slot")
    slot_detection = _detect_object(api, config.slot_object_name)
    if not slot_detection.get("ok"):
        return {
            "ok": True,
            "stage": "done_no_slot",
            "reason": str(slot_detection.get("reason", "no_detection")),
            "fallback_recommended": False,
            "stopped_because": "no_place_target",
            "holding_pose": "slot_observe",
            "slot_observe_pose": list(slot_observe),
            "home_pose": state.home_pose,
            "cycles_done": len(state.cycles),
            "cycles": state.cycles,
            "last_slot_detection": slot_detection,
        }
    slot_pos = _position_from_detection(slot_detection, stage=f"cycle_{state.idx}_detect_slot")
    if isinstance(slot_pos, dict):
        slot_pos["cycles_done"] = len(state.cycles)
        slot_pos["cycles"] = state.cycles
        return slot_pos
    slot_xyz = PoseXYZ(*slot_pos)

    slot_xyz, slot_refine, refined_detection = _maybe_refine_slot(
        api=api,
        strategy=strategy,
        config=config,
        slot_xyz=slot_xyz,
        slot_observe=slot_observe,
        state=state,
    )
    if refined_detection is not None:
        slot_detection = refined_detection

    return {
        "ok": True,
        "slot_detection": slot_detection,
        "slot_xyz": slot_xyz,
        "slot_refine": slot_refine,
        "refined_detection": refined_detection,
    }


def _do_chip_detection(
    api: Any,
    config: SlotPickConfig,
    strategy: SlotPickStrategy,
    state: CycleState,
) -> dict[str, Any]:
    """Move to chip observe pose, detect chip, extract position + pick z.
    Returns ``{ok: True, chip_detection, chip_xyz, approach}``
    on success, or a stop-dict on failure."""
    chip_observe = state.chip_observe
    pick_r = state.pick_r
    state.stage_setter(f"cycle_{state.idx}_chip_observe")
    strategy.goto_transit(chip_observe.x, chip_observe.y, chip_observe.z, chip_observe.r)
    state.stage_setter(f"cycle_{state.idx}_detect_chip")
    chip_detection = _detect_object(api, config.chip_object_name)
    if not chip_detection.get("ok"):
        api.home()
        return {
            "ok": True,
            "stage": "done_no_chip",
            "reason": str(chip_detection.get("reason", "no_detection")),
            "fallback_recommended": False,
            "home_pose": state.home_pose,
            "cycles_done": len(state.cycles),
            "cycles": state.cycles,
            "last_chip_detection": chip_detection,
            "last_slot_detection": state.slot_detection,
            "stopped_because": "no_pick_object_detected",
        }
    chip_pos = _position_from_detection(chip_detection, stage=f"cycle_{state.idx}_detect_chip")
    if isinstance(chip_pos, dict):
        chip_pos["slot_detection"] = state.slot_detection
        chip_pos["cycles_done"] = len(state.cycles)
        chip_pos["cycles"] = state.cycles
        return chip_pos
    chip_xyz = PoseXYZ(*chip_pos)
    chip_pick_z = chip_xyz.z + config.chip_pick_z_offset_mm
    chip_above_z = min(chip_observe.z, chip_pick_z + config.chip_approach_hover_mm)
    approach = GraspApproachParams(chip_xyz.x, chip_xyz.y, chip_pick_z, chip_above_z, pick_r)
    return {
        "ok": True,
        "chip_detection": chip_detection,
        "chip_xyz": chip_xyz,
        "approach": approach,
    }


def _do_pick(
    strategy: SlotPickStrategy,
    config: SlotPickConfig,
    approach: GraspApproachParams,
    state: CycleState,
) -> dict[str, Any] | None:
    """Open gripper, approach, descend, grasp, lift.
    Returns None on success, or a stop-dict on failure."""
    return _execute_grasp_sequence(
        strategy,
        config,
        approach,
        stage_prefix=f"cycle_{state.idx}",
        slot_detection=state.slot_detection,
        chip_detection=state.chip_detection,
        extra_stop_kwargs={"cycles_done": len(state.cycles), "cycles": state.cycles},
    )


def run_slot_pick(
    api: Any,
    config: SlotPickConfig,
    strategy: SlotPickStrategy,
) -> dict[str, Any]:
    """Repeated pick→place cycles, body- and task-agnostic.

    Each cycle: detect the place target (``slot_object_name``), detect the pick
    object (``chip_object_name``) — both via the generic ``get_grasp_info_simple``
    — grasp, place, repeat. Stops cleanly when the place target or the pick object
    is no longer detected, or at ``max_pick_place_cycles``.
    """
    cycles: list[dict[str, Any]] = []
    stage = "start"

    def _set_stage(s: str) -> None:
        nonlocal stage
        stage = s

    poses = _init_cycle_poses(api, config)
    home = poses["home"]
    slot_observe = poses["slot_observe"]
    chip_observe = poses["chip_observe"]
    pick_r, place_r = poses["pick_r"], poses["place_r"]
    home_pose = list(home)

    state = CycleState(
        idx=0,
        cycles=cycles,
        home_pose=home_pose,
        stage_setter=_set_stage,
        slot_observe=slot_observe,
        chip_observe=chip_observe,
        pick_r=pick_r,
        place_r=place_r,
    )

    try:
        for cycle_idx in range(1, config.max_pick_place_cycles + 1):
            state.idx = cycle_idx

            r = _do_slot_detection(api, config, strategy, state)
            if "slot_detection" not in r:
                return r
            state.slot_detection = r["slot_detection"]
            slot_xyz = r["slot_xyz"]
            slot_refine = r["slot_refine"]

            r = _do_chip_detection(api, config, strategy, state)
            if "chip_detection" not in r:
                return r
            state.chip_detection = r["chip_detection"]
            chip_xyz = r["chip_xyz"]
            approach = r["approach"]

            r = _do_pick(strategy, config, approach, state)
            if r is not None:
                return r

            stage = f"cycle_{cycle_idx}_place"
            placed = _place_held_object(
                strategy=strategy,
                config=config,
                place_r=place_r,
                slot_xyz=slot_xyz,
                slot_detection=state.slot_detection,
                chip_detection=state.chip_detection,
                stage_prefix=f"cycle_{cycle_idx}",
            )
            if not placed.get("ok"):
                placed["cycles_done"] = len(cycles)
                placed["cycles"] = cycles
                return placed

            cycles.append(
                {
                    "cycle": cycle_idx,
                    "slot_detection": state.slot_detection,
                    "slot_refine": slot_refine,
                    "chip_detection": state.chip_detection,
                    "slot_xyz": list(slot_xyz),
                    "chip_xyz": list(chip_xyz),
                    "chip_pick_xyz": [chip_xyz.x, chip_xyz.y, approach.chip_pick_z],
                    "pick_r": pick_r,
                    "place_r": place_r,
                    **placed,
                }
            )

        return {
            "ok": True,
            "stage": "done_max_cycles",
            "fallback_recommended": False,
            "home_pose": home_pose,
            "cycles_done": len(cycles),
            "cycles": cycles,
            "stopped_because": "max_pick_place_cycles",
        }
    except Exception as exc:  # noqa: BLE001 - return structured result to the LLM
        return _stop(
            stage,
            f"{type(exc).__name__}: {exc}",
            fallback_recommended=stage.endswith("_detect_slot") or stage.endswith("_detect_chip"),
            slot_detection=state.slot_detection,
            chip_detection=state.chip_detection,
        )


# =============================================================================
# Continuous watch loop (a control program, not an LLM tool)
# =============================================================================
def _do_pick_place(
    strategy: SlotPickStrategy,
    config: SlotPickConfig,
    *,
    approach: GraspApproachParams,
    slot_xyz: PoseXYZ,
    place_r: float,
    chip_detection: dict[str, Any],
    slot_detection: dict[str, Any],
    stage_prefix: str = "watch",
) -> dict[str, Any]:
    """One pick-and-place given already-detected object + target positions.
    Mirrors the pick column of ``run_slot_pick`` + ``_place_held_object``."""
    err = _execute_grasp_sequence(
        strategy,
        config,
        approach,
        stage_prefix=stage_prefix,
        slot_detection=slot_detection,
        chip_detection=chip_detection,
    )
    if err is not None:
        return err
    return _place_held_object(
        strategy=strategy,
        config=config,
        place_r=place_r,
        slot_xyz=slot_xyz,
        slot_detection=slot_detection,
        chip_detection=chip_detection,
        stage_prefix=stage_prefix,
    )


def _act_or_idle(obj: dict[str, Any], target: dict[str, Any]) -> tuple[str, Any]:
    """Given the round's detections, decide whether we *can* act:
    ("act", ((ox,oy,oz),(tx,ty,tz))) — both pick object and target have positions
    ("idle", "no_object")            — pick object not detected
    ("idle", "no_place_target")      — place target not detected
    """
    if not obj.get("ok"):
        return ("idle", "no_object")
    op = _position_from_detection(obj, stage="watch_chip")
    if isinstance(op, dict):
        return ("idle", "no_object")
    if not target.get("ok"):
        return ("idle", "no_place_target")
    tp = _position_from_detection(target, stage="watch_slot")
    if isinstance(tp, dict):
        return ("idle", "no_place_target")
    return ("act", (op, tp))


def _watch_decision(
    obj: dict[str, Any],
    target: dict[str, Any],
    place_done_radius_mm: float,
) -> tuple[str, Any]:
    """Geometric vision judgment for one watch round. ``already_done`` when the pick
    object's detected xy is within ``place_done_radius_mm`` of the target's xy (it is
    sitting on the target); otherwise ``act`` / the relevant idle reason."""
    decision = _act_or_idle(obj, target)
    if decision[0] != "act":
        return decision
    op, tp = decision[1]
    ox, oy, _ = op
    tx, ty, _ = tp
    if math.hypot(ox - tx, oy - ty) <= place_done_radius_mm:
        return ("idle", "already_done")
    return ("act", (op, tp))


def geometric_completion_judge(api: Any, config: SlotPickConfig) -> bool:
    """Default completion judge: detect the pick object + place target and return
    True when the pick object is already sitting on the target (xy within
    ``config.place_done_radius_mm``). Usable as ``is_task_complete`` and as the
    fallback for a VLM judge.
    """
    obj = _detect_object(api, config.chip_object_name)
    target = _detect_object(api, config.slot_object_name)
    return _watch_decision(obj, target, config.place_done_radius_mm)[1] == "already_done"


def run_watch_pick_place(
    api: Any,
    config: SlotPickConfig,
    strategy: SlotPickStrategy,
    *,
    poll_interval_s: float = 1.0,
    max_rounds: int | None = None,
    should_continue: Callable[[], bool] | None = None,
    on_status: Callable[[dict[str, Any]], None] | None = None,
    is_task_complete: Callable[[Any, SlotPickConfig], bool] | None = None,
) -> dict[str, Any]:
    """Continuously watch from the initial (home) observe pose and pick-and-place
    only while the task is NOT yet complete.

    Each round returns to the **single initial pose** (``api.home()`` — e.g. piper's
    operator-set ``home_use_init_pose``), observes ONCE, and judges whether the task
    is already complete. ``is_task_complete(api, config) -> bool`` is the pluggable
    judge: default ``None`` uses the geometric judge (pick object's xy within
    ``place_done_radius_mm`` of the target's xy); pass a VLM judge (see
    ``make_vlm_completion_judge``) for a real visual-understanding answer. The three
    outcomes are:

      * pick object already ON the place target (within ``place_done_radius_mm``) →
        task complete → keep observing (do nothing);
      * pick object present elsewhere → pick it and place it on the target;
      * pick object / place target not detected → keep observing.

    So after a placement the object sits on the target → judged complete → the arm
    holds and waits; once you move the object off the target it is no longer "on Y"
    → the next round picks-and-places it again. At startup with nothing in view it
    just waits at the initial pose. The config's observe poses are NOT used here —
    detection is always at home.

    Runs forever by default; stop via ``KeyboardInterrupt`` / ``should_continue`` /
    ``max_rounds``. Returns a summary ``{rounds, placements, ...}``.
    """
    rounds = 0
    placements = 0
    try:
        while True:
            if should_continue is not None and not should_continue():
                break
            if max_rounds is not None and rounds >= max_rounds:
                break
            rounds += 1
            try:
                api.home()
            except Exception:  # noqa: BLE001 - return-to-initial is best-effort
                logger.warning("[watch] home failed", exc_info=True)
            home = api.get_home_pose()
            home_r = _coerce_float(home.get("r", home.get("rz", 0.0)), "home.r")
            observe_z = _coerce_float(home["z"], "home.z")
            target = _detect_object(api, config.slot_object_name)
            obj = _detect_object(api, config.chip_object_name)
            if is_task_complete is not None:
                action, payload = (
                    ("idle", "already_done") if is_task_complete(api, config) else _act_or_idle(obj, target)
                )
            else:
                action, payload = _watch_decision(obj, target, config.place_done_radius_mm)
            result: dict[str, Any] | None = None
            if action == "act":
                (ox, oy, oz), (tx, ty, tz) = payload
                pick_r = home_r + config.chip_pick_r_offset_deg
                place_r = pick_r + config.place_r_delta_deg
                chip_pick_z = oz + config.chip_pick_z_offset_mm
                chip_above_z = min(observe_z, chip_pick_z + config.chip_approach_hover_mm)
                approach = GraspApproachParams(ox, oy, chip_pick_z, chip_above_z, pick_r)
                result = _do_pick_place(
                    strategy,
                    config,
                    approach=approach,
                    slot_xyz=PoseXYZ(tx, ty, tz),
                    place_r=place_r,
                    chip_detection=obj,
                    slot_detection=target,
                )
                if result.get("ok"):
                    placements += 1
            if on_status is not None:
                on_status(
                    {
                        "round": rounds,
                        "action": action,
                        "reason": None if action == "act" else payload,
                        "placements": placements,
                        "result": result,
                    }
                )
            if action != "act":
                time.sleep(max(0.0, poll_interval_s))
    except KeyboardInterrupt:
        return {"rounds": rounds, "placements": placements, "stopped": "interrupt"}
    return {"rounds": rounds, "placements": placements, "stopped": "condition"}


# =============================================================================
# Tool wrapper (openjiuwen Tool / local fallback)
# =============================================================================
def _tool_description() -> str:
    """Return the user-facing description for the SlotPickSkillTool card."""
    return (
        "Executable loop that repeatedly picks an object and places it at a target "
        "until the pick object is no longer detected (or max_pick_place_cycles). "
        "Detection is generic and task-agnostic: pass chip_object_name (what to "
        "pick) and slot_object_name (where to place) — the words from the user's "
        "task. If ok=False and fallback_recommended=True the failure was in vision; "
        "otherwise stop and report stage/reason."
    )


def _filter_runtime_overrides(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Filter runtime overrides from agent payload; currently no blocked fields."""
    return dict(payload)


class SlotPickSkillTool(Tool):
    def __init__(
        self,
        api: Any,
        strategy: SlotPickStrategy,
        config: SlotPickConfig,
        *,
        name: str = "slot_pick",
        agent_id: str | None = None,
    ) -> None:
        """Initialize the slot-pick tool with api, strategy, and config."""
        self._api = api
        self._strategy = strategy
        self._config = config
        tool_id = f"{name}_{agent_id}" if agent_id else f"{name}_{uuid.uuid4().hex}"
        card = ToolCard(
            id=tool_id,
            name=name,
            description=_tool_description(),
            input_params=_input_params(),
        )
        super().__init__(card)

    async def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> ToolOutput:
        """Execute one slot-pick run; return a ToolOutput with structured result data."""
        payload = inputs or {}
        if not isinstance(payload, dict):
            return ToolOutput(success=False, error="inputs must be an object")
        try:
            config = self._config.merged(_filter_runtime_overrides(payload))
            result = run_slot_pick(self._api, config, self._strategy)
            if inspect.isawaitable(result):
                result = await result
            return ToolOutput(success=True, data=result)
        except Exception as exc:  # noqa: BLE001
            return ToolOutput(success=False, error=f"{type(exc).__name__}: {exc}")

    async def stream(self, inputs: dict[str, Any], **kwargs: Any) -> AsyncIterator[Any]:
        """Streaming interface (no-op; required by the Tool base class)."""
        if False:
            yield None


def build_slot_pick_tool(
    api: Any,
    strategy: SlotPickStrategy,
    config: SlotPickConfig,
    *,
    name: str = "slot_pick",
    agent_id: str | None = None,
) -> Any:
    """Build the openjiuwen ``slot_pick`` Tool bound to ``api`` + ``strategy``.

    ``strategy`` is typically ``GripperStrategy(api, ...)`` for a 6-DoF + parallel
    gripper robot (e.g. piper).
    """
    return SlotPickSkillTool(api, strategy, config, name=name, agent_id=agent_id)


__all__ = [
    "GripperStrategy",
    "SlotPickConfig",
    "SlotPickSkillTool",
    "build_slot_pick_tool",
    "geometric_completion_judge",
    "run_slot_pick",
    "run_watch_pick_place",
]
