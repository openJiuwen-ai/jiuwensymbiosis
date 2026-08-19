# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The shared action vocabulary — what an action *is*, separately from how a body does it.

A planner (or a coding agent authoring a skill) reasons over action **names and
contracts**. If every body were free to invent both, nothing it learned on one
robot would transfer to the next, and a SKILL.md could not be written once and run
anywhere. So the vocabulary is closed and lives here, next to its siblings
``env/base.py:KNOWN_CAPABILITIES`` and ``api/state.py:KNOWN_STATE_TOKENS``:

    one action = one ActionSpec (the contract, body-agnostic) + N body implementations.

Before this module the contract was re-declared on every implementation — a mixin
wrote ``@robot_tool(desc=…, requires=…, provides=…)`` and each adapter that
overrode the method wrote it again. 20 of 39 action names carried 2–4 copies, and
they had already drifted: ``move_joint`` meant "the whole joint vector" on one body
and "one shoulder pitch value" on another. A spec cannot drift because there is
only ever one of it.

**Capability gating reads the spec, not the class.** Which capability an action
belongs to is a property of the action, so it is declared here once. That also
removes a whole failure mode: gating used to be resolved by walking the MRO for
whichever class happened to declare a ``capability`` attribute, which silently
gated every tool an adapter declared alongside its vision tools.

**Params.** A spec lists the parameter *names* a planner may use and which of them
are mandatory; the JSON Schema (types, defaults) is still derived from the
implementation's signature, where the type checker already sees it.
``implements()`` verifies at import time that an implementation accepts every
contract param — a body may add optional params of its own, never drop one.

Adding an action:
  1. Add its ``ActionSpec`` here (name, capability, params, result shape, contract).
  2. Implement it with ``@implements(THE_SPEC)`` on a mixin or an adapter.
  3. Nothing else — tool emission, prompt rendering and sequence validation all
     read the resulting ``ToolMeta`` exactly as before.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from jiuwensymbiosis.api.decorators import ToolMeta, schema_from_signature, schema_from_typeddict
from jiuwensymbiosis.api.state import validate_tokens
from jiuwensymbiosis.env.base import KNOWN_CAPABILITIES
from jiuwensymbiosis.motion.approach import (
    ApproachFailure,
    ApproachResult,
    SearchResult,
)
from jiuwensymbiosis.perception.frame import BasePoint
from jiuwensymbiosis.perception.scene3d import (
    SPATIAL_RELATIONS,
    ObjectGeometryResult,
    SceneScanResult,
    SensingFailure,
    SurfaceGeometryResult,
)
from jiuwensymbiosis.perception.vision import GraspFailure, GraspResult


class UnknownCapability(ValueError):
    """A spec declared a capability outside ``KNOWN_CAPABILITIES``."""


@dataclass(frozen=True)
class ActionSpec:
    """What one action is: its name, gate, parameters, result shape and contract.

    Attributes:
        name: the action name a plan writes as ``op``. Unique across the vocabulary.
        description: what the action does, what it needs and what it establishes.
            This is the ONLY text a planner sees, so it must stay true on every body:
            an implementation has no channel for adding prose of its own.
        capability: the capability gate. ``None`` = every body has it (``home``).
        params: parameter names a planner may pass.
        required_params: the subset that must be passed. Everything else is optional.
        param_schema: per-parameter JSON Schema refining what the signature can express
            — e.g. the field names inside a ``pose`` dict, which every 6-DoF body agrees
            on and a bare ``dict`` annotation would lose. Merged over the derived schema.
        result: a ``TypedDict`` (or a union of them) describing the result fields, so
            a later step can read ``<bind>.field`` and the validator can reject a
            field that does not exist. ``None`` = shape unknown → field checking is
            skipped rather than everything being rejected.
        requires / provides / invalidates: robot self-state, over the closed
            vocabulary in ``api/state.py``.
        produces_location / consumes_location / invalidates_locations: location
            freshness (see ``api/state.py``).
        opens_access / closes_access: this action clears (or restores) whatever blocks
            the thing it acts on — a door, a lid, a stacked crate, a lock. Only the
            barrier side declares anything; which steps reach *through* a barrier is
            derived from the sequence (see ``api/state.py``).
        tags: rail triggers (``VisualFeedbackRail`` / ``RecoveryRail`` watch these).
            NOT a visibility mechanism — that is ``planner_visible``.
        planner_visible: whether the action enters the planner's vocabulary. ``False``
            for bring-up / calibration / diagnostic actions, which stay callable
            through ``robot_control`` and from scripts but must not compete for the
            planner's attention.
    """

    name: str
    description: str
    capability: str | None = None
    params: tuple[str, ...] = ()
    required_params: tuple[str, ...] = ()
    param_schema: Mapping[str, Any] | None = None
    result: Any = None
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    produces_location: bool = False
    consumes_location: bool = False
    invalidates_locations: bool = False
    opens_access: bool = False
    closes_access: bool = False
    tags: tuple[str, ...] = ()
    planner_visible: bool = True

    def __post_init__(self) -> None:
        """Validate the spec against the closed vocabularies it draws on."""
        if self.capability is not None and self.capability not in KNOWN_CAPABILITIES:
            raise UnknownCapability(
                f"action {self.name!r} declares unknown capability {self.capability!r}. "
                f"Add it to KNOWN_CAPABILITIES in jiuwensymbiosis/env/base.py first."
            )
        for field_name in ("requires", "provides", "invalidates"):
            validate_tokens(getattr(self, field_name), field=field_name, owner=f"action {self.name}")
        unknown = set(self.required_params) - set(self.params)
        if unknown:
            raise ValueError(f"action {self.name!r}: required_params {sorted(unknown)} are not in params")
        if self.produces_location and not schema_from_typeddict(self.result).get("properties"):
            raise ValueError(
                f"action {self.name!r} produces a location but declares no readable result shape "
                f"(result={self.result!r}). A plan could not read any field off it, and the sequence "
                f"validator would silently stop checking the fields it invents. Declare a TypedDict."
            )


def _register(*specs: ActionSpec) -> dict[str, ActionSpec]:
    """Index specs by name, rejecting a duplicate rather than silently overwriting."""
    out: dict[str, ActionSpec] = {}
    for spec in specs:
        if spec.name in out:
            raise ValueError(f"duplicate action spec {spec.name!r}")
        out[spec.name] = spec
    return out


# =============================================================================
# Cartesian motion — motion.cartesian
# =============================================================================
GOTO_XYZR = ActionSpec(
    name="goto_xyzr",
    description=(
        "Move the end-effector TIP to absolute (x, y, z[, r]) in mm/deg, base frame. If r is omitted, "
        "the current r is preserved. This is the restricted form of goto_pose: the tool points "
        "down and only the yaw r is yours to choose. Need a different orientation → goto_pose."
    ),
    capability="motion.cartesian",
    params=("x", "y", "z", "r"),
    required_params=("x", "y", "z"),
    invalidates=("body.home",),
    tags=("motion",),
)

GOTO_POSE = ActionSpec(
    name="goto_pose",
    description="Move the end-effector TIP to an absolute 6-DoF pose (x, y, z in mm; rx, ry, rz in deg), "
        "base frame — the same point goto_xyzr and get_pose speak about, so a pose read from one "
        "can be commanded to the other. Use it when the ORIENTATION matters; when top-down with a "
        "yaw will do, goto_xyzr says the same thing with fewer numbers to get wrong. On an arm "
        "with fewer joints than the pose demands, position is enforced and orientation is "
        "best-effort.",
    capability="motion.cartesian",
    params=("pose",),
    required_params=("pose",),
    # The pose dict's own fields. Pinning them here is not cosmetic: the two 6-DoF bodies
    # had drifted onto different key sets (``x_mm/rx_deg`` vs ``x/rx``), so the same plan
    # crashed on one of them — and SafetyRail only ever unpacked ``x/y/z``, silently
    # skipping the bounds check on the other spelling. One contract fixes both.
    param_schema={
        "pose": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "mm, base frame"},
                "y": {"type": "number", "description": "mm, base frame"},
                "z": {"type": "number", "description": "mm, base frame"},
                "rx": {"type": "number", "description": "deg"},
                "ry": {"type": "number", "description": "deg"},
                "rz": {"type": "number", "description": "deg"},
            },
            "required": ["x", "y", "z", "rx", "ry", "rz"],
        }
    },
    invalidates=("body.home",),
    tags=("motion",),
)

MOVE_DIRECTION = ActionSpec(
    name="move_direction",
    description=(
        "Move the end-effector a relative distance in one cardinal direction. "
        "direction ∈ {forward, back, left, right, up, down} (forward=+x, left=+y, up=+z, base frame); "
        "distance_mm is a positive number of millimetres. Orientation is preserved."
    ),
    capability="motion.cartesian",
    params=("direction", "distance_mm"),
    required_params=("direction", "distance_mm"),
    invalidates=("body.home",),
    tags=("motion",),
)

GET_POSE = ActionSpec(
    name="get_pose",
    description="Get the current end-effector TIP pose in mm/deg, base frame — the same point goto_xyzr and "
        "goto_pose command, so a pose read here can be handed straight back to either.",
    capability="motion.cartesian",
    params=(),
)

GET_HOME_POSE = ActionSpec(
    name="get_home_pose",
    description="Get this robot's home-pose constants (read-only).",
    capability="motion.cartesian",
    params=(),
    planner_visible=False,  # a constant, not an observation: nothing to plan around
)

# =============================================================================
# Posture — no capability: every body owes a way back to a safe pose
# =============================================================================
HOME = ActionSpec(
    name="home",
    description=(
        "Return the body to its safe home posture, doing the minimum motion the current state "
        "needs (already home → does nothing). Always safe to call; use it to wrap up a workflow."
    ),
    capability=None,
    params=(),
    provides=("body.home",),
    tags=("motion",),
)
# There is deliberately no second "home_safely" action. Safe homing is ONE thing a body
# owes; how much motion it takes (a 6-DoF arm folding up, a dual-arm torso straightening
# its lifter and neutralising its waist first) is implementation, not a different action.
# Two names for it only gave the planner something to get wrong — and on the one body that
# had both, ``home`` was literally ``return self.home_safely()``.

# =============================================================================
# Joint motion — motion.joint
# =============================================================================
MOVE_JOINT = ActionSpec(
    name="move_joint",
    description=(
        "Move to a joint configuration: q is the FULL joint vector, in the robot's own "
        "convention (rad or deg). To drive a single named joint use move_named_joint."
    ),
    capability="motion.joint",
    params=("q",),
    required_params=("q",),
    invalidates=("body.home",),
    tags=("motion",),
)

MOVE_NAMED_JOINT = ActionSpec(
    name="move_named_joint",
    description="Move ONE named joint to an absolute position in radians.",
    capability="motion.joint",
    params=("joint_name", "position_rad"),
    required_params=("joint_name", "position_rad"),
    invalidates=("body.home",),
    tags=("motion",),
    planner_visible=False,  # joint-level bring-up; task plans work in task space
)

GET_JOINT_POSITIONS = ActionSpec(
    name="get_joint_positions",
    description="Read the latest known joint positions, keyed by joint name.",
    capability="motion.joint",
    params=(),
    planner_visible=False,  # diagnostic; WorldState already carries joints into the prompt
)

# =============================================================================
# End effector — grasp.parallel / grasp.suction
# =============================================================================
OPEN_GRIPPER = ActionSpec(
    name="open_gripper",
    description="Open the parallel gripper, releasing whatever is held. width_mm is a HINT: a gripper "
        "with no width control accepts it and ignores it.",
    capability="grasp.parallel",
    params=("width_mm",),
    provides=("payload.clear",),
    tags=("grasp",),
)

CLOSE_GRIPPER = ActionSpec(
    name="close_gripper",
    description="Close the parallel gripper onto the target. Call it only once the tip is at the grasp "
        "pose. force_n is a HINT: a gripper with no force control accepts it and ignores it.",
    capability="grasp.parallel",
    params=("force_n",),
    requires=("payload.clear",),
    provides=("payload.held",),
    tags=("grasp",),
)

ACTIVATE_SUCTION = ActionSpec(
    name="activate_suction",
    description="Turn suction ON. Call it only once the tip is on/near the target.",
    capability="grasp.suction",
    params=(),
    requires=("payload.clear",),
    provides=("payload.held",),
    tags=("grasp",),
)

DEACTIVATE_SUCTION = ActionSpec(
    name="deactivate_suction",
    description="Turn suction OFF — releases whatever is held.",
    capability="grasp.suction",
    params=(),
    provides=("payload.clear",),
    tags=("grasp",),
)

# =============================================================================
# Camera — vision.camera (a frame grab needs a camera, NOT a detector)
# =============================================================================
GET_IMAGE = ActionSpec(
    name="get_image",
    description="Grab the latest RGB frame as a numpy HxWx3 array.",
    capability="vision.camera",
    params=(),
    planner_visible=False,  # raw pixels are not a plannable quantity
)

PIXEL_TO_BASE_XYZ = ActionSpec(
    name="pixel_to_base_xyz",
    description="Project a pixel (u, v) at a known depth to base-frame XYZ in mm. Requires a loaded calibration.",
    capability="vision.camera",
    params=("u", "v", "depth_m"),
    required_params=("u", "v", "depth_m"),
    result=BasePoint,
    produces_location=True,
    planner_visible=False,  # a lower-level primitive than the detect actions a plan should use
)

# =============================================================================
# Detection — vision.detection
# =============================================================================
GET_GRASP_INFO_SIMPLE = ActionSpec(
    name="get_grasp_info_simple",
    description=(
        "One-shot: detect object_name in the live frame and project it to base XYZ via depth + "
        "calibration, returning a ready-to-use grasp height (grasp_z) and place height (place_z) "
        "for a single gripper. Descend straight to grasp_z — do not re-derive it from position."
    ),
    capability="vision.detection",
    params=("object_name",),
    required_params=("object_name",),
    result=GraspResult | GraspFailure,
    produces_location=True,
    tags=("vision",),
)

LOCATE_FOR_GRASP = ActionSpec(
    name="locate_for_grasp",
    description=(
        "Measure a thing and return what PICKING IT UP needs: base-frame centre, width, height, "
        "near face, top, and the face normal to square up to (mm). Failure → {ok: False, reason}. "
        "Pick this over locate_for_place by what you are about to do, NOT by what the thing is — "
        "the same box is measured this way to carry it and the other way to stack onto it. "
        "Optional reference=<another object> + relation: accept ONLY the target standing in that "
        "spatial relation to the reference — use it when the task names several things of the same "
        "kind, e.g. 'the white box on the brown table' → object_name='white box', "
        "reference='brown table', relation='on'; 'the box beside the hat' → object_name='box', "
        "reference='hat', relation='beside'; 'the apple in the drawer' → object_name='apple', "
        "reference='drawer', relation='in'. relation is one of on / under / in / beside / near, read "
        "as 'object_name <relation> reference', and defaults to on. Pass reference ONLY when the "
        "task actually names one — a reference the scene does not contain fails the measurement "
        "rather than falling back to the plain search."
    ),
    capability="vision.detection",
    params=("object_name", "reference", "relation"),
    param_schema={"relation": {"type": "string", "enum": list(SPATIAL_RELATIONS), "default": "on"}},
    result=ObjectGeometryResult | SensingFailure,
    produces_location=True,
    tags=("vision",),
)

LOCATE_FOR_PLACE = ActionSpec(
    name="locate_for_place",
    description=(
        "Measure a thing and return what PUTTING SOMETHING ON IT needs: the landing height "
        "surface_z_mm, the XY footprint the payload must fit inside, and the near-edge line to "
        "square the base to (mm). object_name defaults to 'table'. Failure → {ok: False, reason}. "
        "**A table, a shelf and another box all use this** — what decides it is that you are about "
        "to place, not what the thing is called. Optional reference=<another object> + relation, "
        "read the same way round as in locate_for_grasp — 'object_name <relation> reference'. So "
        "the surface WITH a cup on it is the surface UNDER the cup: object_name='table', "
        "reference='water cup', relation='under'. relation is one of on / under / in / beside / near "
        "and defaults to on."
    ),
    capability="vision.detection",
    params=("object_name", "reference", "relation"),
    param_schema={"relation": {"type": "string", "enum": list(SPATIAL_RELATIONS), "default": "on"}},
    result=SurfaceGeometryResult | SensingFailure,
    produces_location=True,
    tags=("vision",),
)

ANALYZE_SCENE = ActionSpec(
    name="analyze_scene",
    description=(
        "Scan the scene for EVERY instance of object_name, returning each one's base-frame 3-D "
        "position and distance, nearest first. This is the whole-scene view a plan reads before "
        "committing — it tells you HOW MANY there are, which is what a repeat-until-clear task "
        "needs. For a single target use locate_for_grasp instead."
    ),
    capability="vision.detection",
    params=("object_name",),
    result=SceneScanResult | SensingFailure,
    tags=("vision",),
)

# =============================================================================
# Mobile base — motion.base
# =============================================================================
NAVIGATE_RELATIVE = ActionSpec(
    name="navigate_relative",
    description=(
        "Move the base to a different PLACE: turn by dyaw_rad (+ = left) then advance dx_m metres "
        "(REP-103). A differential base cannot strafe, so dy_m is ignored. dx_m is required "
        "because this action changes where the base stands — to only change which way it faces, "
        "use rotate_base, whose signature cannot translate at all."
    ),
    capability="motion.base",
    params=("dx_m", "dy_m", "dyaw_rad"),
    required_params=("dx_m",),
    invalidates_locations=True,
    tags=("motion",),
)

ROTATE_BASE = ActionSpec(
    name="rotate_base",
    description=(
        "Turn the base in place by dyaw_rad (+ = left): changes the HEADING only, never the "
        "position. It has no translation parameter, so it cannot drive the base away — prefer it "
        "in tight spaces, and over navigate_relative(dx_m=0, ...), where one wrong dx_m really "
        "does drive off. It still stales previously sensed positions: coordinates are base-frame, "
        "and turning moves that frame."
    ),
    capability="motion.base",
    params=("dyaw_rad",),
    required_params=("dyaw_rad",),
    invalidates_locations=True,
    tags=("motion",),
)

DRIVE_ARC = ActionSpec(
    name="drive_arc",
    description=(
        "Drive ONE constant-curvature arc: radius_m (m) and dyaw_rad (rad, + = left). Turns while "
        "advancing, so the base ends up off its original heading line."
    ),
    capability="motion.base",
    params=("radius_m", "dyaw_rad"),
    required_params=("radius_m", "dyaw_rad"),
    invalidates_locations=True,
    tags=("motion",),
    planner_visible=False,  # bring-up / calibration primitive: no perception, keep clear of obstacles
)

# =============================================================================
# Torso — motion.lift / motion.waist
# =============================================================================
SET_LIFT_POSE = ActionSpec(
    name="set_lift_pose",
    description="Set the torso lifter joints to q_lifter (absolute rad per joint).",
    capability="motion.lift",
    params=("q_lifter",),
    required_params=("q_lifter",),
    invalidates=("body.home",),
    tags=("motion",),
    planner_visible=False,  # joint-level; plans use lift_to_clearance, which knows the safe height
)

LIFT_TO_CLEARANCE = ActionSpec(
    name="lift_to_clearance",
    description=(
        "After grasping, raise the held payload to the body's preset travel clearance height and "
        "stand the torso upright, so it can be carried/turned without dragging or blocking the "
        "cameras. Refuses without moving when that height is unreachable."
    ),
    capability="motion.lift",
    params=(),
    requires=("payload.held",),
    provides=("payload.stowed",),
    invalidates=("body.home",),
    tags=("motion",),
)

TURN_WAIST = ActionSpec(
    name="turn_waist",
    description=(
        "Rotate the torso waist by delta_rad (+ = left), holding the arms' current posture. "
        "A waist yaw leaves the base frame fixed, so previously sensed base-frame positions stay valid."
    ),
    capability="motion.waist",
    params=("delta_rad",),
    required_params=("delta_rad",),
    invalidates=("body.home",),
    tags=("motion",),
)

# =============================================================================
# Approach — motion.goal
# =============================================================================
APPROACH_FOR_GRASP = ActionSpec(
    name="approach_for_grasp",
    description=(
        "Find the target and drive the base to a pose the arms can PICK IT UP from — closer than "
        "placing distance, squared to the target's own face. To go and put something ON that thing "
        "instead, use approach_for_place: the choice is what you are about to do, not what the "
        "thing is. Not in view → it sweeps in place to "
        "search; then it squares up to the SENSED bearing and closes in, re-measuring each pass "
        "until the target is centred and inside the working band, so it ends at a workable pose "
        "whether the target started ahead, left or right. On success the converged measurement is "
        "cached for the grasping step: do NOT add a separate locate before grasping. Optional "
        "reference + relation, exactly as in locate_for_grasp, to pin down WHICH one when the task "
        "names several of the same kind. On failure (object_not_found / too_close / "
        "lost_after_move / lidar_blocked / ...) the caller MUST NOT grasp anyway."
    ),
    capability="motion.goal",
    params=("object_name", "reference", "relation"),
    param_schema={"relation": {"type": "string", "enum": list(SPATIAL_RELATIONS), "default": "on"}},
    result=ApproachResult | ApproachFailure,
    produces_location=True,
    invalidates_locations=True,
    tags=("motion", "vision"),
)

APPROACH_FOR_PLACE = ActionSpec(
    name="approach_for_place",
    description=(
        "Find the thing you are about to put something ON, and drive the base to within the arms' "
        "PLACING distance — further back than grasping distance (the arms must reach over its near "
        "edge), squared to that edge. To go and pick that thing UP instead, use approach_for_grasp. "
        "Not in view → it sweeps in place to search. It moves ONLY "
        "when out of reach: already within placing range means no motion at all "
        "(status=in_range). **A table, a shelf and another box all use this** — what decides it is "
        "that you are about to place. Optional reference + relation, exactly as in "
        "locate_for_place, to pin down WHICH one. Any motion stales earlier sensing, so re-run "
        "locate_for_place before placing. On failure (surface_not_found / nav_failed / ...) the "
        "caller MUST NOT place anyway."
    ),
    capability="motion.goal",
    params=("object_name", "reference", "relation"),
    param_schema={"relation": {"type": "string", "enum": list(SPATIAL_RELATIONS), "default": "on"}},
    result=ApproachResult | ApproachFailure,
    produces_location=True,
    invalidates_locations=True,
    tags=("motion", "vision"),
)

SEARCH_TARGET = ActionSpec(
    name="search_target",
    description=(
        "Take one look at the current heading and report whether object_name is in view and at "
        "what BEARING (bearing_rad, + = left). It moves nothing and reports no distance — a "
        "DIRECTION, not a position, so a grasping or placing step cannot act on it. For "
        "coordinates use locate_for_grasp / locate_for_place; to actually go there use "
        "approach_for_grasp / approach_for_place. The typical use is feeding bearing_rad to "
        "rotate_base to turn towards it first. Optional reference + relation as in "
        "locate_for_grasp, though a single 2-D look can only screen the 'on' relation; any other "
        "relation is left to the metric measurement up close."
    ),
    capability="vision.search",
    params=("object_name", "reference", "relation"),
    param_schema={"relation": {"type": "string", "enum": list(SPATIAL_RELATIONS), "default": "on"}},
    result=SearchResult,
    # NOT produces_location: the result carries a bearing, no coordinate. Claiming a
    # location would let the validator accept search_target → dual_arm_grasp, which then
    # reads an empty cache and fails on the robot — a compile-time pass for a run-time bug.
    # NOT invalidates_locations either: reading one frame moves nothing.
    tags=("vision",),
)

# =============================================================================
# Dual-arm grasp — grasp.dual_arm
# =============================================================================
DUAL_ARM_GRASP = ActionSpec(
    name="dual_arm_grasp",
    description=(
        "Clamp an ALREADY-DETECTED target between both arms and confirm the grip by force. "
        "Does NOT detect and does NOT lift: run a detection or approach step first (omit the "
        "argument to use the most recent one), then lift_to_clearance. No detection → "
        "{ok:False, reason:'no_detection'}; no contact force → does not lift."
    ),
    capability="grasp.dual_arm",
    params=("target",),
    requires=("payload.clear",),
    provides=("payload.held",),
    invalidates=("body.home",),
    consumes_location=True,
    tags=("motion",),
)

DUAL_ARM_PLACE = ActionSpec(
    name="dual_arm_place",
    description=(
        "Release a two-arm-held payload onto a sensed support surface and withdraw the arms: "
        "move over a landing point that fits, lower, release, raise. Sense the surface first "
        "(omit the arguments to use the most recent sensing); returning to a safe posture is left to home."
    ),
    capability="grasp.dual_arm",
    params=("target", "surface"),
    requires=("payload.held",),
    provides=("payload.clear",),
    invalidates=("body.home",),
    consumes_location=True,
    tags=("motion",),
)


ACTIONS: Mapping[str, ActionSpec] = _register(
    GOTO_XYZR,
    GOTO_POSE,
    MOVE_DIRECTION,
    GET_POSE,
    GET_HOME_POSE,
    HOME,
    MOVE_JOINT,
    MOVE_NAMED_JOINT,
    GET_JOINT_POSITIONS,
    OPEN_GRIPPER,
    CLOSE_GRIPPER,
    ACTIVATE_SUCTION,
    DEACTIVATE_SUCTION,
    GET_IMAGE,
    PIXEL_TO_BASE_XYZ,
    GET_GRASP_INFO_SIMPLE,
    LOCATE_FOR_GRASP,
    LOCATE_FOR_PLACE,
    ANALYZE_SCENE,
    NAVIGATE_RELATIVE,
    ROTATE_BASE,
    DRIVE_ARC,
    SET_LIFT_POSE,
    LIFT_TO_CLEARANCE,
    TURN_WAIST,
    APPROACH_FOR_GRASP,
    APPROACH_FOR_PLACE,
    SEARCH_TARGET,
    DUAL_ARM_GRASP,
    DUAL_ARM_PLACE,
)


class ContractViolation(TypeError):
    """An implementation's signature does not accept everything its spec promises."""


def _accepts(func: Callable[..., Any], params: Iterable[str]) -> list[str]:
    """Parameter names from ``params`` that ``func`` cannot be called with."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return []  # un-introspectable (C function / partial): trust it
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return []  # **kwargs swallows anything
    return [name for name in params if name not in sig.parameters]


def implements(spec: ActionSpec) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Bind a method as one body's implementation of ``spec``.

    The contract — name, description, capability, params, result shape, pre-conditions
    and effects — comes from the spec and cannot be extended here, not even with prose.
    That is the whole point: a contract with two authors is a contract that drifts.

    So a body has NO channel for telling the planner something about itself. If a fact
    changes what a plan should be and is true of only one robot, the action does not mean
    the same thing there — fix that at the contract (tighten the spec, or have that body
    implement a different action), never by annotating one implementation. If the fact is
    true of every robot, it belongs in the spec description and every body gets it free.
    Everything else — how the motion is produced, which joints move, what the driver does
    — is implementation, and a planner neither needs it nor should be spending attention
    on it.

    Raises:
        ContractViolation: if the signature cannot accept a parameter the spec
            promises. Raised at import time, so a mismatch can never reach the robot.
    """

    def _wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        missing = _accepts(func, spec.params)
        if missing:
            raise ContractViolation(
                f"{func.__qualname__} implements action {spec.name!r} but does not accept "
                f"{missing}; the spec promises a planner may pass {list(spec.params)}. "
                f"Add the parameter (a body may ignore it) or change the spec."
            )
        # Advertise exactly the contract: types/defaults come from the signature (where the
        # type checker sees them), but a param this body added on its own stays unadvertised
        # so the planner cannot come to depend on something another body lacks.
        schema = schema_from_signature(func)
        props = {k: v for k, v in (schema.get("properties") or {}).items() if k in spec.params}
        for name, refinement in (spec.param_schema or {}).items():
            if name in props:
                props[name] = {**props[name], **refinement}
        input_params: dict[str, Any] = {"type": "object", "properties": props}
        if spec.required_params:
            input_params["required"] = list(spec.required_params)
        func.__robot_tool__ = ToolMeta(  # type: ignore[attr-defined]
            name=spec.name,
            description=spec.description,
            input_params=input_params,
            capability=spec.capability,
            tags=list(spec.tags),
            returns=schema_from_typeddict(spec.result),
            requires=spec.requires,
            provides=spec.provides,
            invalidates=spec.invalidates,
            produces_location=spec.produces_location,
            consumes_location=spec.consumes_location,
            invalidates_locations=spec.invalidates_locations,
            opens_access=spec.opens_access,
            closes_access=spec.closes_access,
            planner_visible=spec.planner_visible,
        )
        return func

    return _wrap


def planner_vocabulary(capabilities: Iterable[str]) -> dict[str, ActionSpec]:
    """The specs a planner may use on a body with ``capabilities``.

    The gate is the spec's own capability, so this is exactly what a body advertises
    minus the bring-up / diagnostic actions.
    """
    caps = set(capabilities)
    return {
        name: spec
        for name, spec in ACTIONS.items()
        if spec.planner_visible and (spec.capability is None or spec.capability in caps)
    }
