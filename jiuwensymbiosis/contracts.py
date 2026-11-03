# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Action result types — the data contract between ActionSpec and implementations.

This module is the **single source of truth** for what shape each action returns.

It sits at the package root, belonging to NEITHER side, because both sides read it:
``api/actions.py`` promises these shapes, and ``perception/`` / ``motion/`` build
them. Putting it under ``api/`` would only reverse the old problem — the algorithm
libraries would then import the api package (which imports them back through
``api/components.py``). Owned by no layer, it imports nothing from the package, so
neither direction exists and there is no cycle to reason about. That is the module's
one invariant: **keep this file dependency-free.**

**What goes here:**
  - Result ``TypedDict`` types returned by actions (success + failure shapes)
  - Spatial relation constants read by multiple layers
  - Minimal geometry types that cross module boundaries (``BasePoint``)

**What does NOT go here:**
  - Algorithm implementations (those stay in ``motion/``, ``perception/``)
  - Internal data structures used only within one implementation module
  - Configuration classes (those stay in ``adapters/*/config.py``)

The types here are pure data contracts — no methods, no behavior, no dependencies
on hardware or drivers. A ``TypedDict`` is preferred over a class because it is
already the JSON-like dict the LLM and the validator read; a dataclass would only
add a construction step with no runtime benefit.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

__all__ = [
    # Vision / grasp detection
    "GraspResult",
    "GraspFailure",
    "DetectionReason",
    "DETECTION_REASONS",
    # 3-D scene sensing
    "SensingFailure",
    "ObjectGeometryResult",
    "SurfaceGeometryResult",
    "SceneScanResult",
    "TargetLocationResult",
    "SPATIAL_RELATIONS",
    # Base approach
    "ApproachResult",
    "ApproachFailure",
    "SearchResult",
    "TargetApproachResult",
    "MeasuredBaseMotionResult",
    # Frame projection
    "BasePoint",
    "SimulationSnapshotResult",
    "NavigationResult",
]


class SimulationSnapshotResult(TypedDict):
    """Explicit simulator truth; never a substitute for a sensor observation."""

    source: Literal["simulation_ground_truth"]
    scene_instance_id: str
    reset_generation: int
    snapshot_id: str
    step: int
    sim_time: float
    captured_at: float
    frame: str
    units: dict
    robot: NotRequired[dict]
    entity: NotRequired[dict]
    entities: NotRequired[list[dict]]
    total: NotRequired[int]
    next_offset: NotRequired[int | None]


class MeasuredBaseMotionResult(TypedDict):
    """Blocking base step with observed stopping, including a zero-distance step.

    ``ok`` alone is not evidence of stopping. Only ``settled=True`` together
    with ``stop_confirmation='measured'`` permits target sensing/another step.
    A command acknowledgement or elapsed duration cannot establish these fields.
    """

    ok: bool
    settled: bool
    stop_confirmation: Literal["measured", "unknown"]
    reason: NotRequired[str]


class NavigationResult(TypedDict):
    """Goal completion and measured stopping are separate facts."""

    ok: bool
    goal_reached: bool
    settled: bool
    stop_confirmation: Literal["measured", "unknown"]
    reason: str
    position_error_m: NotRequired[float]
    yaw_error_rad: NotRequired[float]
    final_state: NotRequired[dict]
    plan: NotRequired[dict]
    source: NotRequired[str]
    scene_instance_id: NotRequired[str]
    reset_generation: NotRequired[int]
    recoverable: NotRequired[bool]


# =============================================================================
# Vision / Grasp Detection
# =============================================================================

DETECTION_REASONS = frozenset(
    {
        "no_camera",
        "no_detection",
        "empty_mask",
        "no_valid_depth",
        "detector_unavailable",
    }
)

DetectionReason = Literal[
    "no_camera",
    "no_detection",
    "empty_mask",
    "no_valid_depth",
    "detector_unavailable",
]


class GraspFailure(TypedDict):
    """Failure shape returned by grasp/detection tools."""

    ok: Literal[False]
    reason: DetectionReason
    object: str
    error_code: NotRequired[str]


class GraspResult(TypedDict, total=False):
    """Success shape returned by ``get_grasp_info_simple``.

    ``total=False`` because not every caller populates every field (e.g. some
    adapters omit ``place_z``). ``ok`` is the one always-present key.
    """

    ok: Literal[True]
    object: str
    position: list  # [x, y, z]_mm
    grasp_z: float
    grasp_position: list  # [x, y, z]_mm
    place_z: float
    place_position: list  # [x, y, z]_mm
    score: float
    pixel_uv: list  # [u, v]
    depth_m: float
    grasp_rz: float  # top-surface yaw (deg); only present when top-surface grasp is enabled
    grasp_width_mm: float  # object extent along the gripper closing axis; top-surface grasp only


class CloseEffectorResult(TypedDict, total=False):
    """Result shape of ``close_gripper`` / ``activate_suction``.

    ``ok`` / ``state`` are the command-level shape every body returns. The
    evidence fields are what a body with a measured channel adds — they assert
    the close stopped on object contact, and only they (or the body's
    ``is_grasp_confirmed`` hook) establish ``payload.held``: a successful
    command alone does not. A body with no measured channel declares
    ``trust_command_grasp`` instead (adapter reference, grasp evidence).
    """

    ok: Literal[True]
    state: str
    grasp_confirmed: bool
    contact: bool
    holding_payload: bool


# =============================================================================
# 3-D Scene Sensing
# =============================================================================

# Closed set, deliberately small and deliberately viewpoint-INDEPENDENT: "on" / "under" /
# "in" / "beside" / "near" mean the same thing from wherever the robot happens to stand, so
# a plan carrying one is still true after the base moves. "left of" / "in front of" are NOT
# here — they are only defined relative to an observer, and the observer moves.
SPATIAL_RELATIONS: tuple[str, ...] = ("on", "under", "in", "beside", "near")


class SensingFailure(TypedDict):
    """Shape every 3-D sensing action returns when it cannot answer."""

    ok: Literal[False]
    reason: str
    object: str
    error_code: NotRequired[str]


class ObjectGeometryResult(TypedDict, total=False):
    """Success shape of an object detection — see ``perception.scene3d.object_geometry_fields``.

    ``total=False`` because ``reference`` / ``relation`` are only present on the grounded
    path; every other field is always populated on success.
    """

    ok: Literal[True]
    reason: str
    object: str
    reference: str  # grounded path only: what the target was measured against
    relation: str  # grounded path only: which relation to the reference was confirmed
    center_mm: list[float]  # [x, y, z] base frame
    width_mm: float
    height_mm: float
    front_x_mm: float  # near face (smallest forward X)
    back_x_mm: float
    top_z_mm: float
    n_points: int
    yaw_rad: float  # footprint orientation
    long_mm: float
    short_mm: float
    face_normal: list[float]  # [nx, ny], outward from the near face
    face_flatness: float


class SurfaceGeometryResult(TypedDict, total=False):
    """Success shape of a support-surface sensing — see ``perception.scene3d.surface_footprint_fields``."""

    ok: Literal[True]
    object: str
    reference: str  # grounded path only: what the surface was measured against
    relation: str  # grounded path only: which relation to the reference was confirmed
    surface_z_mm: float  # the height a payload lands on
    center_mm: list[float]
    front_x_mm: float
    back_x_mm: float
    width_mm: float
    n_points: int
    yaw_rad: float
    long_mm: float
    short_mm: float
    face_normal: list[float]
    edge_midpoint_mm: list[float]  # near-edge line the base squares up to
    edge_normal: list[float]
    edge_quality: float
    edge_len_mm: float


class SceneScanResult(TypedDict):
    """Success shape of a whole-scene multi-instance scan."""

    ok: Literal[True]
    object: str
    count: int
    objects: list[dict]  # per-instance geometry, nearest first


# =============================================================================
# Base Approach
# =============================================================================


class ApproachFailure(TypedDict, total=False):
    """Shape an approach returns when it did not reach a workable pose.

    A caller that sees this must NOT act on the target: either nothing was found
    (``object_not_found`` / ``no_detection``) or the base ended up somewhere it
    cannot work from (``too_close`` / ``lidar_blocked`` / ``lost_after_move``).
    """

    ok: Literal[False]
    reason: str
    error_code: str
    turn_rad: float
    forward_m: float
    iters: int


class TargetLocationResult(TypedDict, total=False):
    """RGB-D visible-point measurement; never the unseen object's full centre."""

    ok: bool
    reason: str
    error_code: str
    candidate_count: int
    object: str
    reference: str | None
    relation: str
    detection_label: str | None
    detection_score: float | None
    box_px: list[float] | None
    point_base_mm: list[float]
    distance_m: float
    bearing_rad: float
    near_surface_x_m: float
    n_points: int
    frame_id: str
    captured_monotonic_s: float
    source: Literal["rgbd_perception"]
    frame: Literal["base_forward_left_up"]
    point_kind: Literal["visible_point_median"]


class TargetApproachResult(TypedDict, total=False):
    """Target arrival and measured stopping are independently reported."""

    ok: bool
    goal_reached: bool
    settled: bool
    stop_confirmation: str
    reason: str
    iterations: int
    observations: int
    observation_retries: list[dict]
    target_distance_m: float
    distance_error_m: float
    bearing_error_rad: float
    measurement: TargetLocationResult
    confirmation_failure: TargetLocationResult
    confirmation_frame_ids: list[str]
    steps: list[dict]


class ApproachResult(TypedDict, total=False):
    """Success shape of an approach — the base is at a pose the arms can work from.

    ``status`` is ``in_band`` / ``in_range`` when the convergence check passed, and
    ``max_iters`` when the loop ran out while still improving (still a usable pose,
    but the caller may want to re-sense).
    """

    ok: Literal[True]
    status: str
    turn_rad: float
    forward_m: float
    iters: int


class SearchResult(TypedDict, total=False):
    """Coarse bearing-only search result: direction to the target, never a distance."""

    ok: bool
    found: bool
    verified: bool
    object: str
    reference: str
    camera: str
    bearing_rad: float  # + = target left of the body's heading
    score: float
    overlap: float
    reason: str


# =============================================================================
# Frame Projection
# =============================================================================


class BasePoint(TypedDict):
    """A single base-frame point in millimetres — what a pixel reprojection returns.

    Declared here (same rule as the other result types) so the shape a planner is
    promised and the dict actually built cannot drift.
    """

    x: float
    y: float
    z: float
