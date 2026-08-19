# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Generic base-goal geometry for mobile-manipulation grasp approach.

Pure trigonometry: turn a detected object centre (base-frame mm) into a base
goal ``(turn, forward)`` that faces the object and stops at a configured grasp
distance. Robot-agnostic — reads only config *values* (the grasp-band distances
+ yaw tolerance) off a config-like object, so any mobile base reuses it.
"""

from __future__ import annotations

import math
from typing import Any


def plan_base_goal_for_grasp(center_mm: list[float], cfg: Any) -> tuple[float, float, str]:
    """Detected object centre (base-frame mm) → base goal facing it at grasp distance.

    ``center_mm`` = [x, y, z] (x forward, y left). Returns ``(turn_rad, forward_m, status)``:
      - ``too_close``: object is nearer than the grasp band → caller must NOT drive forward.
      - ``in_band``  : object already in the band and roughly centred → no move needed.
      - ``ok``       : move by (turn_rad, forward_m) — face the object, then advance so it
        lands at the configured forward distance.
    """
    bx = float(center_mm[0]) / 1000.0
    by = float(center_mm[1]) / 1000.0
    rng = math.hypot(bx, by)
    turn = math.atan2(by, bx)
    forward = rng - float(cfg.grasp_target_forward_m)
    if rng < float(cfg.grasp_forward_min_m):
        return (turn, forward, "too_close")
    if float(cfg.grasp_forward_min_m) <= rng <= float(cfg.grasp_forward_max_m) and abs(turn) <= float(
        cfg.base_yaw_tol_rad
    ):
        return (turn, forward, "in_band")
    return (turn, forward, "ok")


def plan_grasp_pose_goal(
    center_mm: list[float], face_normal: tuple[float, float], face_flatness: float, cfg: Any
) -> tuple[float, float, str]:
    """Orientation-aware grasp approach: drive toward a waypoint on the box's **face-normal
    line** so the base ends up SQUARE to the visible face, not on the diagonal to its centre.

    ``center_mm`` = [x, y, z] base mm; ``face_normal`` = (nx, ny) the visible-face normal in the
    ground plane (``face_normal_ground``); ``face_flatness`` = λ_min/λ_mid (small ⇒ a flat,
    front-facing face ⇒ trustworthy normal). When trustworthy we approach ALONG the outward
    normal (so the robot faces the face perpendicular, and the box's sides sit laterally for the
    paddles); otherwise (blob / top surface / ambiguous) we fall back to a plain radial approach
    to the centre. The caller drives + re-detects each iteration (converges like
    ``plan_base_goal_for_grasp``). ``in_band`` needs centred (small bearing) AND square (the
    outward normal within ``grasp_yaw_align_tol`` of the robot→box line).
    """
    bx = float(center_mm[0]) / 1000.0
    by = float(center_mm[1]) / 1000.0
    rng = math.hypot(bx, by)
    grasp_d = float(cfg.grasp_target_forward_m)
    gmin, gmax = float(cfg.grasp_forward_min_m), float(cfg.grasp_forward_max_m)
    bearing = math.atan2(by, bx)
    centred = abs(bearing) <= float(cfg.base_yaw_tol_rad)

    if rng < gmin:  # nearer than the band → caller must NOT drive forward
        return (bearing, rng - grasp_d, "too_close")
    # Phase 1 — not yet in the band: approach the CENTRE radially. We do NOT circle while
    # far (a diagonal shortcut to an off-axis waypoint would cut in too close); get to the
    # right distance first, then square up.
    if rng > gmax:
        return (bearing, rng - grasp_d, "ok")

    # Phase 2 — in the band. Square up to the visible face only if the normal is trustworthy,
    # in bounded steps that preserve the grasp distance (the waypoint stays grasp_d from the
    # box, so stepping toward it never drives net-closer than the band).
    nx, ny = float(face_normal[0]), float(face_normal[1])
    trust = math.hypot(nx, ny) > 0.5 and float(face_flatness) <= float(getattr(cfg, "grasp_face_flatness_max", 0.15))
    if trust:
        if nx * (-bx) + ny * (-by) < 0.0:  # outward normal must point toward the robot (viewer)
            nx, ny = -nx, -ny
        misalign = math.acos(max(-1.0, min(1.0, -(nx * bx + ny * by) / (rng + 1e-9))))
        if misalign > float(getattr(cfg, "grasp_yaw_align_tol", 0.20)):
            px, py = bx + grasp_d * nx, by + grasp_d * ny  # on the face-normal line, grasp_d from box
            step = min(math.hypot(px, py), float(getattr(cfg, "grasp_face_step_m", 0.12)))
            return (math.atan2(py, px), step, "ok")  # small arc step, re-detect next iteration
    if centred:  # in band + (square or normal untrusted) + centred → ready to grasp
        return (bearing, rng - grasp_d, "in_band")
    return (bearing, rng - grasp_d, "ok")  # in band + square but off-centre → just recentre


def plan_grasp_arc_to_entry(center_mm: list[float], face_normal: tuple[float, float], cfg: Any) -> dict:
    """Plan ONE constant-curvature arc that carries a differential base from its current pose
    (origin, heading +X) ONTO the box's face-normal line, at a standoff entry point E.

    On a no-strafe base "square to the face" and "centred on the box" coincide only when the base
    sits on the face's normal line; a discrete in-place turn then straight-in cannot get there
    (it leaves a lateral offset). One arc that lands the base ON the line fixes the initial
    condition, after which the caller's existing square-first loop (rotate to −n in place, then
    turn=0 straight-in) is correct — it preserves square + centred down to the grasp distance.

    ``center_mm`` = [x,y,z] mm (base frame). ``face_normal`` = (nx,ny) unit face normal; its sign is
    fixed here to point toward the robot so the entry point can never land behind the box. The entry
    point is ``E = C + (grasp_d + grasp_arc_standoff_m)·n`` — ON the normal line, one standoff BEYOND
    the grasp distance (a safe distance from the box for the subsequent in-place square).

    The arc is the UNIQUE circle tangent to +X at the origin and passing through E: centre (0, c),
    ``c = (Ex²+Ey²)/(2·Ey)``, radius ``R=|c|``. The heading change to reach E is the inscribed-angle
    identity ``dyaw = 2·atan2(Ey, Ex)`` (twice E's bearing). The arc only has to reach E on the line;
    the final heading is trimmed by the caller's in-place square, so E's heading is not constrained.

    Returns a dict with ``mode``:
      - ``"straight"``: E is dead-ahead (|Ey|≈0) → no arc; caller's loop drives straight-in.
      - ``"arc"``: ``{R, s(+1 CCW/−1 CW), dyaw_arc, E, closest, d_e}`` — drive the arc, then re-detect.
      - ``"reject"``: geometry unsafe/unreachable (see reason) → caller falls back to the discrete loop.

    SAFETY (lidar e-stop is off in production ⇒ geometry must guarantee it): every rejection keeps the
    base on the safe discrete path. ``dyaw`` uses the UNIT normal → no atan2(→0) swing (the old
    ``plan_grasp_pose_goal`` collision mode). The arc is executed only when its whole path stays at
    least ``grasp_d`` from the box centre (``closest ≥ grasp_d``), the radius is not too tight, and the
    swing is small enough to keep the box in the waist FOV — all decided here before any wheel command.
    """
    cx = float(center_mm[0]) / 1000.0
    cy = float(center_mm[1]) / 1000.0
    rng = math.hypot(cx, cy)
    grasp_d = float(cfg.grasp_target_forward_m)
    nx, ny = float(face_normal[0]), float(face_normal[1])
    # Sign-fix: outward normal must point toward the robot (viewer), else E lands behind the box
    # and the arc would cross it (mirrors plan_grasp_pose_goal's flip).
    if nx * (-cx) + ny * (-cy) < 0.0:
        nx, ny = -nx, -ny
    if rng < 1e-6 or (nx * (-cx) + ny * (-cy)) <= 1e-6:
        return {"mode": "reject", "reason": "no_normal_line"}

    d_e = grasp_d + float(getattr(cfg, "grasp_arc_standoff_m", 0.10))
    ex, ey = cx + d_e * nx, cy + d_e * ny          # entry point ON the normal line, one standoff out
    if abs(ey) <= 1e-3:                             # E dead-ahead → no arc needed; loop drives straight-in
        return {"mode": "straight", "E": (ex, ey), "d_e": d_e}

    c = (ex * ex + ey * ey) / (2.0 * ey)           # signed circle centre (0, c)
    radius = abs(c)
    dyaw = 2.0 * math.atan2(ey, ex)                # heading change = twice E's bearing (inscribed angle)

    # Closest approach of the SWEPT arc to the box centre C (circle centre O=(0,c), radius=radius).
    do = math.hypot(cx, cy - c)
    a_start = math.atan2(-c, 0.0)                  # position-angle of the origin about O
    a_c = math.atan2(cy - c, cx)                   # position-angle of C about O
    rel = math.atan2(math.sin(a_c - a_start), math.cos(a_c - a_start))  # a_c relative to a_start
    in_arc = (0.0 <= rel <= dyaw) if dyaw >= 0.0 else (dyaw <= rel <= 0.0)
    closest = abs(do - radius) if in_arc else min(rng, d_e)

    if radius < float(getattr(cfg, "grasp_arc_min_radius_m", 0.35)):
        return {"mode": "reject", "reason": "radius_too_tight", "R": radius, "dyaw_arc": dyaw}
    if abs(dyaw) > float(getattr(cfg, "grasp_arc_max_dyaw_rad", 1.2)):
        return {"mode": "reject", "reason": "dyaw_too_large", "R": radius, "dyaw_arc": dyaw}
    if closest < grasp_d:                          # arc path would enter the grasp_d disk around the box
        return {"mode": "reject", "reason": "arc_enters_grasp_disk", "closest": closest, "dyaw_arc": dyaw}
    return {"mode": "arc", "R": radius, "s": 1.0 if dyaw >= 0.0 else -1.0, "dyaw_arc": dyaw,
            "E": (ex, ey), "closest": closest, "d_e": d_e}


def plan_grasp_right_angle(center_mm: list[float], face_normal: tuple[float, float], cfg: Any) -> dict:
    """Right-angle (L-shaped) route onto the box's FACE-NORMAL LINE, then face the box and approach — the
    close-range-friendly alternative to an arc. An arc needs curving room and is ineffective once the box
    is already near; the right-angle is two PERPENDICULAR straight legs joined by in-place turns, so a
    differential base needs no strafe:
      1. LATERAL leg — turn ⊥ to the normal and drive onto the line. It is perpendicular to the approach,
         so it consumes NO forward distance-to-box (the reason it works when there is little room left).
      2. APPROACH leg — turn to face the box (−n) and drive straight in to the standoff entry point
         E = C + d_e·n (d_e = grasp_d + grasp_arc_standoff_m); the caller's re-detect + align then closes
         the last standoff precisely.

    ``face_normal`` = (nx,ny) unit face normal; its sign is fixed here toward the robot (so the approach
    heading faces the box, never behind it). Returns ``{mode, lat_turn, lat_dist, app_turn, app_dist, E}``
    with turns as ABSOLUTE base-frame headings (from the start heading 0; the caller sequences the second
    turn relative to the first). ``mode``: ``"reject"`` (degenerate — no normal line) → caller falls back
    to the discrete loop; ``"straight"`` (already on the line, |lateral|≤pos_tol) → approach leg only;
    ``"l"`` → lateral leg then approach leg. Unlike the arc this never rejects on geometry — the L reaches
    any pose — so near-square / corner-on boxes are covered here instead of falling through.
    """
    cx = float(center_mm[0]) / 1000.0
    cy = float(center_mm[1]) / 1000.0
    rng = math.hypot(cx, cy)
    grasp_d = float(cfg.grasp_target_forward_m)
    nx, ny = float(face_normal[0]), float(face_normal[1])
    if nx * (-cx) + ny * (-cy) < 0.0:              # sign-fix: normal points toward the robot (viewer)
        nx, ny = -nx, -ny
    if rng < 1e-6 or (nx * (-cx) + ny * (-cy)) <= 1e-6:
        return {"mode": "reject", "reason": "no_normal_line"}

    d_e = grasp_d + float(getattr(cfg, "grasp_arc_standoff_m", 0.25))
    ex, ey = cx + d_e * nx, cy + d_e * ny          # standoff entry point ON the normal line
    app_turn = math.atan2(-ny, -nx)                # heading facing the box along the normal
    s = cx * ny - cy * nx                          # signed ⊥ offset of the robot from the line = (O−C)·(−ny,nx)
    app_dist = max(0.0, -(cx * nx + cy * ny) - d_e)  # along −n from the line foot to E (0 if inside standoff)
    if abs(s) <= float(cfg.base_pos_tol_m):        # already on the line → approach leg only
        return {"mode": "straight", "lat_turn": 0.0, "lat_dist": 0.0,
                "app_turn": app_turn, "app_dist": app_dist, "E": (ex, ey)}
    sgn = 1.0 if s > 0.0 else -1.0                 # lateral direction = −sign(s)·(−ny, nx) = (sgn·ny, −sgn·nx)
    lat_turn = math.atan2(-sgn * nx, sgn * ny)
    return {"mode": "l", "lat_turn": lat_turn, "lat_dist": abs(s),
            "app_turn": app_turn, "app_dist": app_dist, "E": (ex, ey)}


def footprint_face_normal(
    center_mm: list[float], yaw_rad: float, long_mm: float, short_mm: float, cfg: Any
) -> tuple[float, float, float]:
    """Robot-facing face normal of a box from its ground **footprint** PCA — a robust stand-in for
    the 3-D ``face_normal`` when squaring the base to a box for a grasp.

    The 3-D face normal (``face_normal_ground``) is trustworthy only for a flat, fully-visible front
    face; for a box on a table it is often degenerate/untrusted, so the square-up silently never
    fires and the base grasps the box at a twist. The footprint long-axis heading is stable
    regardless of face visibility. ``yaw_rad`` is that heading (base frame); the robot-facing face's
    outward normal is the principal axis (±long / ±short) most aligned with the box→robot direction
    — the SAME near-edge selection as ``plan_surface_square_turn``.

    Returns ``(nx, ny, flatness)`` in the exact convention ``plan_grasp_pose_goal`` consumes, so it
    drops in unchanged: a unit outward normal + a ``flatness`` proxy that is ``0.0`` (trusted) when
    the footprint aspect is decisive, else ``1.0`` (untrusted → caller falls back to a plain radial
    approach). Near-square footprints (aspect below ``grasp_square_min_aspect``) or degenerate spans
    are ambiguous — the near-face axis flips 90° under noise — so they return untrusted; there the
    residual twist barely changes the base-frame clamp width anyway.
    """
    cx = float(center_mm[0]) / 1000.0
    cy = float(center_mm[1]) / 1000.0
    lo, hi = min(long_mm, short_mm), max(long_mm, short_mm)
    if math.hypot(cx, cy) < 1e-6 or lo <= 1e-6:
        return 0.0, 0.0, 1.0
    if hi / lo < float(getattr(cfg, "grasp_square_min_aspect", 1.2)):
        return 0.0, 0.0, 1.0  # near-square footprint → near-face axis ambiguous, skip squaring
    u = (math.cos(yaw_rad), math.sin(yaw_rad))          # long axis
    v = (-math.sin(yaw_rad), math.cos(yaw_rad))         # short axis
    gx, gy = -cx, -cy                                    # box centre → robot
    nx, ny = max(
        ((u[0], u[1]), (-u[0], -u[1]), (v[0], v[1]), (-v[0], -v[1])),
        key=lambda n: n[0] * gx + n[1] * gy,
    )
    return nx, ny, 0.0                                    # unit outward normal, trusted


def plan_surface_square_turn(
    center_mm: list[float], yaw_rad: float, long_mm: float, short_mm: float, cfg: Any
) -> tuple[float, bool]:
    """In-place rotation that squares the base to a support surface's **near edge**.

    A table top is horizontal → its plane normal is vertical (no ground heading), so unlike
    the grasp side we cannot use a face normal. Instead we square to the ground *footprint*:
    ``yaw_rad`` is the footprint's long-axis heading (base frame). The near edge is the
    footprint side facing the robot; its outward normal is the principal axis (±long / ±short)
    most aligned with the table→robot direction. The base should face the OPPOSITE of that
    normal (perpendicular into the near edge).

    Returns ``(turn_rad, applicable)``. ``applicable`` is False when the footprint is nearly
    square (``long/short`` below ``place_square_min_aspect``) or degenerate — the near-edge
    axis then flips 90° under noise, so the caller keeps plain centre-facing instead of
    chasing a bogus normal. ``turn_rad`` is the rotation (base frame, + = left) to align the
    base's forward (+X) with the inward near-edge normal.
    """
    cx = float(center_mm[0]) / 1000.0
    cy = float(center_mm[1]) / 1000.0
    lo, hi = min(long_mm, short_mm), max(long_mm, short_mm)
    if math.hypot(cx, cy) < 1e-6 or lo <= 1e-6:
        return 0.0, False
    if hi / lo < float(getattr(cfg, "place_square_min_aspect", 1.2)):
        return 0.0, False  # near-square footprint → near-edge axis ambiguous, skip squaring
    u = (math.cos(yaw_rad), math.sin(yaw_rad))          # long axis
    v = (-math.sin(yaw_rad), math.cos(yaw_rad))         # short axis
    gx, gy = -cx, -cy                                    # table centre → robot
    # near-edge outward normal = principal axis (either sign) most aligned with table→robot
    nx, ny = max(
        ((u[0], u[1]), (-u[0], -u[1]), (v[0], v[1]), (-v[0], -v[1])),
        key=lambda n: n[0] * gx + n[1] * gy,
    )
    return math.atan2(-ny, -nx), True                    # face into the table (opposite normal)
