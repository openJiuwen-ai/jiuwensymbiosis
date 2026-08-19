# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Differential-drive wheel-speed laws — the maths behind a two-wheel base move.

Every differential base answers the same four questions while executing a move:
how fast should each wheel turn to *rotate* toward a heading, to *advance* a
distance, to *creep while curving* toward a bearing, and to *hold a curvature*.
The answers are geometry and a deceleration profile — no ROS, no vendor SDK, no
topic — so they live here rather than inside any one body's driver.

Each function is a **single control step**: given the state measured this tick and
the tuning, return the wheel pair to command. The caller owns the loop, the
feedback source (odometry) and the actuator, which is what keeps this module free
of any transport.

Sign convention (REP-103): +yaw is left/CCW, and a left turn means the left wheel
runs slower/backwards relative to the right one, so ``(left, right)`` is returned
in that order and a positive turn yields ``left < right``.

.. warning::
   This module must import **nothing** from ``jiuwensymbiosis``. A ROS 2 body's
   wheel worker runs under the system interpreter (rclpy is not importable from
   the agent's conda env) and loads this file by path, bypassing the package
   ``__init__`` — which would otherwise pull in the whole agent stack. Enforced by
   ``tests/unit_tests/adapters/cruzr/test_image_decode_isolation.py``.
"""

from __future__ import annotations

import math

# Guards a division by a tuning value the caller may legitimately set to zero
# ("no deceleration band"), which then degrades to "always command the full speed".
_EPS = 1e-6


def wrap_angle(angle: float) -> float:
    """Normalise ``angle`` to (-π, π].

    Shortest-angle form: a rotation controller steered by the raw difference would
    take the long way round past ±π, and an overshoot would keep spinning instead
    of correcting.
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def ramped_speed(speed: float, min_speed: float, remaining: float, slow_band: float) -> float:
    """Wheel speed for ``remaining`` distance/angle to go, decelerating near the target.

    Full ``speed`` while far, tapering linearly once inside ``slow_band``, but never
    below ``min_speed`` — a differential chassis has a deadband under which the
    wheels simply do not turn, so a profile that decays to zero stalls short of the
    target instead of reaching it.

    Args:
        speed: cruise speed (wheel rad/s) requested for this move.
        min_speed: floor keeping the command above the chassis deadband.
        remaining: distance (m) or angle (rad) still to cover — a **magnitude**.
            The caller owns the direction (it decides which wheel leads), and
            passing a signed error here would taper to ``min_speed`` past the
            target instead of decelerating into it.
        slow_band: how far out to start decelerating; ``0`` disables the taper.
    """
    return min(speed, max(min_speed, speed * remaining / max(slow_band, _EPS)))


def rotate_wheels(remaining_rad: float, *, k_rot: float, k_rot_min: float,
                  k_rot_slow_rad: float) -> tuple[float, float]:
    """Wheel pair for turning in place, decelerating into the target heading.

    ``remaining_rad`` is the *shortest-angle* error (see :func:`wrap_angle`) and is
    re-read every tick, so its sign — not the original command's — picks the
    direction. An overshoot therefore reverses and settles rather than spinning on.
    """
    magnitude = ramped_speed(k_rot, k_rot_min, abs(remaining_rad), k_rot_slow_rad)
    turn = 1.0 if remaining_rad > 0 else -1.0
    return -turn * magnitude, turn * magnitude


def spin_wheels(direction: float, k_rot: float) -> tuple[float, float]:
    """Wheel pair for a constant-speed spin (``direction`` > 0 = left/CCW).

    No deceleration profile: a search spin has no target heading to settle onto —
    it runs until something outside tells it to stop.
    """
    turn = 1.0 if direction >= 0 else -1.0
    return -turn * k_rot, turn * k_rot


def forward_wheels(
    remaining_m: float, direction: float, *, k_fwd: float, k_fwd_min: float, k_fwd_slow_m: float
) -> tuple[float, float]:
    """Wheel pair for driving straight, decelerating into the target distance.

    Both wheels get the same speed; ``direction`` (>0 forward) carries the sign, so
    reversing is the same law with a negative factor.
    """
    magnitude = ramped_speed(k_fwd, k_fwd_min, remaining_m, k_fwd_slow_m)
    sign = 1.0 if direction > 0 else -1.0
    return sign * magnitude, sign * magnitude


def steered_wheels(bearing_rad: float, *, k_fwd: float, k_steer: float, steer_max: float) -> tuple[float, float]:
    """Wheel pair for creeping forward while curving toward ``bearing_rad``.

    A positive bearing (target to the left) slows the left wheel and speeds the
    right one, curving the base toward it. Both wheels are floored at zero so a
    large steer bias curves the base rather than reversing one wheel into a pivot,
    which would stop the approach making forward progress. ``k_steer=0`` degrades
    to a straight creep.
    """
    delta = max(-steer_max, min(steer_max, k_steer * bearing_rad))
    return max(0.0, k_fwd - delta), max(0.0, k_fwd + delta)


def measured_curvature(dyaw: float, ds: float, fallback: float) -> float:
    """Instantaneous curvature |Δyaw|/Δs from one odometry increment.

    ``fallback`` is returned when the base has barely moved, because the ratio is
    numerically meaningless there; feeding the target curvature back makes the
    servo hold its current differential instead of spiking on noise.
    """
    return abs(dyaw) / ds if ds > 1e-4 else fallback


def arc_wheels(
    remaining_rad: float,
    turn_sign: float,
    delta: float,
    *,
    k_fwd: float,
    k_fwd_min: float,
    slow_band_rad: float,
) -> tuple[float, float]:
    """Wheel pair for holding a curvature, decelerating into the target heading change.

    ``delta`` is the wheel-speed differential produced by :func:`arc_delta`; this
    only applies it to the ramped forward speed. Splitting the two means the
    curvature servo integrates on its own cadence while the speed still tapers as
    the arc completes.
    """
    magnitude = ramped_speed(k_fwd, k_fwd_min, remaining_rad, slow_band_rad)
    return max(0.0, magnitude - turn_sign * delta), max(0.0, magnitude + turn_sign * delta)


def arc_delta(delta: float, target_curvature: float, measured: float, *, gain: float, k_fwd: float) -> float:
    """Integrate the wheel-differential that servos measured curvature onto target.

    Integrating the *error* rather than solving for a differential is what removes
    the need for wheel-radius / track calibration: the loop finds whatever
    differential produces the requested curvature on this chassis. Clamped to
    ``[0, 0.9 * k_fwd]`` so the inner wheel can never be driven backwards, which
    would turn the arc into a pivot.
    """
    return max(0.0, min(k_fwd * 0.9, delta + gain * (target_curvature - measured)))


def sector_min_range(
    ranges,
    angle_min: float,
    angle_increment: float,
    *,
    half_sector_rad: float,
    range_min: float,
    range_max: float,
    self_floor: float,
) -> float:
    """Closest valid lidar return within ±``half_sector_rad`` of straight ahead.

    Only the travel sector matters for an e-stop — a wall beside the base is not an
    obstacle. Returns ``inf`` when nothing valid is in the sector, so a caller
    comparing against a safety distance treats "saw nothing" as clear rather than
    as an emergency.

    Args:
        ranges: per-beam distances (m), ordered from ``angle_min``.
        self_floor: returns closer than this are the robot's own body or noise.
    """
    closest = float("inf")
    angle = angle_min
    for distance in ranges:
        in_sector = -half_sector_rad <= angle <= half_sector_rad
        usable = (math.isfinite(distance) and range_min <= distance <= range_max
                  and distance > self_floor)
        if in_sector and usable:
            closest = min(closest, distance)
        angle += angle_increment
    return closest
