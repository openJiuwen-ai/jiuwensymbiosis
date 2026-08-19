# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Orientation-aware grasp approach (phased). ``plan_grasp_pose_goal``:
  1. far  → approach the centre radially (no circling while far — a diagonal shortcut cuts in),
  2. in band + trustworthy face normal + misaligned → small, distance-preserving arc steps that
     square the base to the face (capped, never drives net-closer),
  3. in band + square/centred → grasp.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from jiuwensymbiosis.motion.base_goal import footprint_face_normal, plan_grasp_pose_goal

_CFG = SimpleNamespace(grasp_target_forward_m=0.40, grasp_forward_min_m=0.30,
                       grasp_forward_max_m=0.50, base_yaw_tol_rad=0.08,
                       grasp_yaw_align_tol=0.20, grasp_face_flatness_max=0.15,
                       grasp_face_step_m=0.12, grasp_square_min_aspect=1.2)
_FLAT, _BLOB = 0.05, 0.50


def test_in_band_square_is_ready_to_grasp():
    turn, forward, status = plan_grasp_pose_goal([400.0, 0.0, 500.0], (-1.0, 0.0), _FLAT, _CFG)
    assert status == "in_band"
    assert abs(turn) < 1e-6 and abs(forward) < 1e-6


def test_in_band_rotated_face_takes_a_capped_arc_step():
    nrm = (-math.cos(math.radians(30)), math.sin(math.radians(30)))  # face turned 30°
    turn, forward, status = plan_grasp_pose_goal([400.0, 0.0, 500.0], nrm, _FLAT, _CFG)
    assert status == "ok"
    assert abs(turn) > 0.5                       # steps off to the side, onto the normal line
    assert 0.0 < forward <= _CFG.grasp_face_step_m + 1e-9  # capped — never a big lunge inward


def test_far_box_approaches_radially_without_circling():
    # even with a trustworthy, rotated normal, a FAR box is approached to the band radially first
    nrm = (-math.cos(math.radians(40)), math.sin(math.radians(40)))
    turn, forward, status = plan_grasp_pose_goal([1000.0, 300.0, 500.0], nrm, _FLAT, _CFG)
    assert status == "ok"
    assert abs(turn - math.atan2(0.3, 1.0)) < 1e-6                 # faces the centre, not a side waypoint
    assert abs(forward - (math.hypot(1.0, 0.3) - 0.4)) < 1e-6      # advance to the band


def test_in_band_untrusted_normal_does_not_circle():
    # a bad/ambiguous normal in the band → plain radial (no sideways circling), grasp when centred
    turn, forward, status = plan_grasp_pose_goal([400.0, 0.0, 500.0], (-0.7, 0.7), _BLOB, _CFG)
    assert status == "in_band"
    assert abs(turn) < 1e-6


def test_too_close_flags_no_forward_drive():
    _t, _f, status = plan_grasp_pose_goal([200.0, 0.0, 500.0], (-1.0, 0.0), _FLAT, _CFG)
    assert status == "too_close"


# --- footprint_face_normal: robust square-up signal from the ground footprint PCA ---------------

def test_footprint_normal_straight_box_faces_robot_and_is_grasp_ready():
    # Box dead ahead, long axis along base Y → the robot-facing face normal points straight back
    # (-X); a unit, trusted normal. Fed into the same waypoint machinery → already square → grasp.
    nx, ny, flat = footprint_face_normal([400.0, 0.0, 780.0], math.pi / 2, 800.0, 400.0, _CFG)
    assert abs(nx - (-1.0)) < 1e-6 and abs(ny) < 1e-6 and flat == 0.0
    _t, _f, status = plan_grasp_pose_goal([400.0, 0.0, 780.0], (nx, ny), flat, _CFG)
    assert status == "in_band"


def test_footprint_normal_twisted_box_yields_a_capped_arc_step():
    # Footprint twisted 30° → the near-face normal tilts 30° off -X → drop it into the waypoint
    # machinery and it squares up with a small, distance-preserving arc step (never a big lunge).
    nx, ny, flat = footprint_face_normal([400.0, 0.0, 780.0], math.radians(30), 800.0, 400.0, _CFG)
    assert flat == 0.0 and abs(math.hypot(nx, ny) - 1.0) < 1e-6
    _t, forward, status = plan_grasp_pose_goal([400.0, 0.0, 780.0], (nx, ny), flat, _CFG)
    assert status == "ok"
    assert 0.0 < forward <= _CFG.grasp_face_step_m + 1e-9


def test_footprint_normal_near_square_footprint_is_untrusted():
    # long≈short → the near-edge axis flips 90° under noise; report untrusted so the caller keeps a
    # plain radial approach (the residual twist barely changes the base-frame clamp width anyway).
    assert footprint_face_normal([400.0, 0.0, 780.0], math.radians(30), 600.0, 580.0, _CFG) == (0.0, 0.0, 1.0)


def test_footprint_normal_degenerate_is_untrusted():
    # No footprint spans (or centre at the origin) → nothing to square to → untrusted.
    assert footprint_face_normal([0.0, 0.0, 780.0], math.radians(30), 800.0, 400.0, _CFG) == (0.0, 0.0, 1.0)
    assert footprint_face_normal([400.0, 0.0, 780.0], math.radians(30), 0.0, 0.0, _CFG) == (0.0, 0.0, 1.0)
