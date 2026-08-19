# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ground-plane oriented footprint (PCA) recovers a box's yaw + long/short extents,
so a base approach can square up to a face normal instead of the diagonal to the centre."""

from __future__ import annotations

import numpy as np

from jiuwensymbiosis.perception.object_geometry import oriented_footprint


def _rotated_rect(long_mm, short_mm, theta_rad, n=(40, 20)):
    """Dense samples of a long×short rectangle (long axis along +x) rotated by theta."""
    lx = np.linspace(-long_mm / 2, long_mm / 2, n[0])
    sy = np.linspace(-short_mm / 2, short_mm / 2, n[1])
    gx, gy = np.meshgrid(lx, sy)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return pts @ np.array([[c, -s], [s, c]]).T + np.array([1000.0, 500.0])


def test_recovers_angle_and_extents():
    theta = np.radians(30.0)
    yaw, long_mm, short_mm = oriented_footprint(_rotated_rect(400.0, 200.0, theta))
    assert abs(yaw - theta) < np.radians(3.0)              # long-axis heading recovered
    assert long_mm > short_mm                              # long vs short not swapped
    assert 330.0 < long_mm < 400.0                         # ~0.9·400 (5–95% span)
    assert 160.0 < short_mm < 200.0                        # ~0.9·200


def test_axis_180_ambiguity_normalised():
    # a box at 200° reads the same as 20° (axis is 180°-ambiguous → mapped into (-90,90])
    yaw, _lo, _sh = oriented_footprint(_rotated_rect(400.0, 200.0, np.radians(200.0)))
    assert -np.pi / 2 < yaw <= np.pi / 2
    assert abs(yaw - np.radians(20.0)) < np.radians(3.0)


def test_degenerate_input_is_safe():
    yaw, long_mm, short_mm = oriented_footprint(np.zeros((5, 2)))  # all identical points
    assert (yaw, long_mm, short_mm) == (0.0, 0.0, 0.0) or long_mm == 0.0


# ---- face_normal_ground: 3-D PCA plane normal for squaring the base to a face ----

def test_face_normal_of_a_vertical_front_face():
    from jiuwensymbiosis.perception.object_geometry import face_normal_ground
    # a box front face at x≈500mm: spread in y (width) and z (height), ~flat in x (depth)
    y = np.linspace(-150, 150, 30)
    z = np.linspace(0, 200, 20)
    gy, gz = np.meshgrid(y, z)
    pts = np.stack([np.full(gy.size, 500.0) + np.random.default_rng(0).normal(0, 2, gy.size),
                    gy.ravel(), gz.ravel()], axis=1)
    nx, ny, flat = face_normal_ground(pts)
    assert abs(abs(nx) - 1.0) < 0.05 and abs(ny) < 0.05   # horizontal normal along ±x
    assert flat < 0.15                                    # flat face → trustworthy


def test_horizontal_top_surface_returns_untrusted():
    from jiuwensymbiosis.perception.object_geometry import face_normal_ground
    # a table top: spread in x,y, ~flat in z → normal is vertical → no ground heading
    x = np.linspace(0, 400, 30)
    y = np.linspace(-200, 200, 30)
    gx, gy = np.meshgrid(x, y)
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 600.0)], axis=1)
    nx, ny, flat = face_normal_ground(pts)
    assert (nx, ny, flat) == (0.0, 0.0, 1.0)


def _cube_faces(side=300.0, yaw_deg=30.0, center=(1000.0, 200.0, 150.0), n=16, noise=2.0, seed=0):
    """Visible surface of a CUBE (front + one side + top faces) at ``yaw_deg``, in the base frame.
    A square footprint on purpose — the case the old whole-cloud PCA could not orient."""
    h = side / 2.0
    a = np.linspace(-h, h, n)
    A, B = np.meshgrid(a, a)
    A, B = A.ravel(), B.ravel()
    front = np.stack([np.full(A.size, -h), A, B], axis=1)   # local −x face (faces the robot after yaw)
    face_side = np.stack([A, np.full(A.size, h), B], axis=1)  # local +y face
    top = np.stack([A, B, np.full(A.size, h)], axis=1)      # local +z face (horizontal cap)
    local = np.concatenate([front, face_side, top], axis=0)
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    base = local @ r.T + np.array(center)
    return base + np.random.default_rng(seed).normal(0.0, noise, base.shape)


def test_face_normal_of_a_cube_recovered_aspect_free():
    """The whole point of the edge-based method: a CUBE (square footprint) — whose footprint PCA yaw is
    ill-conditioned noise (the field failure) — still yields a TRUSTWORTHY front-face normal."""
    from jiuwensymbiosis.perception.object_geometry import face_normal_ground, oriented_footprint

    pts = _cube_faces(yaw_deg=30.0)
    # footprint is ~square → PCA can't key off a long axis (this is why the old method failed here)
    _yaw, long_mm, short_mm = oriented_footprint(pts[:, :2])
    assert abs(long_mm - short_mm) < 0.2 * max(long_mm, short_mm)   # aspect ≈ 1 (near-square)
    # edge-based normal recovers the FRONT face (local −x) outward normal: R(30°)·(−1,0)=(−0.866,−0.5)
    nx, ny, flat = face_normal_ground(pts)
    assert flat < 0.15                                              # clean vertical face → trusted (was 1.0)
    assert nx * (-0.866) + ny * (-0.5) > 0.96                      # recovered the front normal (~≤15°)
    # and it is the FRONT face, not the visible side face (whose normal ≈ (−0.5, 0.866))
    assert nx * (-0.5) + ny * 0.866 < 0.7


def test_grounded_normal_ignores_reference_surface_below_min_z():
    """Grounded 'X on Y': if the target mask bleeds onto the reference (a big box below), the edge fit
    would latch onto the REFERENCE's dominant front edge. Filtering to points ABOVE the surface top
    (min_z) recovers the TARGET's own wall normal instead."""
    from jiuwensymbiosis.perception.object_geometry import face_normal_ground

    rng = np.random.default_rng(0)
    # reference brown box: a big vertical front face at x=1200, z 0..480, normal ≈ (−1,0) (angle 180°)
    yb = np.linspace(-300, 300, 40)
    zb = np.linspace(0, 480, 40)
    yb, zb = np.meshgrid(yb, zb)
    brown = np.stack([np.full(yb.size, 1200.0), yb.ravel(), zb.ravel()], axis=1)
    # target bin on top (z 500..650), yaw 30° → its own front normal ≈ (−0.87,−0.5) (angle −150°)
    binpts = _cube_faces(side=200.0, yaw_deg=30.0, center=(1000.0, 150.0, 575.0), n=12)
    allpts = np.vstack([brown, binpts]) + rng.normal(0.0, 2.0, (brown.shape[0] + binpts.shape[0], 3))

    nx0, ny0, _ = face_normal_ground(allpts)                          # no filter → the big reference edge
    assert abs(nx0) > 0.9 and abs(ny0) < 0.2                          # ≈ (−1,0): the REFERENCE, wrong
    nx, ny, flat = face_normal_ground(allpts[allpts[:, 2] > 520.0])   # above the surface → the TARGET wall
    assert flat < 0.15
    assert nx * (-0.866) + ny * (-0.5) > 0.9                          # recovered the bin's own front normal
