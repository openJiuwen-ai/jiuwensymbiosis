# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""near_edge_line: fit the footprint's NEAR edge (the boundary nearest the robot — a table's front lip)
and recover its midpoint + outward normal (toward the robot), so the place-side base can square to the
table edge. Needs no vertical face (unlike face_normal_ground), so it works on a horizontal table top."""

from __future__ import annotations

import math

import numpy as np

from jiuwensymbiosis.perception.object_geometry import near_edge_line


def _filled_table(near_x, depth, half_w, cy=0.0, z=600.0, nd=15, nw=40, seed=0):
    """A filled tabletop footprint: x in [near_x, near_x+depth] (depth), y in [cy±half_w] (width),
    flat at height z. The near edge (nearest the robot at the origin) is the x=near_x lip."""
    xs = np.linspace(near_x, near_x + depth, nd)
    ys = np.linspace(cy - half_w, cy + half_w, nw)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)
    return pts + np.random.default_rng(seed).normal(0.0, 1.5, pts.shape)


def test_axis_aligned_near_edge():
    mx, my, nx, ny, q, elen, ok = near_edge_line(_filled_table(near_x=400.0, depth=500.0, half_w=350.0))
    assert ok
    assert abs(nx - (-1.0)) < 0.08 and abs(ny) < 0.08      # unit normal along −x, toward the robot
    assert abs(mx - 400.0) < 40.0 and abs(my) < 40.0       # midpoint at the near edge's centre
    assert q < 0.1 and elen > 500.0                        # clean, long straight edge → trustworthy


def test_oblique_near_edge_normal_and_midpoint():
    phi = math.radians(20.0)                               # table centre 20° off the robot's +x
    inward = np.array([math.cos(phi), math.sin(phi)])      # robot → into the table
    t = np.array([-math.sin(phi), math.cos(phi)])          # along the near edge
    m = 450.0 * inward                                     # near edge midpoint, 450 mm out
    a = np.linspace(-300.0, 300.0, 40)                     # along the edge
    b = np.linspace(0.0, 400.0, 15)                        # inward depth (b=0 is the near lip)
    A, B = np.meshgrid(a, b)
    xy = m[None, :] + A.ravel()[:, None] * t[None, :] + B.ravel()[:, None] * inward[None, :]
    pts = np.concatenate([xy, np.full((xy.shape[0], 1), 600.0)], axis=1)
    pts = pts + np.random.default_rng(1).normal(0.0, 1.5, pts.shape)

    mx, my, nx, ny, q, elen, ok = near_edge_line(pts)
    assert ok
    assert float(np.array([nx, ny]) @ (-inward)) > 0.98    # normal ≈ −inward (perpendicular to the edge)
    assert abs(mx - m[0]) < 40.0 and abs(my - m[1]) < 40.0  # midpoint on the near edge
    assert q < 0.1 and elen > 500.0


def test_untrusted_fits_return_ok_false():
    assert near_edge_line(np.zeros((10, 3)))[-1] is False                       # too few points
    assert near_edge_line(np.full((60, 3), [500.0, 0.0, 600.0]))[-1] is False   # zero tangential span
    # a clean but tiny edge (80 mm wide < min_span) is not trustworthy → fall back to footprint
    assert near_edge_line(_filled_table(near_x=400.0, depth=300.0, half_w=40.0))[-1] is False
