# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for cruzr pixel->base projection (pure)."""

from __future__ import annotations

import numpy as np

from jiuwensymbiosis.perception.frame import project_to_base

_K = np.array([[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]])


def test_principal_point_identity_extrinsic():
    out = project_to_base((4.0, 4.0), 1.0, _K, np.eye(4))
    np.testing.assert_allclose(out, [0.0, 0.0, 1000.0])


def test_extrinsic_translation_is_applied():
    tf = np.eye(4)
    tf[:3, 3] = [1.0, 2.0, 3.0]
    out = project_to_base((4.0, 4.0), 1.0, _K, tf)
    np.testing.assert_allclose(out, [1.0, 2.0, 1003.0])
