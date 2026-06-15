# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.safety."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.adapters._common.safety import WorkspaceBounds


class TestWorkspaceBounds:
    def test_construction(self):
        wb = WorkspaceBounds(z_min_safe=50.0)
        assert wb.z_min_safe == 50.0

    def test_flange_z_min_safe_tip_frame(self):
        wb = WorkspaceBounds(z_min_safe=50.0, tool_offset_mm=135.8, poses_are_tip_frame=True)
        assert wb.flange_z_min_safe == pytest.approx(50.0 + 135.8)

    def test_flange_z_min_safe_flange_frame(self):
        wb = WorkspaceBounds(z_min_safe=50.0, poses_are_tip_frame=False)
        assert wb.flange_z_min_safe == pytest.approx(50.0)

    def test_check_flange_z_above_floor(self):
        wb = WorkspaceBounds(z_min_safe=50.0)
        wb.check_flange_z(100.0)

    def test_check_flange_z_below_floor_raises(self):
        wb = WorkspaceBounds(z_min_safe=50.0)
        with pytest.raises(RuntimeError, match="refused move"):
            wb.check_flange_z(30.0)

    def test_check_flange_z_at_floor(self):
        wb = WorkspaceBounds(z_min_safe=50.0)
        wb.check_flange_z(50.0)
