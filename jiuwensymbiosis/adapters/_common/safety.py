# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Workspace bounds for cartesian motion gating.

For now this models a single Z floor — enough for SCARA where the only
realistic collision direction is "tool drives into the table". A 6-DoF
arm would extend this with an XYZ bounding box or full collision mesh;
that's deferred until a real adapter needs it.

Two reference frames are tracked so the same bounds object answers both
"is this TIP pose safe?" and "is this FLANGE pose safe?" without the
caller juggling tool_offset_mm:
  * Tip-frame floor (``z_min_safe``) — what the user / task code thinks in.
  * Flange-frame floor (``flange_z_min_safe``) — what the driver actually
    commands. Derived from tip floor + tool_offset_mm IF the adapter
    stores poses in tip frame; otherwise equal to tip floor.

The robot-tag in error messages is parameterized via ``log_prefix`` so
each adapter's user-facing log strings stay unchanged after refactor.
"""

from __future__ import annotations


class WorkspaceBounds:
    """Single-axis Z floor for cartesian motion gating.

    Args:
      z_min_safe: Lowest allowable tip-frame Z (mm).
      tool_offset_mm: Distance along base -Z from flange to tool tip (mm).
        Set to 0 if poses_are_tip_frame=False or no tool offset is configured.
      poses_are_tip_frame: True if the home/calibration poses are stored in
        TIP frame (the high-level convention when a calibration is loaded).
        False if they're in FLANGE frame (no-calibration fallback).
      log_prefix: Tag prepended to RuntimeError text — keep the adapter's
        historical user-visible prefix to avoid breaking log greppers.
    """

    def __init__(
        self,
        *,
        z_min_safe: float,
        tool_offset_mm: float = 0.0,
        poses_are_tip_frame: bool = False,
        log_prefix: str = "[Robot]",
    ) -> None:
        self.z_min_safe = float(z_min_safe)
        self.tool_offset_mm = float(tool_offset_mm)
        self.poses_are_tip_frame = bool(poses_are_tip_frame)
        if self.poses_are_tip_frame:
            self.flange_z_min_safe = self.z_min_safe + self.tool_offset_mm
        else:
            self.flange_z_min_safe = self.z_min_safe
        self._log_prefix = log_prefix

    def check_flange_z(self, z: float) -> None:
        """Raise ``RuntimeError`` if ``z`` (FLANGE frame, mm) violates the floor.

        Error text names both the target flange z, the configured floor, and
        the equivalent tip-frame numbers — diagnostic enough that the
        operator can immediately tell whether the issue is the request or
        the calibration.
        """
        if z < self.flange_z_min_safe:
            raise RuntimeError(
                f"{self._log_prefix} refused move: target flange z={z:.2f}mm < "
                f"flange_z_min_safe={self.flange_z_min_safe:.2f}mm "
                f"(tip would be at {z - self.tool_offset_mm:.2f}mm < "
                f"tip floor {self.z_min_safe:.2f}mm)"
            )
