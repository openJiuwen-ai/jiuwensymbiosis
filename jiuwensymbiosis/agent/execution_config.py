# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Typed fast-execution settings, independent of runners and model clients."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _is_unit_interval(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1


def _is_positive_finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


@dataclass
class ServoConfig:
    """Servo-loop tuning. Conservative defaults are safe for first bring-up."""

    control_hz: float = 30.0  # control-loop rate
    max_lin_step_mm: float = 6.0  # max linear move per tick (slew limit)
    max_ang_step_deg: float = 5.0  # max angular move per tick (slew limit)
    pos_tol_mm: float = 4.0  # "reached" position tolerance
    ang_tol_deg: float = 3.0  # "reached" angular tolerance
    settle_ticks: int = 3  # consecutive in-tolerance ticks to finish
    # Continuous no-progress timeout. This is deliberately not a total move
    # deadline: a long approach may run past it while the live pose is still
    # measurably approaching the latest target.
    timeout_s: float = 20.0
    # Final safety ceiling for a moving target that can keep a servo phase alive
    # indefinitely. None disables the ceiling.
    absolute_timeout_s: float | None = 60.0
    # Minimum accumulated live-pose improvement that refreshes timeout_s.
    progress_pos_epsilon_mm: float = 0.5
    progress_ang_epsilon_deg: float = 0.5
    lost_target_grace_s: float = 3.0  # abort if no target this long

    def __post_init__(self) -> None:
        # Reject non-finite / non-positive tuning; clamp control_hz to the
        # hardware-validated band (<=0 would busy-loop with period=0).
        for name, val in (
            ("control_hz", self.control_hz),
            ("max_lin_step_mm", self.max_lin_step_mm),
            ("max_ang_step_deg", self.max_ang_step_deg),
            ("pos_tol_mm", self.pos_tol_mm),
            ("ang_tol_deg", self.ang_tol_deg),
            ("timeout_s", self.timeout_s),
            ("progress_pos_epsilon_mm", self.progress_pos_epsilon_mm),
            ("progress_ang_epsilon_deg", self.progress_ang_epsilon_deg),
            ("lost_target_grace_s", self.lost_target_grace_s),
        ):
            if isinstance(val, bool) or not (
                isinstance(val, (int, float)) and math.isfinite(float(val)) and float(val) > 0
            ):
                raise ValueError(f"ServoConfig.{name} must be finite and > 0, got {val!r}.")
        if self.absolute_timeout_s is not None and (
            isinstance(self.absolute_timeout_s, bool)
            or not (
                isinstance(self.absolute_timeout_s, (int, float))
                and math.isfinite(float(self.absolute_timeout_s))
                and float(self.absolute_timeout_s) > 0
            )
        ):
            raise ValueError(
                f"ServoConfig.absolute_timeout_s must be None or finite and > 0, got {self.absolute_timeout_s!r}."
            )
        if isinstance(self.settle_ticks, bool) or not isinstance(self.settle_ticks, int):
            raise ValueError(f"ServoConfig.settle_ticks must be int, got {self.settle_ticks!r}.")
        if self.settle_ticks < 1:
            raise ValueError(f"ServoConfig.settle_ticks must be >= 1, got {self.settle_ticks}.")
        if not (1.0 <= float(self.control_hz) <= 200.0):
            raise ValueError(f"ServoConfig.control_hz must be in [1, 200] (hardware-validated), got {self.control_hz}.")


@dataclass
class MaskTrackingConfig:
    """Safety thresholds for fixed-camera mask/depth target filtering.

    Occlusion is intentionally mask-only: when the visible mask becomes
    meaningfully smaller and remains mostly inside a dilation of the trusted
    reference mask, the reference target is frozen.  Centroid and depth are
    still used to distinguish an unchanged full mask from actual movement, but
    never participate in the occlusion decision.  Actual movement needs two
    mutually consistent observations before the trusted target is advanced.
    """

    enabled: bool = True
    min_score: float = 0.35
    dilation_px: int = 4
    static_containment: float = 0.85
    min_visible_ratio: float = 0.25
    occlusion_area_ratio: float = 0.98
    max_static_centroid_shift_px: float = 3.0
    max_static_depth_delta_mm: float = 10.0
    max_depth_span_mm: float = 40.0
    min_valid_depth_ratio: float = 0.50
    motion_min_area_ratio: float = 0.65
    motion_confirm_frames: int = 2
    max_motion_step_mm: float = 35.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("MaskTrackingConfig.enabled must be bool")
        if isinstance(self.dilation_px, bool) or not isinstance(self.dilation_px, int) or self.dilation_px < 0:
            raise ValueError("MaskTrackingConfig.dilation_px must be an integer >= 0")
        for name in ("motion_confirm_frames",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"MaskTrackingConfig.{name} must be an integer >= 1")
        for name in (
            "min_score",
            "static_containment",
            "min_visible_ratio",
            "occlusion_area_ratio",
            "min_valid_depth_ratio",
            "motion_min_area_ratio",
        ):
            value = getattr(self, name)
            if not _is_unit_interval(value):
                raise ValueError(f"MaskTrackingConfig.{name} must be finite and in [0, 1]")
        if not 0.0 < self.motion_min_area_ratio <= 1.0:
            raise ValueError("MaskTrackingConfig.motion_min_area_ratio must be in (0, 1]")
        if not self.min_visible_ratio <= self.motion_min_area_ratio <= 1.0:
            raise ValueError("MaskTrackingConfig.motion_min_area_ratio must be in [min_visible_ratio, 1]")
        if not self.min_visible_ratio <= self.occlusion_area_ratio <= 1.0:
            raise ValueError("MaskTrackingConfig.occlusion_area_ratio must be in [min_visible_ratio, 1]")
        for name in (
            "max_static_centroid_shift_px",
            "max_static_depth_delta_mm",
            "max_depth_span_mm",
            "max_motion_step_mm",
        ):
            value = getattr(self, name)
            if not _is_positive_finite(value):
                raise ValueError(f"MaskTrackingConfig.{name} must be finite and > 0")


@dataclass
class TrackingConfig:
    """Tuning for the fast-path runner (servo / detection only).

    No motion-offset knobs (approach/lift): like the agent path, all working
    heights come from the detection's ``grasp_z`` / ``place_z`` (which already
    embed the calibration offsets ``grasp_z_offset`` / ``chip_thickness``). The
    workflow descends straight to those — no extra hover/lift offset, so there is
    nothing to tune here for motion geometry.
    """

    enabled: bool = True
    detect_hz: float = 5.0  # background detection rate cap
    first_target_timeout_s: float = 8.0  # wait this long for the first detection
    settle_grip_s: float = 0.5  # pause after a gripper command (let it actuate)
    # Cap on post-descend re-align passes before fail-closing (bounds re-servoing
    # when the object keeps moving between detections).
    max_re_align_iters: int = 1
    # A gripper adapter may expose a private ``is_grasp_confirmed`` hook.  When
    # it reports an empty close after ``track_grasp``, return home and run the
    # complete perception/approach/close attempt once more.
    max_grasp_retries: int = 1
    servo: ServoConfig = field(default_factory=ServoConfig)  # track-loop tuning
    # Opt-in at the adapter boundary: only an API exposing the private
    # get_grasp_tracking_sample() hook uses this filter. Other robots keep the
    # original get_grasp_info_simple() tracking path.
    mask_tracking: MaskTrackingConfig = field(default_factory=MaskTrackingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("TrackingConfig.enabled must be bool")
        for name, value in (
            ("detect_hz", self.detect_hz),
            ("first_target_timeout_s", self.first_target_timeout_s),
        ):
            if not (isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0):
                raise ValueError(f"TrackingConfig.{name} must be finite and > 0, got {value!r}.")
        if not (
            isinstance(self.settle_grip_s, (int, float))
            and math.isfinite(float(self.settle_grip_s))
            and float(self.settle_grip_s) >= 0
        ):
            raise ValueError(f"TrackingConfig.settle_grip_s must be finite and >= 0, got {self.settle_grip_s!r}.")
        for name in ("max_re_align_iters", "max_grasp_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"TrackingConfig.{name} must be int, got {value!r}.")
            if value < 0:
                raise ValueError(f"TrackingConfig.{name} must be >= 0, got {value}.")
        if not isinstance(self.mask_tracking, MaskTrackingConfig):
            raise ValueError(
                f"TrackingConfig.mask_tracking must be a MaskTrackingConfig, got {type(self.mask_tracking).__name__}."
            )
