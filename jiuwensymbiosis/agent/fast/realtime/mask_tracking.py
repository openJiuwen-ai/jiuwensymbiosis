# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Mask-aware target filtering for fixed-camera fast grasp tracking.

This module is deliberately opt-in.  The generic fast runner only constructs
the filter when an adapter exposes the private
``get_grasp_tracking_sample()`` hook; ordinary ``get_grasp_info_simple()``
results and robots without that hook keep their existing behaviour.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from jiuwensymbiosis.agent.execution_config import MaskTrackingConfig as MaskTrackingConfig

logger = logging.getLogger(__name__)


class MaskTrackingState:
    """Stable string values attached to accepted internal tracking targets."""

    VISIBLE_STABLE = "visible_stable"
    VISIBLE_MOVING = "visible_moving"
    MOTION_PENDING = "motion_pending"
    OCCLUDED_STATIC = "occluded_static"
    BLIND_LAST_TARGET = "blind_last_target"
    INVALID_OUTLIER = "invalid_outlier"
    LOST = "lost"


@dataclass(frozen=True)
class TrackingSample:
    """A reliable tracking sample handed to the ``_update_*`` classifiers."""

    current: dict[str, Any]
    mask: np.ndarray
    centroid: tuple[float, float]
    depth_mm: float
    depth_span_mm: float
    mask_metrics: dict[str, Any]
    area_ratio: float


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square-dilate a 2-D bool mask without adding an OpenCV dependency."""
    if radius <= 0:
        return mask
    height, width = mask.shape
    padded = np.pad(mask.astype(np.uint8, copy=False), radius)
    # Integral image gives every (2r+1)x(2r+1) window sum in O(HW).
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    # fmt: off
    # Slice colons sit before binary arithmetic operands; ruff format would
    # re-add the leading space (E203), so this block is exempt from formatting.
    window_sum = (
        integral[size: size + height, size: size + width]
        - integral[:height, size: size + width]
        - integral[size: size + height, :width]
        + integral[:height, :width]
    )
    # fmt: on
    return window_sum > 0


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(np.median(xs)), float(np.median(ys))


def _target_xyz(target: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [float(target["x"]), float(target["y"]), float(target["grasp_z"])],
        dtype=np.float64,
    )


class MaskTargetFilter:
    """Keep a trusted target across static occlusion and reject mask outliers."""

    _PRIVATE_SAMPLE_KEYS = frozenset(
        {
            "_tracking_mask",
            "_tracking_depth_span_mm",
            "_tracking_valid_depth_ratio",
        }
    )

    def __init__(self, config: MaskTrackingConfig | None = None, *, name: str = "target") -> None:
        self.config = config or MaskTrackingConfig()
        self._name = name
        self._trusted: dict[str, Any] | None = None
        self._anchor_mask: np.ndarray | None = None
        self._anchor_dilated: np.ndarray | None = None
        self._anchor_centroid: tuple[float, float] | None = None
        self._anchor_depth_mm = 0.0
        self._motion_candidate: dict[str, Any] | None = None
        self._motion_candidate_count = 0
        self._state: str | None = None

    @property
    def state(self) -> str | None:
        return self._state

    def miss(self, reason: str = "not_detected") -> dict[str, Any] | None:
        """Use the last trusted target when no reliable object is detected.

        The filter only lives for one bounded ``track_grasp`` operation, so
        retaining the target here cannot leak into a later grasp attempt.
        """
        return self._blind_or_lost(reason, {})

    def _blind_or_lost(self, reason: str, metrics: dict[str, Any]) -> dict[str, Any] | None:
        self._motion_candidate = None
        self._motion_candidate_count = 0
        detail = {"reason": reason, **metrics}
        if self._trusted is None:
            self._transition(MaskTrackingState.LOST, detail)
            return None
        return self._accepted(self._trusted, MaskTrackingState.BLIND_LAST_TARGET, detail)

    @staticmethod
    def _parse_mask(sample: dict[str, Any]) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
        try:
            mask = np.asarray(sample["_tracking_mask"], dtype=bool)
        except (KeyError, TypeError, ValueError) as exc:
            return None, "malformed_tracking_mask", {"error": str(exc)}
        if mask.ndim != 2 or not mask.any():
            return None, "empty_tracking_mask", {"shape": tuple(mask.shape)}
        return mask.copy(), None, {}

    def update(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Filter one normalized private tracking sample.

        ``None`` means this observation must not advance/refresh the background
        tracker.  An accepted dict never contains the raw NumPy mask.
        """
        mask, mask_reason, mask_metrics = self._parse_mask(sample)
        if mask is None:
            return self._blind_or_lost(mask_reason or "invalid_tracking_mask", mask_metrics)

        mask_result, mask_metrics = self._classify_mask(mask)
        if mask_result is not None:
            return mask_result

        parsed, quality_reason, quality_metrics = self._parse_reliable_target(sample, mask)
        if parsed is None:
            return self._blind_or_lost(quality_reason or "unreliable_tracking_sample", quality_metrics)
        current, centroid, depth_mm, depth_span_mm = parsed
        area_ratio = float(mask_metrics.get("area_ratio", 0.0))
        return self._update_reliable_target(
            TrackingSample(
                current=current,
                mask=mask,
                centroid=centroid,
                depth_mm=depth_mm,
                depth_span_mm=depth_span_mm,
                mask_metrics=mask_metrics,
                area_ratio=area_ratio,
            )
        )

    def _classify_mask(self, mask: np.ndarray) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Handle trusted-mask geometry before reading unreliable depth/score."""
        if self._trusted is None:
            return None, {}
        if self._anchor_mask is None or self._anchor_dilated is None:
            return self._blind_or_lost("missing_anchor_mask", {}), {}
        if mask.shape != self._anchor_mask.shape:
            return (
                self._blind_or_lost(
                    "mask_shape_changed",
                    {"shape": tuple(mask.shape), "anchor_shape": tuple(self._anchor_mask.shape)},
                ),
                {},
            )

        area = int(mask.sum())
        anchor_area = int(self._anchor_mask.sum())
        area_ratio = float(area) / float(anchor_area)
        contained = int(np.logical_and(mask, self._anchor_dilated).sum())
        containment = float(contained) / float(area)
        metrics = {
            "area_ratio": round(area_ratio, 4),
            "containment": round(containment, 4),
        }

        cfg = self.config
        # Occlusion is mask-only and precedes score/depth parsing because those
        # values are expected to degrade once the gripper covers the object.
        if area_ratio < cfg.occlusion_area_ratio and containment >= cfg.static_containment:
            self._motion_candidate = None
            self._motion_candidate_count = 0
            return self._accepted(self._trusted, MaskTrackingState.OCCLUDED_STATIC, metrics), metrics
        if area_ratio < cfg.min_visible_ratio:
            return self._blind_or_lost("visible_mask_too_small", metrics), metrics
        return None, metrics

    def _update_reliable_target(self, sample: TrackingSample) -> dict[str, Any] | None:
        """Classify one sample whose mask, geometry, and depth are reliable."""
        if self._trusted is None:
            self._accept_anchor(sample.current, sample.mask, sample.centroid, sample.depth_mm)
            return self._accepted(sample.current, MaskTrackingState.VISIBLE_STABLE, {"initial": True})

        if self._anchor_centroid is None:
            return self._blind_or_lost("missing_anchor_centroid", sample.mask_metrics)
        containment = float(sample.mask_metrics["containment"])
        centroid_shift = math.hypot(
            sample.centroid[0] - self._anchor_centroid[0],
            sample.centroid[1] - self._anchor_centroid[1],
        )
        depth_delta_mm = abs(sample.depth_mm - self._anchor_depth_mm)
        metrics = {
            **sample.mask_metrics,
            "centroid_shift_px": round(centroid_shift, 3),
            "depth_delta_mm": round(depth_delta_mm, 3),
            "depth_span_mm": round(sample.depth_span_mm, 3),
        }

        cfg = self.config
        if (
            containment >= cfg.static_containment
            and centroid_shift <= cfg.max_static_centroid_shift_px
            and depth_delta_mm <= cfg.max_static_depth_delta_mm
        ):
            self._motion_candidate = None
            self._motion_candidate_count = 0
            # High mask overlap + unchanged centroid/depth means "stationary";
            # preserve the fixed anchor instead of integrating projection noise.
            return self._accepted(self._trusted, MaskTrackingState.VISIBLE_STABLE, metrics)

        return self._update_motion_candidate(replace(sample, mask_metrics=metrics))

    def _update_motion_candidate(self, sample: TrackingSample) -> dict[str, Any] | None:
        """Require consistent observations before advancing a moving target."""
        cfg = self.config
        metrics = sample.mask_metrics
        if sample.area_ratio < cfg.motion_min_area_ratio or sample.area_ratio > (1.0 / cfg.motion_min_area_ratio):
            return self._reject("mask_area_inconsistent_with_motion", metrics)

        if self._motion_candidate is None:
            self._motion_candidate = sample.current
            self._motion_candidate_count = 1
            self._transition(MaskTrackingState.MOTION_PENDING, metrics)
            return None

        step_mm = float(np.linalg.norm(_target_xyz(sample.current) - _target_xyz(self._motion_candidate)))
        metrics["candidate_step_mm"] = round(step_mm, 3)
        if step_mm > cfg.max_motion_step_mm:
            # Treat this observation as a new first candidate.  No output means
            # the old trusted frame is not falsely made "fresh".
            self._motion_candidate = sample.current
            self._motion_candidate_count = 1
            self._transition(MaskTrackingState.INVALID_OUTLIER, {"reason": "motion_step_too_large", **metrics})
            return None

        self._motion_candidate = sample.current
        self._motion_candidate_count += 1
        if self._motion_candidate_count < cfg.motion_confirm_frames:
            self._transition(MaskTrackingState.MOTION_PENDING, metrics)
            return None

        self._accept_anchor(sample.current, sample.mask, sample.centroid, sample.depth_mm)
        return self._accepted(sample.current, MaskTrackingState.VISIBLE_MOVING, metrics)

    def _parse_reliable_target(
        self,
        sample: dict[str, Any],
        mask: np.ndarray,
    ) -> tuple[
        tuple[dict[str, Any], tuple[float, float], float, float] | None,
        str | None,
        dict[str, Any],
    ]:
        cfg = self.config
        try:
            position = sample["position"]
            if not isinstance(position, (list, tuple)) or len(position) < 2:
                raise ValueError("position must contain x/y")
            position_x = float(position[0])
            position_y = float(position[1])
            score = float(sample["score"])
            depth_mm = float(sample["depth_m"]) * 1000.0
            depth_span_mm = float(sample["_tracking_depth_span_mm"])
            valid_depth_ratio = float(sample["_tracking_valid_depth_ratio"])
            _target_xyz(sample)
        except (KeyError, TypeError, ValueError) as exc:
            return None, "malformed_tracking_sample", {"error": str(exc)}
        if not all(
            math.isfinite(v) for v in (position_x, position_y, score, depth_mm, depth_span_mm, valid_depth_ratio)
        ):
            return None, "non_finite_tracking_sample", {}
        if score < cfg.min_score:
            return None, "score_too_low", {"score": round(score, 3)}
        if depth_span_mm > cfg.max_depth_span_mm:
            return (
                None,
                "depth_window_inconsistent",
                {"depth_span_mm": round(depth_span_mm, 3)},
            )
        if valid_depth_ratio < cfg.min_valid_depth_ratio:
            return (
                None,
                "insufficient_valid_depth",
                {"valid_depth_ratio": round(valid_depth_ratio, 3)},
            )
        current = {key: value for key, value in sample.items() if key not in self._PRIVATE_SAMPLE_KEYS}
        return (current, _mask_centroid(mask), depth_mm, depth_span_mm), None, {}

    def _accept_anchor(
        self,
        current: dict[str, Any],
        mask: np.ndarray,
        centroid: tuple[float, float],
        depth_mm: float,
    ) -> None:
        self._trusted = dict(current)
        self._anchor_mask = mask.copy()
        self._anchor_dilated = _binary_dilate(self._anchor_mask, self.config.dilation_px)
        self._anchor_centroid = centroid
        self._anchor_depth_mm = depth_mm
        self._motion_candidate = None
        self._motion_candidate_count = 0

    def _accepted(self, target: dict[str, Any], state: str, metrics: dict[str, Any]) -> dict[str, Any]:
        accepted = dict(target)
        accepted["_tracking_state"] = state
        accepted["_tracking_metrics"] = dict(metrics)
        self._transition(state, metrics)
        return accepted

    def _reject(self, reason: str, metrics: dict[str, Any]) -> None:
        self._motion_candidate = None
        self._motion_candidate_count = 0
        self._transition(MaskTrackingState.INVALID_OUTLIER, {"reason": reason, **metrics})
        return None

    def _transition(self, state: str, metrics: dict[str, Any]) -> None:
        if state == self._state:
            return
        previous = self._state or "none"
        self._state = state
        logger.info(
            "[mask-track-%s] %s -> %s metrics=%s",
            self._name,
            previous,
            state,
            metrics,
        )


__all__ = ["MaskTargetFilter", "MaskTrackingConfig", "MaskTrackingState"]
