"""Associate observed target surfaces across bounded base motions.

Only RGB-D and measured base odometry enter this comparison. No scene entity,
object pose or complete object geometry is needed. Evidence stays host-side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from jiuwensymbiosis.perception.frame import CameraFrame


@dataclass(frozen=True)
class TargetObservation:
    query: tuple[str, str | None, str]
    frame_id: str
    captured_at: float
    odom_frame: str
    points_m: np.ndarray
    color_histogram: np.ndarray

    @classmethod
    def from_frame(cls, query, frame: CameraFrame, mask) -> TargetObservation | None:
        if frame.depth_m is None or frame.intrinsics is None or frame.tf_base_cam is None:
            return None
        base = frame.metadata.get("base_sample")
        reference = frame.metadata.get("scene_instance_id") or (base or {}).get("odom_frame")
        if not base or not reference or not frame.frame_id or frame.captured_monotonic_s is None:
            return None
        mask = np.asarray(mask, dtype=bool)
        valid = mask & np.isfinite(frame.depth_m) & (frame.depth_m > 0)
        y, x = np.nonzero(valid)
        if len(x) < 50:
            return None
        indices = np.linspace(0, len(x) - 1, min(512, len(x)), dtype=int)
        x, y = x[indices], y[indices]
        z, k = frame.depth_m[y, x], frame.intrinsics
        camera = np.column_stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z))
        transform = frame.tf_base_cam
        points = camera @ transform[:3, :3].T + transform[:3, 3] / 1000
        yaw = base["body_yaw_rad"]
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        points = points @ rotation.T + np.asarray(base["position_m"])
        if not np.all(np.isfinite(points)):
            return None
        colors = frame.rgb[mask]
        histogram = np.concatenate([np.histogram(colors[:, i], bins=8, range=(0, 256))[0] for i in range(3)])
        histogram = histogram.astype(float) / max(1, 3 * len(colors))
        return cls(query, frame.frame_id, frame.captured_monotonic_s, str(reference), points, histogram)

    def mismatch(self, newer: TargetObservation) -> str | None:
        if self.query != newer.query or self.odom_frame != newer.odom_frame:
            return "target_reference_changed"
        if self.frame_id == newer.frame_id or newer.captured_at <= self.captured_at:
            return "stale_target_observation"
        # Reciprocal surface overlap tolerates a changing visible median, but
        # rejects jumping to another same-class instance after a move.
        distances = np.linalg.norm(self.points_m[:, None] - newer.points_m[None, :], axis=2)
        overlap = min(float(np.mean(distances.min(axis=0) <= 0.18)), float(np.mean(distances.min(axis=1) <= 0.18)))
        appearance = float(np.minimum(self.color_histogram, newer.color_histogram).sum())
        if overlap < 0.5 or appearance < 0.5:
            return "target_identity_changed"
        return None

    def visible_center_shift_m(self, newer: TargetObservation) -> float:
        """Horizontal median change after compensating for measured base motion.

        Call only after association succeeds. This measures visible-surface
        variation, not the motion of an unseen complete object centre.
        """
        return float(np.linalg.norm(np.median(self.points_m[:, :2], axis=0) - np.median(newer.points_m[:, :2], axis=0)))
