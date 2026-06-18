# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ground-truth mock scene for end-to-end visual pipeline testing.

A ``MockScene`` places named objects on a table (Z=0) and renders them through
a synthetic top-down pinhole camera into RGB + depth frames. The rendered frames
are internally consistent with the real ``detect_and_centroid`` +
``pixel_and_depth_to_camera_xyz`` pipeline, so a scene-aware api can run the
*actual* perception pipeline against synthetic data and recover the object's
base-frame XYZ — closing the perception → projection → motion loop in CI
without any hardware.

Camera model (top-down, fixed, NOT eye-in-hand):
  - Camera at base-frame (0, 0, H) looking straight down.
  - tf_base_cam = rot_x(180°) + translation(0, 0, H).
    The 180° X-rotation flips Y and Z so the CV camera convention
    (x=right, y=down, z=into-scene) maps to base (x=right, y=up→down, z=down→up).
  - Forward projection of base (X, Y, Z):
      depth_mm = H - Z
      u = fx * X / depth_mm + ppx
      v = -fy * Y / depth_mm + ppy    (note the minus: y_cam = -Y_base)
  - Inverse via pixel_and_depth_to_camera_xyz + apply_transform(tf_base_cam, ·)
    recovers (X, Y, Z) exactly (verified by round-trip).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from jiuwensymbiosis.adapters._common.geometry import apply_transform, make_transform


@dataclass
class MockObject:
    """One simulated object on the table.

    The object sits on the table (Z=0); its top surface is at Z = size_mm[2].
    ``base_xy_mm`` is the center of the object's footprint in the base frame.
    """

    name: str
    base_xy_mm: tuple[float, float]
    size_mm: tuple[float, float, float]  # (width_x, depth_y, height_z)
    color: tuple[int, int, int]
    score: float = 0.95


def _rot_x_180() -> np.ndarray:
    """180° rotation about X: flips Y and Z (proper rotation, det=+1)."""
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )


@dataclass
class MockScene:
    """A simulated table-top scene viewed by a fixed top-down pinhole camera.

    Attributes:
        objects: Objects on the table (Z=0 plane).
        image_hw: (height, width) of rendered frames.
        camera_height_mm: Camera Z in the base frame (looks straight down).
        intrinsics: 3x3 pinhole camera matrix [[fx,0,ppx],[0,fy,ppy],[0,0,1]].
        tf_base_cam: 4x4 transform from camera frame to base frame.
    """

    objects: list[MockObject] = field(default_factory=list)
    image_hw: tuple[int, int] = (480, 640)
    camera_height_mm: float = 500.0
    intrinsics: np.ndarray = field(
        default_factory=lambda: np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    )
    tf_base_cam: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        r = _rot_x_180()
        t = np.array([0.0, 0.0, self.camera_height_mm], dtype=np.float64)
        self.tf_base_cam = make_transform(r, t)

    def _project_base_to_pixel(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Project a base-frame (X, Y, Z) point to (u, v, depth_m)."""
        depth_mm = self.camera_height_mm - z
        if depth_mm <= 0:
            raise ValueError(f"point at z={z} is at/above camera height {self.camera_height_mm}")
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        ppx, ppy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        u = fx * x / depth_mm + ppx
        v = -fy * y / depth_mm + ppy
        return u, v, depth_mm / 1000.0

    def _object_pixel_rect(self, obj: MockObject) -> tuple[int, int, int, int]:
        """Projected bounding box [u1, v1, u2, v2] of ``obj``'s top surface."""
        h_img, w_img = self.image_hw
        cx, cy = obj.base_xy_mm
        half_w, half_d = obj.size_mm[0] / 2.0, obj.size_mm[1] / 2.0
        z_top = obj.size_mm[2]
        corners = [
            (cx - half_w, cy - half_d, z_top),
            (cx + half_w, cy - half_d, z_top),
            (cx + half_w, cy + half_d, z_top),
            (cx - half_w, cy + half_d, z_top),
        ]
        us = [self._project_base_to_pixel(*c)[0] for c in corners]
        vs = [self._project_base_to_pixel(*c)[1] for c in corners]
        u1, u2 = max(0, int(min(us))), min(w_img, int(max(us)))
        v1, v2 = max(0, int(min(vs))), min(h_img, int(max(vs)))
        return u1, v1, u2, v2

    def render_rgb(self) -> np.ndarray:
        """Render the scene to an HxWx3 uint8 RGB frame."""
        h_img, w_img = self.image_hw
        rgb = np.full((h_img, w_img, 3), 96, dtype=np.uint8)
        for obj in self.objects:
            u1, v1, u2, v2 = self._object_pixel_rect(obj)
            if u2 > u1 and v2 > v1:
                rgb[v1:v2, u1:u2] = obj.color
        return rgb

    def render_depth_m(self) -> np.ndarray:
        """Render the scene to an HxW float32 depth map in meters.

        Background (table at Z=0) is at camera_height_mm / 1000; each object's
        top surface is at (camera_height_mm - height) / 1000.
        """
        h_img, w_img = self.image_hw
        bg_depth = self.camera_height_mm / 1000.0
        depth = np.full((h_img, w_img), bg_depth, dtype=np.float32)
        for obj in self.objects:
            u1, v1, u2, v2 = self._object_pixel_rect(obj)
            if u2 > u1 and v2 > v1:
                depth[v1:v2, u1:u2] = (self.camera_height_mm - obj.size_mm[2]) / 1000.0
        return depth


def mock_seg_fn_from_scene(scene: MockScene) -> Callable[..., list[dict[str, Any]]]:
    """Build a detector-contract segment function backed by ``scene``.

    Matches ``text_prompt`` against ``MockObject.name`` (case-insensitive exact
    or substring). Returns one detection dict per matching object with a filled
    rectangular mask, bounding box, score, and label — the same shape
    ``detect_and_centroid`` consumes.
    """

    def segment_fn(image: Any, text_prompt: str = "") -> list[dict[str, Any]]:
        h_img, w_img = scene.image_hw
        prompt = (text_prompt or "").strip().lower()
        results: list[dict[str, Any]] = []
        for obj in scene.objects:
            name = obj.name.lower()
            if prompt and prompt != name and prompt not in name and name not in prompt:
                continue
            u1, v1, u2, v2 = scene._object_pixel_rect(obj)
            mask = np.zeros((h_img, w_img), dtype=bool)
            if u2 > u1 and v2 > v1:
                mask[v1:v2, u1:u2] = True
            results.append(
                {
                    "mask": mask,
                    "box": [float(u1), float(v1), float(u2), float(v2)],
                    "score": obj.score,
                    "label": obj.name,
                }
            )
        return results

    return segment_fn
