# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SceneMockApi — a scene-aware mock api that runs the *real* perception pipeline.

Unlike ``MockApi`` (a simple stub returning hardcoded dicts), ``SceneMockApi``
renders RGB + depth frames from a ``MockScene``, runs the actual
``detect_and_centroid`` (from ``adapters._common.vision``) with a scene-backed
segment function, and back-projects via the real
``pixel_and_depth_to_camera_xyz`` + ``apply_transform`` (from
``adapters._common.geometry``). This closes the perception → projection →
motion data-flow loop against known ground truth in CI.

Motion / gripper methods are inherited unchanged from ``MockApi`` — only the
vision methods are overridden. The scene must be attached to the env
(``MockArmEnv(scene=...)``) so that ``get_observation`` / ``get_image`` render
the same frames the pipeline consumes.
"""

from __future__ import annotations

from jiuwensymbiosis.adapters._common.geometry import (
    apply_transform,
    pixel_and_depth_to_camera_xyz,
)
from jiuwensymbiosis.adapters._common.vision import detect_and_centroid
from jiuwensymbiosis.api.decorators import robot_tool
from tests.mocks.mock_api import MockApi
from tests.mocks.mock_scene import MockScene, mock_seg_fn_from_scene


class _PoseShim:
    """Minimal pose shim exposing ``.x/.y/.z/.r`` for ``detect_and_centroid``'s log line."""

    __slots__ = ("x", "y", "z", "r")

    def __init__(self, pose: dict) -> None:
        self.x = float(pose.get("x", 0.0))
        self.y = float(pose.get("y", 0.0))
        self.z = float(pose.get("z", 0.0))
        self.r = float(pose.get("r", 0.0))


class SceneMockApi(MockApi):
    """Mock api that runs the real detect + back-project pipeline against a MockScene.

    The scene is read from ``env._scene`` (set via ``MockArmEnv(scene=...)``).
    Vision methods render frames from the scene, run ``detect_and_centroid``,
    and back-project through the scene's intrinsics + ``tf_base_cam``.
    """

    def __init__(self, env, *, grasp_z_offset_mm: float = 0.0, chip_thickness_mm: float = 5.0) -> None:
        super().__init__(env)
        scene = getattr(env, "_scene", None)
        if scene is None:
            raise ValueError(
                "SceneMockApi requires env to have a scene; construct MockArmEnv(scene=MockScene(...)) first."
            )
        self._scene: MockScene = scene
        self._grasp_z_offset_mm = grasp_z_offset_mm
        self._chip_thickness_mm = chip_thickness_mm

    @robot_tool(
        desc="Detect object_name in the scene and return grasp/place geometry. "
        "Runs the real detect_and_centroid + pinhole back-projection against the "
        'mock scene. Returns {"ok": bool, "object": str, "position": [x,y,z]_mm, '
        '"grasp_z": float, "grasp_position": [x,y,z]_mm, "place_z": float, '
        '"place_position": [x,y,z]_mm, "score": float, "pixel_uv": [u,v], "depth_m": float}.',
    )
    def get_grasp_info_simple(self, object_name: str) -> dict:
        self._call_log.append(f"get_grasp_info_simple({object_name!r})")
        scene = self._scene
        rgb = scene.render_rgb()
        depth = scene.render_depth_m()
        seg_fn = mock_seg_fn_from_scene(scene)
        pose = self.env.get_observation().pose or {}
        det = detect_and_centroid(
            rgb=rgb,
            depth_img_m=depth,
            seg_fn=seg_fn,
            object_name=object_name,
            tcp_at_grab=_PoseShim(pose),
        )
        if not det.get("ok"):
            return det
        u, v, depth_m = det["u"], det["v"], det["depth_m"]
        p_cam = pixel_and_depth_to_camera_xyz((u, v), depth_m, scene.intrinsics)
        xyz = apply_transform(scene.tf_base_cam, p_cam)
        x_f, y_f, top_z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        grasp_z = top_z + self._grasp_z_offset_mm
        z_floor = self.env.z_min_safe
        if z_floor is not None:
            grasp_z = max(grasp_z, float(z_floor))
        place_z = top_z + self._chip_thickness_mm
        return {
            "ok": True,
            "object": object_name,
            "position": [x_f, y_f, top_z],
            "grasp_z": grasp_z,
            "grasp_position": [x_f, y_f, grasp_z],
            "place_z": place_z,
            "place_position": [x_f, y_f, place_z],
            "score": float(det["best"]["score"]),
            "pixel_uv": [u, v],
            "depth_m": depth_m,
        }

    @robot_tool(desc="Pixel (u,v) at depth_m → base-frame XYZ in mm via the mock camera model.")
    def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
        self._call_log.append(f"pixel_to_base_xyz({u},{v},{depth_m})")
        scene = self._scene
        p_cam = pixel_and_depth_to_camera_xyz((u, v), depth_m, scene.intrinsics)
        xyz = apply_transform(scene.tf_base_cam, p_cam)
        return {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}

    @robot_tool(desc="Scene analysis grounded on object_name. Returns detection counts + top scores.")
    def analyze_scene(self, object_name: str | None = None) -> dict:
        target = object_name or ""
        rgb = self._scene.render_rgb()
        seg_fn = mock_seg_fn_from_scene(self._scene)
        results = seg_fn(rgb, text_prompt=target)
        scores = sorted((float(r.get("score", 0.0)) for r in results), reverse=True)
        return {
            "ok": True,
            "object": target,
            "n_detections": len(results),
            "top_scores": scores[:5],
        }
