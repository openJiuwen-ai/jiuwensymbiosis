# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A second dual-arm mobile body, assembled from framework mixins only.

Nothing here imports an adapter. The body declares its capabilities and supplies
the three genuinely body-specific pieces — a calibrated frame, a detector, and the
dual-arm/lifter actuation — then inherits everything else: 3-D scene sensing from
``_Scene3DBody`` and the search sweep, face-normal squaring and approach
convergence from ``_ApproachBody``. A plan written for one dual-arm body running
here unchanged is what makes those layers body-agnostic in fact, not just in
intent.

The world is a set of axis-aligned boxes plus a base pose; the camera is a
forward-looking, down-pitched RGBD rendered by ray-casting those boxes. So the
frames the real perception pipeline consumes are internally consistent with the
projection it inverts, and the base genuinely closes the distance as it drives —
the approach loop converges against measured geometry, not a scripted reply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from jiuwensymbiosis.api import defaults
from jiuwensymbiosis.api.actions import (
    ANALYZE_SCENE,
    APPROACH_FOR_GRASP,
    APPROACH_FOR_PLACE,
    DUAL_ARM_GRASP,
    DUAL_ARM_PLACE,
    LOCATE_FOR_GRASP,
    LOCATE_FOR_PLACE,
    SEARCH_TARGET,
    LIFT_TO_CLEARANCE,
    NAVIGATE_RELATIVE,
    ROTATE_BASE,
    SET_LIFT_POSE,
    TURN_WAIST,
    implements,
)
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.components import Approach, Scene3D
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation
from jiuwensymbiosis.perception.frame import CameraFrame
from jiuwensymbiosis.utils.geometry import make_transform

# base←camera at zero pitch: camera x → base −Y, camera y → base −Z, camera z → base +X.
_CAM_FORWARD = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)
# Reach of the two-arm clamp, measured from the base to the payload's near face.
_GRASP_REACH_MM = 700.0


@dataclass
class WorldBox:
    """One axis-aligned box in the world frame (mm)."""

    name: str
    center_mm: tuple[float, float, float]
    size_mm: tuple[float, float, float]
    color: tuple[int, int, int] = (200, 200, 195)

    def bounds_mm(self) -> tuple[np.ndarray, np.ndarray]:
        c = np.asarray(self.center_mm, dtype=np.float64)
        half = np.asarray(self.size_mm, dtype=np.float64) / 2.0
        return c - half, c + half


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_y(pitch: float) -> np.ndarray:
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


class MockDualArmEnv(BaseRobotEnv):
    """Mobile dual-arm hardware: a planar base, a lifter, a waist, and one RGBD camera.

    Positions are millimetres throughout; only the base-motion verbs speak metres,
    which is the framework's convention (detections mm, base moves m).
    """

    capabilities = frozenset(
        {
            "vision.detection",
            "vision.depth",
            "motion.base",
            "motion.lift",
            "motion.waist",
            "motion.goal",
            # Its one camera is bolted to the base — but the base turns, so it can look
            # around. Searching needs a view it can aim, not a second kind of sensor.
            "vision.search",
            "grasp.dual_arm",
        }
    )
    name = "mock_dual_arm"

    def __init__(
        self,
        boxes: list[WorldBox],
        *,
        image_hw: tuple[int, int] = (240, 320),
        focal_px: float = 200.0,
        camera_height_mm: float = 800.0,
        camera_pitch_rad: float = 0.5,
    ) -> None:
        self.boxes = list(boxes)
        self.base_xy_mm = np.zeros(2, dtype=np.float64)
        self.base_yaw_rad = 0.0
        self.holding_payload = False
        self.lifter_q: dict = {}
        self.waist_rad = 0.0
        self.moves: list[dict] = []
        self._connected = False
        self.image_hw = image_hw
        h, w = image_hw
        self.intrinsics = np.array(
            [[focal_px, 0.0, w / 2.0], [0.0, focal_px, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        self.camera_height_mm = float(camera_height_mm)
        self.tf_base_cam = make_transform(
            _rot_y(float(camera_pitch_rad)) @ _CAM_FORWARD,
            np.array([0.0, 0.0, self.camera_height_mm], dtype=np.float64),
        )

    # --- lifecycle ---

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> RobotObservation:
        return RobotObservation(
            pose={"x": float(self.base_xy_mm[0]), "y": float(self.base_xy_mm[1]), "yaw": self.base_yaw_rad},
            joints=[self.waist_rad],
            extra={
                "holding_payload": self.holding_payload,
                "waist_rad": self.waist_rad,
                "lifter": dict(self.lifter_q),
            },
        )

    # --- mobile-base / torso verbs ---

    def navigate_relative(self, dx_m: float, dy_m: float = 0.0, dyaw_rad: float = 0.0) -> dict:
        """Turn by ``dyaw_rad``, then translate (dx forward, dy left) along the NEW heading —
        the order the approach planner assumes ("face the object, then advance")."""
        self.base_yaw_rad += float(dyaw_rad)
        dx_mm, dy_mm = float(dx_m) * 1000.0, float(dy_m) * 1000.0
        self.base_xy_mm += (_rot_z(self.base_yaw_rad) @ np.array([dx_mm, dy_mm, 0.0]))[:2]
        move = {"ok": True, "dx_m": float(dx_m), "dy_m": float(dy_m), "dyaw_rad": float(dyaw_rad)}
        self.moves.append(move)
        return move

    def navigate_arc(self, radius_m: float, dyaw_rad: float) -> dict:
        """One constant-curvature arc; the turn centre sits on the side ``dyaw_rad`` points to."""
        side = 1.0 if dyaw_rad >= 0 else -1.0
        r_mm, theta = abs(float(radius_m)) * 1000.0, float(dyaw_rad)
        chord = np.array([side * r_mm * math.sin(theta), side * r_mm * (1.0 - math.cos(theta)), 0.0])
        self.base_xy_mm += (_rot_z(self.base_yaw_rad) @ chord)[:2]
        self.base_yaw_rad += theta
        move = {"ok": True, "radius_m": float(radius_m), "dyaw_rad": theta}
        self.moves.append(move)
        return move

    def set_lifter(self, q_lifter: dict) -> dict:
        self.lifter_q = dict(q_lifter)
        return {"ok": True, "q_lifter": dict(self.lifter_q)}

    def turn_waist(self, delta_rad: float) -> dict:
        self.waist_rad += float(delta_rad)
        return {"ok": True, "waist_rad": self.waist_rad}

    def home(self) -> None:
        self.lifter_q = {}
        self.waist_rad = 0.0

    # --- camera ---

    def grab_calibrated_frame(self, camera: str | None = None) -> CameraFrame:
        rgb, depth_m, _owner = self._render()
        return CameraFrame(rgb=rgb, depth_m=depth_m, intrinsics=self.intrinsics, tf_base_cam=self.tf_base_cam)

    def segment(self, image: Any, text_prompt: str = "") -> list[dict]:
        """Open-vocabulary detector contract: name-matched masks over the current view."""
        _rgb, _depth, owner = self._render()
        prompt = (text_prompt or "").strip().lower()
        out: list[dict] = []
        for i, box in enumerate(self.boxes):
            name = box.name.lower()
            if prompt and prompt != name and prompt not in name and name not in prompt:
                continue
            mask = owner == i
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            out.append(
                {
                    "mask": mask,
                    "box": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                    "score": 0.95,
                    "label": box.name,
                }
            )
        return out

    def _camera_rays(self) -> tuple[np.ndarray, np.ndarray]:
        """Ray origin (world mm) and per-pixel direction whose z component is the camera depth."""
        h, w = self.image_hw
        vs, us = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        ppx, ppy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        d_cam = np.stack([(us - ppx) / fx, (vs - ppy) / fy, np.ones_like(us)], axis=-1)
        dirs = d_cam @ (_rot_z(self.base_yaw_rad) @ self.tf_base_cam[:3, :3]).T
        # An exactly axis-parallel ray would divide by zero in the slab test below.
        dirs = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
        origin = np.array([self.base_xy_mm[0], self.base_xy_mm[1], self.camera_height_mm], dtype=np.float64)
        return origin, dirs

    @staticmethod
    def _slab_hit(box: WorldBox, origin: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ray/box slab test → (depth along the camera z axis, hit mask)."""
        lo, hi = box.bounds_mm()
        t0, t1 = (lo - origin) / dirs, (hi - origin) / dirs
        near = np.minimum(t0, t1).max(axis=-1)
        far = np.maximum(t0, t1).min(axis=-1)
        hit = (near <= far) & (far > 0.0)
        return np.where(hit, np.maximum(near, 0.0), 0.0), hit

    def _render(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Z-buffered (rgb uint8, depth m, per-pixel box index) of the current view."""
        origin, dirs = self._camera_rays()
        h, w = self.image_hw
        depth = np.zeros((h, w), dtype=np.float64)
        owner = np.full((h, w), -1, dtype=np.int32)
        for i, box in enumerate(self.boxes):
            d, hit = self._slab_hit(box, origin, dirs)
            closer = hit & ((owner < 0) | (d < depth))
            depth = np.where(closer, d, depth)
            owner = np.where(closer, i, owner)
        rgb = np.full((h, w, 3), 96, dtype=np.uint8)
        for i, box in enumerate(self.boxes):
            rgb[owner == i] = box.color
        return rgb, (depth / 1000.0).astype(np.float32), owner


class MockDualArmApi(BaseRobotApi):
    """The api layer of the second body: two sensing components and its own actuation.

    Every action is declared against the shared spec, exactly as an adapter declares
    it — the base/lifter/waist verbs by forwarding to ``api.defaults`` because nothing
    about them is body-specific, the dual-arm ones by writing the body out. What the
    spec fixes (schema, requires/provides, capability gate) this body cannot restate,
    so a plan compiled for cruzr means the same thing here.
    """

    def __init__(self, env: MockDualArmEnv) -> None:
        super().__init__(env)
        self.env: MockDualArmEnv = env
        # Held, not inherited — this file is the complete list of what the body offers.
        self._scene = Scene3D(self)
        self._nav = Approach(self)
        # Read by Scene3D's detector hook default — no override needed.
        self._seg_fn = env.segment
        self.calls: list[str] = []

    @implements(LOCATE_FOR_GRASP)
    def locate_for_grasp(self, object_name: str = "box", reference: str | None = None,
                         relation: str = "on") -> dict:
        return self._scene.locate_for_grasp(object_name, reference, relation)

    @implements(LOCATE_FOR_PLACE)
    def locate_for_place(self, object_name: str = "table", reference: str | None = None,
                         relation: str = "on") -> dict:
        return self._scene.locate_for_place(object_name, reference, relation)

    @implements(ANALYZE_SCENE)
    def analyze_scene(self, object_name: str = "box") -> dict:
        return self._scene.analyze_scene(object_name)

    @implements(SEARCH_TARGET)
    def search_target(self, object_name: str = "box", reference: str | None = None,
                      relation: str = "on") -> dict:
        return self._nav.search_target(object_name, reference, relation)

    @implements(APPROACH_FOR_GRASP)
    def approach_for_grasp(self, object_name: str = "box", reference: str | None = None,
                           relation: str = "on") -> dict:
        return self._nav.approach_for_grasp(object_name, reference, relation)

    @implements(APPROACH_FOR_PLACE)
    def approach_for_place(self, object_name: str = "table", reference: str | None = None,
                           relation: str = "on") -> dict:
        return self._nav.approach_for_place(object_name, reference, relation)

    @implements(NAVIGATE_RELATIVE)
    def navigate_relative(self, dx_m: float, dy_m: float = 0.0, dyaw_rad: float = 0.0) -> dict:
        return defaults.navigate_relative(self, dx_m, dy_m, dyaw_rad)

    @implements(ROTATE_BASE)
    def rotate_base(self, dyaw_rad: float) -> dict:
        return defaults.rotate_base(self, dyaw_rad)

    @implements(SET_LIFT_POSE)
    def set_lift_pose(self, q_lifter: dict) -> dict:
        return defaults.set_lift_pose(self, q_lifter)

    @implements(TURN_WAIST)
    def turn_waist(self, delta_rad: float) -> dict:
        return defaults.turn_waist(self, delta_rad)

    @implements(LIFT_TO_CLEARANCE)
    def lift_to_clearance(self) -> dict:
        self.calls.append("lift_to_clearance")
        if not self.env.holding_payload:
            return {"ok": False, "reason": "no_payload"}
        self.env.set_lifter({"lift": 0.35})
        return {"ok": True, "clearance_mm": 350.0}

    @implements(DUAL_ARM_GRASP)
    def dual_arm_grasp(self, target: dict | None = None) -> dict:
        self.calls.append("dual_arm_grasp")
        det = target if isinstance(target, dict) else self._last_detection
        if not det or not det.get("ok"):
            return {"ok": False, "reason": "no_detection"}
        front_x = float(det["front_x_mm"])
        if front_x > _GRASP_REACH_MM:
            return {"ok": False, "reason": "out_of_reach", "front_x_mm": front_x}
        self.env.holding_payload = True
        return {"ok": True, "front_x_mm": front_x}

    @implements(DUAL_ARM_PLACE)
    def dual_arm_place(self, target: dict | None = None, surface: dict | None = None) -> dict:
        self.calls.append("dual_arm_place")
        surf = surface if isinstance(surface, dict) else self._last_surface
        if not self.env.holding_payload:
            return {"ok": False, "reason": "no_payload"}
        if not surf or not surf.get("ok"):
            return {"ok": False, "reason": "no_surface"}
        self.env.holding_payload = False
        return {"ok": True, "surface_z_mm": float(surf["surface_z_mm"])}


class MockDualArmSession:
    """Minimal session pairing the two, for the runner and the world-state snapshot."""

    def __init__(self, boxes: list[WorldBox], **env_kwargs: Any) -> None:
        self.env = MockDualArmEnv(boxes, **env_kwargs)
        self.env.connect()
        self.api = MockDualArmApi(self.env)
