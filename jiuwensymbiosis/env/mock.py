# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Mock env for tests / dryrun. No hardware needed."""

from __future__ import annotations

from typing import Optional

import numpy as np

from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation


class MockArmEnv(BaseRobotEnv):
    """A stand-in 4-DOF arm. Tracks pose in memory; renders a dummy 2D scene."""

    capabilities = frozenset(
        {
            "motion.cartesian",
            "grasp.parallel",
            "vision.camera",
            "vision.detection",
        }
    )
    name = "mock_arm"

    def __init__(
        self,
        home_pose: dict | None = None,
        z_min_safe: float = 0.0,
        image_hw: tuple[int, int] = (480, 640),
        workspace_bounds: Optional[tuple[float, float, float, float]] = None,
    ) -> None:
        """Initialize mock arm with given home pose and safety limits."""
        self._home = home_pose or {"x": 200.0, "y": 0.0, "z": 250.0, "r": 0.0}
        self._pose = dict(self._home)
        self._z_min_safe = z_min_safe
        self._workspace_bounds = workspace_bounds
        self._suction = False
        self._connected = False
        self._image_hw = image_hw
        self._move_log: list[dict] = []

    # ------------------------------------------------ formal hardware contract
    @property
    def z_min_safe(self) -> Optional[float]:
        """Tip-frame Z floor (mm). Exposes the ``BaseRobotEnv`` contract property."""
        return self._z_min_safe

    @property
    def workspace_bounds(self) -> Optional[tuple[float, float, float, float]]:
        """XY workspace bounds, or None. Exposes the ``BaseRobotEnv`` contract property."""
        return self._workspace_bounds

    # -------------------------------------------------------------- lifecycle
    def connect(self) -> None:
        """Set connected flag."""
        self._connected = True

    def disconnect(self) -> None:
        """Clear connected flag."""
        self._connected = False

    def reset(self) -> None:
        """Reset pose back to home, release suction, clear move log."""
        self._pose = dict(self._home)
        self._suction = False
        self._move_log.clear()

    # ----------------------------------------------------------------- query
    def get_observation(self) -> RobotObservation:
        """Return a simulated observation with a dummy RGB frame and current pose."""
        h, w = self._image_hw
        rgb = np.full((h, w, 3), 96, dtype=np.uint8)
        # Draw a marker at center to simulate a detection target.
        cy, cx = h // 2, w // 2
        rgb[cy - 8: cy + 8, cx - 8: cx + 8] = (255, 255, 255)
        return RobotObservation(
            pose=dict(self._pose),
            rgb=rgb,
            extra={"suction": self._suction, "z_min_safe": self._z_min_safe},
        )

    # ------------------------------------------------------------------ ops
    def move(self, x: float, y: float, z: float, r: float | None = None) -> None:
        """Move to an absolute Cartesian pose. Raises if z below safe floor."""
        if z < self._z_min_safe:
            raise RuntimeError(f"z={z} below z_min_safe={self._z_min_safe}")
        self._pose["x"] = float(x)
        self._pose["y"] = float(y)
        self._pose["z"] = float(z)
        if r is not None:
            self._pose["r"] = float(r)
        self._move_log.append(dict(self._pose))

    def home(self) -> None:
        """Return to the home pose."""
        self._pose = dict(self._home)
        self._move_log.append(dict(self._pose))

    @property
    def home_pose(self):
        """Return a copy of the home pose."""
        return dict(self._home)

    @property
    def tool_offset_mm(self) -> float:
        """Mock tool offset — zero for a simple arm."""
        return 0.0

    def grab_rgb(self):
        """Override: return the observation RGB directly (avoids full snapshot)."""
        return self.get_observation().rgb

    def set_suction(self, on: bool) -> None:
        """Set the simulated suction state."""
        self._suction = bool(on)
