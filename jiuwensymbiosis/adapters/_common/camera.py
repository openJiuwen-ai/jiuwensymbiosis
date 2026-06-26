# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Intel RealSense camera wrapper — robot-agnostic.

Encapsulates one ``pyrealsense2`` pipeline + color/depth stream alignment +
the 3x3 intrinsics matrix needed by ``adapters._common.geometry``.

Lazy import of ``pyrealsense2`` — if the package isn't installed, ``start()``
logs a warning and returns False, and ``grab_frames()`` returns None.

Construction never raises; failure modes (missing package, device not
found, pipeline start error) all yield ``grab_frames() -> None``. Callers
treat "no camera" the same as "no frames", which keeps the "ok=False,
reason=no_camera" fallback chain intact.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RealSenseCamera:
    """One RealSense device, configured for color + depth at a given resolution.

    Lifecycle:
      * ``__init__`` only stores config.
      * ``start()`` opens the pipeline. Idempotent.
      * ``stop()`` releases it. Idempotent.
      * ``grab_frames()`` returns ``None`` until ``start()`` succeeds, then
        ``(rgb_uint8, depth_m_float32)`` per call (or ``None`` on a transient
        frame-grab error).

    The ``log_prefix`` constructor arg controls the log-line tag so adapters
    can keep their historical user-visible prefix unchanged.
    """

    def __init__(
        self,
        serial: str,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        *,
        log_prefix: str = "[RealSense]",
    ) -> None:
        self.serial = serial
        self.resolution = tuple(resolution)
        self.fps = int(fps)
        self._log_prefix = log_prefix

        self._pipeline = None  # pyrealsense2.pipeline once started
        self._align = None  # pyrealsense2.align(rs.stream.color)
        self._depth_scale: float = 1.0
        self._intrinsics: np.ndarray | None = None

    # ----------------------------------------------------------------- state
    @property
    def is_running(self) -> bool:
        """True once ``start()`` has opened the pipeline."""
        return self._pipeline is not None

    @property
    def depth_scale(self) -> float:
        """Meters per raw depth unit. Set after ``start()`` succeeds; 1.0 before."""
        return self._depth_scale

    @property
    def intrinsics(self) -> np.ndarray | None:
        """3x3 K matrix. None until ``start()`` succeeds."""
        return self._intrinsics

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        """Open the camera pipeline.

        Idempotent. Returns True on success, False on any failure (with a
        warning logged).
        """
        if self._pipeline is not None:
            return True
        try:
            import pyrealsense2 as rs
        except ImportError:
            logger.warning(
                "%s pyrealsense2 not installed — skipping camera.",
                self._log_prefix,
            )
            return False
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial)
            config.enable_stream(
                rs.stream.color,
                self.resolution[0],
                self.resolution[1],
                rs.format.bgr8,
                self.fps,
            )
            config.enable_stream(
                rs.stream.depth,
                self.resolution[0],
                self.resolution[1],
                rs.format.z16,
                self.fps,
            )
            profile = pipeline.start(config)
            self._align = rs.align(rs.stream.color)
            self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self._intrinsics = np.array(
                [
                    [color_intr.fx, 0.0, color_intr.ppx],
                    [0.0, color_intr.fy, color_intr.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            for _ in range(10):
                try:
                    pipeline.wait_for_frames(timeout_ms=2000)
                except Exception as e:  # noqa: BLE001
                    # Transient timeouts are expected while the sensor warms
                    # up; log at debug instead of silently swallowing.
                    logger.debug(
                        "%s warm-up frame grab failed (ignored): %s",
                        self._log_prefix,
                        e,
                    )
            self._pipeline = pipeline
            logger.info(
                "%s Camera SN=%s ready (%dx%d@%d). K=[fx=%.1f, fy=%.1f, ppx=%.1f, ppy=%.1f], depth_scale=%.5fm/unit.",
                self._log_prefix,
                self.serial,
                self.resolution[0],
                self.resolution[1],
                self.fps,
                color_intr.fx,
                color_intr.fy,
                color_intr.ppx,
                color_intr.ppy,
                self._depth_scale,
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "%s Camera init failed (%s); continuing without camera.",
                self._log_prefix,
                e,
            )
            self._pipeline = None
            return False

    def stop(self) -> None:
        """Stop the pipeline. Safe to call multiple times or when never started."""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001
                pass
            self._pipeline = None

    # -------------------------------------------------------------- frame grab
    def grab_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Grab one aligned (rgb, depth_m) pair. depth in meters as float32.

        Returns None if the camera isn't running or the read failed.
        """
        if self._pipeline is None:
            return None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=2000)
            if self._align is not None:
                frames = self._align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                return None
            bgr = np.asanyarray(color.get_data())
            rgb = bgr[:, :, ::-1].copy()
            depth_raw = np.asanyarray(depth.get_data())
            depth_m = depth_raw.astype(np.float32) * float(self._depth_scale)
            return rgb, depth_m
        except Exception as e:  # noqa: BLE001
            logger.warning("%s grab_frames error: %s", self._log_prefix, e)
            return None
