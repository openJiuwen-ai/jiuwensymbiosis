# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""PerceptionEngine — background camera preview + click reprojection engine for the 「感知测试」 tool.

A background thread connects the camera, loops grabbing frames to push a live preview, and reprojects
UI-clicked pixels to base-frame coordinates. The UI drains events via ``drain()``: ``preview_started`` /
``frame`` (data URI) / ``point_result`` (dict) / ``error`` (dict) / ``preview_stopped``.

Usage::

    engine = PerceptionEngine(lambda: body.build_real_session(cfg))
    engine.start()
    engine.request_point(u, v)  # result comes back via drain()'s "point_result" event
    engine.stop()
"""

from __future__ import annotations

import math
import queue
import time
from collections.abc import Callable
from threading import Thread
from typing import TYPE_CHECKING, Any

from jiuwensymbiosis.gui import imaging
from jiuwensymbiosis.utils.logging import get_logger

if TYPE_CHECKING:
    from jiuwensymbiosis.agent.session import RobotSession

logger = get_logger(__name__)

__all__ = ["PerceptionEngine"]

# Preview loop period (s): ~12fps — smooth enough, and a click is serviced within one tick.
_LOOP_PERIOD_S = 0.08


class PerceptionEngine:
    """Background thread + event queue for camera preview and click reprojection.

    Args:
        session_factory: zero-arg callback returning an **unconnected** ``RobotSession``. The UI
            typically passes ``lambda: body.build_real_session(cfg_data)``; tests can inject a
            scene-backed session.
        z_correction_mm: display-layer grasp Z correction (mm). When nonzero, ``point_result`` also
            carries ``z_corrected`` / ``z_correction_mm`` to match the actual grasp descent height.
            Does not change the reprojection itself.
    """

    def __init__(
        self,
        session_factory: Callable[[], RobotSession],
        *,
        z_correction_mm: float = 0.0,
    ) -> None:
        """Store the session factory and z correction; thread and queue start in ``start``."""
        self._session_factory = session_factory
        self._z_correction_mm = float(z_correction_mm)
        self._events: queue.Queue = queue.Queue()
        self._clicks: queue.Queue = queue.Queue()
        self._thread: Thread | None = None
        self._stop = False

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        """Start the background preview thread (idempotent: ignored if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        self._thread = Thread(target=self._run, name="jiuwen-gui-perception", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Request preview stop (the worker exits on its next tick and disconnects the camera)."""
        self._stop = True

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def request_point(self, u: float, v: float) -> None:
        """Request reprojection of one pixel (UI-thread call; only enqueues — the worker does it)."""
        self._clicks.put((float(u), float(v)))

    def drain(self) -> list[tuple[str, Any]]:
        """Non-blocking drain of all queued events, consumed periodically by the view's ``ui.timer``."""
        events: list[tuple[str, Any]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return events

    # ------------------------------------------------------------------ run
    def _run(self) -> None:
        """Worker body: build session, connect camera, loop grabbing frames + servicing clicks, then disconnect."""
        session: RobotSession | None = None
        try:
            session = self._session_factory()
            env = session.env
            api = session.api
            # Camera only: env.connect() rather than ``with session:`` so the session's detector
            # sidecar stays down — click reprojection needs no detection, avoiding GPU startup.
            env.connect()
            self._events.put(("preview_started", {"name": getattr(session, "name", "robot")}))
            while not self._stop:
                obs = env.get_observation()
                rgb = getattr(obs, "rgb", None)
                depth = getattr(obs, "depth", None)
                if rgb is not None:
                    try:
                        self._events.put(("frame", imaging.to_data_uri(rgb)))
                    except Exception as exc:  # a bad frame must not break the preview
                        logger.debug("perception frame encode failed: %s", exc)
                if depth is None:
                    reason = (
                        "相机无画面:请检查相机连接与序列号(camera_serial)。"
                        if rgb is None
                        else "相机未提供深度数据:该本体可能没有深度相机,感知测试需要深度。"
                    )
                    self._events.put(("error", {"reason": reason}))
                    break
                self._service_clicks(api, depth)
                time.sleep(_LOOP_PERIOD_S)
        except Exception as exc:  # report connect/grab failures to the UI instead of crashing
            logger.exception("感知测试预览失败")
            self._events.put(("error", {"reason": str(exc)}))
        finally:
            if session is not None:
                try:
                    session.env.disconnect()
                except Exception as exc:  # best-effort disconnect
                    logger.debug("perception disconnect failed: %s", exc)
            self._events.put(("preview_stopped", {}))

    def _service_clicks(self, api: Any, depth: Any) -> None:
        """Reproject each queued click against the latest frame's depth."""
        while True:
            try:
                u, v = self._clicks.get_nowait()
            except queue.Empty:
                return
            self._events.put(("point_result", self._locate(api, depth, u, v)))

    def _locate(self, api: Any, depth: Any, u: float, v: float) -> dict[str, Any]:
        """Pixel (u,v) + latest depth → base (x,y,z); failures carry a Chinese ``reason``."""
        h, w = depth.shape[0], depth.shape[1]
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            return {"ok": False, "u": ui, "v": vi, "reason": "点击超出画面范围。"}
        depth_m = float(depth[vi, ui])
        if not math.isfinite(depth_m) or depth_m <= 0.0:
            return {"ok": False, "u": ui, "v": vi, "reason": "该点无有效深度(可能超量程/反光),请换一点。"}
        try:
            xyz = api.pixel_to_base_xyz(ui, vi, depth_m)
        except NotImplementedError:
            return {"ok": False, "u": ui, "v": vi, "depth_m": depth_m, "reason": "该本体未实现像素→基座反投影。"}
        except (RuntimeError, ValueError) as exc:
            return {"ok": False, "u": ui, "v": vi, "depth_m": depth_m, "reason": f"反投影失败:{exc}"}
        x, y, z = float(xyz["x"]), float(xyz["y"]), float(xyz["z"])
        result: dict[str, Any] = {"ok": True, "u": ui, "v": vi, "depth_m": depth_m, "x": x, "y": y, "z": z}
        if self._z_correction_mm:
            result["z_corrected"] = z + self._z_correction_mm
            result["z_correction_mm"] = self._z_correction_mm
        return result
