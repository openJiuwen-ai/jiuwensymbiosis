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

from jiuwensymbiosis.agent.lifecycle import CleanupReport, HardwareCleanupError
from jiuwensymbiosis.utils.logging import get_logger
from jiuwensymbiosis_gui.workbench import imaging
from jiuwensymbiosis_gui.workbench.maintenance import (
    BoundedEventQueue,
    MaintenanceOwner,
    append_cleanup_errors,
    cleanup_report_for,
    cleanup_report_message,
    finish_admission,
)

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
        admission: MaintenanceOwner | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Store dependencies; admission and worker startup happen in ``start``."""
        self._session_factory = session_factory
        self._z_correction_mm = float(z_correction_mm)
        self._admission = admission
        self._config_snapshot = config_snapshot
        self._events = BoundedEventQueue()
        self._clicks: queue.Queue = queue.Queue()
        self._thread: Thread | None = None
        self._stop = False
        self._binding: Any = None
        self._lease: Any = None

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        """Admit the actual config, then start preview; busy resources never connect."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self._lease is not None:
            self._emit("error", {"reason": "上次预览清理未确认，资源仍被保留。", "fatal": False})
            self._emit("preview_stopped", {})
            return
        self._stop = False
        if self._admission is not None:
            try:
                if self._config_snapshot is None:
                    raise ValueError("maintenance admission requires an execution config snapshot")
                self._binding, self._lease = self._admission.acquire(
                    self._config_snapshot, "perception_preview", stop=self.stop
                )
            except Exception as exc:
                self._binding = None
                self._lease = None
                self._emit("error", {"reason": f"{type(exc).__name__}: {exc}", "fatal": False})
                self._emit("preview_stopped", {})
                return
        self._thread = Thread(target=self._run, name="jiuwen-gui-perception", daemon=True)
        try:
            self._thread.start()
        except BaseException as exc:
            self._finish_admission(CleanupReport())
            self._emit("error", {"reason": f"后台线程启动失败: {type(exc).__name__}: {exc}", "fatal": False})
            self._emit("preview_stopped", {})
            if not isinstance(exc, Exception):
                raise

    def stop(self) -> None:
        """Request preview stop (the worker exits on its next tick and disconnects the camera)."""
        self._stop = True

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def can_dispose(self) -> bool:
        """Host ownership ends only after worker exit and confirmed release."""
        return not self.is_running() and (self._lease is None or self._lease.released)

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None and getattr(thread, "ident", None) is not None:
            thread.join(timeout)

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
        hardware_cleanup_error: HardwareCleanupError | None = None
        try:
            session = self._binding.build_session() if self._binding is not None else self._session_factory()
            env = session.env
            api = session.api
            # Maintenance binding is prepared with include_sidecars=False. Use the
            # public session lifecycle so connect/disconnect failures are reportable.
            session.connect()
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
        except HardwareCleanupError as exc:
            # Session construction/connect can fail before assigning a usable session.
            # Keep the rollback failure as lease evidence even when there is no object
            # from which cleanup_report_for() can read it.
            logger.exception("感知测试会话硬件清理未确认")
            hardware_cleanup_error = exc
            self._events.put(("error", {"reason": str(exc)}))
        except Exception as exc:  # report connect/grab failures to the UI instead of crashing
            logger.exception("感知测试预览失败")
            self._events.put(("error", {"reason": str(exc)}))
        finally:
            report = cleanup_report_for(session)
            if hardware_cleanup_error is not None:
                report = append_cleanup_errors(report, [f"hardware cleanup unconfirmed: {hardware_cleanup_error}"])
            self._finish_admission(report)
            self._events.put(("preview_stopped", {}))

    def _finish_admission(self, report: CleanupReport) -> None:
        """Persist cleanup evidence; retain an uncertain lease for host shutdown/review."""
        released, report = finish_admission(self._admission, self._lease, report)
        if released:
            self._lease = None
            self._binding = None
            return
        self._emit("error", {"reason": cleanup_report_message(report), "fatal": False})

    def _emit(self, tag: str, payload: Any) -> None:
        """Append one event to the bounded UI stream."""
        self._events.put((tag, payload))

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
