"""GUI-owned standalone process; the rendering interpreter never imports the agent."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from jiuwensymbiosis.simulators._paths import simulator_root
from jiuwensymbiosis.simulators.genesis.camera import OrbitCamera, PreviewSettings
from jiuwensymbiosis.simulators.genesis.ipc import Mailbox
from jiuwensymbiosis.simulators.genesis.scene_catalog import scene_path

_WORKER = Path(__file__).with_name("worker.py")
# Rendering, dynamic libraries, device selection, caches and locale only.
# Provider keys, proxy credentials and arbitrary host variables are not required.
_WORKER_ENV_KEYS = frozenset(
    {
        "HOME",
        "PATH",
        "USER",
        "LOGNAME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XDG_RUNTIME_DIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "CUDA_CACHE_PATH",
        "GS_CACHE_FILE_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "VK_ICD_FILENAMES",
        "VK_DRIVER_FILES",
        "VK_LAYER_PATH",
        "__GLX_VENDOR_LIBRARY_NAME",
        "__NV_PRIME_RENDER_OFFLOAD",
        "EGL_PLATFORM",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "QD_NUM_THREADS",
    }
)


def _worker_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key in _WORKER_ENV_KEYS}
    cache_path = os.environ.get("QD_OFFLINE_CACHE_FILE_PATH")
    if cache_path is not None:
        if not cache_path.strip() or any(char in cache_path for char in ("\0", "\n", "\r")):
            raise ValueError("QD_OFFLINE_CACHE_FILE_PATH must be a non-empty local directory path")
        # Popen runs in state_dir. Resolve relative paths in the parent's cwd,
        # before crossing that boundary, without creating or clearing a cache.
        path = Path(cache_path).expanduser().resolve()
        if path.exists() and not path.is_dir():
            raise ValueError("QD_OFFLINE_CACHE_FILE_PATH must name a directory")
        environment["QD_OFFLINE_CACHE_FILE_PATH"] = str(path)
    cache_size = os.environ.get("QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES")
    if cache_size is not None:
        decimal = cache_size.isascii() and cache_size.isdecimal() and len(cache_size) <= 10
        if not decimal or not 1 <= int(cache_size) <= 2147483647:
            raise ValueError("QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES must be an integer in 1..2147483647 bytes")
        environment["QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES"] = str(int(cache_size))
    # Freeze the same data root for the subprocess, independent of its working directory.
    environment["JIUWENSYMBIOSIS_SIMULATOR_HOME"] = str(simulator_root())
    return environment


class GenesisPreview:
    """One preview scene per GUI process, surviving video reconnects until stopped."""

    def __init__(self, *, state_dir: Path | None = None, settings: PreviewSettings | None = None) -> None:
        self.settings = settings or PreviewSettings()
        self._requested_settings = self.settings
        self._config_revision = 0
        self._observed_revision = -1
        self._switch_started = 0.0
        self.state_dir = (state_dir or Path.home() / ".jiuwensymbiosis" / "genesis-preview").resolve()
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._mailbox: Mailbox | None = None
        self._mailbox_path: Path | None = None
        self._frame: dict[str, Any] | None = None
        self._camera = OrbitCamera()
        self._initial_camera = self._camera
        self._streaming = False
        self._stream_epoch = 0
        self._keyframe_request = 0
        self._pick_request: dict[str, Any] | None = None
        self._base_request: dict[str, Any] | None = None
        self._capture_request: dict[str, Any] | None = None
        self._motion_owner: str | None = None
        self._motion_cancelled = False
        self._motion_request: dict[str, Any] | None = None
        self._motion_binding: tuple[str, int] | None = None
        self._state: dict[str, Any] = {"state": "stopped", "message": "Genesis 后台未启动"}

    def status(self) -> dict[str, Any]:
        """Read the last physical/camera measurements without launching or importing SDKs."""
        with self._lock:
            if self._mailbox is not None:
                observed = self._mailbox.read_status()
                if observed:
                    self._state = observed
                    revision = observed.get("config_revision", 0)
                    if observed.get("state") == "ready" and revision == self._config_revision:
                        if "settings" in observed:
                            self.settings = PreviewSettings(**observed["settings"])
                        if self._observed_revision != revision:
                            self._observed_revision = revision
                            self._requested_settings = self.settings
                            if "camera" in observed:
                                self._camera = OrbitCamera(
                                    **{f.name: observed["camera"][f.name] for f in fields(OrbitCamera)}
                                )
                                initial = observed.get("initial_camera", observed["camera"])
                                self._initial_camera = OrbitCamera(
                                    **{f.name: initial[f.name] for f in fields(OrbitCamera)}
                                )
            if self._process is not None and self._process.poll() is not None and self._state["state"] != "failed":
                self._state = {
                    "state": "failed",
                    "message": f"Genesis 进程已退出（{self._process.returncode}），请查看日志。",
                }
            result = {
                **self._state,
                "backend": "genesis",
                "renderer": "Nyx RefPathTracer" if self.settings.render_mode == "reference" else "Nyx FastPathTracer",
                "settings": asdict(self.settings),
                "requested_settings": asdict(self._requested_settings),
                "requested_config_revision": self._config_revision,
                "log_path": str(self.state_dir / "worker.log"),
                "pid": self._process.pid if self._process is not None else None,
                "requested_camera": self._camera.snapshot(),
            }
            if self._frame is not None:
                result["frame_age_ms"] = round((time.monotonic() - self._frame["captured_at"]) * 1000, 1)
            if (
                self._switch_started
                and self._observed_revision != self._config_revision
                and result.get("state") != "failed"
            ):
                result.update(
                    state="reconfiguring",
                    message="正在切换，视频将自动恢复…",
                    switch_elapsed_s=time.monotonic() - self._switch_started,
                )
            return result

    def start(self) -> dict[str, Any]:
        """Launch a fresh interpreter by file path: multiprocessing spawn reimports the GUI."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self.status()
            self.stop()
            if self.settings.scene_id != "demo":
                scene_path(self.settings.scene_id)
            if sys.platform != "linux":
                self._state = {"state": "failed", "message": "Genesis / Nyx 预览目前支持 Linux + NVIDIA GPU。"}
                return self.status()
            missing = [
                name for name in ("genesis", "gs_nyx_plugin", "aiortc", "av") if importlib.util.find_spec(name) is None
            ]
            if missing:
                self._state = {
                    "state": "failed",
                    "message": f"缺少 {', '.join(missing)}。请在 symbiosis_genesis 环境安装 .[genesis]。",
                }
                return self.status()
            worker_env = _worker_environment()
            self.state_dir.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix="genesis-", suffix=".mmap", dir=self.state_dir)
            os.close(fd)
            self._mailbox_path = Path(name)
            # A live mapping cannot be resized safely while the worker reads it.
            self._mailbox = Mailbox(self._mailbox_path, frame_bytes=1920 * 1080 * 3)
            self._camera, self._frame, self._streaming = OrbitCamera(), None, False
            self._config_revision, self._observed_revision, self._switch_started = 0, -1, 0.0
            self._requested_settings = self.settings
            self._pick_request = None
            self._base_request = None
            self._send_control()
            self._state = {"state": "starting", "message": "正在启动 Genesis / Nyx…"}
            self._mailbox.write_status(self._state)
            try:
                with (self.state_dir / "worker.log").open("a") as log:
                    self._process = subprocess.Popen(
                        [sys.executable, str(_WORKER), str(self._mailbox_path), str(os.getpid())],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        cwd=self.state_dir,
                        env=worker_env,
                    )
            except OSError as exc:
                self._release_mailbox()
                self._state = {"state": "failed", "message": f"无法启动 Genesis：{exc}"}
            return self.status()

    def latest_frame(self) -> dict[str, Any] | None:
        with self._lock:
            if self._mailbox is not None:
                sample = self._mailbox.read_frame(self._frame["sequence"] if self._frame else -1)
                if (
                    sample is not None
                    and sample.get("stream_epoch") == self._stream_epoch
                    and sample.get("config_revision", 0) == self._config_revision
                ):
                    self._frame = sample
            return self._frame

    def acknowledge_frame(self, sequence: int) -> None:
        with self._lock:
            if self._mailbox is not None:
                self._mailbox.acknowledge(sequence)

    def capture_image(
        self, camera_name: str = "observer", *, cancelled: threading.Event | None = None
    ) -> dict[str, Any]:
        """Capture a named camera JPEG and calibration, independently of video.

        Call off the ASGI loop. Bind the request to the scene, configuration and
        requested camera so a concurrent switch cannot silently supply an old view.
        """
        request_id = uuid.uuid4().hex
        path = self.state_dir / f"capture-{request_id}.jpg"
        with self._lock:
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("相机采样已取消。")
            state = self.status()
            if state["state"] != "ready" or time.monotonic() - state.get("measured_at", 0) > 2:
                raise RuntimeError("请启动 Genesis 后台并等待就绪后再识别画面。")
            if self._capture_request is not None:
                raise RuntimeError("另一张画面正在采样，请稍后重试。")
            if camera_name != "observer" and camera_name not in {c["name"] for c in state.get("cameras", [])}:
                raise ValueError(f"当前机器人没有相机：{camera_name}")
            scene_instance_id = state["scene"]["instance_id"]
            config_revision = self._config_revision
            camera_revision = self._camera.revision
            binding = {
                "scene_instance_id": scene_instance_id,
                "config_revision": config_revision,
                "camera_revision": camera_revision,
                "camera_name": camera_name,
            }
            request = {
                "id": request_id,
                **binding,
                "expires_at": time.monotonic() + 12,
            }
            self._capture_request = request
            self._send_control()
        try:
            while time.monotonic() < request["expires_at"]:
                with self._lock:
                    if cancelled is not None and cancelled.is_set():
                        raise RuntimeError("相机采样已取消。")
                    state = self.status()
                    scene_changed = (
                        state.get("scene", {}).get("instance_id") != scene_instance_id
                        or self._config_revision != config_revision
                    )
                    camera_changed = camera_name == "observer" and self._camera.revision != camera_revision
                    if state["state"] != "ready" or scene_changed or camera_changed:
                        raise RuntimeError("采样期间场景或视角已变化，请重新识别。")
                    result = state.get("capture_result", {})
                    if result.get("id") == request_id:
                        if result.get("error"):
                            raise RuntimeError("相机采样失败，请等待画面稳定后重试。")
                        if (
                            any(result.get(key) != value for key, value in binding.items())
                            or time.monotonic() - result.get("captured_at", 0) > 2
                        ):
                            raise RuntimeError("相机采样已过期，请重新识别。")
                        return {**result, "jpeg": path.read_bytes()}
                time.sleep(0.02)
            raise RuntimeError("相机采样超时，请检查 Genesis 后台。")
        finally:
            with self._lock:
                if self._capture_request is request:
                    self._capture_request = None
                    self._send_control()
            path.unlink(missing_ok=True)

    def request_keyframe(self) -> None:
        with self._lock:
            self._keyframe_request += 1
            self._send_control()

    def configure(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        scene_id: str = "demo",
        robot_mode: str = "inspection",
        robot_id: str = "",
        render_quality: str = "reference",
        camera_name: str = "observer",
    ) -> dict:
        with self._lock:
            state = self.status()["state"]
            if state in {"starting", "reconfiguring"}:
                raise RuntimeError("后台正在启动或切换，请稍后再试。")
            if scene_id != "demo":
                scene_path(scene_id)
            same_scene = state == "ready" and (scene_id, robot_mode, robot_id) == (
                self.settings.scene_id,
                self.settings.robot_mode,
                self.settings.robot_id,
            )
            if (
                same_scene
                and camera_name != "observer"
                and camera_name not in {c["name"] for c in self._state.get("cameras", [])}
            ):
                raise ValueError(f"当前机器人没有相机：{camera_name}")
            if render_quality != "reference":
                raise ValueError("网页渲染仅支持 reference。请刷新页面后重试。")
            settings = PreviewSettings(
                width=width,
                height=height,
                fps=fps,
                scene_id=scene_id,
                spp=64,
                render_mode="reference",
                robot_mode=robot_mode,
                robot_id=robot_id,
                camera_name=camera_name,
            )
            if settings == self.settings:
                return self.status()
            self._base_request = None
            if self._process is not None and self._process.poll() is None:
                if state != "ready":
                    raise RuntimeError("后台异常，请先停止后重启。")
                self._requested_settings = settings
                self._config_revision += 1
                self._switch_started = time.monotonic()
                self._frame = None
                self._send_control()
            else:
                self.settings = self._requested_settings = settings
            return self.status()

    def set_streaming(self, enabled: bool) -> None:
        with self._lock:
            if enabled and not self._streaming:
                self._stream_epoch += 1
                self._frame = None
                self._keyframe_request += 1
            self._streaming = enabled
            if not enabled:
                self._base_request = None
            self._send_control()

    def base_command(self, *, vx: float, vy: float, wz: float, scene_instance_id: str, config_revision: int) -> dict:
        """Queue one bounded pulse. Acceptance is not a measured movement result."""
        from jiuwensymbiosis.simulators.genesis.omni_base import validate_twist

        validate_twist(vx, vy, wz)
        with self._lock:
            if self._motion_owner is not None:
                if vx == vy == wz == 0:
                    # A manual stop also revokes future actions from this turn,
                    # including while the Agent is waiting for its model.
                    self._motion_cancelled = True
                    self.cancel_motion(self._motion_owner)
                    return {"queued_id": "cancel", "max_duration_s": 0}
                raise RuntimeError("Agent 正在控制底盘；请先停止任务，再手动操作。")
            state = self.status()
            if state["state"] != "ready" or not self._streaming:
                raise RuntimeError("请等待底盘物理场景就绪并连接视频。")
            if (state.get("robot") or {}).get("control_mode") != "wheel_velocity_physics":
                raise ValueError("当前场景未启用 MOZ1 底盘物理控制。")
            if config_revision != self._config_revision or scene_instance_id != state.get("scene", {}).get(
                "instance_id"
            ):
                raise ValueError("场景已切换，请等待新画面后操作。")
            request_id = uuid.uuid4().hex
            self._base_request = {
                "id": request_id,
                "vx": vx,
                "vy": vy,
                "wz": wz,
                "scene_instance_id": scene_instance_id,
                "config_revision": config_revision,
                "expires_at": time.monotonic() + 1.0,
            }
            self._send_control()
            return {"queued_id": request_id, "max_duration_s": 1.0}

    def acquire_motion(self) -> str:
        """Claim this physical scene without starting it or requiring a video peer."""
        with self._lock:
            state = self.status()
            if state["state"] != "ready" or time.monotonic() - state.get("measured_at", 0) > 2:
                raise RuntimeError("Genesis 物理状态不可用或已过期。")
            if (state.get("robot") or {}).get("control_mode") != "wheel_velocity_physics":
                raise RuntimeError("请先加载 MOZ1 底盘物理场景。")
            if self._motion_owner is not None:
                raise RuntimeError("另一个任务正在控制底盘。")
            self._motion_owner = uuid.uuid4().hex
            self._motion_cancelled = False
            self._motion_binding = (state["scene"]["instance_id"], self._config_revision)
            self._motion_request = None
            self._base_request = None
            self._send_control()
            return self._motion_owner

    def cancel_motion(self, owner: str, request_id: str | None = None) -> None:
        with self._lock:
            if self._motion_owner != owner or self._motion_request is None:
                return
            if request_id is None or self._motion_request["id"] == request_id:
                self._motion_request["cancelled"] = True
                self._send_control()

    def motion_state(self, owner: str) -> dict:
        """A session may only observe the physical scene it acquired."""
        with self._lock:
            state = self.status()
            if self._motion_owner != owner or self._motion_binding != (
                state.get("scene", {}).get("instance_id"),
                self._config_revision,
            ):
                raise RuntimeError("控制会话已失效或场景已切换。")
            return state

    def release_motion(self, owner: str) -> None:
        """Do not release ownership until completion or destruction of the bound scene."""
        self.cancel_motion(owner)
        deadline = time.monotonic() + 12
        while True:
            with self._lock:
                if self._motion_owner != owner:
                    return
                state = self.status()
                request = self._motion_request
                result = state.get("base_motion", {})
                gone = state["state"] in {"stopped", "failed"} or (
                    state["state"] == "ready"
                    and self._motion_binding is not None
                    and state.get("scene", {}).get("instance_id") != self._motion_binding[0]
                )
                done = request is None or (result.get("id") == request["id"] and result.get("status") == "completed")
                if gone or done:
                    if not gone and request is not None and not result.get("settled"):
                        raise RuntimeError("底盘停止尚未确认；保留控制权，请停止后台。")
                    self._motion_owner = self._motion_request = self._motion_binding = None
                    self._motion_cancelled = False
                    self._send_control()
                    return
            if time.monotonic() >= deadline:
                raise RuntimeError("底盘取消未获确认；保留控制权，请停止后台。")
            time.sleep(0.05)

    def navigate_relative(self, owner: str, dx: float, dy: float, dyaw: float, *, cancelled=None) -> dict:
        """Wait for measured completion. Renew the worker lease independently of video."""
        from jiuwensymbiosis.simulators.genesis.base_motion import validate_move

        validate_move(dx, dy, dyaw)
        with self._lock:
            state = self.status()
            if self._motion_owner != owner or self._motion_binding != (
                state.get("scene", {}).get("instance_id"),
                self._config_revision,
            ):
                raise RuntimeError("控制会话已失效或场景已切换。")
            if state["state"] != "ready" or time.monotonic() - state.get("measured_at", 0) > 2:
                raise RuntimeError("Genesis 物理状态不可用或已过期。")
            if self._motion_cancelled or (cancelled is not None and cancelled.is_set()):
                raise RuntimeError("任务已取消。")
            previous = state.get("base_motion", {})
            if self._motion_request and (
                previous.get("id") != self._motion_request["id"] or previous.get("status") != "completed"
            ):
                raise RuntimeError("上一条底盘动作尚未完成。")
            request: dict[str, Any] = {
                "id": uuid.uuid4().hex,
                "scene_instance_id": self._motion_binding[0],
                "config_revision": self._motion_binding[1],
                "dx": dx,
                "dy": dy,
                "dyaw": dyaw,
                "lease_until": time.monotonic() + 2,
                "cancelled": False,
            }
            self._motion_request = request
            self._send_control()
        deadline = time.monotonic() + 100
        try:
            while time.monotonic() < deadline:
                with self._lock:
                    state = self.status()
                    result: dict = state.get("base_motion", {})
                    fresh = time.monotonic() - state.get("measured_at", 0) <= 2
                    if result.get("id") == request["id"] and result.get("status") == "completed" and fresh:
                        return result
                    if state["state"] != "ready" or not fresh or self._config_revision != request["config_revision"]:
                        raise RuntimeError("动作中断：后台状态过期、不可用或配置已切换。")
                    if cancelled is not None and cancelled.is_set():
                        request["cancelled"] = True
                    request["lease_until"] = time.monotonic() + 2
                    self._send_control()
                time.sleep(0.05)
            raise TimeoutError("等待底盘动作结果超时，停止结果尚未确认。")
        finally:
            self.cancel_motion(owner, request["id"])

    def camera_command(self, action: str, **values: float) -> dict:
        with self._lock:
            state = self.status()
            if state["state"] != "ready":
                raise RuntimeError("Genesis 尚未就绪。")
            if self.settings.camera_name != "observer":
                raise ValueError("机器人相机随连杆运动，请切回观察视角后调整相机。")
            if action == "focus":
                selected = state.get("selection")
                if selected and selected["scene_instance_id"] != state["scene"]["instance_id"]:
                    raise ValueError("选中对象已不属于当前场景，请重新点选。")
                bounds = (
                    selected["bounds_m"] if selected else state["scene"].get("bounds_m", [[-1, -0.8, 0], [1, 0.8, 2]])
                )
                self._camera = self._camera.focus(
                    bounds, fov_y=state["camera_fov_y"], aspect=self.settings.width / self.settings.height
                )
            else:
                self._camera = self._camera.command(action, **values)
            if action == "reset":
                self._camera = replace(self._initial_camera, revision=self._camera.revision)
            self._send_control()
            return {"accepted_revision": self._camera.revision, "requested_camera": self._camera.snapshot()}

    def select(
        self, *, action: str, u: float, v: float, scene_instance_id: str, config_revision: int, camera_revision: int
    ) -> dict:
        """Wait off the ASGI loop, releasing the preview lock so video can drain."""
        if action not in {"pick", "clear"}:
            raise ValueError("点选坐标必须在画面内。")
        for coordinate in (u, v):
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)) or not 0 <= coordinate <= 1:
                raise ValueError("点选坐标必须在画面内。")
        request_id = uuid.uuid4().hex
        with self._lock:
            state = self.status()
            if state["state"] != "ready":
                raise RuntimeError("Genesis 尚未就绪。")
            if self.settings.camera_name != "observer":
                raise ValueError("请在观察视角中点选物体。")
            if config_revision != self._config_revision or scene_instance_id != state.get("scene", {}).get(
                "instance_id"
            ):
                raise ValueError("场景已经切换，请等待新画面后重新点选。")
            if action == "pick" and camera_revision != self._camera.revision:
                raise ValueError("视角已经变化，请等待画面稳定后重新点选。")
            self._pick_request = {
                "id": request_id,
                "action": action,
                "u": u,
                "v": v,
                "scene_instance_id": scene_instance_id,
                "config_revision": config_revision,
                "camera_revision": camera_revision,
            }
            self._send_control()
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                state = self.status()
                result = state.get("pick_result", {})
                if result.get("id") == request_id:
                    if result.get("error"):
                        raise ValueError(result["error"])
                    return {
                        "selection": state.get("selection"),
                        "camera": state["camera"],
                        "camera_fov_y": state["camera_fov_y"],
                    }
                if state["state"] != "ready":
                    raise RuntimeError("点选期间场景已停止或切换，请重试。")
                time.sleep(0.01)
            raise RuntimeError("点选未在时限内确认，请重试。")
        finally:
            with self._lock:
                if self._pick_request and self._pick_request["id"] == request_id:
                    self._pick_request = None
                    self._send_control()

    def _send_control(self) -> None:
        if self._mailbox is not None:
            self._mailbox.write_control(
                {
                    "streaming": self._streaming,
                    "camera": self._camera.snapshot(),
                    "settings": asdict(self._requested_settings),
                    "config_revision": self._config_revision,
                    "stream_epoch": self._stream_epoch,
                    "keyframe_request": self._keyframe_request,
                    "pick": self._pick_request,
                    "base_command": self._base_request,
                    "base_motion": self._motion_request,
                    "capture": self._capture_request,
                }
            )

    def stop(self) -> dict:
        """Offload this blocking call from ASGI; only terminate the process this owner started."""
        with self._lock:
            if self._process is not None:
                if self._mailbox is not None:
                    self._mailbox.request_stop()
                try:
                    self._process.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=3)
                self._process = None
            self._release_mailbox()
            self._frame = None
            self._switch_started = 0.0
            self._requested_settings = self.settings
            self._state = {"state": "stopped", "message": "Genesis 后台未运行"}
            self._motion_owner = self._motion_request = self._motion_binding = None
            self._motion_cancelled = False
            return self.status()

    def _release_mailbox(self) -> None:
        if self._mailbox is not None:
            self._mailbox.close()
            self._mailbox = None
        if self._mailbox_path is not None:
            self._mailbox_path.unlink(missing_ok=True)
            self._mailbox_path = None
