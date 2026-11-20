"""Import-light preview modes and backend capabilities shared by host and worker."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

RobotMode = Literal["inspection", "moz1_wheels", "virtual_camera", "jumper", "microduck"]
PhysicsBackend = Literal["genesis", "mujoco"]
BackendId = Literal["mujoco", "genesis_nyx"]


@dataclass(frozen=True)
class PreviewMode:
    label: str
    module: str
    runtime: str
    extra: str = ""
    modules: tuple[str, ...] = ()


MODES = {
    "inspection": PreviewMode("姿态检视（无控制）", "scene_runtime", "RuntimeScene"),
    "moz1_wheels": PreviewMode("MOZ1 底盘物理", "scene_runtime", "RuntimeScene"),
    "virtual_camera": PreviewMode("虚拟相机（无机器人）", "scene_runtime", "RuntimeScene"),
    "jumper": PreviewMode(
        "Jumper 六足步态物理", "jumper_runtime", "JumperRuntime", "jumper", ("mujoco", "onnxruntime")
    ),
    "microduck": PreviewMode(
        "Microduck 双足步态物理", "microduck_runtime", "MicroduckRuntime", "microduck", ("mujoco", "onnxruntime", "bam")
    ),
}
POLICY_MODES = tuple(mode for mode, spec in MODES.items() if spec.extra)


@dataclass(frozen=True)
class PreviewBackend:
    id: BackendId
    renderer: str
    software_encoding: bool
    depth: bool
    truth: bool
    pick: bool

    @property
    def encoder_name(self) -> str:
        return "libx264 / ultrafast / CRF 23" if self.software_encoding else "NVENC H264 / pinned RGBA / QP 20 / P4"

    @property
    def features(self) -> dict[str, bool]:
        return {"depth": self.depth, "truth": self.truth, "pick": self.pick}


MUJOCO = PreviewBackend("mujoco", "MuJoCo OpenGL", True, False, False, False)
NYX = PreviewBackend("genesis_nyx", "Nyx FastPathTracer", False, True, True, True)


def supported_modes(backend: str, platform: str | None = None) -> tuple[str, ...]:
    """Return the robot modes the engine can start on this platform."""
    if backend == "genesis":
        if (platform or sys.platform) != "linux":
            return ()
        return tuple(mode for mode in MODES if mode not in POLICY_MODES)
    if backend == "mujoco":
        return ("inspection", *POLICY_MODES)
    return ()


def validate_backend(backend: str, mode: str, platform: str | None = None) -> None:
    """Validate the explicitly selected engine and controller combination."""
    modes = supported_modes(backend, platform)
    if not modes or mode not in MODES:
        raise ValueError("Unsupported simulation backend or robot mode")
    if mode not in modes:
        raise ValueError(f"{mode} is not supported by {backend}")


def preview_backend(backend: str, platform: str | None = None) -> PreviewBackend:
    if backend == "mujoco":
        return MUJOCO
    if backend != "genesis":
        raise ValueError(f"Unsupported simulation backend: {backend}")
    if (platform or sys.platform) != "linux":
        raise ValueError("Genesis Nyx requires Linux / NVIDIA")
    return NYX


def required_modules(backend: str, mode: str, platform: str | None = None) -> tuple[str, ...]:
    validate_backend(backend, mode, platform)
    media = ("aiortc", "av", "PIL")
    if backend == "mujoco":
        return ("mujoco", *media, *tuple(n for n in MODES[mode].modules if n != "mujoco"))
    return ("genesis", *media, "gs_nyx_plugin")


def installation_extra(backend: str, mode: str, platform: str | None = None) -> str:
    validate_backend(backend, mode, platform)
    if backend == "mujoco":
        return ",".join(filter(None, ("mujoco", MODES[mode].extra)))
    return "genesis"
