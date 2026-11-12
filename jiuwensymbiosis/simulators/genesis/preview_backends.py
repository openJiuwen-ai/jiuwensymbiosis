"""Import-light preview modes and backend capabilities shared by host and worker."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

RobotMode = Literal["inspection", "moz1_wheels", "jumper", "microduck"]
BackendId = Literal["mujoco", "genesis_raster", "genesis_nyx"]


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
RASTER = PreviewBackend("genesis_raster", "Genesis Rasterizer", True, False, False, False)
NYX = PreviewBackend("genesis_nyx", "Nyx", False, True, True, True)


def preview_backend(mode: str, platform: str | None = None) -> PreviewBackend:
    """robot_mode is the legacy preview-mode selector, not a hardware identity."""
    if mode not in MODES:
        raise ValueError(f"Unsupported preview mode: {mode}")
    if mode in POLICY_MODES:
        return MUJOCO
    return RASTER if (platform or sys.platform) == "darwin" else NYX


def required_modules(mode: str, platform: str | None = None) -> tuple[str, ...]:
    # The existing Linux worker still initializes Genesis/Nyx even for a native world.
    modules = ("genesis", "aiortc", "av", *MODES[mode].modules)
    return modules if (platform or sys.platform) == "darwin" else (*modules, "gs_nyx_plugin")


def installation_extra(mode: str, platform: str | None = None) -> str:
    base = "genesis-macos" if (platform or sys.platform) == "darwin" else "genesis"
    return ",".join(filter(None, (base, MODES[mode].extra)))
