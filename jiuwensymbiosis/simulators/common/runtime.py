"""Worker world port and lazy engine dispatch; no simulation SDK imports."""

from __future__ import annotations

import importlib
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol, cast

from .preview_backends import MODES


class SimulationRuntime(Protocol):
    """The worker's world port, implemented by Genesis and native policy worlds."""

    settings: Any
    directory: Path
    info: dict
    pose: dict
    initial_pose: dict
    robot: Any
    renderer: Any
    motion: Any
    arm_motion: Any
    cameras: dict
    bodies: list
    indoor: Any
    steps: int
    cycle: int
    selection: dict | None

    def step(self) -> None:
        pass

    def close(self) -> None:
        pass

    def build_renderer(self, settings: Any) -> None:
        pass

    def close_renderer(self) -> None:
        pass

    def frame_size(self, name: str) -> tuple[int, int]:
        pass

    def update_camera(self, pose: dict) -> None:
        pass

    def selection_snapshot(self) -> dict | None:
        pass

    def pick(self, u: float, v: float) -> Any:
        pass

    def query_truth(self, operation: str, params: dict) -> dict:
        pass


def create_runtime(settings, directory, revision=0, *, startup=None) -> SimulationRuntime:
    if settings.backend == "genesis":
        if sys.platform == "darwin" and settings.scene_id != "demo":
            raise ValueError("Mac Genesis 仅支持动态演示；USD/URDF 请使用 Linux / NVIDIA。")
        from ..genesis.scene_runtime import RuntimeScene

        return RuntimeScene(settings, directory, revision, startup=startup)
    if settings.robot_mode == "inspection":
        from ..mujoco.scene_runtime import RuntimeScene as MujocoScene

        return MujocoScene(settings, directory, revision, startup=startup)
    spec = MODES[settings.robot_mode]
    module = importlib.import_module("..mujoco." + spec.module, __package__)
    return cast(SimulationRuntime, getattr(module, spec.runtime)(settings, directory, revision, startup=startup))


def reset_runtime(runtime, settings, directory, *, phase=nullcontext, stopping=lambda: False, startup=None):
    """Use the scene's checkpoint, or rebuild a native policy world after measured stopping."""
    if settings.backend == "genesis" or settings.robot_mode == "inspection":
        return runtime, runtime.reset(phase=phase, stopping=stopping)
    if runtime.motion.active:
        raise RuntimeError("运动任务仍在执行，不能复位。")
    from ..common.base_motion import measure

    tuning = getattr(runtime.motion, "base", runtime.motion).tuning
    with phase("reset_stop"):
        runtime.stop()
        stable = 0.0
        for _ in range(round(5 / settings.dt)):
            if stopping():
                raise RuntimeError("复位已取消，保留当前世界。")
            runtime.step()
            sample = measure(runtime.robot)
            stopped = (
                math.dist(sample["linear_velocity_m_s"], (0, 0, 0)) < tuning.linear_still
                and math.dist(sample["angular_velocity_rad_s"], (0, 0, 0)) < tuning.angular_still
            )
            stable = stable + settings.dt if stopped else 0.0
            if stable >= tuning.settle_s:
                break
        else:
            raise RuntimeError("复位前未确认停车，保留当前世界。")
    with phase("reset_restore"):
        replacement = create_runtime(settings, directory, runtime.pose["revision"] + 1, startup=startup)
        previous = runtime.info["instance_id"]
        replacement.cycle = runtime.cycle + 1
        replacement.info["reset_generation"] = replacement.cycle
        try:
            runtime.close()
        except BaseException:
            replacement.close()
            raise
    return replacement, {
        "previous_instance_id": previous,
        "scene_instance_id": replacement.info["instance_id"],
        "reset_generation": replacement.cycle,
        "stop_confirmation": "measured",
    }
