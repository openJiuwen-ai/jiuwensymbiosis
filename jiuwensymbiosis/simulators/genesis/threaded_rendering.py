"""Rigid-scene snapshots keep concurrent Nyx renders off live Genesis state.

The physical thread publishes completed immutable buffers. The independent main
loop owns rendering and consumes the borrowed RGB view before its next render.
"""

# This companion is the isolated bridge to the pinned Nyx plugin's buffers.
# ruff: noqa: SLF001

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class RigidSnapshot:
    positions: Any
    rotations: Any
    camera_name: str
    observer_pose: dict
    sensor_pose: dict | None
    rectified: bool
    completed_at: float
    ready_at: float


class ThreadedSceneRenderer:
    def __init__(self, renderer, dispatch, pose):
        self._renderer = renderer
        self._dispatch = dispatch
        self._pose = {k: pose[k] for k in ("pos", "lookat", "up")}
        plugin = renderer._renderer
        if any(
            getattr(plugin, name, 0)
            for name in (
                "_num_fem_verts",
                "_num_pbd_verts",
                "_num_mpm_visual_verts",
                "_mpm_recon_slots",
                "_sph_recon_slots",
            )
        ):
            raise ValueError("Split-thread rendering currently requires a rigid-only scene")
        self.fov_y = renderer.fov_y
        self.appearance = renderer.appearance
        self.last_camera = renderer.last_camera
        self.render_counts = renderer.render_counts
        self._completed = 0
        self._overlap_steps = 0
        self._latest: dict = {}
        # Rendering, readback and encoding share their own GPU output stream.
        import torch

        self._output_stream = dispatch.call(torch.cuda.Stream)

    def output_context(self):
        import torch

        return torch.cuda.stream(self._output_stream)

    def update_camera_pose(self, *, pos, lookat, up):
        self._pose = {"pos": pos, "lookat": lookat, "up": up}

    def _snapshot(self, camera_name, rectified=False):
        import quadrants as qd
        from genesis.utils.misc import qd_to_torch

        renderer = self._renderer
        if camera_name not in renderer._camera_indices:
            raise ValueError(f"当前场景没有相机：{camera_name}")
        renderer._scene.visualizer.update_visual_states()
        qd.sync()
        completed_at = time.monotonic()
        solver = renderer._renderer._rigid_solver
        # copy=True is essential: a zero-copy view would change during step().
        positions = qd_to_torch(solver.dyn_state.vgeoms.pos, transpose=True, copy=True)
        rotations = qd_to_torch(solver.dyn_state.vgeoms.quat, transpose=True, copy=True)
        sensor_pose = None if camera_name == "observer" else renderer._sensor_pose(camera_name)
        return RigidSnapshot(
            positions,
            rotations,
            camera_name,
            self._pose.copy(),
            sensor_pose,
            rectified,
            completed_at,
            time.monotonic(),
        )

    def completed_snapshot(self, camera_name="observer"):
        """Physics-owner call; returns owned copies of completed CPU state."""
        return self._snapshot(camera_name)

    def render_completed(self, snapshot, *, observer_pose=None):
        """Main-only call; no reads from live Genesis solver state."""
        if observer_pose is not None:
            snapshot = replace(snapshot, observer_pose={k: observer_pose[k] for k in ("pos", "lookat", "up")})
        with self.output_context():
            return self._render(snapshot)

    def _render(self, snapshot):
        import torch

        started = time.monotonic()
        before = self._dispatch.physics_steps
        renderer = self._renderer
        plugin = renderer._renderer
        if not hasattr(self, "_upload_stream"):
            self._upload_stream = torch.cuda.Stream()
        with torch.cuda.stream(self._upload_stream):
            if plugin._geom_pos_tensor_backend is not None:
                plugin._geom_pos_tensor_backend.copy_(snapshot.positions)
                plugin._geom_rot_tensor_backend.copy_(snapshot.rotations)
            plugin.update_camera_tensor(0, **snapshot.observer_pose)
            if snapshot.sensor_pose is not None:
                name = snapshot.camera_name
                pose = snapshot.sensor_pose
                plugin.update_camera_tensor(renderer._camera_indices[name], pose["pos"], pose["lookat"], pose["up"])
                renderer.last_camera.clear()
                renderer.last_camera.update({**renderer.cameras[name].metadata(), **pose})
            self._upload_stream.synchronize()
        uploaded = time.monotonic()
        # Pinned gs-nyx-plugin bridge: do not call update_scene(), which reads
        # mutable Genesis buffers. All writes above are complete before Nyx.
        plugin._renderer.update_scene()
        rgb = renderer.render_camera(snapshot.camera_name)
        rgb = renderer.project_rgb(rgb, snapshot.camera_name, rectified=snapshot.rectified)
        torch.cuda.current_stream().synchronize()
        finished = time.monotonic()
        steps = max(0, self._dispatch.physics_steps - before)
        self._completed += 1
        self._overlap_steps += steps
        self._latest = {
            "upload_ms": (uploaded - started) * 1000,
            "native_render_ms": (finished - uploaded) * 1000,
            "render_wall_ms": (finished - started) * 1000,
            "physics_steps_during_render": steps,
        }
        return rgb

    def read_rgb(self, camera_name="observer", *, rectified=False):
        snapshot = self._snapshot(camera_name, rectified)

        def capture():
            with self.output_context():
                return self._render(snapshot).cpu().numpy().copy()

        return self._dispatch.call(capture)

    def read_rgbd(self, camera_name, *, stride=8, check=lambda: None, region=None):
        # Synchronous capture is a barrier: physics waits until RGB and depth
        # from the same snapshot have both completed, preserving calibration.
        check()
        snapshot = self._snapshot(camera_name, rectified=True)

        def capture():
            with self.output_context():
                rgb = self._render(snapshot).cpu().numpy().copy()
                depth = self._renderer.read_depth(camera_name, stride=stride, check=check, region=region)
                return rgb, depth

        return self._dispatch.call(capture)

    def pick(self, u, v):
        return self._dispatch.call(self._renderer.pick, u, v)

    def threading_snapshot(self):
        return {
            **self._dispatch.snapshot(),
            "completed_frames": self._completed,
            "physics_steps_during_render_total": self._overlap_steps,
            **self._latest,
        }

    def close(self):
        self._dispatch.call(self._renderer.close)
