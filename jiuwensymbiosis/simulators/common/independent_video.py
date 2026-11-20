"""Completed-state exchange and main-thread rendering/readback/encoding loop.

Simulation time is always the completed physics step times dt. Presentation
uses monotonic wall time; repeating a state never advances simulation time.
The exchange holds only the newest completed state. A renderer's local reference
keeps the previous immutable CPU buffers alive until its frame is consumed.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompletedFrame:
    geometry: Any
    step: int
    dt: float
    completed_at: float
    ready_at: float
    scene_instance_id: str
    config_revision: int
    stream_epoch: int
    reset_id: str | None
    reset_generation: int
    camera_revision: int

    @property
    def sim_time(self) -> float:
        return self.step * self.dt

    def matches(self, command: dict) -> bool:
        return (
            self.config_revision == command.get("config_revision", 0)
            and self.stream_epoch == command["stream_epoch"]
            and self.reset_id == (command.get("reset") or {}).get("id")
        )


class CompletedStates:
    """One replacement slot, never a backlog of obsolete physics states."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest: CompletedFrame | None = None
        self._completed_step = 0

    def publish(self, state: CompletedFrame) -> None:
        with self._lock:
            if self._latest is not None and state.step < self._latest.step:
                raise ValueError("Physics time regressed without clearing the state exchange")
            self._latest = state
            self._completed_step = max(self._completed_step, state.step)

    def note_step(self, step: int) -> None:
        with self._lock:
            self._completed_step = step

    def read(self) -> tuple[CompletedFrame | None, int]:
        with self._lock:
            return self._latest, self._completed_step

    def clear(self) -> None:
        with self._lock:
            self._latest = None
            self._completed_step = 0


class IndependentVideoLoop:
    """All renderer output, readback and codec operations stay on main.

    Construct on main through the dispatcher. Physics only publishes completed
    snapshots and reads copied telemetry. clear_state/close are main-thread
    barriers, so reset and teardown cannot race an in-flight render.
    """

    def __init__(self, dispatch, mailbox, renderer, encoder, settings, *, monotonic=time.monotonic):
        self.dispatch = dispatch
        self.mailbox = mailbox
        self.renderer = renderer
        self.encoder = encoder
        self.settings = settings
        self.states = CompletedStates()
        self._now = monotonic
        self._next_frame = monotonic()
        self._started = monotonic()
        self._published = 0
        self._keyframe_request = -1
        self._camera_revision = -1
        self._last_state = None
        self._closed = False
        self._lock = threading.Lock()
        self._times: deque = deque(maxlen=600)
        self._fresh_times: deque = deque(maxlen=600)
        self._frames = self._bytes = 0
        self._latest: dict = {}
        self._timings: dict[str, deque] = {
            name: deque(maxlen=600)
            for name in ("render", "encode", "copy", "nvenc", "ipc", "frame_age", "snapshot", "state_age")
        }
        dispatch.set_tick(self.tick)

    def clear_state(self):
        def clear():
            self.states.clear()
            self._camera_revision = -1
            self._next_frame = self._now()

        self.dispatch.call(clear)

    def tick(self) -> float:
        if self.mailbox.stopping():
            return 0.05
        now = self._now()
        if now < self._next_frame:
            return self._next_frame - now
        state, _ = self.states.read()
        command = self.mailbox.read_control()
        if state is None or not state.matches(command):
            return 0.005
        pose = command["camera"]
        camera_revision = max(state.camera_revision, pose["revision"])
        streaming = command["streaming"]
        if not streaming and self._camera_revision == camera_revision:
            return 0.01
        if streaming and self._published and self.mailbox.acknowledged() < self._published:
            return 0.001

        started = self._now()
        # This callback never consults mutable Genesis state. Observer-camera
        # edits can be shown even while the physics owner is inside a long step.
        rgb = self.renderer.render_completed(
            state.geometry, observer_pose=pose if pose["revision"] > state.camera_revision else None
        )
        rendered_at = self._now()
        self.dispatch.frame_sequence += 1
        sequence = self.dispatch.frame_sequence
        _, latest_step = self.states.read()
        metadata = {
            "sequence": sequence,
            "captured_at": started,
            "state_completed_at": state.completed_at,
            "snapshot_ready_at": state.ready_at,
            "rendered_at": rendered_at,
            "step": state.step,
            "sim_time": state.sim_time,
            "dt_s": state.dt,
            "reset_generation": state.reset_generation,
            "camera_revision": camera_revision,
            "camera_name": self.settings.camera_name,
            "stream_epoch": state.stream_epoch,
            "config_revision": state.config_revision,
            "scene_instance_id": state.scene_instance_id,
            "latest_physics_step": latest_step,
            "state_lag_steps": max(0, latest_step - state.step),
            "simulation_lag_s": max(0, latest_step - state.step) * state.dt,
        }
        costs = {
            "render": (rendered_at - started) * 1000,
            "snapshot": (state.ready_at - state.completed_at) * 1000,
        }
        size = 0
        if streaming:
            with self.renderer.output_context():
                packet = self.encoder.encode(
                    rgb, metadata, force_keyframe=command["keyframe_request"] != self._keyframe_request
                )[0]
            encoded_at = self._now()
            costs.update(
                encode=(encoded_at - rendered_at) * 1000, copy=self.encoder.copy_ms, nvenc=self.encoder.encode_ms
            )
            # A configuration/reset may arrive while the GPU is rendering.
            # Never publish a completed packet under the replacement world's ID.
            if state.matches(self.mailbox.read_control()) and not self.mailbox.stopping():
                payload = packet.pop("h264")
                metadata.update(published_at=encoded_at, state_age_ms=(encoded_at - state.completed_at) * 1000)
                self.mailbox.write_frame({**packet, **metadata}, payload)
                size = len(payload)
                self._published = sequence
                self._keyframe_request = command["keyframe_request"]
            costs["ipc"] = (self._now() - encoded_at) * 1000
        del rgb  # Borrowed Nyx view must be consumed before any later render.
        finished = self._now()
        costs["frame_age"] = (finished - started) * 1000
        costs["state_age"] = (finished - state.completed_at) * 1000
        self._camera_revision = camera_revision
        identity = (state.scene_instance_id, state.reset_generation, state.step)
        with self._lock:
            self._frames += 1
            self._bytes += size
            self._times.append(finished)
            if identity != self._last_state:
                self._fresh_times.append(finished)
            self._last_state = identity
            self._latest = {**metadata, "state_age_ms": costs["state_age"]}
            for name, value in costs.items():
                self._timings[name].append(value)
        # No physics-blocking percentage budget here. Missed presentation
        # deadlines are skipped, rather than accumulating catch-up render jobs.
        self._next_frame = max(started + 1 / self.settings.fps, finished)
        return max(0.001, self._next_frame - finished)

    def snapshot(self) -> dict:
        import numpy as np

        now = self._now()
        with self._lock:
            times = [t for t in self._times if now - t <= 5]
            fresh = [t for t in self._fresh_times if now - t <= 5]
            latest = self._latest.copy()
            return {
                "rendered_frames": self._frames,
                "sequence": self.dispatch.frame_sequence,
                "encoded_bytes": self._bytes,
                "camera_revision": self._camera_revision,
                "video_schedule": {
                    "mode": "independent_completed_states",
                    "target_fps": self.settings.fps,
                    "recent_source_fps": (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0,
                    "fresh_state_fps": (len(fresh) - 1) / (fresh[-1] - fresh[0]) if len(fresh) > 1 else 0.0,
                },
                "frame_clock": {
                    **latest,
                    "simulation_clock": "completed_physics_step_times_dt",
                    "presentation_clock": "monotonic_wall_time",
                    "current_state_age_ms": max(0.0, (now - latest["state_completed_at"]) * 1000) if latest else None,
                },
                "timings_ms": {
                    name: {"p50": round(float(np.percentile(v, 50)), 3), "p95": round(float(np.percentile(v, 95)), 3)}
                    for name, v in self._timings.items()
                    if v
                },
            }

    def progress(self) -> tuple[int, int]:
        with self._lock:
            return self._frames, self._camera_revision

    def close(self):
        def close():
            if self._closed:
                return
            self._closed = True
            self.dispatch.set_tick(None)
            self.states.clear()
            self.encoder.close()
            self.encoder = self.renderer = None

        self.dispatch.call(close)
