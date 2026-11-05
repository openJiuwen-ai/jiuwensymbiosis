"""Convert a validated Genesis head-camera capture to simulator-neutral evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from jiuwensymbiosis.spatial_memory.models import Observation, canonical


def observation_from_capture(capture: dict, *, map_frame: str, allow_motion: bool = False) -> Observation:
    """Use the frozen optical-camera pose, never the current/commanded base pose."""
    camera = capture["camera"]
    if (
        capture.get("camera_name") != "head_cam"
        or camera.get("depth_type") != "optical_z"
        or camera.get("depth_unit") != "m"
    ):
        raise ValueError("Require head_cam optical-Z depth in metres")
    if camera.get("distortion_model") != "rectified_pinhole" or np.any(np.asarray(camera["D"]) != 0):
        raise ValueError("Memory requires rectified RGB-D")
    if allow_motion:
        sync = capture.get("synchronization", {})
        step = capture.get("step")
        if (
            capture.get("capture_mode") != "spatial_memory"
            or sync.get("kind") != "single_physics_state"
            or type(step) is not int
            or step < 0
            or any(
                type(sync.get(key)) is not int or sync[key] != step for key in ("rgb_step", "depth_step", "pose_step")
            )
        ):
            raise ValueError("Moving capture requires synchronized RGB, depth and measured pose")
    for key, limit in (("linear_velocity_m_s", 0.008), ("angular_velocity_rad_s", 0.02)):
        velocity = np.asarray(capture["base_sample"][key])
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("Capture velocity is invalid")
        if not allow_motion and np.linalg.norm(velocity) >= limit:
            raise ValueError("Capture was not stationary")
    calibration = {
        key: camera[key] for key in ("K", "D", "width", "height", "distortion_model", "depth_type", "depth_unit")
    }
    calibration_id = hashlib.sha256(canonical(calibration).encode()).hexdigest()
    return Observation(
        observation_id=capture["id"],
        scene_epoch=f"{capture['scene_instance_id']}:{capture['config_revision']}",
        map_frame=map_frame,
        calibration_id=calibration_id,
        timestamp_s=capture["captured_at"],
        clock_id=f"genesis-monotonic:{capture['scene_instance_id']}",
        pose_source="simulator_readback",
        rgb=capture["rgb"],
        depth_m=capture["depth_m"],
        intrinsics=np.asarray(camera["K"]),
        map_from_camera=np.asarray(camera["T_world_camera"]),
    )


class GenesisMemoryRecorder:
    """Own a recording of validated captures from the connected preview.

    Recording validates camera/pose synchronization and persists raw evidence.
    Motion execution and inference belong to their respective callers.
    """

    def __init__(self, output: Path, *, map_frame: str, allow_motion: bool = False):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.map_frame = map_frame
        self.allow_motion = allow_motion
        self.frames: list[str] = []
        self.observation_ids: set[str] = set()
        self.scope: tuple[str, str, str] | None = None

    def __call__(self, capture: dict) -> None:
        self.record(capture)

    def record(self, capture: dict) -> Observation:
        """Persist evidence first, then return the exact immutable input for ingestion."""
        from jiuwensymbiosis.spatial_memory.artifacts import write_observation

        observation = observation_from_capture(capture, map_frame=self.map_frame, allow_motion=self.allow_motion)
        scope = observation.scene_epoch, observation.calibration_id, observation.clock_id
        if self.scope is not None and scope != self.scope:
            raise ValueError("Capture scope changed during recording")
        if observation.observation_id in self.observation_ids:
            raise ValueError("Duplicate capture ID in sequence")
        name = f"{len(self.frames):03d}"
        write_observation(self.output / name, observation)
        metadata = {
            k: capture[k]
            for k in (
                "id",
                "camera_name",
                "camera",
                "scene_instance_id",
                "config_revision",
                "step",
                "base_sample",
                "captured_at",
            )
        }
        for key in ("capture_mode", "synchronization", "motion_active", "motion_id", "capture_duration_ms"):
            if key in capture:
                metadata[key] = capture[key]
        (self.output / name / "capture.json").write_text(json.dumps(metadata, indent=2))
        manifest = {
            "schema_version": 1,
            "frames": [*self.frames, name],
            "capture_mode": "moz1_head_cam_moving" if self.allow_motion else "moz1_head_cam",
            "pose_source": "simulator_readback",
            "map_frame": self.map_frame,
            "scan_complete": None,
        }
        temporary = self.output / "sequence.tmp.json"
        temporary.write_text(canonical(manifest) + "\n")
        temporary.replace(self.output / "sequence.json")
        self.frames.append(name)
        self.observation_ids.add(observation.observation_id)
        self.scope = scope
        return observation


def capture_sequence(
    preview,
    output: Path,
    *,
    map_frame: str,
    frames: int,
    interval_s: float = 0.5,
    depth_stride: int = 4,
    allow_motion: bool = False,
    cancelled=None,
) -> dict:
    """Record from an already connected preview without issuing motion commands.

    Each completed bundle is published before the next capture. An interrupted
    recording remains replayable up to its last committed frame.
    """
    import time

    if type(frames) is not int or frames < 1:
        raise ValueError("frames must be a positive integer")
    if not np.isfinite(interval_s) or interval_s < 0:
        raise ValueError("interval_s must be finite and nonnegative")
    if type(depth_stride) is not int or not 4 <= depth_stride <= 32:
        raise ValueError("depth_stride must be in [4, 32]")
    if not isinstance(map_frame, str) or not map_frame.strip():
        raise ValueError("map_frame is required")
    recorder = GenesisMemoryRecorder(output, map_frame=map_frame, allow_motion=allow_motion)
    report: dict[str, object] = {"requested_frames": frames, "captured_frames": 0, "completed": False}
    try:
        for index in range(frames):
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("Recording cancelled")
            raw = preview.capture_image(
                "head_cam",
                include_depth=True,
                depth_stride=depth_stride,
                capture_mode="spatial_memory" if allow_motion else "action",
                cancelled=cancelled,
            )
            recorder.record(raw)
            report["captured_frames"] = len(recorder.frames)
            if index + 1 < frames:
                if cancelled is None:
                    time.sleep(interval_s)
                elif cancelled.wait(interval_s):
                    raise RuntimeError("Recording cancelled")
        report["completed"] = True
        return report
    except BaseException as exc:
        report["error"] = str(exc)
        raise
    finally:
        (recorder.output / "capture-report.json").write_text(canonical(report) + "\n")
