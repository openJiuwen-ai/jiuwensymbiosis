"""Resident worker: sole owner of Simulation, rendering and scene mutations."""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import tempfile
import time
from collections import deque
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

# File entry keeps the worker independent of core/Agent import hooks.
if not __package__:
    import importlib.util

    _PACKAGE_NAME = "_jiuwensymbiosis_simulators"
    _PACKAGE_ENTRY = Path(__file__).resolve().parents[1] / "__init__.py"
    _PACKAGE_SPEC = importlib.util.spec_from_file_location(
        _PACKAGE_NAME, _PACKAGE_ENTRY, submodule_search_locations=[str(_PACKAGE_ENTRY.parent)]
    )
    if _PACKAGE_SPEC is None or _PACKAGE_SPEC.loader is None:
        raise ImportError("Cannot load the simulator package")
    _PACKAGE_MODULE = importlib.util.module_from_spec(_PACKAGE_SPEC)
    sys.modules[_PACKAGE_NAME] = _PACKAGE_MODULE
    _PACKAGE_SPEC.loader.exec_module(_PACKAGE_MODULE)
    _ENTRY_SPEC = importlib.util.spec_from_file_location(_PACKAGE_NAME + ".common.worker", __file__)
    if _ENTRY_SPEC is None:
        raise ImportError("Cannot identify the standalone simulator module")
    # Seed the file entry's import context from its loader, without core import hooks.
    globals().update({"__package__": _ENTRY_SPEC.parent, "__spec__": _ENTRY_SPEC})

from .base_motion import measure

# Absolute-file entry avoids the agent's dynamic module hooks during Torch compilation.
from .camera import PreviewSettings
from .encoding import NvEncoder, rgb_array
from .ipc import Mailbox
from .preview_backends import preview_backend
from .render_schedule import RenderSchedule
from .runtime import create_runtime, reset_runtime
from .scene_catalog import resolve_scene, validate_material_bundle
from .scheduling import PhysicsClock
from .startup_diagnostics import StartupDiagnostics

logger = logging.getLogger("jiuwensymbiosis.simulators.common.worker")

# OpenCV reads this at its first import, which happens inside this process when
# the renderer bakes EXR material textures; setting it later has no effect.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def _encoder_type(settings):
    if preview_backend(settings.backend).software_encoding:
        from .encoding import SoftwareEncoder

        return SoftwareEncoder
    return NvEncoder


def _publish_configuration(
    mailbox: Mailbox, revision: int, settings: PreviewSettings, startup: StartupDiagnostics, data: dict
) -> None:
    mailbox.write_status(
        {
            "state": "reconfiguring",
            "message": "正在切换场景或画质…",
            "config_revision": revision,
            "settings": asdict(settings),
            "startup": startup.snapshot(),
            "loading": data,
        }
    )


def run_preview(mailbox: Mailbox, directory: Path, *, render_dispatch=None) -> None:
    """Keep Simulation initialized while replacing only the resources that changed."""
    settings = PreviewSettings(**mailbox.read_control()["settings"])
    if render_dispatch is None and settings.backend == "genesis":
        from .split_threads import run_split

        run_split(lambda dispatch: run_preview(mailbox, directory, render_dispatch=dispatch))
        return
    # Report the same physics device that the SDK owner resolves, independently of rendering.
    from ..genesis.engine import physics_backend as resolve_physics_backend

    physics_backend = resolve_physics_backend()
    profile = preview_backend(settings.backend)
    renderer_name = profile.renderer
    encoder_name = profile.encoder_name
    sdk = ExitStack()
    encoder = runtime = None

    def make_encoder(config, scene):
        def construct():
            codec = _encoder_type(config)(config, resolution=scene.frame_size(config.camera_name))
            if render_dispatch is None:
                return codec
            from .independent_video import IndependentVideoLoop

            return IndependentVideoLoop(render_dispatch, mailbox, scene.renderer, codec, config)

        return construct() if render_dispatch is None else render_dispatch.call(construct)

    def discard_video():
        if render_dispatch is not None and encoder is not None:
            # Main-thread barrier: finish consuming the borrowed RGB view, then
            # invalidate completed states before reset/capture/renderer teardown.
            encoder.clear_state()

    capture_path: Path | None = None
    depth_path: Path | None = None
    query_path: Path | None = None
    revision = mailbox.read_control().get("config_revision", 0)
    error = None
    switch_ms = 0.0
    sequence = published = encoded_bytes = 0
    timings: dict[str, deque] = {
        k: deque(maxlen=600) for k in ("physics", "render", "encode", "copy", "nvenc", "ipc", "snapshot", "frame_age")
    }
    startup = StartupDiagnostics(
        logger=logger,
        publish=lambda data: mailbox.write_status(
            {
                "state": "starting",
                "message": f"正在初始化 Simulation / {renderer_name}…",
                "startup": data,
                "loading": data,
            }
        ),
    )
    loading = startup
    try:
        import numpy as np

        if settings.backend == "genesis":
            from ..genesis.engine import engine_session

            sdk.enter_context(engine_session(startup, render_dispatch=render_dispatch))
        else:
            with startup.phase("mujoco_init"):
                import mujoco

                if mujoco.__version__ != "4.0.1":
                    raise RuntimeError("MuJoCo backend requires mujoco-core 4.0.1; use a fresh .[mujoco] environment")
        with startup.phase("runtime_scene"):
            runtime = create_runtime(settings, directory, startup=startup, render_dispatch=render_dispatch)
        logger.info("Simulation world and %s renderer ready", renderer_name)
        with startup.phase("encoder_init" if profile.software_encoding else "nvenc_init"):
            encoder = make_encoder(settings, runtime)
        logger.info("%s encoder ready", encoder_name)
        stream_epoch = -1
        keyframe_request = -1
        camera_dirty = True
        pick_result: dict = {}
        capture_result: dict = {}
        base_result: dict = {}
        query_result: dict = {}
        reset_result: dict = {}
        physics_clock = PhysicsClock(settings.dt)
        video_clock = RenderSchedule(settings.fps, monotonic=time.monotonic)
        next_frame = next_status = time.monotonic()
        next_snapshot = 0.0
        while not mailbox.stopping():
            if render_dispatch is not None:
                render_dispatch.check_error()
            command = mailbox.read_control()
            if query_path is not None and (command.get("scene_query") or {}).get("id") != query_result.get("id"):
                query_path.unlink(missing_ok=True)
                query_path = None
            # The requester may cancel before a slow render finishes writing.
            if (capture_path is not None or depth_path is not None) and (command.get("capture") or {}).get(
                "id"
            ) != capture_result.get("id"):
                if capture_path is not None:
                    capture_path.unlink(missing_ok=True)
                    capture_path = None
                if depth_path is not None:
                    depth_path.unlink(missing_ok=True)
                    depth_path = None
            requested_revision = command.get("config_revision", 0)
            if requested_revision != revision:
                discard_video()
                base_result = {}
                if runtime.info["mode"] == "robot_mobile":
                    runtime.motion.cancel("configuration_changed")
                    if getattr(runtime, "arm_motion", None) is not None:
                        runtime.arm_motion.cancel("configuration_changed")
                    runtime.robot.stop()
                started = time.monotonic()
                requested = PreviewSettings(**command["settings"])
                loading = StartupDiagnostics(
                    logger=logger,
                    operation="configure",
                    publish=partial(_publish_configuration, mailbox, revision, settings, startup),
                )
                candidate = new_encoder = None
                renderer_changed = False
                old_settings = settings
                error = None
                try:
                    if requested.scene_id != settings.scene_id and requested.scene_id != "demo":
                        asset = resolve_scene(requested.scene_id)
                        if asset.asset.preview_mode == "urdf":
                            from ..genesis.urdf_compat import validate_urdf

                            validate_urdf(asset.preview_path)
                            validate_material_bundle(asset)
                        elif asset.asset.preview_mode == "static_usd":
                            # Reject missing/empty/oversized assets before releasing the current renderer.
                            with tempfile.TemporaryDirectory(prefix="check-", dir=directory) as check:
                                from ..genesis.usd_scene import prepare_static_stage

                                prepare_static_stage(asset.preview_path, Path(check) / "background.usda")
                    encoder.close()
                    encoder = None
                    if (
                        requested.scene_id,
                        requested.robot_mode,
                        requested.robot_id,
                        requested.planning_profile,
                        requested.floor_height_m,
                    ) != (
                        settings.scene_id,
                        settings.robot_mode,
                        settings.robot_id,
                        settings.planning_profile,
                        settings.floor_height_m,
                    ):
                        renderer_changed = True
                        runtime.close_renderer()
                        candidate = create_runtime(
                            requested,
                            directory,
                            runtime.pose["revision"] + 1,
                            startup=loading,
                            render_dispatch=render_dispatch,
                        )
                    elif (requested.width, requested.height, requested.spp, requested.render_mode) != (
                        settings.width,
                        settings.height,
                        settings.spp,
                        settings.render_mode,
                    ):
                        renderer_changed = True
                        runtime.close_renderer()
                        with loading.phase("nyx_renderer"):
                            runtime.build_renderer(requested)
                    # Open NVENC only after Nyx finishes replacing CUDA resources:
                    # opening it before teardown fails on repeated 720p/1080p
                    # switches in the real GPU regression test.
                    with loading.phase(
                        "encoder_init" if preview_backend(requested.backend).software_encoding else "nvenc"
                    ):
                        new_encoder = make_encoder(requested, candidate or runtime)
                    if candidate is not None:
                        runtime.close()
                        runtime = candidate
                        candidate = None
                    encoder = new_encoder
                    new_encoder = None
                    settings = requested
                    profile = preview_backend(settings.backend)
                    renderer_name, encoder_name = profile.renderer, profile.encoder_name
                    runtime.settings = requested
                except Exception as exc:
                    loading.finish(error=True)
                    logger.exception("Simulation reconfiguration failed; restoring previous scene")
                    error = f"切换失败，保留原场景和配置：{str(exc)[:1000]}"
                    if candidate is not None:
                        candidate.close()
                    if new_encoder is not None:
                        new_encoder.close()
                    if renderer_changed:
                        runtime.close_renderer()
                        runtime.build_renderer(old_settings)
                    if encoder is None:
                        encoder = make_encoder(old_settings, runtime)
                revision = requested_revision
                switch_ms = round((time.monotonic() - started) * 1000, 1)
                published = 0
                keyframe_request = -1
                camera_dirty = True
                for values in timings.values():
                    values.clear()
                # Rebuilding CUDA resources is an explicit simulation pause.
                # Begin a new measurement epoch, rather than hiding ordinary
                # frame stalls by resetting the physics deadline every loop.
                physics_clock = PhysicsClock(settings.dt)
                video_clock = RenderSchedule(settings.fps, monotonic=time.monotonic)
                next_frame = next_status = time.monotonic()
                next_snapshot = 0.0
            reset = command.get("reset")
            if reset and reset["id"] != reset_result.get("id"):
                discard_video()
                reset_result = {"id": reset["id"], "complete": False}

                def publish_reset(data, revision, settings, runtime, reset_result):
                    mailbox.write_status(
                        {
                            "state": "resetting",
                            "message": "正在复位当前世界…",
                            "config_revision": revision,
                            "settings": asdict(settings),
                            "scene": runtime.info,
                            "loading": data,
                            "startup": startup.snapshot(),
                            "reset_result": reset_result,
                        }
                    )

                loading = StartupDiagnostics(
                    logger=logger,
                    operation="reset",
                    publish=partial(
                        publish_reset, revision=revision, settings=settings, runtime=runtime, reset_result=reset_result
                    ),
                )
                try:
                    if (
                        reset["scene_instance_id"] != runtime.info["instance_id"]
                        or reset["config_revision"] != revision
                    ):
                        raise ValueError("复位请求所属世界已经变化。")
                    runtime, result = reset_runtime(
                        runtime, settings, directory, phase=loading.phase, stopping=mailbox.stopping, startup=loading
                    )
                    if render_dispatch is not None:
                        encoder.close()
                        encoder = make_encoder(settings, runtime)
                        next_snapshot = 0.0
                    reset_result.update(result)
                    pick_result = capture_result = base_result = query_result = {}
                    published = 0
                    camera_dirty = True
                    physics_clock = PhysicsClock(settings.dt)
                    video_clock = RenderSchedule(settings.fps, monotonic=time.monotonic)
                    next_frame = next_status = time.monotonic()
                except Exception as exc:
                    loading.finish(error=True)
                    reset_result.update(error=str(exc)[:1000], complete=True)
                    next_status = 0
            if command["stream_epoch"] != stream_epoch:
                camera_name = command["settings"].get("camera_name", "observer")
                camera_changed = camera_name != settings.camera_name
                if camera_changed:
                    runtime.frame_size(camera_name)  # Validate against this built world's cameras.
                discard_video()
                if camera_changed:
                    settings = replace(settings, camera_name=camera_name)
                    runtime.settings = settings
                if stream_epoch >= 0 or camera_changed:
                    encoder.close()
                    encoder = make_encoder(settings, runtime)
                next_snapshot = 0.0
                stream_epoch = command["stream_epoch"]
                published = 0
                keyframe_request = -1
                camera_dirty = True
                next_status = 0
            streaming = command["streaming"]
            if command["camera"]["revision"] > runtime.pose["revision"]:
                runtime.update_camera(command["camera"])
                camera_dirty = True
            # Encoder/camera replacement can block. Validate movement against
            # fresh control and lease timestamps immediately before each batch.
            command = mailbox.read_control()
            if command.get("config_revision", 0) != revision or command["stream_epoch"] != stream_epoch:
                continue
            streaming = command["streaming"]
            rejected_motion = {}
            if runtime.info["mode"] == "robot_mobile":
                motion_request = command.get("base_motion")
                arm_request = bool(motion_request and motion_request.get("kind") == "arm")
                motion = getattr(runtime, "arm_motion", None) if arm_request else runtime.motion
                if motion is None:
                    rejected_motion = {
                        "id": (motion_request or {}).get("id"),
                        "status": "completed",
                        "ok": False,
                        "settled": False,
                        "stop_confirmation": "unknown",
                        "reason": "Arm planning not configured",
                    }
                    motion, motion_request = runtime.motion, None
                if getattr(runtime, "arm_motion", None) is not None and not arm_request:
                    runtime.arm_motion.accept_request(
                        None,
                        scene_id=runtime.info["instance_id"],
                        revision=revision,
                        now=time.monotonic(),
                        step=runtime.steps,
                    )
                if arm_request:
                    runtime.motion.accept_request(
                        None,
                        scene_id=runtime.info["instance_id"],
                        revision=revision,
                        now=time.monotonic(),
                        step=runtime.steps,
                    )
                motion.accept_request(
                    motion_request,
                    scene_id=runtime.info["instance_id"],
                    revision=revision,
                    now=time.monotonic(),
                    step=runtime.steps,
                )
                base: dict = command.get("base_command") or {}
                valid = (
                    base
                    and command["streaming"]
                    and base["scene_instance_id"] == runtime.info["instance_id"]
                    and base["config_revision"] == revision
                    and base["expires_at"] > time.monotonic()
                )
                if motion.active or (getattr(runtime, "arm_motion", None) is not None and runtime.arm_motion.active):
                    pass  # Motion owns wheel targets until measured completion, including braking.
                elif rejected_motion or not valid:
                    runtime.robot.stop()
                elif base["id"] != base_result.get("id"):
                    base_result = {"id": base["id"], "applied": False}
                    try:
                        runtime.robot.command(
                            base["vx"], base["vy"], base["wz"], min(1.0, base["expires_at"] - time.monotonic())
                        )
                        base_result["applied"] = True
                    except ValueError as exc:
                        runtime.robot.stop()
                        base_result["error"] = str(exc)
                    next_status = 0
            started = time.monotonic()
            completed_steps = physics_clock.run_batch(runtime.step, stopping=mailbox.stopping)
            if render_dispatch is not None:
                render_dispatch.physics_steps += completed_steps
                encoder.states.note_step(runtime.steps)
            if completed_steps:
                timings["physics"].append((time.monotonic() - started) * 1000)
            if mailbox.stopping():
                break
            now = time.monotonic()
            query = command.get("scene_query")
            if query and query["id"] != query_result.get("id"):
                query_result = {"id": query["id"]}
                try:
                    if (
                        query["config_revision"] != revision
                        or query["scene_instance_id"] != runtime.info["instance_id"]
                        or now > query["expires_at"]
                    ):
                        raise ValueError("scene_query_expired_or_scene_changed")
                    result = runtime.query_truth(query["operation"], query["params"])
                    if time.monotonic() > query["expires_at"]:
                        raise ValueError("scene_query_expired")
                    query_path = directory / f"query-{query['id']}.json"
                    query_path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
                except Exception as exc:
                    if isinstance(exc, ValueError):
                        query_result["error"] = str(exc)[:300]
                    else:
                        logger.exception("Simulation scene query failed")
                        query_result["error"] = "scene_query_failed"
                next_status = 0
            capture: dict = command.get("capture") or {}
            capture_pending = capture and capture["id"] != capture_result.get("id")
            capture_invalid = capture_pending and (
                capture["config_revision"] != revision
                or capture["scene_instance_id"] != runtime.info["instance_id"]
                or (
                    (
                        capture.get("camera_name", "observer") == "observer"
                        or runtime.settings.robot_mode == "virtual_camera"
                    )
                    and capture["camera_revision"] != runtime.pose["revision"]
                )
                or now > capture["expires_at"]
            )
            if capture_pending and (capture_invalid or render_dispatch is not None or physics_clock.render_ready()):
                discard_video()
                capture_result = {"id": capture["id"]}
                try:
                    if capture_invalid:
                        raise ValueError("Capture request expired or view changed")
                    from PIL import Image

                    captured_at = time.monotonic()
                    captured_utc = datetime.now(UTC).isoformat(timespec="milliseconds")
                    try:
                        name = capture.get("camera_name", "observer")
                        if capture.get("include_depth", False):
                            from ..genesis.rgbd import base_body_bounds, base_from_camera

                            memory_capture = capture.get("capture_mode") == "spatial_memory"
                            if memory_capture and name != "head_cam":
                                raise ValueError("Spatial memory requires head_cam")
                            virtual_capture = runtime.settings.robot_mode == "virtual_camera"
                            if memory_capture and virtual_capture:
                                raise ValueError("Robot spatial_memory capture does not accept virtual motion evidence")
                            if not virtual_capture and (
                                runtime.motion is None
                                or (runtime.motion.active and not memory_capture)
                                or (getattr(runtime, "arm_motion", None) is not None and runtime.arm_motion.active)
                            ):
                                raise ValueError("RGB-D requires a stationary supported mobile robot")
                            deadline = min(capture["expires_at"], captured_at + 2.0)

                            from ..genesis.capture_status import refresh_capture_status

                            last_capture_status = time.monotonic()

                            def check_capture(
                                deadline=deadline,
                                capture=capture,
                                revision=revision,
                                memory_capture=memory_capture,
                                runtime=runtime,
                            ):
                                nonlocal last_capture_status
                                current = mailbox.read_control()
                                expired = mailbox.stopping() or time.monotonic() > deadline
                                if (
                                    expired
                                    or (current.get("capture") or {}).get("id") != capture["id"]
                                    or current.get("config_revision", 0) != revision
                                ):
                                    raise ValueError("RGB-D capture cancelled or expired")
                                if memory_capture and time.monotonic() - last_capture_status >= 0.2:
                                    refresh_capture_status(runtime, mailbox, measure_base=measure, now=time.monotonic)
                                    last_capture_status = time.monotonic()

                            sensor_step = runtime.steps
                            pixels, depth = runtime.renderer.read_rgbd(
                                name,
                                stride=capture.get("depth_stride", 8),
                                **({"region": capture["depth_region"]} if capture.get("depth_region") else {}),
                                check=check_capture,
                            )
                            base_sample = None if virtual_capture else measure(runtime.robot)
                            if runtime.steps != sensor_step:
                                raise ValueError("Physics advanced during RGB-D capture")
                            if memory_capture:
                                capture_result.update(
                                    capture_mode="spatial_memory",
                                    synchronization={
                                        "kind": "single_physics_state",
                                        "rgb_step": sensor_step,
                                        "depth_step": sensor_step,
                                        "pose_step": sensor_step,
                                    },
                                    motion_active=runtime.motion.active,
                                )
                            if virtual_capture:
                                capture_result.update(
                                    pose_source="virtual_camera_pose",
                                    robot_loaded=False,
                                    synchronization={
                                        "kind": "single_virtual_pose",
                                        "rgb_step": sensor_step,
                                        "depth_step": sensor_step,
                                        "pose_step": sensor_step,
                                    },
                                )
                            else:
                                runtime.renderer.last_camera["T_base_camera_mm"] = base_from_camera(
                                    np.asarray(runtime.renderer.last_camera["T_world_camera"]), base_sample
                                ).tolist()
                                if runtime.arm_motion is not None:
                                    runtime.renderer.last_camera["T_planning_base_camera_m"] = (
                                        np.linalg.inv(runtime.arm_motion.base_pose())
                                        @ np.asarray(runtime.renderer.last_camera["T_world_camera"])
                                    ).tolist()
                                capture_result.update(
                                    base_sample=base_sample,
                                    body_bounds_m=base_body_bounds(runtime.robot.bounds, base_sample).tolist(),
                                    motion_id=runtime.motion.result.get("id"),
                                )
                            depth_path = directory / f"capture-{capture['id']}.npz"
                            np.savez(depth_path, rgb=pixels, depth_m=depth)
                            check_capture()
                        else:
                            pixels = rgb_array(runtime.renderer.read_rgb(name))
                    finally:
                        physics_clock.mark_render()
                    if time.monotonic() > capture["expires_at"]:
                        raise ValueError("Capture expired during rendering")
                    height, width = pixels.shape[:2]
                    capture_path = directory / f"capture-{capture['id']}.jpg"
                    Image.fromarray(pixels).save(capture_path, quality=95)
                    del pixels
                    capture_result.update(
                        captured_at=captured_at,
                        captured_utc=captured_utc,
                        scene_instance_id=runtime.info["instance_id"],
                        config_revision=revision,
                        camera_revision=capture["camera_revision"],
                        camera_name=name,
                        camera=runtime.pose.copy() if name == "observer" else runtime.renderer.last_camera.copy(),
                        step=runtime.steps,
                        width=width,
                        height=height,
                        capture_duration_ms=(time.monotonic() - captured_at) * 1000,
                    )
                except Exception as exc:
                    logger.warning("Simulation image capture failed (%s): %s", type(exc).__name__, exc)
                    capture_result["error"] = "capture_failed"
                next_status = 0
            if mailbox.stopping():
                break
            available = not published or mailbox.acknowledged() >= published
            now = time.monotonic()
            # Video has its own bounded wall-time budget. Waiting for all
            # physics debt to clear can starve it indefinitely near capacity.
            # Every iteration still advances a bounded physics batch first.
            render_requested = streaming or camera_dirty
            if not render_requested:
                video_clock.pause()
            frame_due = render_requested and available and video_clock.ready()
            if render_dispatch is not None:
                from .independent_video import CompletedFrame

                if render_requested and now >= next_snapshot:
                    snapshot_started = time.monotonic()
                    geometry = runtime.renderer.completed_snapshot(settings.camera_name)
                    encoder.states.publish(
                        CompletedFrame(
                            geometry=geometry,
                            step=runtime.steps,
                            dt=settings.dt,
                            completed_at=geometry.completed_at,
                            ready_at=geometry.ready_at,
                            scene_instance_id=runtime.info["instance_id"],
                            config_revision=revision,
                            stream_epoch=stream_epoch,
                            reset_id=reset_result.get("id"),
                            reset_generation=runtime.cycle,
                            camera_revision=runtime.pose["revision"],
                        )
                    )
                    del geometry
                    timings["snapshot"].append((time.monotonic() - snapshot_started) * 1000)
                    next_snapshot = time.monotonic() + 1 / settings.fps
                frames, rendered_camera_revision = encoder.progress()
                camera_dirty = rendered_camera_revision != runtime.pose["revision"]
                if frames:
                    startup.mark_first_frame()
                    loading.mark_first_frame()
                    loading.finish()
                    if reset_result and not reset_result.get("error"):
                        reset_result["complete"] = True
            if frame_due and render_dispatch is None:
                started = time.monotonic()
                with loading.phase("initial_render") if not loading.finished else nullcontext():
                    rgb = runtime.renderer.read_rgb(settings.camera_name)
                frame_step, frame_camera_revision = runtime.steps, runtime.pose["revision"]
                render_ms = (time.monotonic() - started) * 1000
                startup.mark_first_frame()
                loading.mark_first_frame()
                loading.finish()
                if reset_result and not reset_result.get("error"):
                    reset_result["complete"] = True
                physics_clock.mark_render()
                timings["render"].append(render_ms)
                timings["frame_age"].append((time.monotonic() - started) * 1000)
                sequence += 1
                if streaming:
                    encode_start = time.monotonic()
                    request = command["keyframe_request"]
                    packet = encoder.encode(
                        rgb,
                        {
                            "sequence": sequence,
                            "captured_at": started,
                            "step": frame_step,
                            "camera_revision": frame_camera_revision,
                            "camera_name": settings.camera_name,
                            "stream_epoch": stream_epoch,
                            "config_revision": revision,
                            "scene_instance_id": runtime.info["instance_id"],
                        },
                        force_keyframe=request != keyframe_request,
                    )[0]
                    keyframe_request = request
                    timings["encode"].append((time.monotonic() - encode_start) * 1000)
                    timings["copy"].append(encoder.copy_ms)
                    timings["nvenc"].append(encoder.encode_ms)
                    ipc_start = time.monotonic()
                    payload = packet.pop("h264")
                    mailbox.write_frame(packet, payload)
                    encoded_bytes += len(payload)
                    published = packet["sequence"]
                    available = False
                    timings["ipc"].append((time.monotonic() - ipc_start) * 1000)
                del rgb  # release the borrowed CUDA view before any renderer replacement
                camera_dirty = frame_camera_revision != runtime.pose["revision"]
                video_clock.finish(started)
                next_frame = video_clock.next_frame
            pick = command.get("pick")
            if pick and pick["id"] != pick_result.get("id") and not camera_dirty:
                discard_video()
                pick_started = time.monotonic()
                pick_result = {"id": pick["id"]}
                try:
                    if pick["config_revision"] != revision or pick["scene_instance_id"] != runtime.info["instance_id"]:
                        raise ValueError("场景已经切换，请重新点选。")
                    if pick["action"] == "clear":
                        runtime.selection = None
                    elif pick["camera_revision"] != runtime.pose["revision"]:
                        raise ValueError("视角已经变化，请重新点选。")
                    else:
                        runtime.pick(pick["u"], pick["v"])
                except Exception as exc:
                    logger.exception("Simulation picking failed")
                    pick_result["error"] = str(exc)[:500]
                pick_result["duration_ms"] = round((time.monotonic() - pick_started) * 1000, 2)
                next_status = 0
            now = time.monotonic()
            if now >= next_status:
                video_stats = encoder.snapshot() if render_dispatch is not None else {}
                startup_data = startup.snapshot()
                mailbox.write_status(
                    {
                        "state": "ready"
                        if loading.finished
                        else {"startup": "starting", "configure": "reconfiguring", "reset": "resetting"}[
                            loading.operation
                        ],
                        "startup": startup_data,
                        "loading": loading.snapshot(),
                        "reset_result": reset_result,
                        "message": f"{runtime.info.get('engine', 'Simulation')} / "
                        f"{runtime.info.get('renderer', renderer_name)} 已就绪 · {settings.scene_id}"
                        + (
                            "（静态场景，仅地面物理碰撞）"
                            if runtime.info["mode"] == "static_background"
                            else "（固定姿态检视，无碰撞）"
                            if runtime.info["mode"] == "robot_preview"
                            else runtime.info.get("control_description", "（轮速驱动、地面接触、受限关节控制）")
                            if runtime.info["mode"] == "robot_mobile"
                            else ""
                        )
                        if loading.finished
                        else "正在等待当前世界的首帧…",
                        "config_revision": revision,
                        "stream_epoch": stream_epoch,
                        "settings": asdict(settings),
                        "reconfigure_error": error,
                        "switch_ms": switch_ms,
                        "scene": runtime.info,
                        "measured_at": time.monotonic(),
                        "steps": runtime.steps,
                        "sim_time": runtime.steps * settings.dt,
                        "physics_clock": {"config_revision": revision, **physics_clock.snapshot()},
                        "video_schedule": video_stats.get("video_schedule", video_clock.snapshot()),
                        "frame_clock": video_stats.get("frame_clock"),
                        "execution_threads": runtime.renderer.threading_snapshot()
                        if render_dispatch is not None
                        else {"mode": "serial"},
                        "cycle": runtime.cycle,
                        "rendered_frames": video_stats.get("sequence", sequence),
                        "encoded_bytes": video_stats.get("encoded_bytes", encoded_bytes),
                        "render_ms": round(timings["render"][-1], 2) if timings["render"] else 0,
                        "timings_ms": {
                            **video_stats.get("timings_ms", {}),
                            **{
                                k: {
                                    "p50": round(float(np.percentile(v, 50)), 3),
                                    "p95": round(float(np.percentile(v, 95)), 3),
                                }
                                for k, v in timings.items()
                                if v
                            },
                        },
                        "encoder": encoder_name,
                        "physics_backend": "MuJoCo CPU" if settings.backend == "mujoco" else physics_backend,
                        "raw_cpu_readback": True,
                        "raw_ipc": False,
                        "camera": runtime.pose,
                        "initial_camera": runtime.initial_pose,
                        "camera_fov_y": runtime.renderer.fov_y,
                        "cameras": [camera.metadata() for camera in runtime.cameras.values()],
                        "camera_render_counts": runtime.renderer.render_counts,
                        "selection": runtime.selection_snapshot(),
                        "pick_result": pick_result,
                        "capture_result": capture_result,
                        "scene_query_result": query_result,
                        "base_command_result": base_result,
                        "base_motion": rejected_motion
                        or (
                            runtime.arm_motion.snapshot()
                            if getattr(runtime, "arm_motion", None) is not None
                            and (command.get("base_motion") or {}).get("kind") == "arm"
                            else runtime.motion.result
                            if runtime.motion is not None
                            else {}
                        ),
                        "effectors": runtime.arm_motion.effectors.report()
                        if getattr(runtime, "arm_motion", None) is not None
                        else {},
                        "base_observation": (measure(runtime.robot) if runtime.motion is not None else None),
                        "streaming": streaming,
                        "object_positions": [body.get_pos()[0].cpu().tolist() for body in runtime.bodies],
                        "robot": runtime.robot.snapshot() if runtime.robot is not None else None,
                        "scene_contacts": runtime.indoor.contacts(runtime.robot)
                        if runtime.indoor is not None and runtime.robot is not None
                        else None,
                    }
                )
                next_status = now + 0.5
            deadline = min(physics_clock.next_step, next_frame) if streaming and available else physics_clock.next_step
            time.sleep(max(0.0, min(0.001, deadline - time.monotonic())))
    except Exception as exc:
        logger.exception("Simulation preview worker failed")
        mailbox.write_status(
            {
                "state": "failed",
                "message": f"Simulation / {renderer_name} 启动或运行失败：{str(exc)[:3000]}",
                "startup": startup.finish(error=True),
                "loading": loading.finish(error=True),
            }
        )
    finally:
        try:
            discard_video()
        except Exception:
            logger.exception("Pending render failed during cleanup")
        for resource in (encoder, runtime):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    logger.exception("Simulation resource cleanup failed")
        encoder = runtime = resource = None
        gc.collect()
        sdk.close()
        if capture_path is not None:
            capture_path.unlink(missing_ok=True)
        if depth_path is not None:
            depth_path.unlink(missing_ok=True)
        if query_path is not None:
            query_path.unlink(missing_ok=True)


def main() -> None:
    import ctypes
    import faulthandler
    import signal

    faulthandler.enable()

    # Linux parent-death signal also frees CUDA if the GUI crashes without shutdown hooks.
    expected_parent = int(sys.argv[2])
    if sys.platform == "linux" and ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
        raise OSError(ctypes.get_errno(), "Cannot set worker parent-death signal")
    if os.getppid() != expected_parent:
        return
    if sys.platform == "darwin":
        import threading

        def watch_parent() -> None:
            # macOS has no prctl; terminate the worker if its owning GUI dies.
            while os.getppid() == expected_parent:
                time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=watch_parent, daemon=True).start()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = Path(sys.argv[1])
    mailbox = Mailbox(path)
    try:
        run_preview(mailbox, path.parent)
    finally:
        mailbox.close()


if __name__ == "__main__":
    main()
