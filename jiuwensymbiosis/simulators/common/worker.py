"""Resident worker: sole owner of Simulation, rendering and scene mutations."""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from collections import deque
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

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
from .preview_backends import MUJOCO_VERSION, preview_backend
from .protocol import (
    base_command_valid,
    binding_mismatch,
    build_base_command_reply,
    build_capture_error,
    build_capture_reply,
    build_pick_reply,
    build_rejected_reply,
    capture_request_invalid,
    parse_base_command_request,
    parse_base_motion_request,
    parse_capture_request,
    parse_pick_request,
    parse_reset_request,
    parse_scene_query_request,
    pick_binding_error,
)
from .reconfigurator import Reconfigurator, SwitchResources, precheck_scene
from .runtime import create_runtime, reset_runtime
from .scheduling import PhysicsClock
from .startup_diagnostics import StartupDiagnostics

logger = logging.getLogger("jiuwensymbiosis.simulators.common.worker")


def _message_id(value: Any) -> Any:
    """Read a request id for an in-band rejection without requiring an object."""
    return value.get("id") if isinstance(value, dict) else None


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


def run_preview(mailbox: Mailbox, directory: Path) -> None:
    """Keep Simulation initialized while replacing only the resources that changed."""
    settings = PreviewSettings(**mailbox.read_control()["settings"])
    macos = sys.platform == "darwin"
    # Single-robot stepping on Apple Silicon is dominated by Metal dispatch
    # overhead. CPU keeps the same solver/timestep and leaves the GPU to GL.
    physics_backend = os.environ.get("JIUWEN_GENESIS_MACOS_BACKEND", "cpu") if macos else "cuda"
    reconfigurator = Reconfigurator(
        precheck=precheck_scene,
        runtime_factory=create_runtime,
        encoder_factory=lambda config, *, resolution: _encoder_type(config)(config, resolution=resolution),
    )
    profile = preview_backend(settings.backend)
    renderer_name = profile.renderer
    encoder_name = profile.encoder_name
    sdk = ExitStack()
    encoder = runtime = None
    capture_path: Path | None = None
    depth_path: Path | None = None
    query_path: Path | None = None
    revision = mailbox.read_control().get("config_revision", 0)
    error = None
    switch_ms = 0.0
    sequence = published = encoded_bytes = 0
    timings: dict[str, deque] = {k: deque(maxlen=600) for k in ("physics", "render", "encode", "copy", "nvenc", "ipc")}
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

            sdk.enter_context(engine_session(startup))
        else:
            with startup.phase("mujoco_init"):
                import mujoco

                if mujoco.__version__ != MUJOCO_VERSION:
                    raise RuntimeError(f"MuJoCo 后台加载了 {mujoco.__version__}，需要 mujoco-core {MUJOCO_VERSION}。")
        with startup.phase("runtime_scene"):
            runtime = create_runtime(settings, directory, startup=startup)
        logger.info("Simulation world and %s renderer ready", renderer_name)
        with startup.phase("encoder_init" if profile.software_encoding else "nvenc_init"):
            encoder = reconfigurator.replace_encoder(
                None, settings, resolution=runtime.frame_size(settings.camera_name)
            )
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
        next_frame = next_status = time.monotonic()
        while not mailbox.stopping():
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
                base_result = {}
                if runtime.info["mode"] == "robot_mobile":
                    runtime.motion.cancel("configuration_changed")
                    if getattr(runtime, "arm_motion", None) is not None:
                        runtime.arm_motion.cancel("configuration_changed")
                    runtime.robot.stop()
                requested = PreviewSettings(**command["settings"])
                loading = StartupDiagnostics(
                    logger=logger,
                    operation="configure",
                    publish=partial(_publish_configuration, mailbox, revision, settings, startup),
                )
                switch = reconfigurator.apply(
                    SwitchResources(runtime=runtime, encoder=encoder, settings=settings),
                    requested,
                    requested_revision,
                    directory=directory,
                    loading=loading,
                )
                runtime, encoder, settings = (
                    switch.resources.runtime,
                    switch.resources.encoder,
                    switch.resources.settings,
                )
                error = switch.error
                revision = switch.revision
                switch_ms = switch.switch_ms
                if switch.replaced_world:
                    logger.info("Simulation world replaced during reconfiguration")
                profile = preview_backend(settings.backend)
                renderer_name, encoder_name = profile.renderer, profile.encoder_name
                published = 0
                keyframe_request = -1
                camera_dirty = True
                for values in timings.values():
                    values.clear()
                # Rebuilding CUDA resources is an explicit simulation pause.
                # Begin a new measurement epoch, rather than hiding ordinary
                # frame stalls by resetting the physics deadline every loop.
                physics_clock = PhysicsClock(settings.dt)
                next_frame = next_status = time.monotonic()
            reset = command.get("reset")
            if reset:
                try:
                    reset = parse_reset_request(reset)
                except ValueError as exc:
                    logger.warning("Simulation reset request rejected (%s)", exc)
                    reset_result = build_rejected_reply(
                        "reset", _message_id(reset), error=str(exc)[:1000], complete=True
                    )
                    reset = None
                    next_status = 0
            if reset and reset.get("id") != reset_result.get("id"):
                reset_result = {"id": reset.get("id"), "complete": False}

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
                    if binding_mismatch(reset, config_revision=revision, scene_instance_id=runtime.info["instance_id"]):
                        raise ValueError("复位请求所属世界已经变化。")
                    runtime, result = reset_runtime(
                        runtime, settings, directory, phase=loading.phase, stopping=mailbox.stopping, startup=loading
                    )
                    reset_result.update(result)
                    pick_result = capture_result = base_result = query_result = {}
                    published = 0
                    camera_dirty = True
                    physics_clock = PhysicsClock(settings.dt)
                    next_frame = next_status = time.monotonic()
                except Exception as exc:
                    loading.finish(error=True)
                    reset_result.update(error=str(exc)[:1000], complete=True)
                    next_status = 0
            if command["stream_epoch"] != stream_epoch:
                if stream_epoch >= 0:
                    encoder = reconfigurator.replace_encoder(
                        encoder, settings, resolution=runtime.frame_size(settings.camera_name)
                    )
                stream_epoch = command["stream_epoch"]
                published = 0
                keyframe_request = -1
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
                if motion_request:
                    try:
                        motion_request = parse_base_motion_request(motion_request)
                    except ValueError as exc:
                        logger.warning("Simulation base motion request rejected (%s)", exc)
                        rejected_motion = {
                            "id": _message_id(motion_request),
                            "status": "completed",
                            "ok": False,
                            "settled": False,
                            "stop_confirmation": "unknown",
                            "reason": "invalid_request",
                        }
                        motion_request = None
                        next_status = 0
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
                if base:
                    try:
                        base = parse_base_command_request(base)
                    except ValueError as exc:
                        logger.warning("Simulation base command rejected (%s)", exc)
                        base_result = build_rejected_reply(
                            "base_command", _message_id(base), error=str(exc)[:300], applied=False
                        )
                        base = {}
                        next_status = 0
                valid = base_command_valid(
                    base,
                    config_revision=revision,
                    scene_instance_id=runtime.info["instance_id"],
                    streaming=command["streaming"],
                    now=time.monotonic(),
                )
                if motion.active or (getattr(runtime, "arm_motion", None) is not None and runtime.arm_motion.active):
                    pass  # Motion owns wheel targets until measured completion, including braking.
                elif rejected_motion or not valid:
                    runtime.robot.stop()
                elif base.get("id") != base_result.get("id"):
                    base_result = build_base_command_reply(base.get("id"), applied=False)
                    try:
                        runtime.robot.command(
                            base.get("vx", 0.0),
                            base.get("vy", 0.0),
                            base.get("wz", 0.0),
                            min(1.0, base.get("expires_at", 0.0) - time.monotonic()),
                        )
                        base_result["applied"] = True
                    except ValueError as exc:
                        runtime.robot.stop()
                        base_result["error"] = str(exc)
                    next_status = 0
            started = time.monotonic()
            if physics_clock.run_batch(runtime.step, stopping=mailbox.stopping):
                timings["physics"].append((time.monotonic() - started) * 1000)
            if mailbox.stopping():
                break
            now = time.monotonic()
            query = command.get("scene_query")
            if query:
                try:
                    query = parse_scene_query_request(query)
                except ValueError as exc:
                    logger.warning("Simulation scene query request rejected (%s)", exc)
                    query_result = build_rejected_reply("scene_query", _message_id(query), error=str(exc)[:300])
                    next_status = 0
                    query = None
            if query and query.get("id") != query_result.get("id"):
                query_result = {"id": query.get("id")}
                try:
                    if binding_mismatch(
                        query, config_revision=revision, scene_instance_id=runtime.info["instance_id"]
                    ) or now > query.get("expires_at", 0.0):
                        raise ValueError("scene_query_expired_or_scene_changed")
                    result = runtime.query_truth(query.get("operation", ""), query.get("params", {}))
                    if time.monotonic() > query.get("expires_at", 0.0):
                        raise ValueError("scene_query_expired")
                    query_path = directory / f"query-{query.get('id')}.json"
                    query_path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
                except Exception as exc:
                    if isinstance(exc, ValueError):
                        query_result["error"] = str(exc)[:300]
                    else:
                        logger.exception("Simulation scene query failed")
                        query_result["error"] = "scene_query_failed"
                next_status = 0
            capture: dict = command.get("capture") or {}
            if capture:
                try:
                    capture = parse_capture_request(capture)
                except ValueError as exc:
                    logger.warning("Simulation capture request rejected (%s)", exc)
                    capture_result = build_rejected_reply("capture", _message_id(capture))
                    capture = {}
                    next_status = 0
            capture_pending = capture and capture.get("id") != capture_result.get("id")
            capture_invalid = capture_pending and capture_request_invalid(
                capture,
                config_revision=revision,
                scene_instance_id=runtime.info["instance_id"],
                camera_revision=runtime.pose["revision"],
                now=now,
            )
            capture_due = bool(capture_pending and capture_invalid)
            if capture_pending and not capture_due:
                capture_due = physics_clock.render_ready()
            if capture_due:
                capture_result = build_capture_error(capture.get("id"))
                try:
                    if capture_invalid:
                        raise ValueError("Capture request expired or view changed")
                    from PIL import Image

                    captured_at = time.monotonic()
                    captured_utc = datetime.now(UTC).isoformat(timespec="milliseconds")
                    capture_base_sample = capture_body_bounds = capture_motion_id = None
                    try:
                        name = capture.get("camera_name", "observer")
                        if capture.get("include_depth", False):
                            from ..genesis.rgbd import base_body_bounds, base_from_camera

                            arm_active = getattr(runtime, "arm_motion", None) is not None and runtime.arm_motion.active
                            if runtime.motion is None or runtime.motion.active or arm_active:
                                raise ValueError("RGB-D requires a stationary supported mobile robot")
                            deadline = min(capture.get("expires_at", captured_at + 2.0), captured_at + 2.0)

                            def check_capture(deadline=deadline, capture=capture, revision=revision):
                                current = mailbox.read_control()
                                current_capture = current.get("capture") or {}
                                expired = mailbox.stopping() or time.monotonic() > deadline
                                if (
                                    expired
                                    or current_capture.get("id") != capture.get("id")
                                    or current.get("config_revision", 0) != revision
                                ):
                                    raise ValueError("RGB-D capture cancelled or expired")

                            pixels, depth = runtime.renderer.read_rgbd(
                                name,
                                stride=capture.get("depth_stride", 8),
                                check=check_capture,
                            )
                            base_sample = measure(runtime.robot)
                            runtime.renderer.last_camera["T_base_camera_mm"] = base_from_camera(
                                np.asarray(runtime.renderer.last_camera["T_world_camera"]), base_sample
                            ).tolist()
                            capture_base_sample = base_sample
                            capture_body_bounds = base_body_bounds(runtime.robot.bounds, base_sample).tolist()
                            capture_motion_id = runtime.motion.result.get("id")
                            depth_path = directory / f"capture-{capture.get('id')}.npz"
                            np.savez(depth_path, rgb=pixels, depth_m=depth)
                            check_capture()
                        else:
                            pixels = rgb_array(runtime.renderer.read_rgb(name))
                    finally:
                        physics_clock.mark_render()
                    if time.monotonic() > capture.get("expires_at", 0.0):
                        raise ValueError("Capture expired during rendering")
                    height, width = pixels.shape[:2]
                    capture_path = directory / f"capture-{capture.get('id')}.jpg"
                    Image.fromarray(pixels).save(capture_path, quality=95)
                    del pixels
                    capture_result = build_capture_reply(
                        capture.get("id"),
                        captured_at=captured_at,
                        captured_utc=captured_utc,
                        scene_instance_id=runtime.info["instance_id"],
                        config_revision=revision,
                        camera_revision=capture.get("camera_revision", 0),
                        camera_name=name,
                        camera=runtime.pose.copy() if name == "observer" else runtime.renderer.last_camera.copy(),
                        step=runtime.steps,
                        width=width,
                        height=height,
                        capture_duration_ms=(time.monotonic() - captured_at) * 1000,
                        base_sample=capture_base_sample,
                        body_bounds_m=capture_body_bounds,
                        motion_id=capture_motion_id,
                    )
                except Exception as exc:
                    logger.warning("Simulation image capture failed (%s)", type(exc).__name__)
                    capture_result["error"] = "capture_failed"
                next_status = 0
            if mailbox.stopping():
                break
            available = not published or mailbox.acknowledged() >= published
            now = time.monotonic()
            # A render is allowed to create debt, but another render must wait
            # for fixed-step catch-up, except the bounded overload heartbeat.
            # Each batch refreshes control/leases; no second physics owner.
            render_requested = streaming or camera_dirty
            frame_due = render_requested and available and now >= next_frame
            if frame_due and physics_clock.render_ready():
                started = time.monotonic()
                with loading.phase("initial_render") if not loading.finished else nullcontext():
                    rgb = runtime.renderer.read_rgb(settings.camera_name)
                    if not preview_backend(settings.backend).software_encoding:
                        import torch

                        torch.cuda.current_stream().synchronize()
                startup.mark_first_frame()
                loading.mark_first_frame()
                loading.finish()
                if reset_result and not reset_result.get("error"):
                    reset_result["complete"] = True
                physics_clock.mark_render()
                timings["render"].append((time.monotonic() - started) * 1000)
                sequence += 1
                if streaming:
                    encode_start = time.monotonic()
                    request = command["keyframe_request"]
                    packet = encoder.encode(
                        rgb,
                        {
                            "sequence": sequence,
                            "captured_at": started,
                            "step": runtime.steps,
                            "camera_revision": runtime.pose["revision"],
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
                camera_dirty = False
                next_frame = max(next_frame + 1 / settings.fps, time.monotonic())
            pick = command.get("pick")
            if pick:
                try:
                    pick = parse_pick_request(pick)
                except ValueError as exc:
                    logger.warning("Simulation pick request rejected (%s)", exc)
                    pick_result = build_rejected_reply("pick", _message_id(pick), error=str(exc)[:500], duration_ms=0.0)
                    next_status = 0
                    pick = None
            if pick and pick.get("id") != pick_result.get("id") and not camera_dirty:
                pick_started = time.monotonic()
                try:
                    stale = pick_binding_error(
                        pick,
                        config_revision=revision,
                        scene_instance_id=runtime.info["instance_id"],
                        camera_revision=runtime.pose["revision"],
                    )
                    if stale == "scene":
                        raise ValueError("场景已经切换，请重新点选。")
                    if stale == "camera":
                        raise ValueError("视角已经变化，请重新点选。")
                    if pick.get("action") == "clear":
                        runtime.selection = None
                    else:
                        runtime.pick(pick.get("u", 0.0), pick.get("v", 0.0))
                    pick_result = build_pick_reply(pick.get("id"), duration_ms=(time.monotonic() - pick_started) * 1000)
                except Exception as exc:
                    logger.exception("Simulation picking failed")
                    pick_result = {
                        **build_pick_reply(pick.get("id"), duration_ms=(time.monotonic() - pick_started) * 1000),
                        "error": str(exc)[:500],
                    }
                next_status = 0
            now = time.monotonic()
            if now >= next_status:
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
                            "（静态场景，碰撞已加载）"
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
                        "settings": asdict(settings),
                        "reconfigure_error": error,
                        "switch_ms": switch_ms,
                        "scene": runtime.info,
                        "measured_at": time.monotonic(),
                        "steps": runtime.steps,
                        "sim_time": runtime.steps * settings.dt,
                        "physics_clock": {"config_revision": revision, **physics_clock.snapshot()},
                        "cycle": runtime.cycle,
                        "rendered_frames": sequence,
                        "encoded_bytes": encoded_bytes,
                        "render_ms": round(timings["render"][-1], 2) if timings["render"] else 0,
                        "timings_ms": {
                            k: {
                                "p50": round(float(np.percentile(v, 50)), 3),
                                "p95": round(float(np.percentile(v, 95)), 3),
                            }
                            for k, v in timings.items()
                            if v
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
