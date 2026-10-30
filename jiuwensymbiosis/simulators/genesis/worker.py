"""Resident worker: sole owner of Genesis, CUDA, Nyx and scene mutations."""

from __future__ import annotations

import gc
import logging
import os
import sys
import tempfile
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from base_motion import measure

# Absolute-file entry avoids the agent's dynamic module hooks during Torch compilation.
from camera import PreviewSettings
from encoding import NvEncoder
from ipc import Mailbox
from scene_catalog import resolve_scene, validate_material_bundle
from scene_runtime import RuntimeScene
from scheduling import PhysicsClock
from startup_diagnostics import StartupDiagnostics
from usd_scene import prepare_static_stage

logger = logging.getLogger("jiuwensymbiosis.simulators.genesis.worker")


def run_preview(mailbox: Mailbox, directory: Path) -> None:
    """Keep Genesis initialized while replacing only the resources that changed."""
    settings = PreviewSettings(**mailbox.read_control()["settings"])
    initialized = sdk_initialized = False
    encoder = runtime = None
    capture_path: Path | None = None
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
                "message": "正在初始化 Genesis / Nyx…",
                "startup": data,
            }
        ),
    )
    try:
        with startup.phase("imports"):
            import genesis as gs
            import gs_nyx.nyx_py_sdk as nps
            import numpy as np
            import quadrants
            import torch
        # Bake scene tensor shapes into kernels; keep physical timestep,
        # substeps and collision geometry unchanged. New scenes may compile.
        with startup.phase("genesis_init"):
            gs.init(backend=gs.cuda, logging_level="warning", performance_mode=True)
            initialized = True
        startup.collect_runtime(quadrants)
        with startup.phase("nyx_startup"):
            nps.startup()
            sdk_initialized = True
        with startup.phase("runtime_scene"):
            runtime = RuntimeScene(settings, directory, startup=startup)
        logger.info("Genesis world and Nyx renderer ready")
        with startup.phase("nvenc_init"):
            encoder = NvEncoder(settings, resolution=runtime.frame_size(settings.camera_name))
        logger.info("NVENC encoder ready")
        stream_epoch = -1
        keyframe_request = -1
        camera_dirty = True
        pick_result: dict = {}
        capture_result: dict = {}
        base_result: dict = {}
        physics_clock = PhysicsClock(settings.dt)
        next_frame = next_status = time.monotonic()
        while not mailbox.stopping():
            command = mailbox.read_control()
            # The requester may cancel before a slow render finishes writing.
            if capture_path is not None and (command.get("capture") or {}).get("id") != capture_result.get("id"):
                capture_path.unlink(missing_ok=True)
                capture_path = None
            requested_revision = command.get("config_revision", 0)
            if requested_revision != revision:
                base_result = {}
                if runtime.info["mode"] == "robot_mobile":
                    runtime.motion.cancel("configuration_changed")
                    runtime.robot.stop()
                started = time.monotonic()
                requested = PreviewSettings(**command["settings"])
                mailbox.write_status(
                    {
                        "state": "reconfiguring",
                        "message": "正在切换场景或画质…",
                        "config_revision": revision,
                        "settings": asdict(settings),
                    }
                )
                candidate = new_encoder = None
                renderer_changed = False
                old_settings = settings
                error = None
                try:
                    if requested.scene_id != settings.scene_id and requested.scene_id != "demo":
                        asset = resolve_scene(requested.scene_id)
                        if asset.asset.preview_mode == "urdf":
                            from urdf_compat import validate_urdf

                            validate_urdf(asset.preview_path)
                            validate_material_bundle(asset)
                        else:
                            # Reject missing/empty/oversized assets before releasing the current renderer.
                            with tempfile.TemporaryDirectory(prefix="check-", dir=directory) as check:
                                prepare_static_stage(asset.preview_path, Path(check) / "background.usda")
                    encoder.close()
                    encoder = None
                    if (requested.scene_id, requested.robot_mode, requested.robot_id) != (
                        settings.scene_id,
                        settings.robot_mode,
                        settings.robot_id,
                    ):
                        renderer_changed = True
                        runtime.close_renderer()
                        candidate = RuntimeScene(requested, directory, runtime.pose["revision"] + 1)
                    elif (requested.width, requested.height, requested.spp, requested.render_mode) != (
                        settings.width,
                        settings.height,
                        settings.spp,
                        settings.render_mode,
                    ):
                        renderer_changed = True
                        runtime.close_renderer()
                        runtime.build_renderer(requested)
                    # Open NVENC only after Nyx finishes replacing CUDA resources:
                    # opening it before teardown fails on repeated 720p/1080p
                    # switches in the real GPU regression test.
                    new_encoder = NvEncoder(
                        requested, resolution=(candidate or runtime).frame_size(requested.camera_name)
                    )
                    if candidate is not None:
                        runtime.close()
                        runtime = candidate
                        candidate = None
                    encoder = new_encoder
                    new_encoder = None
                    settings = requested
                    runtime.settings = requested
                except Exception as exc:
                    logger.exception("Genesis reconfiguration failed; restoring previous scene")
                    error = f"切换失败，保留原场景和配置：{str(exc)[:1000]}"
                    if candidate is not None:
                        candidate.close()
                    if new_encoder is not None:
                        new_encoder.close()
                    if renderer_changed:
                        runtime.close_renderer()
                        runtime.build_renderer(old_settings)
                    if encoder is None:
                        encoder = NvEncoder(old_settings, resolution=runtime.frame_size(old_settings.camera_name))
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
                next_frame = next_status = time.monotonic()
            if command["stream_epoch"] != stream_epoch:
                if stream_epoch >= 0:
                    encoder.close()
                    encoder = NvEncoder(settings, resolution=runtime.frame_size(settings.camera_name))
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
            if runtime.info["mode"] == "robot_mobile":
                motion_request = command.get("base_motion")
                motion = runtime.motion
                motion.accept_request(
                    motion_request,
                    scene_id=runtime.info["instance_id"],
                    revision=revision,
                    now=time.monotonic(),
                    step=runtime.steps,
                )
                base = command.get("base_command")
                valid = (
                    base
                    and command["streaming"]
                    and base["scene_instance_id"] == runtime.info["instance_id"]
                    and base["config_revision"] == revision
                    and base["expires_at"] > time.monotonic()
                )
                if motion.active:
                    pass  # Motion owns wheel targets until measured completion, including braking.
                elif not valid:
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
            if physics_clock.run_batch(runtime.step, stopping=mailbox.stopping):
                timings["physics"].append((time.monotonic() - started) * 1000)
            if mailbox.stopping():
                break
            now = time.monotonic()
            capture = command.get("capture")
            capture_pending = capture and capture["id"] != capture_result.get("id")
            capture_invalid = capture_pending and (
                capture["config_revision"] != revision
                or capture["scene_instance_id"] != runtime.info["instance_id"]
                or (
                    capture.get("camera_name", "observer") == "observer"
                    and capture["camera_revision"] != runtime.pose["revision"]
                )
                or now > capture["expires_at"]
            )
            if capture_pending and (capture_invalid or physics_clock.render_ready()):
                capture_result = {"id": capture["id"]}
                try:
                    if capture_invalid:
                        raise ValueError("Capture request expired or view changed")
                    from PIL import Image

                    captured_at = time.monotonic()
                    captured_utc = datetime.now(UTC).isoformat(timespec="milliseconds")
                    try:
                        name = capture.get("camera_name", "observer")
                        pixels = runtime.renderer.read_rgb(name).cpu().numpy()
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
                    )
                except Exception as exc:
                    logger.warning("Genesis image capture failed (%s)", type(exc).__name__)
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
                with startup.phase("initial_render") if not startup.finished else nullcontext():
                    rgb = runtime.renderer.read_rgb(settings.camera_name)
                    torch.cuda.current_stream().synchronize()
                startup.mark_first_frame()
                startup.finish()
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
            if pick and pick["id"] != pick_result.get("id") and not camera_dirty:
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
                    logger.exception("Genesis picking failed")
                    pick_result["error"] = str(exc)[:500]
                pick_result["duration_ms"] = round((time.monotonic() - pick_started) * 1000, 2)
                next_status = 0
            now = time.monotonic()
            if now >= next_status:
                mailbox.write_status(
                    {
                        "state": "ready" if startup.finished else "starting",
                        "startup": startup.snapshot(),
                        "message": f"Genesis / Nyx / NVENC 已就绪 · {settings.scene_id}"
                        + (
                            "（静态背景，无碰撞）"
                            if runtime.info["mode"] == "static_background"
                            else "（固定姿态检视，无碰撞）"
                            if runtime.info["mode"] == "robot_preview"
                            else "（轮速驱动、地面接触、上身锁定）"
                            if runtime.info["mode"] == "robot_mobile"
                            else ""
                        )
                        if startup.finished
                        else "正在等待首帧…",
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
                        "encoder": "NVENC H264 / pinned RGBA / QP 20 / P4",
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
                        "base_command_result": base_result,
                        "base_motion": runtime.motion.result if runtime.motion is not None else {},
                        "base_observation": (measure(runtime.robot) if runtime.motion is not None else None),
                        "streaming": streaming,
                        "object_positions": [body.get_pos()[0].cpu().tolist() for body in runtime.bodies],
                        "robot": runtime.robot.snapshot() if runtime.robot is not None else None,
                        "scene_contacts": runtime.indoor.contacts(runtime.robot)
                        if runtime.indoor is not None
                        else None,
                    }
                )
                next_status = now + 0.5
            deadline = min(physics_clock.next_step, next_frame) if streaming and available else physics_clock.next_step
            time.sleep(max(0.0, min(0.001, deadline - time.monotonic())))
    except Exception as exc:
        logger.exception("Genesis preview worker failed")
        mailbox.write_status(
            {
                "state": "failed",
                "message": f"Genesis / Nyx 启动或运行失败：{str(exc)[:3000]}",
                "startup": startup.finish(error=True),
            }
        )
    finally:
        for resource in (encoder, runtime):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    logger.exception("Genesis resource cleanup failed")
        encoder = runtime = resource = None
        gc.collect()
        if initialized:
            gs.destroy()
        if sdk_initialized:
            nps.shutdown()
        if capture_path is not None:
            capture_path.unlink(missing_ok=True)


def main() -> None:
    import ctypes
    import faulthandler
    import signal

    faulthandler.enable()

    # Linux parent-death signal also frees CUDA if the GUI crashes without shutdown hooks.
    expected_parent = int(sys.argv[2])
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
        raise OSError(ctypes.get_errno(), "Cannot set worker parent-death signal")
    if os.getppid() != expected_parent:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = Path(sys.argv[1])
    mailbox = Mailbox(path)
    try:
        run_preview(mailbox, path.parent)
    finally:
        mailbox.close()


if __name__ == "__main__":
    main()
