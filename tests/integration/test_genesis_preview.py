"""Opt-in real CUDA + Nyx + H264 WebRTC acceptance, independent from browser QA.

JIUWEN_GENESIS_INTEGRATION=1 python -m pytest -q tests/integration/test_genesis_preview.py
Set JIUWEN_GENESIS_EVIDENCE to retain decoded images and measurements.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("JIUWEN_GENESIS_INTEGRATION") != "1", reason="Opt-in CUDA/Nyx acceptance"),
]


async def test_real_nyx_motion_camera_reconnect_and_cleanup(tmp_path):
    import numpy as np
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    from PIL import Image

    from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview
    from jiuwensymbiosis.simulators.genesis.stream import PreviewStream

    evidence = Path(os.environ.get("JIUWEN_GENESIS_EVIDENCE", tmp_path / "evidence"))
    await asyncio.to_thread(evidence.mkdir, parents=True, exist_ok=True)
    preview = GenesisPreview(state_dir=tmp_path / "runtime")
    service = PreviewStream(preview)
    clients = []
    report = {}

    async def wait_until(predicate, seconds=15):
        async with asyncio.timeout(seconds):
            while not predicate():
                if preview.status()["state"] == "failed":
                    pytest.fail(preview.status()["message"])
                await asyncio.sleep(0.05)

    async def connect():
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        clients.append(pc)
        incoming = asyncio.Queue()
        pc.on("track", incoming.put_nowait)
        pc.addTransceiver("video", direction="recvonly")
        await pc.setLocalDescription(await pc.createOffer())
        answer = await service.offer(pc.localDescription.sdp)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type="answer"))
        track = await asyncio.wait_for(incoming.get(), 5)
        frame = await asyncio.wait_for(track.recv(), 10)
        assert (frame.width, frame.height) == (1280, 720)
        return answer["session_id"], track

    started = time.monotonic()
    try:
        await service.start()
        await wait_until(lambda: preview.status()["state"] == "ready", 180)
        pid = preview.status()["pid"]
        report["startup_seconds"] = round(time.monotonic() - started, 2)
        session, track = await connect()
        start = time.monotonic()
        frames = 0
        heights = []
        first = previous = None
        max_image_change = 0.0
        while time.monotonic() - start < 8:
            frame = await asyncio.wait_for(track.recv(), 5)
            service.heartbeat(session)
            image = frame.to_ndarray(format="rgb24")
            if first is None:
                first = image.copy()
                Image.fromarray(first).save(evidence / "nyx-first-decoded.png")
            if previous is not None:
                change = float(np.abs(image.astype(float) - previous).mean())
                if change > max_image_change:
                    max_image_change = change
                    Image.fromarray(image).save(evidence / "nyx-motion-decoded.png")
            previous = image.astype(float)
            heights.append(preview.status()["object_positions"][2][2])
            frames += 1
        report.update(
            decoded_frames=frames,
            decoded_fps=round(frames / (time.monotonic() - start), 2),
            object_height_range_m=[min(heights), max(heights)],
            max_frame_difference=max_image_change,
        )
        assert max(heights) - min(heights) > 0.2
        assert max_image_change > 0.1
        report["camera_commands"] = []
        for action, params in [
            ("orbit", {"dx": 0.15}),
            ("pan", {"dx": 0.05, "dy": 0.04}),
            ("zoom", {"delta": -0.2}),
            ("reset", {}),
        ]:
            before = preview.status()["camera"]
            accepted = await service.camera(session, action, **params)
            revision = accepted["accepted_revision"]
            await wait_until(
                lambda revision=revision: (
                    preview.status().get("camera", {}).get("revision") == revision
                    and preview.latest_frame()["camera_revision"] == revision
                )
            )
            assert preview.status()["camera"]["pos"] != before["pos"]
            report["camera_commands"].append(
                {"action": action, "revision": revision, "applied": preview.status()["camera"]}
            )
        await service.disconnect(session)
        await wait_until(lambda: preview.status().get("streaming") is False)
        step = preview.status()["steps"]
        frame_count = preview.status()["rendered_frames"]
        await wait_until(lambda: preview.status()["steps"] > step, seconds=2)
        assert preview.status()["rendered_frames"] == frame_count
        second_session, second_track = await connect()
        assert preview.status()["pid"] == pid
        await service.disconnect(session)  # stale client cannot close its replacement
        assert service.status()["connection"] == "connected"
        await asyncio.wait_for(second_track.recv(), 5)
        report["reconnect_preserved_pid"] = pid
        service.heartbeat(second_session)
        await service.stop()
        assert service.status()["state"] == "stopped"
        assert not await asyncio.to_thread(Path(f"/proc/{pid}").exists)
        report["worker_exited"] = True
        report["status"] = "passed"
        (evidence / "gpu-webrtc-acceptance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    finally:
        await service.stop()
        for pc in clients:
            await pc.close()
        if (preview.state_dir / "worker.log").exists():
            (evidence / "acceptance-worker.log").write_bytes((preview.state_dir / "worker.log").read_bytes())


def test_real_nvenc_quality_and_forced_idr():
    """Validate color/range and recovery from a fresh decoder, using actual NVENC."""
    import av
    import numpy as np
    import torch

    from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings
    from jiuwensymbiosis.simulators.genesis.encoding import NvEncoder

    settings = PreviewSettings()
    yy, xx = np.mgrid[: settings.height, : settings.width]
    source = np.stack((xx / settings.width * 255, yy / settings.height * 255, xx * 0 + 128), axis=-1).astype(np.uint8)
    encoder = NvEncoder(settings)
    try:
        gpu = torch.from_numpy(source).cuda()
        first = encoder.encode(gpu, {"sequence": 1}, force_keyframe=True)[0]
        middle = encoder.encode(gpu, {"sequence": 2}, force_keyframe=False)[0]
        recovery = encoder.encode(gpu, {"sequence": 3}, force_keyframe=True)[0]
        assert first["keyframe"] and recovery["keyframe"] and not middle["keyframe"]
        decoder = av.CodecContext.create("h264", "r")
        image = decoder.decode(av.Packet(recovery["h264"]))[0].to_ndarray(format="rgb24")
        psnr = 10 * np.log10(255**2 / np.mean((source.astype(float) - image) ** 2))
        assert psnr > 40
        assert decoder.profile == "Constrained Baseline"
        assert (decoder.width, decoder.height) == (settings.width, settings.height)
    finally:
        encoder.close()


async def test_real_hot_switch_keeps_worker_peer_and_physics(tmp_path):
    """Same PID/RTP session across FPS, resolution, USD and failed-scene changes."""
    import numpy as np
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    from PIL import Image

    from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview
    from jiuwensymbiosis.simulators.genesis.stream import PreviewStream

    evidence = Path(os.environ.get("JIUWEN_GENESIS_EVIDENCE", tmp_path / "hot-evidence"))
    await asyncio.to_thread(evidence.mkdir, parents=True, exist_ok=True)
    preview = GenesisPreview(state_dir=tmp_path / "hot-runtime")
    service = PreviewStream(preview)
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    incoming = asyncio.Queue()
    pc.on("track", incoming.put_nowait)
    arrivals = []
    latest = None
    session = None
    pump = heartbeat = None
    report = {"switches": []}

    async def wait_ready():
        async with asyncio.timeout(180):
            while preview.status()["state"] != "ready":
                if preview.status()["state"] == "failed":
                    pytest.fail(preview.status()["message"])
                await asyncio.sleep(0.05)

    async def consume(track):
        nonlocal latest
        while True:
            frame = await track.recv()
            latest = (time.monotonic(), frame)
            arrivals.append(latest[0])

    async def keepalive():
        while True:
            service.heartbeat(session)
            await asyncio.sleep(1)

    try:
        await service.start()
        await wait_ready()
        pid = preview.status()["pid"]
        report["pid"] = pid
        pc.addTransceiver("video", direction="recvonly")
        await pc.setLocalDescription(await pc.createOffer())
        answer = await service.offer(pc.localDescription.sdp)
        session = answer["session_id"]
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type="answer"))
        track = await asyncio.wait_for(incoming.get(), 10)
        pump = asyncio.create_task(consume(track))
        heartbeat = asyncio.create_task(keepalive())
        await asyncio.sleep(1)
        scenarios = [
            ("fps30", "demo", "1280x720", 30, False),
            ("fps60", "demo", "1280x720", 60, False),
            ("demo1080", "demo", "1920x1080", 60, False),
            ("demo720", "demo", "1280x720", 60, False),
            ("kitchen", "library://geniesim/background/kitchen/kitchen_1/background", "1280x720", 60, False),
            ("kitchen1080", "library://geniesim/background/kitchen/kitchen_1/background", "1920x1080", 60, False),
            ("kitchen720", "library://geniesim/background/kitchen/kitchen_1/background", "1280x720", 60, False),
            ("office", "library://geniesim/background/office/office_00/background", "1280x720", 60, False),
            ("empty-rejected", "library://geniesim/background/room/room_2/background", "1280x720", 60, True),
            ("large-rejected", "library://geniesim/background/common/factory/background", "1280x720", 60, True),
            ("demo-return", "demo", "1280x720", 60, False),
        ]
        previous_image = None
        for label, scene_id, resolution, fps, expected_failure in scenarios:
            before = preview.status()
            start = time.monotonic()
            requested = await service.configure({"resolution": resolution, "fps": fps, "scene_id": scene_id})
            revision = requested["requested_config_revision"]
            await wait_ready()
            ready = time.monotonic()
            after = preview.status()
            assert after["pid"] == pid
            assert after["config_revision"] == revision
            assert service.status()["connection"] == pc.connectionState == "connected"
            assert service._session == session
            assert bool(after["reconfigure_error"]) == expected_failure
            expected_scene = before["settings"]["scene_id"] if expected_failure else scene_id
            assert after["settings"]["scene_id"] == expected_scene
            retained_world = expected_scene == before["settings"]["scene_id"]
            assert (after["scene"]["instance_id"] == before["scene"]["instance_id"]) == retained_world
            if retained_world:
                assert after["steps"] >= before["steps"]
                assert after["camera"] == before["camera"]
            width, height = after["settings"]["width"], after["settings"]["height"]
            async with asyncio.timeout(15):
                while latest is None or latest[0] < ready or (latest[1].width, latest[1].height) != (width, height):
                    if pump.done():
                        await pump
                    await asyncio.sleep(0.01)
            first_frame = latest[0]
            # Settle decoder queues before measuring delivered cadence.
            await asyncio.sleep(1)
            sample_start = time.monotonic()
            await asyncio.sleep(4)
            times = [t for t in arrivals if t >= sample_start]
            assert len(times) > 20
            measured = (len(times) - 1) / (times[-1] - times[0])
            if resolution == "1280x720" and not expected_failure:
                assert measured >= fps * 0.85
            image = latest[1].to_ndarray(format="rgb24")
            assert image.std() > 5
            Image.fromarray(image).save(evidence / f"{label}.png")
            if previous_image is not None and not retained_world and image.shape == previous_image.shape:
                assert np.abs(image.astype(float) - previous_image).mean() > 5
            previous_image = image.astype(float)
            report["switches"].append(
                {
                    "label": label,
                    "settings": after["settings"],
                    "pid": after["pid"],
                    "same_peer": True,
                    "retained_world": retained_world,
                    "ready_ms": round((ready - start) * 1000, 1),
                    "video_resumed_ms": round((first_frame - start) * 1000, 1),
                    "decoded_fps": round(measured, 2),
                    "error": after["reconfigure_error"],
                    "scene_instance": after["scene"]["instance_id"],
                }
            )
            (evidence / "hot-switch.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        report["status"] = "passed"
    finally:
        for task in (pump, heartbeat):
            if task:
                task.cancel()
        await asyncio.gather(*(t for t in (pump, heartbeat) if t), return_exceptions=True)
        await service.stop()
        await pc.close()
        report["worker_stopped"] = preview.status()["pid"] is None
        (evidence / "hot-switch.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        if (preview.state_dir / "worker.log").exists():
            (evidence / "hot-worker.log").write_bytes((preview.state_dir / "worker.log").read_bytes())


async def test_real_asset_library_sources_render_and_switch(tmp_path):
    """Real indexed USD/URDF → Genesis/Nyx → NVENC → decoded WebRTC, same worker."""
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    from PIL import Image

    from jiuwensymbiosis.simulators.assets import AssetLibrary
    from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview
    from jiuwensymbiosis.simulators.genesis.stream import PreviewStream

    library = AssetLibrary()
    identifiers = [
        "library://spiritai/moz1/moz1_omni_gripper_double_narrow",
        "library://geniesim/background/kitchen/kitchen_1/background",
    ]
    rooms = [
        asset
        for asset in library.search(kind="environment", query="simple_room", limit=1000).items
        if asset.name == "simple_room"
    ]
    assert len(rooms) == 1, "Index exactly one simple_room environment"
    identifiers.append(rooms[0].id)
    for identifier in identifiers:
        library.resolve(identifier)
    evidence = Path(os.environ.get("JIUWEN_GENESIS_EVIDENCE", tmp_path / "evidence"))
    await asyncio.to_thread(evidence.mkdir, parents=True, exist_ok=True)
    preview = GenesisPreview(state_dir=tmp_path / "library-runtime")
    service = PreviewStream(preview)
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    incoming = asyncio.Queue()
    pc.on("track", incoming.put_nowait)
    latest = None
    arrivals = []
    pump = heartbeat = None
    report = {"index": library.status(), "scenes": []}

    async def ready():
        async with asyncio.timeout(180):
            while preview.status()["state"] != "ready":
                if preview.status()["state"] == "failed":
                    pytest.fail(preview.status()["message"])
                await asyncio.sleep(0.05)

    async def consume(track):
        nonlocal latest
        while True:
            frame = await track.recv()
            latest = (time.monotonic(), frame)
            arrivals.append(latest[0])

    async def keepalive(session):
        while True:
            service.heartbeat(session)
            await asyncio.sleep(1)

    try:
        await service.start({"scene_id": identifiers[0], "resolution": "1280x720", "fps": 60})
        await ready()
        pid = preview.status()["pid"]
        pc.addTransceiver("video", direction="recvonly")
        await pc.setLocalDescription(await pc.createOffer())
        answer = await service.offer(pc.localDescription.sdp)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type="answer"))
        track = await asyncio.wait_for(incoming.get(), 10)
        pump = asyncio.create_task(consume(track))
        heartbeat = asyncio.create_task(keepalive(answer["session_id"]))
        for index, identifier in enumerate(identifiers):
            if index:
                await service.configure({"scene_id": identifier, "resolution": "1280x720", "fps": 60})
                await ready()
            status = preview.status()
            assert status["pid"] == pid
            assert not status.get("reconfigure_error")
            assert status["scene"]["id"] == identifier
            since = time.monotonic()
            async with asyncio.timeout(15):
                while latest is None or latest[0] < since:
                    if pump.done():
                        await pump
                    await asyncio.sleep(0.01)
            await asyncio.sleep(1)
            begin = time.monotonic()
            await asyncio.sleep(3)
            times = [t for t in arrivals if t >= begin]
            assert len(times) >= 10
            frame = latest[1]
            pixels = frame.to_ndarray(format="rgb24")
            assert (frame.width, frame.height) == (1280, 720)
            assert pixels.std() > 5
            root_id = library.get(identifier).root_id
            Image.fromarray(pixels).save(evidence / f"{root_id}.png")
            report["scenes"].append(
                {
                    "id": identifier,
                    "pid": pid,
                    "same_peer": pc.connectionState == "connected",
                    "decoded_fps": round((len(times) - 1) / (times[-1] - times[0]), 2),
                    "scene": status["scene"],
                    "robot": status.get("robot"),
                }
            )
            (evidence / "asset-library-render.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        report["passed"] = True
    finally:
        for task in (pump, heartbeat):
            if task:
                task.cancel()
        await asyncio.gather(*(t for t in (pump, heartbeat) if t), return_exceptions=True)
        await service.stop()
        await pc.close()
        report["worker_stopped"] = preview.status()["pid"] is None
        (evidence / "asset-library-render.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        if (preview.state_dir / "worker.log").exists():
            (evidence / "asset-library-worker.log").write_bytes((preview.state_dir / "worker.log").read_bytes())
