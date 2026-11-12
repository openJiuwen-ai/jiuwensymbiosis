"""Opt-in Apple Silicon acceptance using real physics, rendering and WebRTC."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform != "darwin" or os.environ.get("JIUWEN_GENESIS_MACOS_INTEGRATION") != "1",
        reason="Opt-in Apple Silicon / OpenGL acceptance",
    ),
]


async def test_macos_demo_video_camera_resize_reconnect_and_cleanup(tmp_path):
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

    async def wait_until(predicate, seconds=30):
        async with asyncio.timeout(seconds):
            while not predicate():
                state = preview.status()
                assert state["state"] != "failed", state.get("message")
                await asyncio.sleep(0.1)

    async def connect():
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        clients.append(pc)
        incoming = asyncio.Queue()
        pc.on("track", incoming.put_nowait)
        pc.addTransceiver("video", direction="recvonly")
        await pc.setLocalDescription(await pc.createOffer())
        answer = await service.offer(pc.localDescription.sdp)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type="answer"))
        track = await asyncio.wait_for(incoming.get(), 10)
        frame = await asyncio.wait_for(track.recv(), 30)
        return answer["session_id"], track, frame

    started = time.monotonic()
    try:
        await service.start()
        await wait_until(lambda: preview.status()["state"] == "ready", 300)
        pid = preview.status()["pid"]
        report["startup_seconds"] = round(time.monotonic() - started, 2)
        session, track, frame = await connect()
        assert (frame.width, frame.height) == (960, 540)
        first = frame.to_ndarray(format="rgb24")
        assert first.std() > 10, "Rendered image must contain visible geometry"
        Image.fromarray(first).save(evidence / "macos-first-frame.png")
        frames, heights, differences = 0, [], []
        video_started = time.monotonic()
        while time.monotonic() - video_started < 6:
            frame = await asyncio.wait_for(track.recv(), 15)
            service.heartbeat(session)
            pixels = frame.to_ndarray(format="rgb24")
            differences.append(float(np.abs(pixels.astype(float) - first).mean()))
            heights.append(preview.status()["object_positions"][2][2])
            frames += 1
        assert max(heights) - min(heights) > 0.2
        assert max(differences) > 0.1
        report.update(decoded_frames=frames, decoded_fps=frames / (time.monotonic() - video_started))
        before = preview.status()["camera"]["pos"]
        accepted = await service.camera(session, "orbit", dx=0.15)
        await wait_until(lambda: preview.status()["camera"]["revision"] == accepted["accepted_revision"])
        assert preview.status()["camera"]["pos"] != before
        capture = await asyncio.to_thread(preview.capture_image)
        assert capture["width"] == 960 and capture["height"] == 540
        report["jpeg_capture"] = True
        await service.disconnect(session)
        second_session, second_track, _ = await connect()
        assert preview.status()["pid"] == pid
        await asyncio.wait_for(second_track.recv(), 15)
        await service.disconnect(second_session)
        preview.configure(width=1280, height=720, fps=30)
        await wait_until(lambda: preview.status().get("config_revision") == 1)
        _, _, resized = await connect()
        assert (resized.width, resized.height) == (1280, 720)
        assert preview.status()["pid"] == pid
        report["final_status"] = preview.status()
        await service.stop()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        report["worker_exited"] = True
        report["status"] = "passed"
        (evidence / "macos-acceptance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    finally:
        await service.stop()
        for pc in clients:
            await pc.close()
        log = preview.state_dir / "worker.log"
        if log.exists():
            (evidence / "worker.log").write_bytes(log.read_bytes())
