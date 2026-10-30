"""Media lifecycle and real CPU H264 transport; no GUI, Genesis or GPU imports."""

import asyncio
import time

import pytest

from jiuwensymbiosis.simulators.genesis.camera import OrbitCamera
from jiuwensymbiosis.simulators.genesis.stream import PreviewStream, StreamError

pytest.importorskip("aiortc", reason="Install the simulator-media extra for WebRTC tests")
pytest.importorskip("av", reason="Install the simulator-media extra for H264 tests")


class CpuPreview:
    def __init__(self):
        self.state = "stopped"
        self.streaming = False
        self.camera = OrbitCamera()
        self.sequence = 0
        self.starts = self.stops = 0
        self.encoder = None
        self.keyframe_requests = 0
        self.last_frame = 0
        self.configuration = {}

    def configure(self, **values):
        self.configuration = values
        return self.status()

    def status(self):
        return {"state": self.state}

    def start(self):
        self.starts += 1
        self.state = "ready"
        return self.status()

    def stop(self):
        self.stops += 1
        self.state = "stopped"
        return self.status()

    def set_streaming(self, enabled):
        if enabled:
            self.encoder = None
        self.streaming = enabled

    def camera_command(self, action, **values):
        self.camera = self.camera.command(action, **values)
        return {"accepted_revision": self.camera.revision}

    def acknowledge_frame(self, sequence):
        pass

    def request_keyframe(self):
        self.keyframe_requests += 1

    def latest_frame(self):
        from fractions import Fraction

        import av
        import numpy as np

        if time.monotonic() - self.last_frame < 1 / 60:
            return None
        self.last_frame = time.monotonic()

        if self.encoder is None:
            self.encoder = av.CodecContext.create("libx264", "w")
            self.encoder.width = self.encoder.height = 64
            self.encoder.pix_fmt = "yuv420p"
            self.encoder.time_base = Fraction(1, 60)
            self.encoder.options = {"tune": "zerolatency", "crf": "20"}
        self.sequence += 1
        frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), self.sequence % 255, dtype=np.uint8), format="rgb24")
        frame.pts = self.sequence
        return {
            "h264": b"".join(bytes(p) for p in self.encoder.encode(frame)),
            "sequence": self.sequence,
            "captured_at": time.monotonic(),
        }


async def test_selection_requires_live_session_and_dispatches_off_event_loop():
    import threading
    from types import SimpleNamespace

    backend = CpuPreview()
    service = PreviewStream(backend)
    main_thread = threading.get_ident()

    def pick(**values):
        assert threading.get_ident() != main_thread
        return {"selection": {"name": "cube"}, **values}

    backend.select = pick
    with pytest.raises(StreamError, match="会话已结束"):
        await service.select("a" * 32, action="pick")
    service._session = "a" * 32
    service._peer = SimpleNamespace(connectionState="connected")
    result = await service.select("a" * 32, action="pick", u=0.2, v=0.3)
    assert result["selection"]["name"] == "cube"
    assert result["u"] == 0.2


async def test_offer_requires_ready_scene():
    service = PreviewStream(CpuPreview())
    with pytest.raises(StreamError, match="先启动"):
        await service.offer("v=0")
    assert service.status()["connection"] == "closed"


async def test_real_cpu_webrtc_decodes_frames_and_reconnect_keeps_scene():
    rtc = pytest.importorskip("aiortc")
    backend = CpuPreview()
    service = PreviewStream(backend)
    await service.start()
    clients = []

    async def connect():
        pc = rtc.RTCPeerConnection(rtc.RTCConfiguration(iceServers=[]))
        clients.append(pc)
        incoming = asyncio.Queue()
        pc.on("track", incoming.put_nowait)
        pc.addTransceiver("video", direction="recvonly")
        await pc.setLocalDescription(await pc.createOffer())
        answer = await service.offer(pc.localDescription.sdp)
        assert "VP8" not in answer["sdp"]
        assert "H264/90000" in answer["sdp"]
        assert "profile-level-id=42e033" in answer["sdp"]
        await pc.setRemoteDescription(rtc.RTCSessionDescription(sdp=answer["sdp"], type="answer"))
        track = await asyncio.wait_for(incoming.get(), 5)
        frame = await asyncio.wait_for(track.recv(), 5)
        assert (frame.width, frame.height) == (64, 64)
        return answer["session_id"], pc.localDescription.sdp

    try:
        old, sdp = await connect()
        assert backend.streaming
        from aiortc.rtp import RTCP_PSFB_PLI, RtcpPsfbPacket

        sender = service._peer.getSenders()[0]
        before = backend.keyframe_requests
        await sender._handle_rtcp_packet(RtcpPsfbPacket(fmt=RTCP_PSFB_PLI, ssrc=1, media_ssrc=sender._ssrc))
        assert backend.keyframe_requests == before + 1
        with pytest.raises(StreamError, match="已有视频"):
            await service.offer(sdp)
        with pytest.raises(StreamError, match="会话已结束"):
            await service.camera("f" * 32, "reset")
        assert (await service.camera(old, "orbit", dx=0.2))["accepted_revision"] == 1
        assert (await service.camera(old, "fly", dx=0.1, forward=0.1, strafe=0.02, lift=-0.01))[
            "accepted_revision"
        ] == 2
        await service.disconnect(old)
        assert backend.state == "ready" and not backend.streaming
        new, _ = await connect()
        await service.disconnect(old)
        service.heartbeat(new)
        assert service.status()["connection"] == "connected"
        assert backend.starts == 1
        await service.stop()
        assert service.status() == {"state": "stopped", "connection": "closed"}
    finally:
        await service.stop()
        for pc in clients:
            await pc.close()


async def test_invalid_offer_releases_viewer_lease():
    pytest.importorskip("aiortc")
    service = PreviewStream(CpuPreview())
    await service.start()
    try:
        with pytest.raises(StreamError):
            await service.offer("v=0\r\n")
        assert service.status()["connection"] == "closed"
        assert not service.preview.streaming
    finally:
        await service.stop()


async def test_unanswered_offer_expires_even_with_heartbeats():
    rtc = pytest.importorskip("aiortc")
    backend = CpuPreview()
    service = PreviewStream(backend)
    await service.start()
    pc = rtc.RTCPeerConnection(rtc.RTCConfiguration(iceServers=[]))
    try:
        pc.addTransceiver("video", direction="recvonly")
        await pc.setLocalDescription(await pc.createOffer())
        answer = await service.offer(pc.localDescription.sdp)
        service._offered_at -= 16
        service.heartbeat(answer["session_id"])
        await asyncio.wait_for(asyncio.shield(service._watchdog), 4)
        assert service.status()["connection"] == "closed"
        assert not backend.streaming
        assert backend.state == "ready"
    finally:
        await service.stop()
        await pc.close()


@pytest.mark.parametrize(
    "state,elapsed,closed", [("reconfiguring", 15, False), ("reconfiguring", 181, True), ("ready", 0, True)]
)
async def test_watchdog_allows_frame_gap_only_during_bounded_switch(monkeypatch, state, elapsed, closed):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    backend = CpuPreview()
    service = PreviewStream(backend)
    peer = SimpleNamespace(connectionState="connected", close=AsyncMock())
    service._peer, service._session = peer, "viewer"
    service._last_seen = time.monotonic()
    monkeypatch.setattr(backend, "status", lambda: {"state": state, "switch_elapsed_s": elapsed, "frame_age_ms": 15000})
    iterations = 0

    async def next_tick(_):
        nonlocal iterations
        iterations += 1
        if iterations == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr("jiuwensymbiosis.simulators.genesis.stream.asyncio.sleep", next_tick)
    if closed:
        await service._watch("viewer")
        peer.close.assert_awaited_once()
        assert service._session is None
    else:
        with pytest.raises(asyncio.CancelledError):
            await service._watch("viewer")
        peer.close.assert_not_awaited()
        assert service._session == "viewer"


async def test_stale_packet_is_acknowledged_and_waits_for_fresh_idr():
    from collections import deque

    from jiuwensymbiosis.simulators.genesis.stream import make_video_track

    class StalledPreview:
        def __init__(self):
            self.acked = []
            self.requests = 0
            self.packets = deque(
                [
                    {"sequence": 1, "captured_at": time.monotonic() - 3, "keyframe": True, "h264": b"old"},
                    {"sequence": 2, "captured_at": time.monotonic(), "keyframe": False, "h264": b"dependent"},
                    {"sequence": 3, "captured_at": time.monotonic(), "keyframe": True, "h264": b"recovery"},
                ]
            )

        def latest_frame(self):
            return self.packets.popleft()

        def status(self):
            return {"state": "ready"}

        def acknowledge_frame(self, sequence):
            self.acked.append(sequence)

        def request_keyframe(self):
            self.requests += 1

    preview = StalledPreview()
    track = make_video_track(preview)
    try:
        packet = await asyncio.wait_for(track.recv(), 1)
        assert bytes(packet) == b"recovery"
        assert preview.acked == [1, 2, 3]
        assert preview.requests == 1
        assert packet.pts == 0
    finally:
        track.stop()


@pytest.mark.parametrize("action", ["start", "configure"])
async def test_partial_stream_options_default_to_reference(action):
    backend = CpuPreview()
    service = PreviewStream(backend)
    await getattr(service, action)({"resolution": "1280x720", "fps": 30})
    assert backend.configuration["render_quality"] == "reference"


async def test_media_track_waits_through_reconfiguration():
    from jiuwensymbiosis.simulators.genesis.stream import make_video_track

    backend = CpuPreview()
    backend.start()
    track = make_video_track(backend)
    first = await track.recv()
    backend.state = "reconfiguring"
    pending = asyncio.create_task(track.recv())
    try:
        await asyncio.sleep(0.05)
        assert not pending.done()
        backend.state = "ready"
        second = await asyncio.wait_for(pending, 2)
        assert second.pts > first.pts
    finally:
        track.stop()
        if not pending.done():
            pending.cancel()


async def test_disconnected_backend_does_not_skip_local_media_cleanup():
    from unittest.mock import AsyncMock, Mock

    preview = CpuPreview()
    preview.set_streaming = Mock(side_effect=RuntimeError("backend lost"))
    service = PreviewStream(preview)
    peer, track = AsyncMock(), Mock()
    service._peer, service._track = peer, track
    service._watchdog = asyncio.create_task(asyncio.sleep(60))
    watcher = service._watchdog
    with pytest.raises(RuntimeError, match="backend lost"):
        await service.stop()
    assert watcher.cancelled()
    track.stop.assert_called_once()
    peer.close.assert_awaited_once()
