"""Media lifecycle and real CPU H264 transport; no GUI, Genesis or GPU imports."""

import asyncio
import threading
import time

import pytest

from jiuwensymbiosis.simulators.common.camera import OrbitCamera
from jiuwensymbiosis.simulators.common.stream import PreviewStream, StreamError

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
        self.keyframe_event = threading.Event()
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

    def close(self):
        self.detached = True

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
        self.keyframe_event.set()

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


@pytest.mark.parametrize("operation", ["select", "camera", "base_command"])
async def test_selection_requires_live_session_and_dispatches_off_event_loop(operation):
    import threading
    from types import SimpleNamespace

    backend = CpuPreview()
    service = PreviewStream(backend)
    main_thread = threading.get_ident()

    def pick(*args, **values):
        assert threading.get_ident() != main_thread
        return {"selection": {"name": "cube"}, **values}

    setattr(backend, "camera_command" if operation == "camera" else operation, pick)
    call = getattr(service, operation)
    with pytest.raises(StreamError, match="会话已结束"):
        await call("a" * 32, action="pick")
    service._session = "a" * 32
    service._peer = SimpleNamespace(connectionState="connected")
    result = await call("a" * 32, action="pick", u=0.2, v=0.3)
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
        backend.keyframe_event.clear()
        await sender._handle_rtcp_packet(RtcpPsfbPacket(fmt=RTCP_PSFB_PLI, ssrc=1, media_ssrc=sender._ssrc))
        assert await asyncio.to_thread(backend.keyframe_event.wait, 1)
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

    monkeypatch.setattr("jiuwensymbiosis.simulators.common.stream.asyncio.sleep", next_tick)
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
    import threading
    from collections import deque

    main_thread = threading.get_ident()

    from jiuwensymbiosis.simulators.common.stream import make_video_track

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
            assert threading.get_ident() != main_thread
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
@pytest.mark.parametrize(
    "options,expected",
    [
        (
            {"resolution": "1280x720", "fps": 30},
            {
                "width": 1280,
                "height": 720,
                "fps": 30,
                "scene_id": "demo",
                "robot_mode": "inspection",
                "robot_id": "",
                "render_quality": "reference",
                "camera_name": "observer",
                "backend": "genesis",
            },
        ),
        (
            {
                "resolution": " 01920x+1080 ",
                "fps": 60,
                "scene_id": "room",
                "robot_mode": "robot",
                "robot_id": "test-robot",
                "render_quality": "fast",
                "camera_name": "head",
                "ignored": "unused",
            },
            {
                "width": 1920,
                "height": 1080,
                "fps": 60,
                "scene_id": "room",
                "robot_mode": "robot",
                "robot_id": "test-robot",
                "render_quality": "fast",
                "camera_name": "head",
                "backend": "genesis",
            },
        ),
        (
            {
                "resolution": "64x32",
                "fps": None,
                "scene_id": "",
                "robot_mode": None,
                "robot_id": None,
                "render_quality": "",
                "camera_name": None,
            },
            {
                "width": 64,
                "height": 32,
                "fps": None,
                "scene_id": "",
                "robot_mode": None,
                "robot_id": None,
                "render_quality": "",
                "camera_name": None,
                "backend": "genesis",
            },
        ),
    ],
    ids=["defaults", "explicit", "falsey-values"],
)
async def test_stream_options_reach_preview_before_start(monkeypatch, action, options, expected):
    backend = CpuPreview()
    service = PreviewStream(backend)
    original_options = options.copy()
    main_thread = threading.get_ident()
    events = []
    configure = backend.configure
    start = backend.start

    def record_configuration(**values):
        assert threading.get_ident() != main_thread
        events.append("configure")
        return configure(**values)

    def record_start():
        assert threading.get_ident() != main_thread
        assert backend.configuration == expected
        events.append("start")
        return start()

    monkeypatch.setattr(backend, "configure", record_configuration)
    monkeypatch.setattr(backend, "start", record_start)
    result = await getattr(service, action)(options)
    assert backend.configuration == expected
    assert options == original_options
    assert events == (["configure", "start"] if action == "start" else ["configure"])
    assert result == {"state": "ready" if action == "start" else "stopped"}


async def test_start_without_options_keeps_existing_configuration(monkeypatch):
    from unittest.mock import Mock

    backend = CpuPreview()
    backend.configuration = {"width": 640, "height": 480, "fps": 24}
    configure = Mock(side_effect=AssertionError("unexpected reconfiguration"))
    monkeypatch.setattr(backend, "configure", configure)
    assert await PreviewStream(backend).start() == {"state": "ready"}
    configure.assert_not_called()
    assert backend.configuration == {"width": 640, "height": 480, "fps": 24}
    assert backend.starts == 1


@pytest.mark.parametrize("action", ["start", "configure"])
@pytest.mark.parametrize(
    "options,error",
    [
        ({"resolution": "bad", "fps": 30}, ValueError),
        ({"resolution": "640x480x32", "fps": 30}, ValueError),
        ({"fps": 30}, KeyError),
        ({"resolution": "640x480"}, KeyError),
    ],
)
async def test_invalid_stream_options_fail_before_preview(monkeypatch, action, options, error):
    from unittest.mock import Mock

    backend = CpuPreview()
    configure = Mock()
    monkeypatch.setattr(backend, "configure", configure)
    with pytest.raises(error):
        await getattr(PreviewStream(backend), action)(options)
    configure.assert_not_called()
    assert backend.starts == 0


@pytest.mark.parametrize("action", ["start", "configure"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, TimeoutError])
async def test_preview_configuration_failure_preserves_entry_point_boundary(monkeypatch, action, error_type):
    from unittest.mock import Mock

    backend = CpuPreview()
    error = error_type("configuration failed")
    configure = Mock(side_effect=error)
    monkeypatch.setattr(backend, "configure", configure)
    with pytest.raises(StreamError if action == "configure" else error_type) as caught:
        await getattr(PreviewStream(backend), action)({"resolution": "640x480", "fps": 30})
    if action == "configure":
        assert caught.value.status_code == 409
        assert isinstance(caught.value.__cause__, error_type)
        assert str(caught.value.__cause__) == "configuration failed"
    assert str(caught.value) == "configuration failed"
    configure.assert_called_once()
    assert backend.starts == 0


async def test_media_track_waits_through_reconfiguration():
    from jiuwensymbiosis.simulators.common.stream import make_video_track

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
    assert preview.stops == 1
    track.stop.assert_called_once()
    peer.close.assert_awaited_once()


async def test_shutdown_detaches_without_stopping_resident_backend():
    preview = CpuPreview()
    service = PreviewStream(preview)
    await service.start()
    await service.close()
    assert preview.detached and preview.stops == 0


async def test_reset_between_frame_and_status_discards_old_world_packet():
    from unittest.mock import Mock

    from jiuwensymbiosis.simulators.common.stream import make_video_track

    preview = CpuPreview()
    now = time.monotonic()
    samples = iter(
        [
            {
                "sequence": 1,
                "h264": b"old-world",
                "scene_instance_id": "before",
                "config_revision": 0,
                "stream_epoch": 0,
                "keyframe": True,
                "captured_at": now,
            },
            {
                "sequence": 2,
                "h264": b"new-world",
                "scene_instance_id": "after",
                "config_revision": 0,
                "stream_epoch": 1,
                "keyframe": True,
                "captured_at": now,
            },
        ]
    )
    preview.latest_frame = lambda: next(samples)
    preview.status = lambda: {
        "state": "ready",
        "scene": {"instance_id": "after"},
        "config_revision": 0,
        "stream_epoch": 1,
    }
    preview.acknowledge_frame = Mock()
    track = make_video_track(preview)
    try:
        assert bytes(await track.recv()) == b"new-world"
        assert preview.keyframe_requests == 1
        assert preview.acknowledge_frame.call_count == 2
    finally:
        track.stop()


@pytest.mark.parametrize("stop_fails", [False, True])
async def test_peer_failure_still_stops_backend_and_preserves_errors(stop_fails):
    from unittest.mock import AsyncMock, Mock

    preview = CpuPreview()
    preview.stop = Mock(side_effect=RuntimeError("stop failed") if stop_fails else None)
    service = PreviewStream(preview)
    service._peer = AsyncMock()
    service._peer.close.side_effect = RuntimeError("peer failed")
    with pytest.raises(RuntimeError, match="stop failed" if stop_fails else "peer failed") as caught:
        await service.stop()
    preview.stop.assert_called_once()
    if stop_fails:
        assert "peer failed" in str(caught.value.__context__)
