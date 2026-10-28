"""One WebRTC viewer lease over a GUI-owned preview; optional media imports are lazy."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from fractions import Fraction
from typing import Any

from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)


class StreamError(Exception):
    """A recoverable media/lifecycle error that the viewer can display."""

    def __init__(self, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


def make_video_track(preview: GenesisPreview) -> Any:
    """Create a track without importing any simulator SDK into the GUI process."""
    from aiortc import VideoStreamTrack
    from aiortc.mediastreams import MediaStreamError
    from av import Packet

    class PreviewTrack(VideoStreamTrack):
        def __init__(self) -> None:
            super().__init__()
            self.sequence = -1
            self.epoch: float | None = None
            self.keyframes = 0
            self.waiting_keyframe = False

        async def recv(self):
            await asyncio.sleep(0)
            while self.readyState == "live":
                sample = preview.latest_frame()
                state = preview.status()["state"]
                if state == "reconfiguring":
                    await asyncio.sleep(0.01)
                    continue
                if state != "ready":
                    raise MediaStreamError
                if sample is not None and sample["sequence"] != self.sequence:
                    self.sequence = sample["sequence"]
                    if time.monotonic() - sample["captured_at"] >= 2:
                        preview.request_keyframe()
                        self.waiting_keyframe = True
                    if self.waiting_keyframe:
                        if not sample.get("keyframe") or time.monotonic() - sample["captured_at"] >= 2:
                            preview.acknowledge_frame(self.sequence)
                            await asyncio.sleep(0)
                            continue
                        self.waiting_keyframe = False
                    if self.epoch is None:
                        self.epoch = sample["captured_at"]
                    frame = Packet(sample["h264"])
                    self.keyframes += bool(sample.get("keyframe"))
                    preview.acknowledge_frame(self.sequence)
                    frame.pts = round((sample["captured_at"] - self.epoch) * 90000)
                    frame.time_base = Fraction(1, 90000)
                    return frame
                await asyncio.sleep(0.001)
            raise MediaStreamError

    return PreviewTrack()


class PreviewStream:
    """Own signaling, a single viewer and cleanup, independently of the scene lifetime."""

    def __init__(self, preview: GenesisPreview | None = None) -> None:
        self.preview = preview or GenesisPreview()
        self._lock = asyncio.Lock()
        self._peer: Any = None
        self._track: Any = None
        self._session: str | None = None
        self._watchdog: asyncio.Task | None = None
        self._last_seen = 0.0
        self._offered_at = 0.0

    def status(self) -> dict[str, Any]:
        status = {**self.preview.status(), "connection": self._peer.connectionState if self._peer else "closed"}
        if self._track is not None:
            status["media"] = {"source_epoch": self._track.epoch, "keyframes": self._track.keyframes}
        return status

    async def start(self, options: dict | None = None) -> dict[str, Any]:
        async with self._lock:
            if options is not None:
                width, height = map(int, options["resolution"].split("x"))
                self.preview.configure(
                    width=width,
                    height=height,
                    fps=options["fps"],
                    scene_id=options.get("scene_id", "demo"),
                    robot_mode=options.get("robot_mode", "inspection"),
                    robot_id=options.get("robot_id", ""),
                    render_quality=options.get("render_quality", "reference"),
                    camera_name=options.get("camera_name", "observer"),
                )
            return await asyncio.to_thread(self.preview.start)

    async def stop(self) -> dict[str, Any]:
        async with self._lock:
            await self._close_peer()
            return await asyncio.to_thread(self.preview.stop)

    async def configure(self, options: dict) -> dict[str, Any]:
        """Keep the peer and RTP clock; the worker resumes with a new IDR/SPS."""
        async with self._lock:
            width, height = map(int, options["resolution"].split("x"))
            try:
                return await asyncio.to_thread(
                    self.preview.configure,
                    width=width,
                    height=height,
                    fps=options["fps"],
                    scene_id=options.get("scene_id", "demo"),
                    robot_mode=options.get("robot_mode", "inspection"),
                    robot_id=options.get("robot_id", ""),
                    render_quality=options.get("render_quality", "reference"),
                    camera_name=options.get("camera_name", "observer"),
                )
            except (ValueError, RuntimeError) as exc:
                raise StreamError(str(exc), 409) from exc

    async def offer(self, sdp: str) -> dict[str, str]:
        """Answer a browser receive-only offer; release failed negotiations immediately."""
        async with self._lock:
            if self.preview.status()["state"] != "ready":
                raise StreamError("请先启动 Genesis，等待后台就绪。")
            if self._peer is not None:
                raise StreamError("已有视频连接，请先断开原窗口；关闭窗口后最多等待 20 秒。")
            from aiortc import RTCConfiguration, RTCPeerConnection, RTCRtpSender, RTCSessionDescription

            peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            session = uuid.uuid4().hex
            self._peer, self._session = peer, session
            self._last_seen = self._offered_at = time.monotonic()
            succeeded = False

            @peer.on("connectionstatechange")
            async def state_changed() -> None:
                if peer.connectionState in {"failed", "closed", "disconnected"}:
                    await self.disconnect(session)

            try:
                async with asyncio.timeout(12):
                    from aiortc.sdp import SessionDescription

                    remote = SessionDescription.parse(sdp)
                    if sum(m.kind == "video" for m in remote.media) != 1:
                        raise StreamError("连接请求必须包含一个视频通道。", 400)
                    codecs = [c for c in RTCRtpSender.getCapabilities("video").codecs if c.mimeType == "video/H264"]
                    peer.addTransceiver("video", direction="sendonly").setCodecPreferences(codecs)
                    # Preferences must precede remote negotiation; applying them after
                    # setRemoteDescription leaves VP8 selected in aiortc 1.15.
                    await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
                    self._track = make_video_track(self.preview)
                    sender = peer.addTrack(self._track)
                    # aiortc 1.15's pre-encoded Packet path ignores its keyframe flag.
                    # No public PLI/FIR callback exists in the pinned version; bridge
                    # this sender only. The real CPU WebRTC test exercises RTCP PLI.
                    # pylint: disable=protected-access
                    sender._send_keyframe = self.preview.request_keyframe  # type: ignore[method-assign]  # noqa: SLF001
                    # pylint: enable=protected-access
                    answer = SessionDescription.parse((await peer.createAnswer()).sdp)
                    for media in answer.media:
                        for codec in media.rtp.codecs:
                            if codec.mimeType == "video/H264":
                                if codec.parameters.get("level-asymmetry-allowed") != "1":
                                    raise StreamError("客户端必须支持 H264 level-asymmetry。", 400)
                                # Baseline level 5.1 covers 1080p120. Do not advertise
                                # aiortc's software default (3.1) for the NVENC stream.
                                profile = str(codec.parameters["profile-level-id"])
                                codec.parameters["profile-level-id"] = profile[:4] + "33"
                    await peer.setLocalDescription(RTCSessionDescription(sdp=str(answer), type="answer"))
                self.preview.set_streaming(True)
                self._watchdog = asyncio.create_task(self._watch(session))
                succeeded = True
                return {"sdp": peer.localDescription.sdp, "type": "answer", "session_id": session}
            except StreamError:
                raise
            except Exception as exc:
                logger.warning("Genesis WebRTC negotiation failed: %s", exc)
                raise StreamError("视频协商失败，请重新连接。", 400) from exc
            finally:
                if not succeeded:
                    await self._close_peer()

    def heartbeat(self, session: str) -> None:
        self._require_session(session)
        self._last_seen = time.monotonic()

    async def camera(self, session: str, action: str, **values: float) -> dict:
        async with self._lock:
            self._require_session(session)
            if self._peer.connectionState != "connected":
                raise StreamError("视频连接尚未建立。")
            try:
                return self.preview.camera_command(action, **values)
            except (ValueError, RuntimeError) as exc:
                raise StreamError(str(exc), 400) from exc

    async def select(self, session: str, **values) -> dict:
        async with self._lock:
            self._require_session(session)
            if self._peer.connectionState != "connected":
                raise StreamError("视频连接尚未建立。")
            try:
                return await asyncio.to_thread(self.preview.select, **values)
            except (ValueError, RuntimeError) as exc:
                raise StreamError(str(exc), 400) from exc

    async def base_command(self, session: str, **values) -> dict:
        async with self._lock:
            self._require_session(session)
            if self._peer.connectionState != "connected":
                raise StreamError("视频连接尚未建立。")
            try:
                return self.preview.base_command(**values)
            except (ValueError, RuntimeError) as exc:
                raise StreamError(str(exc), 400) from exc

    def _require_session(self, session: str) -> None:
        if session != self._session or self._peer is None:
            raise StreamError("视频会话已结束，请重新连接。")

    async def disconnect(self, session: str) -> None:
        """An old page's late disconnect must never close its replacement connection."""
        async with self._lock:
            if session == self._session:
                await self._close_peer()

    async def _watch(self, session: str) -> None:
        while session == self._session:
            await asyncio.sleep(2)
            status = self.preview.status()
            viewer_expired = time.monotonic() - self._last_seen > 20 or (
                self._peer is not None
                and self._peer.connectionState != "connected"
                and time.monotonic() - self._offered_at > 15
            )
            preview_stale = (
                status["state"] not in {"ready", "reconfiguring"}
                or (status["state"] == "ready" and status.get("frame_age_ms", 0) > 10000)
                or status.get("switch_elapsed_s", 0) > 180
            )
            if viewer_expired or preview_stale:
                logger.info("Closing stale Genesis viewer")
                await self.disconnect(session)
                return

    async def _close_peer(self) -> None:
        peer, track, watcher = self._peer, self._track, self._watchdog
        self._peer = self._track = self._watchdog = self._session = None
        error = None
        try:
            await asyncio.to_thread(self.preview.set_streaming, False)
        except (RuntimeError, OSError) as exc:
            error = exc
        if watcher is not None and watcher is not asyncio.current_task():
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
        if track is not None:
            track.stop()
        if peer is not None:
            await peer.close()
        if error is not None:
            raise error
