"""Worker-owned NVENC and reusable staging buffers; GUI receives only H264.

Imported only by the standalone worker, without agent/package initialization.
"""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Any


class NvEncoder:
    """Encode RGBA with NVENC, wrapping pinned host memory without a NumPy copy.

    PyAV still requires a device-to-host transfer and NVENC upload. Color conversion
    runs in NVENC; no raw pixels cross IPC or enter the GUI's media pipeline.
    """

    def __init__(self, settings: Any, *, resolution: tuple[int, int] | None = None) -> None:
        import av
        import torch
        from av.video.codeccontext import VideoCodecContext

        self._av = av
        self._torch = torch
        width, height = resolution or (settings.width, settings.height)
        self._rgba = torch.empty((height, width, 4), dtype=torch.uint8, device="cuda")
        self._rgba[..., 3] = 255
        self._host = torch.empty_like(self._rgba, device="cpu", pin_memory=True)
        self._array = self._host.numpy()
        codec = av.CodecContext.create("h264_nvenc", "w")
        if not isinstance(codec, VideoCodecContext):
            raise RuntimeError("h264_nvenc must be a video encoder")
        self._encoder: VideoCodecContext | None = codec
        self._encoder.width = width
        self._encoder.height = height
        self._encoder.pix_fmt = "rgba"
        self._encoder.framerate = Fraction(settings.fps)
        self._encoder.time_base = Fraction(1, settings.fps)
        self._encoder.max_b_frames = 0
        self._encoder.gop_size = max(1, settings.fps // 2)
        self._encoder.options = {
            "preset": "p4",
            "tune": "ull",
            "rc": "constqp",
            "qp": "20",
            "profile": "baseline",
            "level": "5.1",
            "zerolatency": "1",
            "delay": "0",
            "forced-idr": "1",
            "rc-lookahead": "0",
        }
        # Probe hardware support during startup, before advertising ready.
        torch.cuda.empty_cache()
        with av.logging.Capture() as messages:
            try:
                self._encoder.open()
            except av.FFmpegError as exc:
                detail = "; ".join(message.strip() for _, _, message in messages)
                free, total = torch.cuda.mem_get_info()
                raise RuntimeError(
                    f"NVENC 初始化失败（显存空闲 {free / 2**20:.0f}/{total / 2**20:.0f} MiB）：{detail or exc}"
                ) from exc
        self.copy_ms = self.encode_ms = 0.0

    def encode(self, rgb: Any, metadata: dict, *, force_keyframe: bool) -> list[dict]:
        encoder = self._encoder
        if encoder is None:
            raise RuntimeError("NVENC is closed")
        if not rgb.is_cuda or rgb.dtype != self._torch.uint8:
            raise RuntimeError("Nyx must return a CUDA uint8 RGB tensor for NVENC")
        started = time.monotonic()
        self._rgba[..., :3].copy_(rgb)
        self._host.copy_(self._rgba, non_blocking=True)
        self._torch.cuda.current_stream().synchronize()
        self.copy_ms = (time.monotonic() - started) * 1000
        started = time.monotonic()
        frame = self._av.VideoFrame.from_numpy_buffer(self._array, format="rgba")
        frame.pts = metadata["sequence"]
        frame.time_base = encoder.time_base
        if force_keyframe:
            frame.pict_type = self._av.video.frame.PictureType.I
        packets = encoder.encode(frame)
        # Reusing pinned input is safe only with immediate, non-reordered output.
        if len(packets) != 1 or packets[0].pts != frame.pts:
            raise RuntimeError("NVENC must emit exactly one immediate access unit per input frame")
        self.encode_ms = (time.monotonic() - started) * 1000
        packet = packets[0]
        return [{**metadata, "h264": bytes(packet), "keyframe": packet.is_keyframe}]

    def close(self) -> None:
        if self._encoder is not None:
            try:
                self._encoder.encode(None)
            finally:
                self._encoder = None
