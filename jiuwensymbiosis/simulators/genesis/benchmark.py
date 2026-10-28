"""Measure real Nyx/NVENC/WebRTC delivery on the local GPU (no browser).

python -m jiuwensymbiosis.simulators.genesis.benchmark --resolution 1080p --fps 120 --output outputs/genesis-bench
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", choices=["720p", "1080p"], default="720p")
    parser.add_argument("--fps", type=int, choices=[30, 60, 120], default=120)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="demo", help="GenieSim catalog ID, or demo")
    parser.add_argument("--robot", default="", help="Indexed robot ID to add to a USD background")
    parser.add_argument("--robot-mode", choices=["inspection", "moz1_wheels"], default="inspection")
    parser.add_argument("--spp", type=int, default=8, help="Samples per pixel (1–64)")
    parser.add_argument("--render-mode", choices=["fast", "reference"], default="fast")
    parser.add_argument("--min-fps", type=float, default=0, help="Fail if real decoded FPS is below this value")
    args = parser.parse_args()
    if not 5 <= args.seconds <= 3600:
        parser.error("--seconds must be 5–3600")
    return args


async def benchmark(args: argparse.Namespace) -> None:
    """Own the entire test scene and receiver; stop both on failure or cancellation."""
    import numpy as np
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    from PIL import Image

    from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings
    from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview
    from jiuwensymbiosis.simulators.genesis.stream import PreviewStream

    width, height = (1920, 1080) if args.resolution == "1080p" else (1280, 720)
    settings = PreviewSettings(
        width=width,
        height=height,
        fps=args.fps,
        scene_id=args.scene,
        spp=args.spp,
        robot_id=args.robot,
        robot_mode=args.robot_mode,
        render_mode=args.render_mode,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    preview = GenesisPreview(state_dir=output / "runtime", settings=settings)
    service = PreviewStream(preview)
    client = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    incoming: asyncio.Queue = asyncio.Queue()
    client.on("track", incoming.put_nowait)
    report = {"settings": asdict(settings), "seconds": args.seconds, "warmup_seconds": 4}
    times: list[float] = []
    delays: list[float] = []
    try:
        await service.start()
        async with asyncio.timeout(600):
            while preview.status()["state"] != "ready":
                if preview.status()["state"] == "failed":
                    raise RuntimeError(preview.status()["message"])
                await asyncio.sleep(0.1)
        client.addTransceiver("video", direction="recvonly")
        await client.setLocalDescription(await client.createOffer())
        answer = await service.offer(client.localDescription.sdp)
        await client.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type="answer"))
        track = await asyncio.wait_for(incoming.get(), 5)
        warm = time.monotonic() + 4
        end = warm + args.seconds
        initial = None
        source_epoch = None
        last = None
        while time.monotonic() < end:
            frame = await asyncio.wait_for(track.recv(), 10)
            now = time.monotonic()
            service.heartbeat(answer["session_id"])
            if now < warm:
                continue
            if initial is None:
                initial = preview.status()
                source_epoch = service.status()["media"]["source_epoch"]
            if source_epoch is None:
                raise RuntimeError("Decoded video has no capture timestamp origin")
            times.append(now)
            delays.append((now - source_epoch - frame.pts * float(frame.time_base)) * 1000)
            last = frame
        if initial is None or last is None or len(times) < 2:
            raise RuntimeError("No measurable decoded video")
        final = preview.status()
        duration = times[-1] - times[0]
        measured_duration = final["measured_at"] - initial["measured_at"]
        Image.fromarray(last.to_ndarray(format="rgb24")).save(output / "decoded.png")
        stats = await client.getStats()
        decoded_fps = (len(times) - 1) / duration
        metrics = {
            "decoded_fps": decoded_fps,
            "source_fps": (final["rendered_frames"] - initial["rendered_frames"]) / measured_duration,
            "physics_hz": (final["steps"] - initial["steps"]) / measured_duration,
            "bitrate_mbps": (final["encoded_bytes"] - initial["encoded_bytes"]) * 8 / measured_duration / 1e6,
        }
        report.update(
            **metrics,
            frames=len(times),
            interval_ms_p50_p95=np.percentile(np.diff(times) * 1000, [50, 95]).tolist(),
            receive_delay_ms_p50_p95=np.percentile(delays, [50, 95]).tolist(),
            initial=initial,
            final=final,
            rtp_stats=[vars(x) for x in stats.values()],
            status="passed",
        )
        if decoded_fps < args.min_fps:
            report["status"] = "below_fps_target"
            raise RuntimeError(f"Decoded {decoded_fps:.2f} FPS, required {args.min_fps:.2f}")
        json.dump(metrics, sys.stdout, indent=2)
    finally:
        await service.stop()
        await client.close()
        report["worker_stopped"] = preview.status()["pid"] is None
        (output / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    asyncio.run(benchmark(parse_args()))
