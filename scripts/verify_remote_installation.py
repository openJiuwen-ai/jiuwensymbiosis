# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Verify an installed client wheel without Torch against real loopback HTTP.

Run from outside the source checkout with the clean client environment's Python:
    /path/to/client-env/bin/python /checkout/scripts/verify_remote_installation.py

The HTTP responses are deterministic protocol stubs, not model predictions.
No camera, microphone, speaker, credentials or external service is required.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def verify() -> dict:
    """Exercise real HTTP codecs, geometry and playback composition in a clean wheel install."""
    forbidden = ("torch", "torchvision", "torchaudio", "funasr", "ChatTTS", "lerobot")
    for name in forbidden:
        if importlib.util.find_spec(name) is not None:
            raise RuntimeError(f"model module is installed: {name}")
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise RuntimeError(f"model distribution is installed: {name}")

    import numpy as np

    import jiuwensymbiosis
    from jiuwensymbiosis.errors import InferenceServiceError
    from jiuwensymbiosis.perception.config import DetectorConfig
    from jiuwensymbiosis.perception.detector_client import create_detector_client, encode_mask_png
    from jiuwensymbiosis.perception.scene3d import detect_object_geometry
    from jiuwensymbiosis.utils.service_http import HttpEndpointConfig
    from jiuwensymbiosis.voice.asr import RemoteASRBackend
    from jiuwensymbiosis.voice.protocol import decode_audio, encode_audio
    from jiuwensymbiosis.voice.tts import RemoteTTSBackend

    package_path = Path(jiuwensymbiosis.__file__).resolve()
    if not package_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"expected an installed wheel under {sys.prefix}, imported {package_path}")
    mask = np.zeros((64, 64), dtype=bool)
    mask[12:52, 12:52] = True
    audio = np.arange(240, dtype=np.int16)
    requests_seen: list[str] = []

    class Stub(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def reply(self, status: int, data: dict) -> None:
            raw = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            requests_seen.append(self.path)
            if self.path != "/v1/health":
                raise ValueError(f"unexpected path {self.path}")
            self.reply(
                200,
                {
                    "schema_version": 1,
                    "request_id": self.headers.get("X-Request-ID"),
                    "result": {"service": "vision", "status": "ready"},
                },
            )

        def do_POST(self) -> None:
            requests_seen.append(self.path)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            request_id = body.get("request_id")
            if body.get("text_prompt") == "__unavailable__":
                self.reply(
                    503,
                    {
                        "schema_version": 1,
                        "request_id": request_id,
                        "error": {"code": "inference_unavailable", "message": "loading", "retryable": True},
                    },
                )
                return
            detection = {"box": [12, 12, 52, 52], "score": 0.9, "label": "box"}
            if self.path == "/v1/segment":
                result = {
                    "frame_id": body["frame_id"],
                    "width": 64,
                    "height": 64,
                    "detections": [{**detection, "mask": encode_mask_png(mask)}],
                }
            elif self.path == "/v1/transcribe":
                pcm, rate = decode_audio(body["audio"])
                if rate != 16000 or pcm.size != 1600:
                    raise ValueError("ASR client did not preserve PCM input")
                result = {"utterance_id": body["utterance_id"], "text": "定位盒子"}
            elif self.path == "/v1/synthesize":
                result = {"audio": encode_audio(audio, 24000), "sample_rate": 24000, "channels": 1, "duration_s": 0.01}
            else:
                raise ValueError(f"unexpected path {self.path}")
            self.reply(200, {"schema_version": 1, "request_id": request_id, "result": result})

    class Player:
        def __init__(self) -> None:
            self.played: list[tuple] = []

        def play(self, pcm, sample_rate) -> None:
            self.played.append((pcm.copy(), sample_rate))

        def close(self) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    checks = []
    try:
        endpoint = HttpEndpointConfig(url=url)
        with create_detector_client(DetectorConfig(mode="remote", endpoint=endpoint)) as detector:
            detector.readiness()
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            results = detector(rgb, "box")
            if not np.array_equal(results[0]["mask"], mask):
                raise RuntimeError("mask changed during HTTP transfer")
            geometry = detect_object_geometry(
                rgb,
                np.full((64, 64), 0.5),
                np.array([[100, 0, 32], [0, 100, 32], [0, 0, 1]], dtype=float),
                np.eye(4),
                seg_fn=detector,
                object_name="box",
            )
            if not geometry.get("ok") or abs(geometry["center_mm"][2] - 500) > 1e-6:
                raise RuntimeError(f"geometry reconstruction failed: {geometry}")
            try:
                detector(rgb, "__unavailable__")
            except InferenceServiceError as exc:
                if exc.code != "inference_unavailable":
                    raise
            else:
                raise RuntimeError("server failure became an empty detection")
            # Capture-age rejection is opt-in; the default still uses the
            # endpoint's finite request timeout for older captures.
            count = len(requests_seen)
            results = detector(rgb, "box", captured_monotonic_s=time.monotonic() - 30)
            if not np.array_equal(results[0]["mask"], mask) or len(requests_seen) != count + 1:
                raise RuntimeError("default frame-age policy rejected or bypassed detection")
        with create_detector_client(DetectorConfig(mode="remote", endpoint=endpoint, max_frame_age_s=10)) as detector:
            count = len(requests_seen)
            try:
                detector(rgb, "box", captured_monotonic_s=time.monotonic() - 30)
            except InferenceServiceError as exc:
                if exc.code != "inference_result_stale":
                    raise
            else:
                raise RuntimeError("expired frame was accepted")
            if len(requests_seen) != count:
                raise RuntimeError("expired frame made a network request")
        checks.append("vision v1: readiness, detection, mask, 3-D geometry, failure, default/opt-in freshness")
        asr = RemoteASRBackend(endpoint)
        try:
            if asr.transcribe(np.zeros(1600, dtype=np.int16)) != "定位盒子":
                raise RuntimeError("ASR transcript mismatch")
        finally:
            asr.close()
        checks.append("remote ASR: PCM16 WAV through real HTTP")
        player = Player()
        tts = RemoteTTSBackend(endpoint, player=player, async_play=False)
        try:
            tts.speak("已找到盒子")
            tts.wait()
            if (
                len(player.played) != 1
                or player.played[0][1] != 24000
                or not np.array_equal(player.played[0][0], audio)
            ):
                raise RuntimeError("TTS playback samples or sample rate changed")
        finally:
            tts.close()
        checks.append("remote TTS: HTTP WAV, real sample rate, injected memory player")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
    if any(name in sys.modules for name in forbidden):
        raise RuntimeError("a model package was imported during the client run")
    return {
        "ok": True,
        "python": sys.executable,
        "package": str(package_path),
        "model_packages_absent": list(forbidden),
        "http_requests": len(requests_seen),
        "checks": checks,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
