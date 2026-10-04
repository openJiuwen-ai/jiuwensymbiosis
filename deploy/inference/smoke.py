"""Exercise configured remote models without a body, microphone, or speaker.

Run from a checkout after installing .[remote]. The built-in image and silence
check the transport; supply --image / --audio to assess model outputs.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import uuid
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from jiuwensymbiosis.perception.config import parse_detector_config
from jiuwensymbiosis.perception.detector_client import create_detector_client
from jiuwensymbiosis.utils.service_http import HttpServiceClient
from jiuwensymbiosis.voice.asr import build_asr_backend
from jiuwensymbiosis.voice.config import VoiceConfig
from jiuwensymbiosis.voice.protocol import decode_audio, response_result, validate_audio_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="red object")
    parser.add_argument("--audio", type=Path, help="Mono PCM16 WAV; must match the deployed ASR sample rate")
    parser.add_argument("--text", default="远程语音服务连接正常。")
    parser.add_argument("--output-wav", type=Path, help="Optionally save TTS audio; never plays it")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("Configuration must be a YAML mapping")
    detector = parse_detector_config(config)
    voice = VoiceConfig.from_dict(config.get("voice"))
    report = {}
    if detector.mode == "remote":
        started = time.monotonic()
        with create_detector_client(detector) as client:
            client.readiness()
            if args.image:
                with Image.open(args.image) as source:
                    frame = np.asarray(source.convert("RGB"))
            else:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                frame[60:180, 80:240, 0] = 255
            detections = client(frame, args.prompt)
        report["vision"] = {"detections": len(detections), "elapsed_s": round(time.monotonic() - started, 3)}
    if voice.asr_backend == "remote":
        started = time.monotonic()
        backend = build_asr_backend(voice)
        try:
            if args.audio:
                if args.audio.stat().st_size > voice.asr_endpoint.max_request_bytes:
                    raise ValueError("Input WAV exceeds the configured request limit")
                audio, rate = decode_audio(
                    {"mime_type": "audio/wav", "data_base64": base64.b64encode(args.audio.read_bytes()).decode()},
                    max_duration_s=voice.max_audio_duration_s,
                )
            else:
                rate = voice.sample_rate
                audio = np.zeros(rate, dtype=np.int16)
            text = backend.transcribe_utterance(audio, rate, uuid.uuid4().hex)
        finally:
            backend.close()
        report["asr"] = {"text": text, "elapsed_s": round(time.monotonic() - started, 3)}
    if voice.tts_backend == "remote":
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        with HttpServiceClient(voice.tts_endpoint, service="tts") as client:
            data = client.request_json(
                "POST",
                "/v1/synthesize",
                {"request_id": request_id, "text": args.text, "voice": voice.tts_voice},
                request_id=request_id,
            )
        result = response_result(data, request_id, service="tts")
        audio, rate = decode_audio(
            result.get("audio"),
            max_duration_s=voice.tts_max_audio_duration_s,
            max_bytes=voice.tts_endpoint.max_response_bytes,
        )
        validate_audio_metadata(result, audio, rate)
        if args.output_wav:
            args.output_wav.write_bytes(base64.b64decode(result["audio"]["data_base64"], validate=True))
        report["tts"] = {
            "sample_rate": rate,
            "duration_s": len(audio) / rate,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    if not report:
        parser.error("Configure at least one remote backend")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
