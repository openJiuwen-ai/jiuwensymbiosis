# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reference HTTP server with deterministic model doubles and no audio devices."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from jiuwensymbiosis.serving.speech_server import create_app
from jiuwensymbiosis.voice.protocol import decode_audio, encode_audio


class ASR:
    sample_rate = 16000

    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        if self.fail:
            raise RuntimeError("private model path /weights/secret")
        return "拿起杯子"


class Synthesizer:
    voices = ("default",)

    def synthesize(self, text, voice):
        return np.ones(240, np.int16), 24000


def test_server_asr_tts_and_health_contract():
    with TestClient(create_app(asr=ASR(), synthesizer=Synthesizer())) as client:
        health = client.get("/v1/health").json()["result"]
        assert health["ready"] is True
        assert health["capabilities"] == ["asr", "tts"]
        result = client.post(
            "/v1/transcribe",
            json={"request_id": "r1", "utterance_id": "u1", "audio": encode_audio(np.ones(160, np.int16), 16000)},
        ).json()
        assert result == {"schema_version": 1, "request_id": "r1", "result": {"utterance_id": "u1", "text": "拿起杯子"}}
        response = client.post("/v1/synthesize", json={"request_id": "r2", "text": "完成"})
        assert response.status_code == 200
        result = response.json()["result"]
        audio, rate = decode_audio(result["audio"])
        assert rate == 24000
        assert len(audio) == 240


def test_server_rejects_bad_audio_rate_and_oversized_text_before_inference():
    asr = ASR()
    with TestClient(create_app(asr=asr, synthesizer=Synthesizer(), max_text_length=2)) as client:
        assert (
            client.post(
                "/v1/transcribe",
                json={"request_id": "r1", "utterance_id": "u1", "audio": encode_audio(np.ones(160, np.int16), 8000)},
            ).status_code
            == 422
        )
        assert client.post("/v1/synthesize", json={"request_id": "r2", "text": "too long"}).status_code == 422
        assert (
            client.post("/v1/synthesize", json={"request_id": "r2", "text": "ok", "voice": "unknown"}).status_code
            == 422
        )
        assert asr.calls == 0


def test_disabled_backend_and_model_failures_are_not_empty_success():
    with TestClient(create_app(asr=ASR(fail=True))) as client:
        assert client.post("/v1/synthesize", json={"request_id": "r2", "text": "完成"}).status_code == 503
        response = client.post(
            "/v1/transcribe",
            json={"request_id": "r1", "utterance_id": "u1", "audio": encode_audio(np.ones(160, np.int16), 16000)},
        )
        assert response.status_code == 503
        assert response.json()["request_id"] == "r1"
        assert response.json()["error"]["code"] == "inference_unavailable"
        assert "private model path" not in response.text


def test_raw_request_bound():
    with TestClient(create_app(asr=ASR(), max_request_bytes=100)) as client:
        assert client.get("/v1/health").status_code == 200
        response = client.post("/v1/transcribe", content=b"x" * 101)
        assert response.status_code == 413


def test_chattts_adapter_loads_prepared_weights_and_returns_pcm(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from jiuwensymbiosis.serving.speech_models import ChatTTSSynthesizer

    calls = []

    class Chat:
        InferCodeParams = SimpleNamespace

        def load(self, **kwargs):
            calls.append(kwargs)
            return True

        def sample_random_speaker(self):
            return "test-speaker"

        def infer(self, text, **kwargs):
            assert kwargs["stream"] is False
            assert kwargs["split_text"] is False
            return [np.array([-0.5, 0.0, 0.5], dtype=np.float32)]

        def unload(self):
            calls.append("unload")

    monkeypatch.setitem(sys.modules, "ChatTTS", SimpleNamespace(Chat=Chat))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda name: name))
    model = ChatTTSSynthesizer(tmp_path, device="cpu")
    audio, rate = model.synthesize("test")
    assert audio.dtype == np.int16
    assert audio.tolist() == [-16383, 0, 16383]
    assert rate == 24000
    assert calls[0] == {"source": "custom", "custom_path": str(tmp_path), "device": "cpu", "compile": False}
    model.close()
    assert calls[-1] == "unload"


def test_local_funasr_load_is_explicit_and_does_not_enable_other_models(monkeypatch):
    import sys
    from types import SimpleNamespace

    from jiuwensymbiosis.voice.asr import FunASRBackend

    calls = []
    monkeypatch.setitem(
        sys.modules, "funasr", SimpleNamespace(AutoModel=lambda **kwargs: calls.append(kwargs) or object())
    )
    model = FunASRBackend("/weights/asr", device="cpu")
    assert calls == []
    model.load()
    model.load()
    assert calls == [{"model": "/weights/asr", "disable_update": True, "device": "cpu"}]
    model.close()


@pytest.mark.parametrize("owned", [True, False])
def test_partial_startup_closes_every_attempted_owned_backend(owned):
    events = []

    class Backend:
        def __init__(self, name, *, fail=False):
            self.name, self.fail = name, fail

        def load(self):
            events.append((self.name, "load"))
            if self.fail:
                raise RuntimeError("partial load failed")

        def close(self):
            events.append((self.name, "close"))

    app = create_app(asr=Backend("asr"), synthesizer=Backend("tts", fail=True), owns_backends=owned)
    with pytest.raises(RuntimeError, match="partial load failed"), TestClient(app):
        pytest.fail("partial startup cannot become ready")
    expected = [("asr", "load"), ("tts", "load")]
    if owned:
        expected += [("tts", "close"), ("asr", "close")]
    assert events == expected


@pytest.mark.parametrize("failure", ["load_false", "load_error", "speaker_error"])
def test_failed_chattts_load_retains_a_cleanup_handle_without_becoming_ready(monkeypatch, tmp_path, failure):
    import sys
    from types import SimpleNamespace

    from jiuwensymbiosis.serving.speech_models import ChatTTSSynthesizer

    unloaded = []

    class Chat:
        def load(self, **kwargs):
            if failure == "load_error":
                raise RuntimeError("partial model load")
            return failure != "load_false"

        def sample_random_speaker(self):
            raise RuntimeError("speaker load failed")

        def unload(self):
            unloaded.append(True)

    monkeypatch.setitem(sys.modules, "ChatTTS", SimpleNamespace(Chat=Chat))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda name: name))
    model = ChatTTSSynthesizer(tmp_path)
    with pytest.raises(RuntimeError):
        model.load()
    with pytest.raises(RuntimeError):
        model.load()  # A retained partial model must never count as ready.
    model.close()
    assert unloaded == [True]


def test_startup_cleanup_attempts_other_backends_when_one_close_fails():
    closed = []

    class ASRBackend:
        def close(self):
            closed.append("asr")

    class TTSBackend:
        def load(self):
            raise RuntimeError("partial load failed")

        def close(self):
            closed.append("tts")
            raise RuntimeError("model release failed")

    app = create_app(asr=ASRBackend(), synthesizer=TTSBackend(), owns_backends=True)
    with pytest.raises(ExceptionGroup, match="cleanup is incomplete") as raised, TestClient(app):
        pytest.fail("partial startup cannot become ready")
    assert closed == ["tts", "asr"]
    assert str(raised.value.exceptions[0]) == "model release failed"
