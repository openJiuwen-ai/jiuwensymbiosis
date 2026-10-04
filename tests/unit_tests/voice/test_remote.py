# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Speech peer contracts, exercised without a model, microphone, or GPU."""

from __future__ import annotations

import _thread
import asyncio
import base64
import io
import json
import threading
import wave

import httpx
import numpy as np
import pytest

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.errors import InferenceServiceError
from jiuwensymbiosis.utils.service_http import HttpEndpointConfig, HttpServiceClient
from jiuwensymbiosis.voice.asr import RemoteASRBackend
from jiuwensymbiosis.voice.audio import FileAudioSource, RecordTuning
from jiuwensymbiosis.voice.config import VoiceConfig
from jiuwensymbiosis.voice.loop import VoiceLoop
from jiuwensymbiosis.voice.protocol import decode_audio, encode_audio
from jiuwensymbiosis.voice.tts import ChatTTSBackend, NullTTS, RemoteTTSBackend


class MemoryPlayer:
    def __init__(self):
        self.played = []
        self.closed = False

    def play(self, audio, sample_rate):
        self.played.append((audio.copy(), sample_rate))

    def close(self):
        self.closed = True


@pytest.mark.parametrize("over_budget", [False, True])
def test_voice_loop_cancellation_closes_owned_remote_asr_within_budget(monkeypatch, over_budget):
    from jiuwensymbiosis.voice import asr as module

    entered, cleaning = threading.Event(), threading.Event()
    release = asyncio.Event()
    errors, commands = [], []

    async def handle(request):
        try:
            entered.set()
            await asyncio.sleep(30)
        finally:
            cleaning.set()
            if over_budget:
                await release.wait()
            else:
                await asyncio.sleep(0.2)
        return httpx.Response(200, json={})

    monkeypatch.setattr(
        module,
        "HttpServiceClient",
        lambda endpoint, **kwargs: HttpServiceClient(endpoint, transport=httpx.MockTransport(handle), **kwargs),
    )
    loop = VoiceLoop(
        VoiceConfig(
            asr_backend="remote",
            asr_endpoint=HttpEndpointConfig(url="http://speech"),
            shutdown_timeout_s=0.02 if over_budget else 1.0,
        ),
        commands.append,
        audio=FileAudioSource([np.ones(480, dtype=np.int16)]),
        tts=NullTTS(),
    )
    # Let the real factory create ASR so VoiceLoop and ASR own their resources.
    client = loop.asr._client

    def run():
        try:
            with loop:
                loop.run_forever()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(2)
        pool, http_thread = client._client, client._thread
        loop.stop()
        assert cleaning.wait(1)
        worker.join(2)
        assert not worker.is_alive()
        assert commands == []
        if over_budget:
            assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
            assert "cleanup is incomplete" in str(errors[0])
            assert loop.cancel_token.pending_work
            assert not client.closed and not pool.is_closed and http_thread.is_alive()
            client._loop.call_soon_threadsafe(release.set)
            assert loop.cancel_token.wait_for_idle(2)
            loop.config.shutdown_timeout_s = 1.0
            assert loop.close(cancelled=True).released
        else:
            assert errors == []
        assert not loop.cancel_token.pending_work
        assert client.closed and pool.is_closed and not http_thread.is_alive()
        assert loop.close(cancelled=True).released
    finally:
        if client._loop is not None and not client._loop.is_closed():
            client._loop.call_soon_threadsafe(release.set)
        loop.stop()
        worker.join(3)
        client.close()


def test_ctrl_c_during_remote_asr_releases_owned_http_resources(monkeypatch):
    from jiuwensymbiosis.voice import asr as module

    entered, cancelled, finished = threading.Event(), threading.Event(), threading.Event()
    commands, handles = [], []

    async def handle(request):
        handles.append((client._client, threading.current_thread()))
        entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            await asyncio.sleep(0.05)
            raise
        finally:
            finished.set()

    monkeypatch.setattr(
        module,
        "HttpServiceClient",
        lambda endpoint, **kwargs: HttpServiceClient(endpoint, transport=httpx.MockTransport(handle), **kwargs),
    )
    loop = VoiceLoop(
        VoiceConfig(asr_backend="remote", asr_endpoint=HttpEndpointConfig(url="http://speech")),
        commands.append,
        audio=FileAudioSource([np.ones(480, dtype=np.int16)]),
        tts=NullTTS(),
    )
    client = loop.asr._client

    def interrupt():
        if entered.wait(2):
            # Raise KeyboardInterrupt on the listening/main thread; do not set
            # the token first, since VoiceLoop.stop() runs only after unwinding.
            _thread.interrupt_main()

    interrupter = threading.Thread(target=interrupt)
    interrupter.start()
    try:
        with loop:
            loop.run_forever()
        assert entered.is_set() and cancelled.is_set() and finished.is_set()
        assert loop.cancel_token.is_set() and not loop.cancel_token.pending_work
        assert commands == []
        assert client.closed
        pool, http_thread = handles[0]
        assert pool.is_closed and not http_thread.is_alive()
    finally:
        interrupter.join(3)
        client.close()


def test_http_voice_roundtrip_preserves_source_and_playback_rates():
    endpoint = HttpEndpointConfig(url="https://speech.example/prefix")
    incoming_rates = []
    spoken = []

    def respond(request):
        body = json.loads(request.content)
        if request.url.path == "/prefix/v1/transcribe":
            audio, rate = decode_audio(body["audio"])
            incoming_rates.append((len(audio), rate))
            result = {"utterance_id": body["utterance_id"], "text": "九问九问拿起杯子"}
        else:
            assert request.url.path == "/prefix/v1/synthesize"
            spoken.append(body["text"])
            result = {
                "audio": encode_audio(np.arange(240, dtype=np.int16), 24000),
                "sample_rate": 24000,
                "channels": 1,
                "duration_s": 0.01,
            }
        return httpx.Response(200, json={"schema_version": 1, "request_id": body["request_id"], "result": result})

    player = MemoryPlayer()
    commands = []
    with (
        HttpServiceClient(endpoint, service="asr", transport=httpx.MockTransport(respond)) as ac,
        HttpServiceClient(endpoint, service="tts", transport=httpx.MockTransport(respond)) as tc,
    ):
        asr = RemoteASRBackend(endpoint, client=ac)
        tts = RemoteTTSBackend(endpoint, client=tc, player=player, async_play=True)
        with VoiceLoop(
            VoiceConfig(),
            lambda text: (commands.append(text), "完成")[1],
            asr=asr,
            tts=tts,
            audio=FileAudioSource([np.ones(800, np.int16)], sample_rate=8000),
        ) as loop:
            command = loop.run_once()
            loop.handle_command(command)
            loop.handle_command(command)  # one captured utterance is dispatched at most once
            loop.wait()
        tts.close()
    assert commands == ["拿起杯子"]
    assert incoming_rates == [(800, 8000)]
    assert spoken == ["收到指令，开始执行", "完成"]
    assert [rate for _, rate in player.played] == [24000, 24000]
    assert not player.closed  # injected resources remain caller-owned


@pytest.mark.parametrize("corrupt", ["id", "utterance", "text"])
def test_asr_rejects_wrong_identity_and_invalid_text(corrupt):
    class Client:
        def request_json(self, method, path, payload, *, request_id):
            return {
                "schema_version": 1,
                "request_id": "wrong" if corrupt == "id" else request_id,
                "result": {
                    "utterance_id": "wrong" if corrupt == "utterance" else payload["utterance_id"],
                    "text": 123 if corrupt == "text" else "command",
                },
            }

    backend = RemoteASRBackend(HttpEndpointConfig(url="http://speech"), client=Client())
    with pytest.raises(InferenceServiceError, match="Invalid"):
        backend.transcribe(np.ones(20, np.int16))


def test_tts_rejects_metadata_before_playback():
    class Client:
        def request_json(self, method, path, payload, *, request_id):
            return {
                "schema_version": 1,
                "request_id": request_id,
                "result": {
                    "audio": encode_audio(np.ones(240, np.int16), 24000),
                    "sample_rate": 16000,
                    "channels": 1,
                    "duration_s": 0.01,
                },
            }

    player = MemoryPlayer()
    backend = RemoteTTSBackend(
        HttpEndpointConfig(url="http://speech"), client=Client(), player=player, async_play=False
    )
    with pytest.raises(InferenceServiceError):
        backend.speak("test")
    assert not player.played
    backend.close()


def test_wav_rejects_stereo_truncation_and_oversize():
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00" * 40)
    with pytest.raises(ValueError, match="mono"):
        decode_audio({"mime_type": "audio/wav", "data_base64": base64.b64encode(stream.getvalue()).decode()})
    payload = encode_audio(np.ones(1600, np.int16), 16000)
    with pytest.raises(ValueError):
        decode_audio(payload, max_duration_s=0.01)
    with pytest.raises(ValueError):
        decode_audio(payload, max_bytes=100)
    data = base64.b64decode(payload["data_base64"])
    payload["data_base64"] = base64.b64encode(data[:-2]).decode()
    with pytest.raises(ValueError, match="Truncated"):
        decode_audio(payload)


def test_capture_default_has_finite_frame_budget():
    cfg = VoiceConfig(max_audio_duration_s=0.06)
    assert RecordTuning.from_config(cfg).max_frames == 2


def test_wav_file_uses_real_rate_and_rejects_duration_before_reading(tmp_path):
    path = tmp_path / "utterance.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(np.ones(800, dtype="<i2").tobytes())
    source = FileAudioSource([path], sample_rate=16000)
    assert len(source.record_segment()) == 800
    assert source.sample_rate == 8000
    with pytest.raises(ValueError, match="duration"):
        FileAudioSource([path], max_audio_duration_s=0.01).record_segment()


def test_ordered_queue_bound_and_pending_cleanup_are_observable():
    backend = ChatTTSBackend("/unused", queue_size=1)
    token = CancelToken()
    backend.bind_cancel_token(token)
    gate = threading.Event()
    started = threading.Event()
    spoken = []
    backend._loaded = True

    def speaker(text):
        started.set()
        gate.wait(2)
        spoken.append(text)

    backend._speaker = speaker
    backend.speak("first")
    assert started.wait(1)
    backend.speak("second")
    try:
        with pytest.raises(InferenceServiceError, match="queue"):
            backend.speak("third")
        with pytest.raises(TimeoutError):
            backend.close(discard=True, timeout_s=0.01)
        assert token.pending_work == ("voice.tts.playback",)
    finally:
        gate.set()
        backend.close(timeout_s=1)
    assert spoken == ["first"]
    assert not token.pending_work


@pytest.mark.parametrize("owned_player", [True, False])
def test_remote_playback_cancel_wakes_waiter_and_preserves_ownership(monkeypatch, owned_player):
    from jiuwensymbiosis.voice import tts as module

    gate = threading.Event()
    started = threading.Event()
    waiting = threading.Event()
    finished = threading.Event()
    errors = []

    class Player(MemoryPlayer):
        def play(self, audio, sample_rate):
            started.set()
            assert gate.wait(2)

        def close(self):
            super().close()
            gate.set()

    class Client:
        def __init__(self):
            self.closed = False

        def bind_cancel_token(self, token):
            pass

        def close(self):
            self.closed = True

        def request_json(self, method, path, payload, *, request_id):
            return {
                "schema_version": 1,
                "request_id": request_id,
                "result": {
                    "audio": encode_audio(np.ones(240, np.int16), 24000),
                    "sample_rate": 24000,
                    "channels": 1,
                    "duration_s": 0.01,
                },
            }

    player, client = Player(), Client()
    monkeypatch.setattr(module, "SoundDevicePlayer", lambda **kwargs: player)
    backend = RemoteTTSBackend(
        HttpEndpointConfig(url="http://speech"), client=client, player=None if owned_player else player
    )
    token = CancelToken()
    backend.bind_cancel_token(token)

    def wait_for_tts():
        waiting.set()
        try:
            backend.wait()
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    backend.speak("feedback")
    assert started.wait(1)
    waiter = threading.Thread(target=wait_for_tts)
    waiter.start()
    assert waiting.wait(1)
    try:
        token.set()
        assert finished.wait(0.2), "cancellation must interrupt the normal playback wait"
        assert len(errors) == 1 and isinstance(errors[0], RunCancelled)
        assert player.closed is owned_player
        if owned_player:
            backend.close(discard=True, timeout_s=1)
            assert not token.pending_work
        else:
            with pytest.raises(TimeoutError):
                backend.close(discard=True, timeout_s=0.01)
            assert token.pending_work == ("voice.tts.playback",)
        assert not client.closed
    finally:
        gate.set()
        waiter.join(1)
        backend.close(discard=True, timeout_s=1)
    assert not token.pending_work


@pytest.mark.parametrize("discard", [False, True])
@pytest.mark.parametrize("explicit_timeout", [False, True])
def test_owned_remote_tts_close_shares_budget_and_retains_resources_for_retry(monkeypatch, discard, explicit_timeout):
    from jiuwensymbiosis.voice import tts as module

    entered, cleaning, closed = threading.Event(), threading.Event(), threading.Event()
    release = asyncio.Event()
    token = CancelToken()
    requests, errors = [], []
    player = MemoryPlayer()

    async def handle(request):
        requests.append(json.loads(request.content)["text"])
        entered.set()
        try:
            await asyncio.sleep(30)
        finally:
            cleaning.set()
            await release.wait()
        return httpx.Response(200, json={})

    monkeypatch.setattr(
        module,
        "HttpServiceClient",
        lambda endpoint, **kwargs: HttpServiceClient(endpoint, transport=httpx.MockTransport(handle), **kwargs),
    )
    monkeypatch.setattr(module, "SoundDevicePlayer", lambda **kwargs: player)
    backend = RemoteTTSBackend(HttpEndpointConfig("http://speech"), timeout_s=5.0 if explicit_timeout else 0.02)
    backend.bind_cancel_token(token)
    client = backend._client

    def close():
        try:
            kwargs = {"timeout_s": 0.02} if explicit_timeout else {}
            backend.close(discard=discard, **kwargs)
        except Exception as exc:
            errors.append(exc)
        finally:
            closed.set()

    closer = threading.Thread(target=close)
    try:
        backend.speak("first")
        assert entered.wait(2)
        pool, http_thread, worker = client._client, client._thread, backend._worker
        if discard:
            backend.speak("queued")
        closer.start()
        # The HTTP cleanup is held open: closing must report incomplete work
        # within the short budget, without silently starting another 5s wait.
        assert closed.wait(0.5)
        assert len(errors) == 1 and isinstance(errors[0], (TimeoutError, RuntimeError))
        assert token.pending_work
        assert not client.closed and not pool.is_closed and http_thread.is_alive()
        assert player.closed
        with pytest.raises(RuntimeError, match="closed"):
            backend.speak("after close")
        if discard:
            assert cleaning.is_set()
        token.set()
        client._loop.call_soon_threadsafe(release.set)
        backend.close(discard=True, timeout_s=1)
        assert not token.pending_work
        assert client.closed and pool.is_closed
        assert not http_thread.is_alive() and not worker.is_alive()
        assert player.played == [] and requests == ["first"]
        backend.close(discard=True, timeout_s=0)
    finally:
        token.set()
        if client._loop is not None and not client._loop.is_closed():
            client._loop.call_soon_threadsafe(release.set)
        if closer.ident is not None:
            closer.join(2)
        backend.close(discard=True, timeout_s=1)
