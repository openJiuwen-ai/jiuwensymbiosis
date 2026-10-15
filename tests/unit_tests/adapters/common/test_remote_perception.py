"""HTTP deployment behavior at the adapter/session boundary, without hardware."""

import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.piper.api import PiperApi
from jiuwensymbiosis.adapters.piper.config import PiperConfig
from jiuwensymbiosis.adapters.so101.api import So101Api
from jiuwensymbiosis.agent.cancel import RunCancelled
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.env.mock import MockArmEnv
from jiuwensymbiosis.errors import InferenceServiceError
from jiuwensymbiosis.perception.frame import CameraFrame
from jiuwensymbiosis.utils.service_http import HttpServiceClient


def test_maintenance_reconnect_keeps_inference_lazy_and_usable():
    class Env(MockArmEnv):
        def __init__(self, cfg):
            super().__init__()

    session = make_builder(PiperConfig, Env, PiperApi, managed_detector=True).from_dict(
        {"detector": {"mode": "remote", "endpoint": {"url": "http://service"}}},
        include_sidecars=False,
    )
    calls = []

    def handle(request):
        calls.append(request.headers.get("authorization"))
        return httpx.Response(
            200,
            json={
                "schema_version": 1,
                "request_id": request.headers["x-request-id"],
                "result": {"service": "vision", "status": "ready"},
            },
        )

    detector = session.api._seg_fn
    detector._http = HttpServiceClient(detector.endpoint, service="vision", transport=httpx.MockTransport(handle))
    for cycle in range(2):
        with session:
            assert len(calls) == cycle
            assert detector.readiness()["status"] == "ready"
        assert detector.cleanup_report().released
        with pytest.raises(InferenceServiceError, match="closed"):
            detector.readiness()
    assert calls == [None, None]


def test_maintenance_connect_does_not_require_remote_inference_connection():
    class Env(MockArmEnv):
        def __init__(self, cfg):
            super().__init__()

    session = make_builder(PiperConfig, Env, PiperApi, managed_detector=True).from_dict(
        {
            "detector": {
                "mode": "remote",
                "endpoint": {
                    "url": "https://gpu.invalid/vision",
                },
            },
        },
        include_sidecars=False,
    )
    with session:
        assert session._connected
    assert session.api._seg_fn.cleanup_report().released


@pytest.mark.parametrize("body", ["piper", "so101", "cruzr"])
def test_official_builder_honors_remote_config_without_model_start(body, monkeypatch, tmp_path):
    import importlib

    from jiuwensymbiosis.perception import detector_sidecar

    monkeypatch.setattr(
        detector_sidecar, "detector_subprocess", lambda **kwargs: pytest.fail("Remote mode must not start a model")
    )
    module = importlib.import_module(f"jiuwensymbiosis.adapters.{body}")
    builder = getattr(module, f"build_{body}_session")
    source = Path(__file__).resolve().parents[4] / "jiuwensymbiosis_gui/workbench/data/configs" / body / f"{body}.yaml"
    data = yaml.safe_load(source.read_text())
    data["detector"] = {
        "mode": "remote",
        "endpoint": {
            "url": "https://gpu.invalid/vision",
        },
    }
    session = builder.from_dict(data)
    assert session.env.cfg.detector.mode == "remote"
    assert session.env.cfg.detector.spawn is False
    assert session.env.cfg.detector.local is None
    assert session.api._seg_fn.config.endpoint.url == "https://gpu.invalid/vision"
    events = []
    monkeypatch.setattr(session.env, "connect", lambda: events.append("connect"))
    monkeypatch.setattr(session.env, "disconnect", lambda: events.append("disconnect"))
    session.motion_log_dir = str(tmp_path)

    def handle(request):
        events.append("health")
        return httpx.Response(
            200,
            json={
                "schema_version": 1,
                "request_id": request.headers["x-request-id"],
                "result": {"service": "vision", "status": "ready"},
            },
        )

    detector = session.api._seg_fn
    client = HttpServiceClient(detector.endpoint, service="vision", transport=httpx.MockTransport(handle))
    detector._http = client
    with session:
        assert detector.readiness()["status"] == "ready"
        pool, thread = client._client, client._thread
        assert not pool.is_closed and thread.is_alive()
    assert events == ["connect", "health", "disconnect"]
    assert session.cleanup_report().released
    assert client.closed and pool.is_closed and not thread.is_alive()


@pytest.mark.parametrize("api_cls", [PiperApi, So101Api])
def test_analyze_scene_preserves_service_failure_code_and_invalidates_cache(api_cls, monkeypatch):
    failure = InferenceServiceError("unavailable", code="inference_timeout", service="vision")

    def detect(*args, **kwargs):
        raise failure

    api = api_cls(MockArmEnv(), detector_client=detect)
    api.last_detection = {"cached": True}
    monkeypatch.setattr(api, "get_image", lambda: np.zeros((2, 2, 3), dtype=np.uint8))
    result = api.analyze_scene("box")
    assert result == {"ok": False, "reason": "detector_unavailable", "error_code": "inference_timeout"}
    assert api.last_detection is None


@pytest.mark.parametrize("api_cls", [PiperApi, So101Api])
def test_analyze_scene_propagates_cancellation(api_cls, monkeypatch):
    def detect(*args, **kwargs):
        raise RunCancelled

    api = api_cls(MockArmEnv(), detector_client=detect)
    monkeypatch.setattr(api, "get_image", lambda: np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(RunCancelled):
        api.analyze_scene("box")


@pytest.mark.parametrize("api_cls", [PiperApi, So101Api])
def test_unmanaged_api_with_no_detector_url_reports_disabled(api_cls, monkeypatch):
    api = api_cls(MockArmEnv(), detector_service_url=None)
    monkeypatch.setattr(api, "get_image", lambda: np.zeros((2, 2, 3), dtype=np.uint8))
    result = api.analyze_scene("box")
    assert result["reason"] == "detector_unavailable"
    assert result["error_code"] == "inference_unavailable"


@pytest.mark.parametrize("api_cls", [PiperApi, So101Api, CruzrApi])
@pytest.mark.parametrize("status", [200, 503])
def test_unmanaged_api_releases_http_resources_after_each_call(api_cls, status, monkeypatch):
    clients = []
    original_init = HttpServiceClient.__init__

    def init(self, endpoint, **kwargs):
        def respond(request):
            assert request.url.path == "/v1/segment"
            body = json.loads(request.content)
            return httpx.Response(
                status,
                json={
                    "schema_version": 1,
                    "request_id": body["request_id"],
                    "result": {"frame_id": body["frame_id"], "width": 2, "height": 2, "detections": []},
                },
            )

        transport = httpx.MockTransport(respond)
        original_init(self, endpoint, transport=transport, **kwargs)
        clients.append(self)

    monkeypatch.setattr(HttpServiceClient, "__init__", init)
    env = MockArmEnv()
    api = api_cls(env, detector_service_url="http://vision")
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    if api_cls is CruzrApi:
        monkeypatch.setattr(
            api, "_grab_calibrated_frame", lambda camera: CameraFrame(rgb, np.ones((2, 2)), np.eye(3), np.eye(4))
        )
    else:
        monkeypatch.setattr(api, "get_image", lambda: rgb)
    session = RobotSession(env, api)
    try:
        for _ in range(2):
            with session:
                for _ in range(2):
                    result = api.analyze_scene("box")
                    assert result["ok"] is (status == 200)
                    if status != 200:
                        assert result["reason"] == "detector_unavailable"
                    assert clients
                    assert all(
                        client.closed and client._thread is None and client._client is None for client in clients
                    )
            assert session.cleanup_report().released
        assert len(clients) == 4
    finally:
        for client in clients:
            client.close()
