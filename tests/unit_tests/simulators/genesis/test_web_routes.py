"""Local HTTP controls and browser resources without Genesis or a GPU."""

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jiuwensymbiosis.simulators.genesis.camera import OrbitCamera
from jiuwensymbiosis.simulators.genesis.stream import PreviewStream
from jiuwensymbiosis.simulators.genesis.web.routes import install_genesis_routes


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


@pytest.fixture
def client():
    app = FastAPI()
    backend = CpuPreview()
    service = install_genesis_routes(app, PreviewStream(backend))
    with TestClient(app, base_url="http://127.0.0.1:8770", client=("127.0.0.1", 12345)) as client:
        yield client, backend, service


def headers(action):
    return {"Origin": "http://127.0.0.1:8770", "X-Genesis-Action": action}


def test_library_routes_use_independent_index_and_accept_asset_ids(client, tmp_path, monkeypatch):
    import yaml

    from jiuwensymbiosis.simulators.assets import build_index, load_config
    from jiuwensymbiosis.simulators.assets import index as asset_index

    http, backend, _ = client
    source = tmp_path / "assets/background/room/one/background.usda"
    source.parent.mkdir(parents=True)
    source.touch()
    config = tmp_path / "library.yaml"
    config.write_text(
        yaml.safe_dump(
            {"index": "index.sqlite", "roots": [{"id": "geniesim", "scanner": "geniesim", "path": "assets"}]}
        )
    )
    monkeypatch.setattr(asset_index, "DEFAULT_INDEX_PATH", tmp_path / "index.sqlite")
    assert http.get("/genesis/library").status_code == 409
    build_index(load_config(config))
    config.unlink()
    response = http.get("/genesis/library", params={"query": "room/one", "limit": 1})
    assert response.status_code == 200
    assert response.json()["total"] == 1
    identifier = response.json()["items"][0]["id"]
    assert http.get("/genesis/scenes").json()[1]["id"] == identifier
    response = http.post("/genesis/backend/start", headers=headers("start"), json={"scene_id": identifier})
    assert response.status_code == 200
    assert backend.configuration["scene_id"] == identifier
    assert http.get("/genesis/library?limit=1001").status_code == 422


@pytest.mark.parametrize(
    "path,action,body",
    [
        ("backend/start", "start", {}),
        ("backend/stop", "stop", {}),
        ("backend/configure", "configure", {"fps": 30}),
        ("offer", "offer", {"type": "offer", "sdp": "v=0"}),
        ("camera", "camera", {"session_id": "a" * 32, "action": "reset"}),
        ("disconnect", "disconnect", {"session_id": "a" * 32}),
        ("heartbeat", "heartbeat", {"session_id": "a" * 32}),
        (
            "selection",
            "selection",
            {
                "session_id": "a" * 32,
                "action": "pick",
                "u": 0.5,
                "v": 0.5,
                "scene_instance_id": "b" * 32,
                "config_revision": 0,
                "camera_revision": 0,
            },
        ),
    ],
)
@pytest.mark.parametrize("origin", [None, "http://evil.example", "https://127.0.0.1:8770"])
def test_cross_origin_actions_are_rejected_before_side_effects(client, path, action, body, origin):
    http, backend, _ = client
    request_headers = headers(action)
    if origin is None:
        request_headers.pop("Origin")
    else:
        request_headers["Origin"] = origin
    assert http.post(f"/genesis/{path}", headers=request_headers, json=body).status_code == 403
    assert backend.starts == backend.stops == 0


@pytest.mark.parametrize("host,peer", [("127.0.0.1", "192.168.1.5"), ("evil.example", "127.0.0.1")])
def test_rebinding_or_remote_peer_cannot_start(host, peer):
    app = FastAPI()
    backend = CpuPreview()
    install_genesis_routes(app, PreviewStream(backend))
    http = TestClient(app, base_url=f"http://{host}", client=(peer, 12345))
    assert (
        http.post(
            "/genesis/backend/start", headers={"Origin": f"http://{host}", "X-Genesis-Action": "start"}
        ).status_code
        == 403
    )
    assert backend.starts == 0


def test_assets_status_and_local_lifecycle(client):
    http, backend, _ = client
    assert http.get("/genesis/backend").json()["state"] == "stopped"
    response = http.get("/genesis/viewer")
    assert response.status_code == 200
    assert "frame-ancestors 'self'" in response.headers["content-security-policy"]
    assert http.get("/genesis/assets/viewer.js").status_code == 200
    assert http.get("/genesis/assets/../../pyproject.toml").status_code == 404
    assert http.post("/genesis/backend/start", headers=headers("start")).json()["state"] == "ready"
    assert http.post("/genesis/backend/stop", headers=headers("stop")).json()["state"] == "stopped"
    assert backend.starts == backend.stops == 1


@pytest.mark.parametrize("value", [1.01, -1.01, "NaN", True])
@pytest.mark.parametrize("field", ["dx", "forward", "strafe", "lift"])
def test_invalid_camera_commands_are_rejected(client, value, field):
    http, _, _ = client
    assert (
        http.post(
            "/genesis/camera", headers=headers("camera"), json={"session_id": "a" * 32, "action": "fly", field: value}
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "field,value",
    [("u", -0.01), ("v", 1.01), ("u", True), ("v", "NaN"), ("camera_revision", True), ("scene_instance_id", "bad")],
)
def test_selection_rejects_invalid_pixel_and_view_identity(client, field, value):
    http, _, _ = client
    body = {
        "session_id": "a" * 32,
        "action": "pick",
        "u": 0.5,
        "v": 0.5,
        "scene_instance_id": "b" * 32,
        "config_revision": 0,
        "camera_revision": 0,
        field: value,
    }
    assert http.post("/genesis/selection", headers=headers("selection"), json=body).status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"resolution": "7680x4320"},
        {"fps": 120},
        {"fps": 1000},
        {"fps": True},
        {"spp": 1},
        {"render_quality": "balanced"},
        {"render_quality": "high"},
        {"render_quality": "fast"},
    ],
)
def test_start_rejects_unsupported_quality(client, body):
    http, backend, _ = client
    assert http.post("/genesis/backend/start", headers=headers("start"), json=body).status_code == 422
    assert backend.starts == 0


def test_running_configuration_route_preserves_backend(client):
    http, backend, _ = client
    http.post("/genesis/backend/start", headers=headers("start"))
    response = http.post(
        "/genesis/backend/configure", headers=headers("configure"), json={"resolution": "1920x1080", "fps": 30}
    )
    assert response.status_code == 200
    assert backend.configuration == {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "scene_id": "demo",
        "robot_mode": "inspection",
        "robot_id": "",
        "render_quality": "reference",
        "camera_name": "observer",
    }
    assert backend.starts == 1 and backend.stops == 0

    assert http.post("/genesis/backend/configure", headers=headers("configure"), json={"fps": 120}).status_code == 422


def test_base_route_requires_local_origin_live_session_and_finite_values(client):
    from types import SimpleNamespace

    http, backend, service = client
    body = {"session_id": "a" * 32, "scene_instance_id": "b" * 32, "config_revision": 0, "vx": 0.15}
    assert http.post("/genesis/base", json=body).status_code == 403
    assert http.post("/genesis/base", headers=headers("base"), json=body).status_code == 409
    calls = []
    backend.base_command = lambda **kw: calls.append(kw) or {"queued_id": "c" * 32}
    service._session = body["session_id"]
    service._peer = SimpleNamespace(connectionState="connected")
    try:
        for invalid in (True, "nan", 0.31):
            assert http.post("/genesis/base", headers=headers("base"), json={**body, "vx": invalid}).status_code == 422
        assert calls == []
        response = http.post("/genesis/base", headers=headers("base"), json=body)
        assert response.status_code == 200
        assert calls == [{"vx": 0.15, "vy": 0, "wz": 0, "scene_instance_id": "b" * 32, "config_revision": 0}]
        service._peer.connectionState = "disconnected"
        assert http.post("/genesis/base", headers=headers("base"), json=body).status_code == 409
    finally:
        service._peer = service._session = None


@pytest.mark.parametrize("quality", ["reference"])
def test_indoor_background_and_robot_ids_reach_backend_together(client, quality):
    http, backend, _ = client
    room = "library://geniesim/background/home/home_0/background"
    robot = "library://spiritai/moz1/moz1_omni_gripper_double_narrow"
    response = http.post(
        "/genesis/backend/start",
        headers=headers("start"),
        json={"scene_id": room, "robot_id": robot, "robot_mode": "moz1_wheels", "render_quality": quality},
    )
    assert response.status_code == 200
    assert backend.configuration["scene_id"] == room
    assert backend.configuration["robot_id"] == robot
    assert backend.configuration["render_quality"] == quality
    assert backend.configuration["robot_mode"] == "moz1_wheels"
    assert (
        http.post(
            "/genesis/backend/configure", headers=headers("configure"), json={"robot_id": "/tmp/robot.urdf"}
        ).status_code
        == 422
    )


@pytest.mark.parametrize("name", ["viewer.js", "viewer.css", "workbench.css", "metrics.mjs", "navigation.mjs"])
def test_local_assets_revalidate_including_not_modified_responses(client, name):
    http, _, _ = client
    url = f"/genesis/assets/{name}"
    response = http.get(url)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    cached = http.get(url, headers={"If-None-Match": response.headers["etag"]})
    assert cached.status_code == 304
    assert cached.headers["cache-control"] == "no-cache"


def test_viewer_entrypoints_use_current_content_fingerprints(client):
    from hashlib import sha256

    from jiuwensymbiosis.simulators.genesis.web.routes import _ASSETS

    http, _, _ = client
    response = http.get("/genesis/viewer")
    assert response.headers["cache-control"] == "no-store"
    for name in ("viewer.js", "viewer.css"):
        version = sha256((_ASSETS / name).read_bytes()).hexdigest()[:16]
        assert f'/genesis/assets/{name}?v={version}"' in response.text


def test_named_camera_capture_returns_pixels_and_calibration_without_video(client, monkeypatch):
    import json

    http, backend, _ = client

    def capture(name):
        assert name == "head_cam"
        return {
            "jpeg": b"pixels",
            "camera_name": name,
            "step": 42,
            "camera": {"K": [[400, 0, 320], [0, 400, 240], [0, 0, 1]]},
        }

    monkeypatch.setattr(backend, "capture_image", capture, raising=False)
    body = {"camera_name": "head_cam"}
    assert http.post("/genesis/capture", json=body).status_code == 403
    response = http.post("/genesis/capture", headers=headers("capture"), json=body)
    assert response.status_code == 200 and response.content == b"pixels"
    assert json.loads(response.headers["X-Genesis-Camera"])["step"] == 42
    assert response.headers["Cache-Control"] == "no-store"
    assert http.post("/genesis/capture", headers=headers("capture"), json={"camera_name": "../bad"}).status_code == 422
