"""Cross-client lifecycle and ownership over the real Unix transport, without a GPU."""

import socket
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.resident import ResidentPreview
from jiuwensymbiosis.simulators.genesis.resident_server import ResidentHost, ResidentServer
from jiuwensymbiosis.simulators.genesis.resident_transport import MAX_RESPONSE, receive_message, send_message


@pytest.fixture
def service(tmp_path):
    preview = Mock()
    preview.status.return_value = {"state": "ready", "scene": {"instance_id": "same-world"}, "pid": 123}
    preview.start.side_effect = preview.status
    preview.stop.return_value = {"state": "stopped"}
    preview.acquire_motion.return_value = "owner"
    preview.capture_image.return_value = {
        "jpeg": b"image",
        "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
        "depth_m": np.ones((2, 3), dtype=np.float32),
    }
    host = ResidentHost(preview)
    # Unix socket paths are bounded independently of pytest's long test names.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="genesis-test-") as directory:
        root = Path(directory)
        server = ResidentServer(root / "service.sock", host)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield SimpleNamespace(host=host, preview=preview, root=root)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_gui_reconnect_preserves_world_and_cross_gui_viewer_is_exclusive(service):
    first, second = ResidentPreview(state_dir=service.root), ResidentPreview(state_dir=service.root)
    try:
        assert first.start()["scene"]["instance_id"] == "same-world"
        second.start()
        first.set_streaming(True)
        with pytest.raises(RuntimeError, match="另一个 GUI"):
            second.set_streaming(True)
        second.set_streaming(False)
        assert service.host.viewer == first._client_id
        first.close()
        service.preview.stop.assert_not_called()
        second.set_streaming(True)
        assert second.status()["scene"]["instance_id"] == "same-world"
        service.preview.configure.assert_not_called()
        second.stop()
        service.preview.stop.assert_called_once()
    finally:
        first.close()
        second.close()


def test_capture_roundtrip_supports_jpeg_rgb_and_depth_without_pickle(service):
    client = ResidentPreview(state_dir=service.root)
    client.start()
    try:
        result = client.capture_image("head_cam", include_depth=True)
        assert result["jpeg"] == b"image"
        np.testing.assert_array_equal(result["depth_m"], np.ones((2, 3), dtype=np.float32))
        assert result["rgb"].dtype == np.uint8
        assert service.preview.capture_image.call_args.kwargs["camera_name"] == "head_cam"
    finally:
        client.close()


def dispatch(host, client, method, **params):
    return host.dispatch({"client_id": client, "id": uuid.uuid4().hex, "method": method, "params": params})


def test_client_cannot_use_another_clients_motion_owner(service):
    first, second = uuid.uuid4().hex, uuid.uuid4().hex
    for client in (first, second):
        dispatch(service.host, client, "attach")
    owner = dispatch(service.host, first, "acquire_motion")
    with pytest.raises(RuntimeError, match="控制权"):
        dispatch(service.host, second, "navigate_relative", owner=owner, dx=0.1, dy=0, dyaw=0)
    service.preview.navigate_relative.assert_not_called()


def test_policy_action_roundtrip_enforces_owner_and_passes_cancellation(service):
    service.preview.policy_action.return_value = {"ok": True, "settled": True}
    service.preview.release_motion.return_value = None
    first, second = ResidentPreview(state_dir=service.root), ResidentPreview(state_dir=service.root)
    try:
        first.start()
        second.start()
        owner = first.acquire_motion()
        with pytest.raises(RuntimeError, match="控制权"):
            second.policy_action(owner, "jump")
        service.preview.policy_action.assert_not_called()
        assert first.policy_action(owner, "jump") == {"ok": True, "settled": True}
        assert isinstance(service.preview.policy_action.call_args.kwargs["cancelled"], threading.Event)
        first.release_motion(owner)
    finally:
        first.close()
        second.close()


def test_expired_gui_cancels_inflight_request_and_confirms_stop(service):
    now = [0.0]
    host = ResidentHost(service.preview, clock=lambda: now[0], lease_s=6)
    client = uuid.uuid4().hex
    dispatch(host, client, "attach")
    dispatch(host, client, "acquire_motion")
    cancel = threading.Event()
    host.clients[client].requests["inflight"] = cancel
    now[0] = 7
    host.expire_clients()
    assert cancel.is_set()
    service.preview.release_motion.assert_called_once_with("owner")
    service.preview.stop.assert_not_called()
    with pytest.raises(RuntimeError, match="失效"):
        dispatch(host, client, "status")


def test_unconfirmed_detach_stop_retains_ownership_and_reports_error(service):
    host = service.host
    client = uuid.uuid4().hex
    dispatch(host, client, "attach")
    dispatch(host, client, "acquire_motion")
    service.preview.release_motion.side_effect = RuntimeError("stop unknown")
    assert dispatch(host, client, "detach") == {"detached": False, "error": "stop unknown"}
    assert host.clients[client].owner == "owner"
    assert host.clients[client].closed
    assert host.status(client)["resident"]["cleanup_errors"] == ["stop unknown"]


def test_failed_explicit_stop_does_not_shut_down_service(service):
    client = uuid.uuid4().hex
    dispatch(service.host, client, "attach")
    service.preview.stop.side_effect = RuntimeError("process alive")
    with pytest.raises(RuntimeError):
        dispatch(service.host, client, "stop")
    assert not service.host.closing
    assert not service.host.shutdown_ready
    assert dispatch(service.host, client, "status")["state"] == "ready"


def test_transport_rejects_oversized_header_without_reading_body():
    import struct

    sender, receiver = socket.socketpair()
    try:
        sender.sendall(struct.pack("!I", MAX_RESPONSE + 1))
        with pytest.raises(ValueError, match="size limit"):
            receive_message(receiver, limit=MAX_RESPONSE, timeout=1)
        with pytest.raises(ValueError, match="size limit"):
            send_message(sender, {"large": "0123456789"}, limit=2)
    finally:
        sender.close()
        receiver.close()


def test_status_never_launches_a_backend(tmp_path, monkeypatch):
    client = ResidentPreview(state_dir=tmp_path)
    launch = Mock(side_effect=AssertionError("status must not launch"))
    monkeypatch.setattr("subprocess.Popen", launch)
    assert client.status()["state"] == "stopped"
    assert not client.path.exists()
    client.close()


def test_long_home_does_not_break_optional_gui_routes(tmp_path):
    client = ResidentPreview(state_dir=tmp_path / ("long-home-" * 20))
    assert client.status()["state"] == "stopped"
    with pytest.raises(ValueError, match="too long"):
        client.start()
    assert client.status()["state"] == "failed"


def test_release_completion_cannot_clear_newly_acquired_owner(service):
    client = uuid.uuid4().hex
    dispatch(service.host, client, "attach")
    dispatch(service.host, client, "acquire_motion")

    def release(owner):
        service.preview.acquire_motion.return_value = "new-owner"
        dispatch(service.host, client, "acquire_motion")

    service.preview.release_motion.side_effect = release
    dispatch(service.host, client, "release_motion", owner="owner")
    assert service.host.clients[client].owner == "new-owner"
    dispatch(service.host, client, "motion_state", owner="new-owner")


def test_cancel_arriving_before_request_registration_is_retried(service):
    client = ResidentPreview(state_dir=service.root)
    client.start()
    owner = client.acquire_motion()
    cancelled, allow_registration = threading.Event(), threading.Event()
    original_dispatch = service.host.dispatch
    attempts = []

    def reordered(message):
        if message["method"] == "navigate_relative":
            cancelled.set()
            allow_registration.wait(2)
        result = original_dispatch(message)
        if message["method"] == "cancel_request":
            attempts.append(result["cancelled"])
            allow_registration.set()
        return result

    service.host.dispatch = reordered
    service.preview.navigate_relative.side_effect = lambda **kw: {"cancelled": kw["cancelled"].wait(2)}
    try:
        assert client.navigate_relative(owner, 0.1, 0, 0, cancelled=cancelled) == {"cancelled": True}
        assert attempts[0] is False and attempts[-1] is True
    finally:
        client.close()


def test_already_cancelled_work_is_not_dispatched(service):
    client = ResidentPreview(state_dir=service.root)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError, match="取消"):
        client.navigate_relative("unused", 0.1, 0, 0, cancelled=cancelled)
    service.preview.navigate_relative.assert_not_called()


def test_heartbeat_failure_is_visible_and_terminal(tmp_path):
    client = ResidentPreview(state_dir=tmp_path)
    client._attached = True
    client._rpc = Mock(side_effect=RuntimeError("lease lost"))
    client._heartbeat_stop.wait = Mock(return_value=False)
    client._heartbeat()
    assert not client._attached
    assert client.status() == {"state": "failed", "message": "lease lost"}
    with pytest.raises(RuntimeError, match="失效"):
        ResidentPreview._rpc(client, "acquire_motion")


@pytest.mark.parametrize("operation", ["start", "configure"])
def test_slow_lifecycle_call_does_not_block_heartbeat(service, operation):
    from concurrent.futures import ThreadPoolExecutor

    client = uuid.uuid4().hex
    dispatch(service.host, client, "attach")
    entered, release = threading.Event(), threading.Event()

    def slow():
        entered.set()
        assert release.wait(3)

    getattr(service.preview, operation).side_effect = slow
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(dispatch, service.host, client, operation)
        try:
            assert entered.wait(1)
            heartbeat = pool.submit(dispatch, service.host, client, "heartbeat")
            assert heartbeat.result(timeout=1) == {"alive": True}
        finally:
            release.set()
            pending.result(timeout=1)
