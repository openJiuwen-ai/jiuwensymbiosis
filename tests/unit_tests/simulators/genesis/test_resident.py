"""Cross-client lifecycle and ownership over the real Unix transport, without a GPU."""

import socket
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jiuwensymbiosis.simulators.common.resident import ResidentPreview
from jiuwensymbiosis.simulators.common.resident_server import ResidentHost, ResidentServer
from jiuwensymbiosis.simulators.common.resident_transport import MAX_RESPONSE, receive_message, send_message


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
    client = ResidentPreview(state_dir=service.root)
    client.start()
    try:
        service.preview.stop.side_effect = RuntimeError("process alive")
        with pytest.raises(RuntimeError, match="process alive"):
            client.stop()
        assert not service.host.closing
        assert not service.host.shutdown_ready
        assert client.status()["state"] == "ready"
    finally:
        client.close()


def test_status_reports_stopping_through_worker_and_listener_shutdown(service):
    import fcntl

    client = ResidentPreview(state_dir=service.root)
    entered, exit_worker = threading.Event(), threading.Event()

    def stop_worker():
        entered.set()
        assert exit_worker.wait(3)
        return {"state": "stopped"}

    service.preview.stop.side_effect = stop_worker
    client.start()
    with ThreadPoolExecutor(max_workers=1) as pool, (service.root / "service.lock").open("a") as ownership:
        fcntl.flock(ownership, fcntl.LOCK_EX)
        stopping = pool.submit(client.stop)
        try:
            assert entered.wait(2)
            assert client.status()["state"] == "stopping"
            exit_worker.set()
            assert client._heartbeat_stop.wait(2)
            assert client.status()["state"] == "stopping"
            assert not stopping.done()
        finally:
            exit_worker.set()
            fcntl.flock(ownership, fcntl.LOCK_UN)
        assert stopping.result(timeout=2)["state"] == "stopped"
    assert client.status()["state"] == "stopped"


@pytest.mark.parametrize("failed_reply", [False, True])
def test_status_reply_in_flight_cannot_overwrite_confirmed_stop(service, monkeypatch, failed_reply):
    client = ResidentPreview(state_dir=service.root)
    status_entered, deliver_status = threading.Event(), threading.Event()
    client.start()
    fetch = client._object

    def delayed_status(method, **kwargs):
        if method == "status":
            status_entered.set()
            assert deliver_status.wait(3)
            if failed_reply:
                raise RuntimeError("late response from the released service")
            return {"state": "ready"}
        return fetch(method, **kwargs)

    monkeypatch.setattr(client, "_object", delayed_status)
    with ThreadPoolExecutor(max_workers=1) as pool:
        status = pool.submit(client.status)
        try:
            assert status_entered.wait(2)
            assert client.stop()["state"] == "stopped"
        finally:
            deliver_status.set()
        assert status.result(timeout=2)["state"] == "stopped"
    assert client.status()["state"] == "stopped"


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


def test_sensor_connection_only_attaches_and_preserves_capture_mode(service):
    client = ResidentPreview(state_dir=service.root)
    try:
        assert client.connect()["state"] == "ready"
        client.capture_image("head_cam", include_depth=True, capture_mode="spatial_memory")
        assert service.preview.capture_image.call_args.kwargs["capture_mode"] == "spatial_memory"
        service.preview.start.assert_not_called()
        service.preview.acquire_motion.assert_not_called()
    finally:
        client.close()
    service.preview.stop.assert_not_called()


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


@pytest.mark.parametrize("motion_fails", [False, True])
def test_detach_retries_only_pending_cleanup(service, motion_fails):
    host = service.host
    client = uuid.uuid4().hex
    dispatch(host, client, "attach")
    dispatch(host, client, "acquire_motion")
    dispatch(host, client, "set_streaming", enabled=True)
    cancelled = threading.Event()
    host.clients[client].requests["inflight"] = cancelled
    service.preview.set_streaming.side_effect = RuntimeError("stream failed")
    if motion_fails:
        service.preview.release_motion.side_effect = RuntimeError("stop unknown")
    result = dispatch(host, client, "detach")
    assert not result["detached"] and "stream failed" in result["error"]
    assert cancelled.is_set()
    service.preview.release_motion.assert_called_once_with("owner")
    assert host.clients[client].owner == ("owner" if motion_fails else None)
    assert host.viewer == client
    assert host.status(client)["resident"]["cleanup_errors"] == [result["error"]]
    if motion_fails:
        assert "stop unknown" in result["error"]
    with pytest.raises(RuntimeError, match="失效"):
        dispatch(host, client, "heartbeat")
    replacement = None
    if not motion_fails:
        replacement = uuid.uuid4().hex
        dispatch(host, replacement, "attach")
        service.preview.acquire_motion.return_value = "replacement-owner"
        dispatch(host, replacement, "acquire_motion")
    service.preview.set_streaming.side_effect = None
    service.preview.release_motion.side_effect = None
    host.expire_clients()
    assert client not in host.clients and host.viewer is None
    if replacement is not None:
        assert host.clients[replacement].owner == "replacement-owner"
    assert service.preview.release_motion.call_count == (2 if motion_fails else 1)
    assert host.status(client)["resident"]["cleanup_errors"] == []
    assert dispatch(host, client, "detach") == {"detached": True}


def test_old_detach_retry_preserves_reconnected_viewer(service):
    host = service.host
    old, new = uuid.uuid4().hex, uuid.uuid4().hex
    dispatch(host, old, "attach")
    dispatch(host, old, "acquire_motion")
    dispatch(host, old, "set_streaming", enabled=True)
    service.preview.release_motion.side_effect = RuntimeError("stop unknown")
    assert not dispatch(host, old, "detach")["detached"]
    dispatch(host, new, "attach")
    dispatch(host, new, "set_streaming", enabled=True)
    service.preview.set_streaming.reset_mock()
    service.preview.release_motion.side_effect = None
    assert dispatch(host, old, "detach") == {"detached": True}
    assert host.viewer == new
    service.preview.set_streaming.assert_not_called()


def test_expiration_failure_does_not_skip_other_clients(service, monkeypatch):
    host = service.host
    clients = [uuid.uuid4().hex for _ in range(2)]
    for client in clients:
        dispatch(host, client, "attach")
        host.clients[client].last_seen -= 10
    original = host.detach

    def detach(client):
        if client == clients[0]:
            raise RuntimeError("unexpected cleanup failure")
        return original(client)

    monkeypatch.setattr(host, "detach", detach)
    host.expire_clients()
    assert clients[1] not in host.clients
    assert host.clients[clients[0]].closed
    assert "unexpected cleanup failure" in host.status(clients[0])["resident"]["cleanup_errors"][0]
    monkeypatch.setattr(host, "detach", original)
    host.expire_clients()
    assert not host.clients


@pytest.mark.parametrize("joint_goal", [None, {"arm_joint": 0.3}])
def test_pick_lift_roundtrip_preserves_payload(service, joint_goal):
    client = ResidentPreview(state_dir=service.root)
    client.connect()
    trial = {"lift_m": 0.1, "hold_s": 0.2}
    if joint_goal is not None:
        trial["joint_goal"] = joint_goal
    service.preview.move_arm.return_value = {"settled": True}
    try:
        owner = client.acquire_motion()
        assert client.move_arm(owner, group="left", pick_lift=trial) == {"settled": True}
        values = service.preview.move_arm.call_args.kwargs
        assert values["pick_lift"] == trial
        assert values["owner"] == owner and values["group"] == "left"
        assert isinstance(values["cancelled"], threading.Event)
    finally:
        client.close()


def test_lease_watch_recovers_after_unexpected_failure(service, monkeypatch, caplog):
    host = service.host
    client = uuid.uuid4().hex
    dispatch(host, client, "attach")
    host.clients[client].last_seen -= 10
    expire = host.expire_clients
    calls = []

    def tick():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("watch failed")
        expire()

    monkeypatch.setattr(host, "expire_clients", tick)
    finished = Mock()
    finished.wait.side_effect = [False, False, True]
    with caplog.at_level("ERROR", logger="jiuwensymbiosis"):
        host.watch_clients(finished)
    assert len(calls) == 2 and client not in host.clients
    assert "watch failed" in caplog.text


def test_concurrent_detach_does_not_repeat_pending_release(service):
    from concurrent.futures import ThreadPoolExecutor

    host = service.host
    client = uuid.uuid4().hex
    dispatch(host, client, "attach")
    dispatch(host, client, "acquire_motion")
    entered, finish = threading.Event(), threading.Event()

    def release(owner):
        entered.set()
        assert finish.wait(3)

    service.preview.release_motion.side_effect = release
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(dispatch, host, client, "detach")
        try:
            assert entered.wait(1)
            assert not dispatch(host, client, "detach")["detached"]
            host.expire_clients()
            service.preview.release_motion.assert_called_once_with("owner")
        finally:
            finish.set()
        assert pending.result(timeout=1) == {"detached": True}
    assert client not in host.clients
