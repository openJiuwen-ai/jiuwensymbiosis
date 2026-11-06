"""Bounded scene RPC ownership and stale-scene checks, without a worker process."""

import json
import time
from unittest.mock import Mock

import pytest

from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview


@pytest.fixture
def transport(tmp_path, monkeypatch):
    preview = GenesisPreview(state_dir=tmp_path)
    state = {"state": "ready", "measured_at": time.monotonic(), "scene": {"instance_id": "scene"}}
    monkeypatch.setattr(preview, "status", lambda: state)
    return preview, state


def test_readonly_rpc_returns_bound_snapshot_and_removes_response(transport, monkeypatch):
    preview, state = transport

    def worker():
        request = preview._scene_query
        if request:
            (preview.state_dir / f"query-{request['id']}.json").write_text(
                json.dumps({"scene_instance_id": "scene", "snapshot_id": "fresh", "step": 17})
            )
            state["scene_query_result"] = {"id": request["id"]}

    monkeypatch.setattr(preview, "_send_control", worker)
    assert preview.query_scene("list_entities")["snapshot_id"] == "fresh"
    assert preview._motion_owner is None and preview._motion_request is None
    assert preview._scene_query is None and not list(preview.state_dir.glob("query-*.json"))


def test_changed_scene_and_cancelled_queries_do_not_reuse_response(transport, monkeypatch):
    preview, state = transport

    def switch():
        if preview._scene_query:
            state["scene"]["instance_id"] = "other"

    monkeypatch.setattr(preview, "_send_control", switch)
    with pytest.raises(RuntimeError, match="Scene changed"):
        preview.query_scene("robot_state")
    assert preview._scene_query is None


def test_enqueue_failure_and_oversized_parameters_do_not_lock_future_queries(transport, monkeypatch):
    preview, _ = transport
    send = Mock(side_effect=ValueError("Mailbox budget"))
    monkeypatch.setattr(preview, "_send_control", send)
    with pytest.raises(ValueError, match="bounded request"):
        preview.query_scene("entity_state", {"entity_id": "x" * 3000})
    send.assert_not_called()
    assert preview._scene_query is None
    with pytest.raises(ValueError, match="Mailbox budget"):
        preview.query_scene("robot_state")
    assert preview._scene_query is None
