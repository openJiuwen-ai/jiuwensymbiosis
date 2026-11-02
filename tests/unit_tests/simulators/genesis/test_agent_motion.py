"""Shared actions, body constraints and control ownership without SDK/network."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jiuwensymbiosis.simulators.genesis.agent import make_session
from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview
from jiuwensymbiosis.tools.builder import list_tool_meta


@pytest.fixture
def preview(monkeypatch, tmp_path):
    p = GenesisPreview(state_dir=tmp_path)
    state = {
        "state": "ready",
        "measured_at": time.monotonic(),
        "scene": {"instance_id": "scene"},
        "robot": {"control_mode": "wheel_velocity_physics"},
        "base_motion": {},
    }
    monkeypatch.setattr(p, "status", lambda: state)
    monkeypatch.setattr(p, "_send_control", Mock())
    return p, state


def test_only_one_owner_no_video_required_and_manual_cannot_override(preview):
    p, _ = preview
    assert not p._streaming
    owner = p.acquire_motion()
    with pytest.raises(RuntimeError, match="另一个"):
        p.acquire_motion()
    with pytest.raises(RuntimeError, match="Agent"):
        p.base_command(vx=0.1, vy=0, wz=0, scene_instance_id="scene", config_revision=0)
    p.release_motion("other-owner")
    assert p._motion_owner == owner
    p.release_motion(owner)
    assert p._motion_owner is None


def test_stale_observation_or_scene_rejects_motion(preview):
    p, state = preview
    state["measured_at"] -= 3
    with pytest.raises(RuntimeError, match="过期"):
        p.acquire_motion()
    state["measured_at"] = time.monotonic()
    owner = p.acquire_motion()
    state["scene"]["instance_id"] = "changed"
    with pytest.raises(RuntimeError, match="切换"):
        p.navigate_relative(owner, 0.1, 0, 0)
    with pytest.raises(RuntimeError, match="切换"):
        p.motion_state(owner)
    p.release_motion(owner)


def test_no_overlap_even_when_status_contains_previous_completion(preview):
    p, state = preview
    owner = p.acquire_motion()
    p._motion_request = {"id": "pending", "cancelled": False}
    state["base_motion"] = {"id": "previous", "status": "completed", "settled": True}
    with pytest.raises(RuntimeError, match="尚未完成"):
        p.navigate_relative(owner, 0.1, 0, 0)
    assert p._motion_request["id"] == "pending"


def test_cancelled_session_cannot_send_new_motion(preview):
    p, _ = preview
    event = threading.Event()
    session = make_session(p, cancelled=event)
    session.env.connect()
    event.set()
    with pytest.raises(RuntimeError, match="取消"):
        session.api.navigate_relative(0, 0.2)
    session.env.disconnect()


@pytest.mark.parametrize("completed_action", [False, True])
def test_manual_stop_revokes_owner_between_actions_until_release(preview, monkeypatch, completed_action):
    p, state = preview
    sent = []

    def worker():
        request = p._motion_request
        if request and not request["cancelled"]:
            sent.append(request["id"])
            state["base_motion"] = {"id": request["id"], "status": "completed", "ok": True, "settled": True}

    monkeypatch.setattr(p, "_send_control", worker)
    owner = p.acquire_motion()
    if completed_action:
        assert p.navigate_relative(owner, 0.1, 0, 0)["ok"]
    before_stop = len(sent)
    p.base_command(vx=0, vy=0, wz=0, scene_instance_id="scene", config_revision=0)

    with pytest.raises(RuntimeError, match="取消"):
        p.navigate_relative(owner, 0.1, 0, 0)
    assert len(sent) == before_stop
    with pytest.raises(RuntimeError, match="另一个"):
        p.acquire_motion()

    p.release_motion(owner)
    new_owner = p.acquire_motion()
    assert p.navigate_relative(new_owner, 0.1, 0, 0)["ok"]
    assert len(sent) == before_stop + 1
    p.release_motion(new_owner)


def test_shared_tool_metadata_and_body_commands_include_lateral():
    preview = SimpleNamespace(navigate_relative=Mock(return_value={"ok": True, "settled": True}))
    session = make_session(preview)
    session.env.low_level.owner = "owner"
    meta = {m["name"]: m for m in list_tool_meta(session.api, env=session.env)}
    assert {"navigate_relative", "rotate_base"} <= meta.keys()
    assert session.describe()["base_motion"]["supports_lateral"] is True
    session.api.navigate_relative(0, 0.2)
    assert preview.navigate_relative.call_args.args[:4] == ("owner", 0.0, 0.2, 0.0)
    session.api.rotate_base(0.5)
    assert preview.navigate_relative.call_args.args[:4] == ("owner", 0.0, 0.0, 0.5)


def test_failed_action_blocks_followup_in_the_same_session(preview, monkeypatch):
    p, _ = preview
    session = make_session(p)
    session.env.connect()
    original = p.navigate_relative
    monkeypatch.setattr(p, "navigate_relative", Mock(return_value={"ok": False, "settled": True}))
    assert session.api.navigate_relative(0.1)["ok"] is False
    monkeypatch.setattr(p, "navigate_relative", original)
    with pytest.raises(RuntimeError, match="取消"):
        session.api.navigate_relative(0.1)
    session.env.disconnect()
