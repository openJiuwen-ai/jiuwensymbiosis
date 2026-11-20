"""Shared actions, body constraints and control ownership without SDK/network."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jiuwensymbiosis.simulators.common.agent import make_session, simulation_tools
from jiuwensymbiosis.simulators.common.preview import SimulationPreview
from jiuwensymbiosis.tools.builder import list_tool_meta


@pytest.fixture
def preview(monkeypatch, tmp_path):
    p = SimulationPreview(state_dir=tmp_path)
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
    preview = SimpleNamespace(
        status=lambda: {"state": "ready"}, navigate_relative=Mock(return_value={"ok": True, "settled": True})
    )
    session = make_session(preview)
    session.env.low_level.owner = "owner"
    meta = {m["name"]: m for m in list_tool_meta(session.api, env=session.env)}
    assert {"navigate_relative", "rotate_base"} <= meta.keys()
    assert session.describe()["base_motion"]["supports_lateral"] is True
    session.api.navigate_relative(0, 0.2)
    assert preview.navigate_relative.call_args.args[:4] == ("owner", 0.0, 0.2, 0.0)
    session.api.rotate_base(0.5)
    assert preview.navigate_relative.call_args.args[:4] == ("owner", 0.0, 0.0, 0.5)


@pytest.mark.parametrize("enabled", [True, False])
async def test_roller_policy_tools_connect_without_relative_navigation(preview, monkeypatch, enabled):
    p, state = preview
    state["settings"] = {"robot_mode": "microduck"}
    state["robot"] = {
        "control_mode": "policy_locomotion",
        "relative_navigation": False,
        "supports_lateral": False,
        "policy_actions": ["roller_skate", "mouth"],
    }
    action = Mock(return_value={"kind": "roller_skate", "ok": True, "settled": True, "reason": "reached"})
    monkeypatch.setattr(p, "policy_action", action)
    session = make_session(p, actions_enabled=enabled)
    session.strict_capabilities = True
    with session:
        names = {m["name"] for m in list_tool_meta(session.api, env=session.env)}
        assert {"navigate_relative", "rotate_base"}.isdisjoint(names)
        tools = {tool.card.name: tool for tool in simulation_tools(session.api)}
        assert tools.keys() == ({"roller_skate", "mouth"} if enabled else set())
        if enabled:
            owner = p._motion_owner
            assert owner
            assert (await tools["roller_skate"].invoke({"forward": 0.6}))["ok"]
            assert action.call_args.args == (owner, "roller_skate")
        else:
            assert p._motion_owner is None
            action.assert_not_called()
    assert p._motion_owner is None


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


@pytest.mark.parametrize("settled", [True, False])
def test_missed_kick_replans_only_after_fresh_head_image(preview, monkeypatch, settled):
    p, _ = preview
    session = make_session(p)
    driver = session.env.low_level
    kick = Mock(return_value={"kind": "kick_left", "ok": False, "settled": settled, "reason": "kick_unconfirmed"})
    move = Mock(return_value={"ok": True, "settled": True, "reason": "reached"})
    monkeypatch.setattr(p, "policy_action", kick)
    monkeypatch.setattr(p, "navigate_relative", move)

    with session:
        before_kick = time.monotonic()
        feedback = driver.policy_action("kick_left")
        assert feedback["reason"] == "kick_unconfirmed"
        assert feedback["recoverable"] is settled
        if settled:
            assert feedback["next_observation"] == "capture_camera(head_cam)"
            driver.accept_camera_observation("head_cam", before_kick)
            driver.accept_camera_observation("observer", time.monotonic())
            with pytest.raises(RuntimeError, match="capture_camera"):
                driver.navigate_relative(0.1)
            move.assert_not_called()
            driver.accept_camera_observation("head_cam", time.monotonic())
            assert driver.navigate_relative(0.1)["ok"]
        else:
            driver.accept_camera_observation("head_cam", time.monotonic())
            with pytest.raises(RuntimeError, match="取消"):
                driver.navigate_relative(0.1)
            move.assert_not_called()


def test_settled_short_sprint_returns_measured_failure_to_agent(preview, monkeypatch):
    p, _ = preview
    session = make_session(p)
    driver = session.env.low_level
    monkeypatch.setattr(
        p,
        "policy_action",
        Mock(
            return_value={
                "kind": "sprint_2m",
                "ok": False,
                "settled": True,
                "reason": "sprint_short",
                "forward_displacement_m": 1.999,
                "after": {"position_m": [1.999, 0, 0.116]},
            }
        ),
    )
    with session:
        feedback = driver.policy_action("sprint_2m")
        assert feedback["reason"] == "sprint_short" and not feedback["ok"]
        assert feedback["settled"] and feedback["forward_displacement_m"] == 1.999
        assert "after" not in feedback and not driver.cancelled.is_set()
