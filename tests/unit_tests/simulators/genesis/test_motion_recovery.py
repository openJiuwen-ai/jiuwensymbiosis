"""Recover only stopped timeouts, with a new head image delivered to the Agent."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from jiuwensymbiosis.agent.config import ModelSpec
from jiuwensymbiosis.simulators.genesis import base_motion
from jiuwensymbiosis.simulators.genesis.agent import GenesisDriver, run_agent_query
from jiuwensymbiosis.simulators.genesis.agent_camera import CameraObservation
from jiuwensymbiosis.simulators.genesis.visual_policy import OBSERVATION_POLICY


def test_timeout_reports_recoverable_stop_only_after_measured_stability(monkeypatch):
    state = {
        "position_m": [0.0, 0.0, 0.25],
        "body_yaw_rad": 0.0,
        "linear_velocity_m_s": [0.1, 0.0, 0.0],
        "angular_velocity_rad_s": [0.0, 0.0, 0.0],
    }
    monkeypatch.setattr(base_motion, "measure", lambda robot: state)
    robot = SimpleNamespace(command=Mock(), stop=Mock())
    motion = base_motion.RelativeMotion(robot)
    motion.start("timeout", 0.25, 0, 0, step=0)
    motion.elapsed = 39.99
    motion.tick(0.02, 1)
    assert motion.failure == "motion_timeout" and motion.phase == "brake"
    for step in range(2, 22):
        motion.tick(0.05, step)
    assert motion.active and not motion.result.get("settled")
    state["linear_velocity_m_s"] = [0.0, 0.0, 0.0]
    for step in range(22, 33):
        motion.tick(0.05, step)
    assert motion.active  # 0.55 seconds is not the required 0.6-second stop window.
    motion.tick(0.05, 33)
    assert not motion.active
    assert motion.result["settled"] and not motion.result["ok"]
    assert motion.result["reason"] == "motion_timeout"
    robot.command.assert_not_called()


def make_driver(result):
    preview = SimpleNamespace(
        acquire_motion=Mock(return_value="owner"),
        navigate_relative=Mock(return_value=result),
        status=lambda: {
            "state": "ready",
            "scene": {"instance_id": "room"},
            "config_revision": 1,
            "cameras": [{"name": "head_cam"}],
        },
    )
    driver = GenesisDriver(preview)
    driver.connect()
    return driver, preview


async def test_stopped_timeout_requires_delivered_head_image_before_next_motion(tmp_path):
    driver, preview = make_driver({"ok": False, "settled": True, "reason": "motion_timeout"})
    result = driver.navigate_relative(0.25)
    assert not driver.cancelled.is_set()
    assert result["recoverable"] is True
    with pytest.raises(RuntimeError, match="head_cam"):
        driver.navigate_relative(-0.1)
    assert preview.navigate_relative.call_count == 1
    preview.capture_image = Mock(
        side_effect=lambda name, **kw: {
            "camera_name": name,
            "captured_at": time.monotonic(),
            "jpeg": b"head-pixels",
        }
    )
    camera = CameraObservation(
        preview, driver.cancelled, tmp_path, "recovery", None, on_observation=driver.accept_camera_observation
    )
    await camera.capture("head_cam")
    # Capturing alone is insufficient: the model has not received the image yet.
    with pytest.raises(RuntimeError, match="head_cam"):
        driver.navigate_relative(-0.1)
    ctx = SimpleNamespace(context=SimpleNamespace(add_messages=AsyncMock()), request_force_finish=Mock())
    await camera.before_model_call(ctx)
    assert camera.frames[0]["injected"]
    preview.navigate_relative.return_value = {"ok": True, "settled": True}
    assert driver.navigate_relative(-0.1)["ok"]
    assert not driver.cancelled.is_set()


def test_old_or_non_head_observation_does_not_unlock_recovery():
    driver, preview = make_driver({"ok": False, "settled": True, "reason": "motion_timeout"})
    old = time.monotonic()
    driver.navigate_relative(0.25)
    driver.accept_camera_observation("head_cam", old)
    driver.accept_camera_observation("observer", time.monotonic())
    with pytest.raises(RuntimeError, match="head_cam"):
        driver.navigate_relative(0.1)
    assert preview.navigate_relative.call_count == 1


@pytest.mark.parametrize(
    "result",
    [
        {"ok": False, "settled": False, "reason": "motion_timeout"},
        {"ok": False, "reason": "motion_timeout"},
        {"ok": False, "settled": True, "reason": "cancelled_or_lease_expired"},
        {"ok": False, "settled": True, "reason": "unknown"},
    ],
)
def test_unconfirmed_stop_or_other_failure_cancels(result):
    driver, preview = make_driver(result)
    driver.navigate_relative(0.25)
    assert driver.cancelled.is_set()
    with pytest.raises(RuntimeError, match="取消"):
        driver.navigate_relative(0.1)
    assert preview.navigate_relative.call_count == 1


def test_user_cancel_cannot_be_cleared_by_stopped_timeout():
    driver, preview = make_driver({"ok": False, "settled": True, "reason": "motion_timeout"})

    def cancel_during_motion(*args, **kwargs):
        driver.cancelled.set()
        return {"ok": False, "settled": True, "reason": "motion_timeout"}

    preview.navigate_relative.side_effect = cancel_during_motion
    result = driver.navigate_relative(0.25)
    assert driver.cancelled.is_set()
    assert not result.get("recoverable")


@pytest.mark.parametrize("error", [TimeoutError("unknown stop"), RuntimeError("ownership lost")])
def test_unknown_outcome_exception_prevents_retries(error, caplog):
    driver, preview = make_driver({})
    preview.navigate_relative.side_effect = error
    with pytest.raises(type(error)):
        driver.navigate_relative(0.25)
    assert driver.cancelled.is_set()
    assert "genesis.motion.failed" in caplog.text
    assert type(error).__name__ in caplog.text
    assert str(error) not in caplog.text


async def test_failed_image_delivery_keeps_recovery_locked(tmp_path):
    driver, preview = make_driver({"ok": False, "settled": True, "reason": "motion_timeout"})
    driver.navigate_relative(0.25)
    preview.capture_image = Mock(
        side_effect=lambda name, **kw: {
            "camera_name": name,
            "captured_at": time.monotonic(),
            "jpeg": b"head-pixels",
        }
    )
    camera = CameraObservation(
        preview, driver.cancelled, tmp_path, "failed", None, on_observation=driver.accept_camera_observation
    )
    await camera.capture("head_cam")
    ctx = SimpleNamespace(
        context=SimpleNamespace(add_messages=AsyncMock(side_effect=RuntimeError())), request_force_finish=Mock()
    )
    await camera.before_model_call(ctx)
    with pytest.raises(RuntimeError, match="head_cam"):
        driver.navigate_relative(0.1)
    assert preview.navigate_relative.call_count == 1


def test_agent_tool_loop_observes_then_replans_after_timeout(tmp_path):
    """A local model fixture drives the real Agent/SDK, not direct driver calls."""
    _, preview = make_driver({})
    preview.navigate_relative.side_effect = [
        {
            "ok": False,
            "settled": True,
            "reason": "motion_timeout",
            "before": "PRIVATE_BEFORE",
            "target_world": "PRIVATE_TARGET",
        },
        {"ok": True, "settled": True, "reason": "reached", "after": "PRIVATE_AFTER", "position_error_m": 123.456789},
    ]
    preview.release_motion = Mock()
    preview.motion_state = lambda owner: {
        **preview.status(),
        "measured_at": time.monotonic(),
        "base_observation": {"position_m": [123.456789, 987.654321, 0]},
        "scene_contacts": {"object": "PRIVATE_CONTACT"},
        "steps": 1,
        "sim_time": 1,
    }
    preview.capture_image = Mock(
        side_effect=lambda name, **kw: {
            "camera_name": name,
            "captured_at": time.monotonic(),
            "jpeg": b"new-head-image",
            "camera": {"T_world_camera": "PRIVATE_EXTRINSICS"},
            "scene_instance_id": "PRIVATE_ROOM",
        }
    )
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            calls = [m for m in request["messages"] if m["role"] == "tool"]
            sequence = [
                ("navigate_relative", {"dx_m": 0.25}),
                ("capture_camera", {"camera_name": "head_cam"}),
                ("navigate_relative", {"dx_m": -0.1, "dy_m": 0.1}),
            ]
            if len(calls) < len(sequence):
                name, params = sequence[len(calls)]
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"step-{len(calls)}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(params)},
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": "已重新观察并调整移动，尚未宣称到达。"}
            body = json.dumps(
                {
                    "id": "recovery-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = run_agent_query(
                preview,
                ModelSpec(api_base=f"http://127.0.0.1:{server.server_port}/v1", model_name="test"),
                "靠近目标；超时停车后重新看图，调整路线继续。",
                output_dir=tmp_path,
                history=[
                    {"role": "assistant", "content": "PRIVATE_LEGACY_STATE"},
                    {"role": "user", "content": "SAFE_VISUAL_GOAL", "observation_policy": OBSERVATION_POLICY},
                ],
            )
        finally:
            server.shutdown()
            thread.join(2)
    assert result["execution_status"] == "completed"
    assert len(result["actions"]) == 2
    assert result["actions"][0]["recoverable"] and result["actions"][1]["ok"]
    assert result["frames"][0]["injected"]
    assert preview.navigate_relative.call_args_list[1].args[1:] == (-0.1, 0.1, 0.0)
    assert any(
        isinstance(m.get("content"), list) and any(c.get("type") == "image_url" for c in m["content"])
        for r in requests[2:]
        for m in r["messages"]
    )
    wire = json.dumps(requests)
    for forbidden in (
        "PRIVATE_",
        "position_m",
        "target_world",
        "position_error_m",
        "scene_contacts",
        "123.456789",
        "T_world_camera",
        "scene_instance_id",
    ):
        assert forbidden not in wire
    assert "SAFE_VISUAL_GOAL" in wire
    assert {t["function"]["name"] for t in next(r["tools"] for r in requests if r.get("tools"))} == {
        "navigate_relative",
        "rotate_base",
        "capture_camera",
    }
    assert result["actions"][0]["target_world"] == "PRIVATE_TARGET"
    assert result["before"]["position_m"][0] == 123.456789
    preview.release_motion.assert_called_once()


@pytest.mark.parametrize(
    "result",
    [
        {"ok": True, "settled": True, "reason": "reached", "after": "PRIVATE_WORLD_POSE"},
        {"ok": False, "settled": True, "reason": "PRIVATE_MESH_COLLISION"},
    ],
)
def test_motion_outcome_allowlist_does_not_forward_private_fields(result):
    driver, _ = make_driver(result)
    feedback = driver.navigate_relative(0.1)
    assert "PRIVATE_" not in json.dumps(feedback)
    assert driver.results[0] == {**result, **({"recoverable": False} if not result["ok"] else {})}
