"""Opt-in native robot policies + shared Agent + real camera/WebRTC acceptance."""

import asyncio
import io
import json
import math
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="Apple Silicon simulation acceptance",
    ),
]


@pytest.mark.parametrize(
    ("body", "skills"),
    [("jumper", False), ("jumper", True), ("microduck", False)],
    ids=["locomotion", "jump-and-claws", "microduck"],
)
async def test_jumper_shared_agent_motion_camera_video_and_stop(tmp_path, monkeypatch, body, skills):
    prefix = "JIUWEN_" + body.upper()
    if os.environ.get(prefix + "_INTEGRATION") != "1":
        pytest.skip("Opt-in " + body + " acceptance")
    import numpy as np
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    from PIL import Image

    from jiuwensymbiosis.agent.config import ModelSpec
    from jiuwensymbiosis.simulators.assets.robot_assets import register_robot
    from jiuwensymbiosis.simulators.common.agent import make_session, run_agent_query
    from jiuwensymbiosis.simulators.common.preview import SimulationPreview
    from jiuwensymbiosis.simulators.common.resident import ResidentPreview
    from jiuwensymbiosis.simulators.common.stream import PreviewStream

    source = await asyncio.to_thread(Path(os.environ[prefix + "_SOURCE"]).resolve)
    entry = await asyncio.to_thread(register_robot, body, source, tmp_path)
    monkeypatch.setenv("JIUWENSYMBIOSIS_SIMULATOR_HOME", str(tmp_path))
    evidence = Path(os.environ.get(prefix + "_EVIDENCE", tmp_path / "evidence"))
    await asyncio.to_thread(evidence.mkdir, parents=True, exist_ok=True)
    preview_type = ResidentPreview if os.environ.get("JIUWEN_SIM_RESIDENT_INTEGRATION") == "1" else SimulationPreview
    runtime_directory = tempfile.TemporaryDirectory(prefix="sym-policy-", dir="/tmp")
    preview = preview_type(state_dir=Path(runtime_directory.name))
    service = PreviewStream(preview)
    peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    report = {"model_service": "deterministic local HTTP fixture; real Agent and physics", "source": str(source)}

    async def wait(predicate, seconds=60):
        async with asyncio.timeout(seconds):
            while not predicate():
                state = preview.status()
                assert state["state"] != "failed", state.get("message")
                await asyncio.sleep(0.1)

    requests = []
    sequence = [
        ("capture_camera", {"camera_name": "head_cam"}),
        ("navigate_relative", {"dx_m": 0.2}),
        ("navigate_relative", {"dx_m": 0.0, "dy_m": 0.15}),
        ("rotate_base", {"dyaw_rad": 0.5}),
        ("navigate_relative", {"dx_m": -0.15}),
        ("capture_camera", {"camera_name": "head_cam"}),
    ]

    if skills:
        sequence = [
            ("capture_camera", {"camera_name": "head_cam"}),
            ("jump", {}),
            ("close_gripper", {"effector": "left"}),
            ("open_gripper", {"effector": "left"}),
            ("close_gripper", {"effector": "right"}),
            ("open_gripper", {"effector": "right"}),
            ("navigate_relative", {"dx_m": 0.15}),
            ("jump", {}),
            ("capture_camera", {"camera_name": "head_cam"}),
        ]
        evidence = evidence / "skills"
        await asyncio.to_thread(evidence.mkdir, parents=True, exist_ok=True)

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            count = sum(m.get("role") == "tool" for m in body.get("messages", []))
            call = bool(body.get("tools")) and count < len(sequence)
            if body.get("tools"):
                requests.append(body)
            message = {"role": "assistant", "content": "已完成测试动作并重新观察。"}
            if call:
                name, params = sequence[count]
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{count}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(params)},
                        }
                    ],
                }
            payload = json.dumps(
                {
                    "id": "jumper-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "jumper-test",
                    "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if call else "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    try:
        preview.configure(width=960, height=540, fps=30, scene_id=entry["scene_id"], backend="mujoco", robot_mode=body)
        await asyncio.to_thread(preview.start)
        await wait(lambda: preview.status()["state"] == "ready")
        initial = preview.status()
        pid = initial["pid"]
        assert initial["robot"]["control_mode"] == "policy_locomotion"
        assert initial["robot"]["dofs"] == (14 if body == "microduck" else 22)
        if body == "jumper":
            assert initial["scene"]["policy"]["reference_check"]["frames"] == 24
        assert initial["base_observation"]["position_m"][2] > 0.07
        report["initial"] = initial
        for name in ("observer", "head_cam"):
            frame = await asyncio.to_thread(preview.capture_image, name)
            (evidence / f"{name}.jpg").write_bytes(frame["jpeg"])
            pixels = np.asarray(Image.open(io.BytesIO(frame["jpeg"])))
            assert pixels.std() > 10
            if name == "head_cam":
                assert pixels.shape == (480, 640, 3)
                if body == "microduck":
                    camera = frame["camera"]
                    direction = np.asarray(camera["lookat"]) - camera["pos"]
                    yaw = initial["base_observation"]["body_yaw_rad"]
                    assert np.dot(direction, [math.cos(yaw), math.sin(yaw), 0]) > 0.8
                    assert camera["up"][2] > 0.8  # Exported camera must not look back into its own lens.

        # Real shared RobotSession, SDK tool dispatch and camera image injection.
        with ThreadingHTTPServer(("127.0.0.1", 0), Model) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = await asyncio.to_thread(
                    run_agent_query,
                    preview,
                    ModelSpec(api_base=f"http://127.0.0.1:{server.server_port}/v1", model_name="jumper-test"),
                    "观察后依次前进20厘米、左移15厘米、左转0.5弧度、后退15厘米，再看相机。",
                    output_dir=evidence / "agent",
                )
            finally:
                server.shutdown()
                thread.join(2)
        assert result["execution_status"] == "completed" and result["control_released"]
        assert len(result["actions"]) == len(sequence) - 2 and len(result["frames"]) == 2
        for action in result["actions"]:
            assert action["ok"] and action["settled"], action
            if "position_error_m" in action:
                assert action["position_error_m"] <= (0.04 if body == "microduck" else 0.03)
                assert abs(action["yaw_error_rad"]) <= (0.1 if body == "microduck" else 0.08)
        if skills:
            for action in (result["actions"][0], result["actions"][-1]):
                assert action["kind"] == "jump" and action["flight_s"] > 0.1
                assert action["max_height_m"] - action["before"]["position_m"][2] > 0.08
                assert action["support_contacts"] >= 3
            for close, opened, sign in (
                (result["actions"][1], result["actions"][2], 1),
                (result["actions"][3], result["actions"][4], -1),
            ):
                assert sign * (close["joint_angle_rad"] - opened["joint_angle_rad"]) > 0.9
                assert abs(opened["joint_velocity_rad_s"]) < 0.03
                assert not close["grasp_verified"] and not opened["grasp_verified"]
            assert {"jump", "open_gripper", "close_gripper"} <= {t["function"]["name"] for t in requests[0]["tools"]}
        else:
            forward = result["actions"][0]
            assert forward["after"]["position_m"][0] - forward["before"]["position_m"][0] > 0.15
            lateral = result["actions"][1]
            assert lateral["after"]["position_m"][1] - lateral["before"]["position_m"][1] > 0.1
            turn = result["actions"][2]
            assert turn["after"]["body_yaw_rad"] - turn["before"]["body_yaw_rad"] > 0.4
        assert any("data:image/jpeg;base64," in json.dumps(request) for request in requests)
        assert all("set_gripper" not in {t["function"]["name"] for t in req["tools"]} for req in requests)
        assert "target_world" not in json.dumps(requests)
        assert "joint_angle_rad" not in json.dumps(requests)
        assert "max_height_m" not in json.dumps(requests)
        if body == "microduck":
            assert {"jump", "open_gripper", "close_gripper"}.isdisjoint(
                t["function"]["name"] for t in requests[0]["tools"]
            )
        report["agent"] = result

        # Receive the robot camera through the same H264/WebRTC endpoint as the UI.
        configured = preview.configure(
            width=960,
            height=540,
            fps=30,
            scene_id=entry["scene_id"],
            backend="mujoco",
            robot_mode=body,
            camera_name="head_cam",
        )
        await wait(lambda: preview.status().get("config_revision") == configured["requested_config_revision"])
        incoming = asyncio.Queue()
        peer.on("track", incoming.put_nowait)
        peer.addTransceiver("video", direction="recvonly")
        await peer.setLocalDescription(await peer.createOffer())
        answer = await service.offer(peer.localDescription.sdp)
        await peer.setRemoteDescription(RTCSessionDescription(answer["sdp"], answer["type"]))
        track = await asyncio.wait_for(incoming.get(), 15)
        video = await asyncio.wait_for(track.recv(), 15)
        assert (video.width, video.height) == (640, 480)
        await service.disconnect(answer["session_id"])
        report["head_camera_webrtc"] = True

        # Cancelling a moving Agent session must command zero and confirm a stop.
        cancelled = threading.Event()
        with make_session(preview, cancelled=cancelled) as session:
            task = asyncio.create_task(asyncio.to_thread(session.api.navigate_relative, 0.8))
            await wait(lambda: preview.status().get("base_motion", {}).get("status") == "running")
            await asyncio.sleep(0.5)
            cancelled.set()
            stopped = await task
            assert not stopped["ok"] and stopped["settled"]
            report["cancelled_motion"] = stopped
        assert preview.status()["pid"] == pid
        if skills:
            cancelled = threading.Event()
            with make_session(preview, cancelled=cancelled) as session:
                task = asyncio.create_task(asyncio.to_thread(session.api.jump))
                await wait(lambda: preview.status().get("robot", {}).get("policy_mode") == "jump")
                cancelled.set()
                stopped = await task
                assert not stopped["ok"] and stopped["settled"] and stopped["airborne"]
                report["cancelled_jump"] = stopped
        # Resizing does not replace the world or teleport the robot.
        before = preview.status()["base_observation"]["position_m"]
        instance = preview.status()["scene"]["instance_id"]
        configured = preview.configure(
            width=1280, height=720, fps=30, scene_id=entry["scene_id"], backend="mujoco", robot_mode=body
        )
        await wait(lambda: preview.status().get("config_revision") == configured["requested_config_revision"])
        assert preview.status()["scene"]["instance_id"] == instance
        after = preview.status()["base_observation"]["position_m"]
        assert math.dist(before, after) < 0.02
        capture = await asyncio.to_thread(preview.capture_image)
        assert (capture["width"], capture["height"]) == (1280, 720)
        await asyncio.sleep(2)
        report["final"] = preview.status()
        assert report["final"]["physics_clock"]["real_time_factor"] > 0.9
        # New upstream reset must also reset native-policy worlds and revoke old bindings.
        await asyncio.to_thread(preview.reset_scene)
        await wait(
            lambda: preview.status()["state"] == "ready" and preview.status()["scene"]["instance_id"] != instance
        )
        reset = preview.status()
        assert reset["pid"] == pid
        assert reset["reset_result"]["stop_confirmation"] == "measured"
        assert math.dist(reset["base_observation"]["position_m"][:2], (0, 0)) < 0.04
        report["reset"] = reset
        report["resident"] = preview_type is ResidentPreview
        report["status"] = "passed"
    finally:
        await service.stop()
        await peer.close()
        (evidence / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        log = preview.state_dir / "worker.log"
        if log.exists():
            (evidence / "worker.log").write_bytes(log.read_bytes())
        runtime_directory.cleanup()
