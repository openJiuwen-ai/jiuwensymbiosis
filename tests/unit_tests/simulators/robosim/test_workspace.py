"""Exercise workspace/tool boundaries over loopback HTTP with a stub RoboSim service."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from jiuwensymbiosis.simulators.robosim import RobosimConfig, RobosimWorkspace, bind_robosim_session
from jiuwensymbiosis.simulators.robosim.client import RobosimClient
from jiuwensymbiosis.simulators.robosim.lowlevel import RobosimLowLevel
from jiuwensymbiosis.simulators.robosim.operations import RobosimError
from jiuwensymbiosis.simulators.robosim.tools import build_workspace_tools
from scripts.robosim_identity import RuntimeIdentity
from tests.unit_tests.simulators.robosim.test_adapter import observation


@pytest.fixture
def backend():
    state = SimpleNamespace(identity=RuntimeIdentity(), robot=None, robot_type="moz1", calls=[], overrides={})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            state.calls.append((self.command, self.path, json.loads(raw) if raw else None))
            if self.path in state.overrides:
                status, payload = state.overrides[self.path]
            else:
                status = 200
                if self.path == "/symbiosis/workspace":
                    payload = state.identity.snapshot(state.robot)
                elif self.path == "/sim/health":
                    payload = {"ready": True}
                elif self.path == "/robot" and self.command == "POST":
                    state.robot = object()
                    payload = {"success": True, "data": {"robot_type": state.robot_type}}
                elif self.path == "/robot/state":
                    payload = {"robot_type": state.robot_type}
                elif self.path.startswith("/robot/gripper/"):
                    payload = {
                        "success": True,
                        "data": {
                            "robot_type": state.robot_type,
                            "gripper_open": self.path.endswith("open"),
                            "holding": None,
                            "holding_arm_id": None,
                        },
                    }
                elif self.path == "/skills/reset_to_home":
                    payload = {
                        "success": True,
                        "data": {
                            "arms_reset": ["left"],
                            "partial_success": False,
                            "gripper_released": False,
                            "failed_arms": {},
                        },
                    }
                elif self.path.startswith("/symbiosis/observation"):
                    payload = observation()
                else:
                    payload = {"success": True, "data": {"items": []}}
            self.send_response(status)
            if 300 <= status < 400:
                self.send_header("Location", "/redirect-target")
            self.end_headers()
            self.wfile.write(payload if isinstance(payload, bytes) else json.dumps(payload).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    state.cfg = RobosimConfig(service_url=f"http://127.0.0.1:{server.server_port}")
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_empty_workspace_queries_do_not_load_or_connect_robot(backend):
    workspace = RobosimWorkspace(backend.cfg)
    assert workspace.refresh()["robot_state"] is None
    workspace.list_robots()
    workspace.robot_spec()
    assert workspace.scene_state()["source"] == "robosim_registry"
    with pytest.raises(RobosimError) as exc:
        workspace.observe()
    assert exc.value.code == "robot_not_loaded"
    assert all(method == "GET" and path != "/robot/state" for method, path, _ in backend.calls)


async def test_named_tool_loads_once_then_measured_observation_is_available(backend):
    workspace = RobosimWorkspace(backend.cfg)
    tools = {tool.card.name: tool for tool in build_workspace_tools(workspace)}
    assert set(tools) == {
        "robosim_status",
        "robosim_list_robots",
        "robosim_robot_spec",
        "robosim_scene_state",
        "robosim_observe",
        "robosim_load_robot",
    }
    result = await tools["robosim_load_robot"].invoke({})
    assert result["robot_state"]["robot_type"] == "moz1"
    snapshot = workspace.load_robot()
    assert snapshot["robot_state"]["robot_type"] == "moz1"
    assert snapshot["busy"] is False
    assert snapshot["command"]["status"] == "completed"
    assert len([call for call in backend.calls if call[0] == "POST"]) == 1
    observed = await tools["robosim_observe"].invoke({})
    assert observed["simulation_time_s"] == 1.0
    assert workspace.observe()["simulation_time_s"] == 1.0
    assert "robosim_load_robot" not in {tool.card.name for tool in build_workspace_tools(workspace, allow_load=False)}


def test_load_preserves_other_body(backend):
    backend.robot = object()
    backend.robot_type = "another_robot"
    with pytest.raises(RobosimError) as exc:
        RobosimWorkspace(backend.cfg).load_robot()
    assert exc.value.code == "robosim_robot_mismatch"
    assert all(call[0] == "GET" for call in backend.calls)


def test_replacement_same_model_and_service_restart_change_revision(backend):
    workspace = RobosimWorkspace(backend.cfg)
    first = workspace.load_robot()
    assert workspace.refresh()["revision"] == first["revision"]
    backend.robot = object()
    second = workspace.refresh()
    assert second["revision"] > first["revision"]
    assert second["robot_instance_id"] != first["robot_instance_id"]
    backend.identity = RuntimeIdentity()
    third = workspace.refresh()
    assert third["revision"] > second["revision"]
    assert third["service_instance_id"] != second["service_instance_id"]


def test_unknown_write_survives_client_reset_but_restart_reconciles(backend):
    workspace = RobosimWorkspace(backend.cfg)
    workspace.refresh()
    backend.overrides["/robot"] = (200, b"truncated json")
    with pytest.raises(RobosimError) as exc:
        workspace.load_robot()
    assert exc.value.outcome_unknown
    workspace.close()
    replacement = RobosimWorkspace(backend.cfg)
    assert replacement.refresh()["write_blocked"] is True
    before = len(backend.calls)
    with pytest.raises(RobosimError, match="Previous command"):
        replacement.client.request("POST", "/robot/gripper/open", {})
    assert len(backend.calls) == before
    backend.identity = RuntimeIdentity()
    del backend.overrides["/robot"]
    assert replacement.load_robot()["write_blocked"] is False


@pytest.mark.parametrize(
    "status,payload,unknown",
    [
        (200, {"success": False, "error": "IK failed", "failure_reason": "ik_failed"}, False),
        (422, {"detail": {"failure_reason": "invalid_target"}}, False),
        (500, {"detail": {"failure_reason": "execution_error"}}, True),
        (302, {}, True),
        (200, b"broken", True),
        (200, ["not an object"], True),
    ],
)
def test_transport_errors_never_retry_or_follow_redirect(backend, status, payload, unknown):
    backend.overrides["/test"] = (status, payload)
    client = RobosimClient(backend.cfg)
    with pytest.raises(RobosimError) as exc:
        client.request("POST", "/test", {})
    assert exc.value.outcome_unknown is unknown
    assert client.commands.snapshot()["write_blocked"] is unknown
    assert len(backend.calls) == 1
    if isinstance(payload, dict) and "failure_reason" in payload:
        assert exc.value.code == payload["failure_reason"]
    if isinstance(payload, dict) and "detail" in payload:
        assert exc.value.code == payload["detail"]["failure_reason"]


def test_read_error_and_invalid_input_do_not_poison_write_lease(backend):
    client = RobosimClient(backend.cfg)
    backend.overrides["/query"] = (200, b"bad json")
    with pytest.raises(RobosimError):
        client.request("GET", "/query")
    with pytest.raises(ValueError):
        client.request("POST", "/test", {"x": float("nan")})
    with pytest.raises(ValueError):
        client.request("POST", "http://outside/test", {})
    assert len(backend.calls) == 1
    assert not client.commands.snapshot()["write_blocked"]
    client.request("POST", "/test", {})


def test_missing_identity_bridge_explains_launcher_upgrade(backend):
    backend.overrides["/symbiosis/workspace"] = (404, {"detail": "Not found"})
    with pytest.raises(RobosimError) as exc:
        RobosimWorkspace(backend.cfg).refresh()
    assert exc.value.code == "robosim_identity_required"


def test_bound_driver_rejects_same_model_replacement_and_borrows_transport(backend):
    workspace = RobosimWorkspace(backend.cfg)
    snapshot = workspace.load_robot()
    driver = RobosimLowLevel(
        backend.cfg,
        client=workspace.client,
        expected_identity=(snapshot["service_instance_id"], snapshot["robot_instance_id"]),
    )
    backend.robot = object()
    before = len([call for call in backend.calls if call[0] == "POST"])
    with pytest.raises(RobosimError) as exc:
        driver.set_gripper(False)
    assert exc.value.code == "robosim_robot_changed"
    assert len([call for call in backend.calls if call[0] == "POST"]) == before
    driver.close()
    assert workspace.refresh()["robot_state"]["robot_type"] == "moz1"


def test_identity_tracks_unload_and_object_replacement():
    identity = RuntimeIdentity()
    assert identity.snapshot(None)["robot_instance_id"] is None
    robot = object()
    first = identity.snapshot(robot)
    assert identity.snapshot(robot) == first
    assert identity.snapshot(None)["robot_instance_id"] is None
    assert identity.snapshot(robot)["robot_instance_id"] != first["robot_instance_id"]
    assert identity.snapshot(robot)["service_instance_id"] == first["service_instance_id"]


def test_port_zero_is_not_a_service_origin():
    with pytest.raises(ValueError):
        RobosimConfig(service_url="http://localhost:0")


def test_bound_session_disconnect_keeps_workspace_available(backend):
    workspace = RobosimWorkspace(backend.cfg)
    session, tools = bind_robosim_session(workspace, workspace.load_robot())
    session.connect()
    driver = session.env.low_level
    assert driver.client is workspace.client
    assert session.api.get_joint_positions() == {"wrist": 0.0}
    assert "robosim_load_robot" not in {tool.card.name for tool in tools}
    session.disconnect()
    assert workspace.refresh()["robot_state"]["robot_type"] == "moz1"
    writes = [call for call in backend.calls if call[0] == "POST"]
    with pytest.raises(RobosimError) as exc:
        driver.set_gripper(False)
    assert exc.value.code == "robosim_disconnected"
    assert [call for call in backend.calls if call[0] == "POST"] == writes
    workspace.close()


def test_replacement_after_binding_cannot_connect_or_reload_old_session(backend):
    workspace = RobosimWorkspace(backend.cfg)
    session, _ = bind_robosim_session(workspace, workspace.load_robot())
    backend.robot = object()
    with pytest.raises(RobosimError) as exc:
        session.connect()
    assert exc.value.code == "robosim_robot_changed"
    assert len([call for call in backend.calls if call[0] == "POST"]) == 1
    assert workspace.refresh()["robot_instance_id"]
    workspace.close()


def test_standalone_driver_also_rejects_replaced_robot(backend):
    backend.robot = object()
    driver = RobosimLowLevel(backend.cfg)
    driver.connect()
    backend.robot = object()
    with pytest.raises(RobosimError) as exc:
        driver.set_gripper(False)
    assert exc.value.code == "robosim_robot_changed"
    assert all(method == "GET" for method, *_ in backend.calls)
    driver.close()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_redirects_are_not_followed_for_reads_or_writes(backend, status, method):
    backend.overrides["/redirect"] = (status, {})
    client = RobosimClient(backend.cfg)
    with pytest.raises(RobosimError) as exc:
        client.request(method, "/redirect")
    assert exc.value.code == "robosim_http_error"
    assert exc.value.outcome_unknown is (method != "GET")
    assert client.commands.snapshot()["write_blocked"] is (method != "GET")
    assert backend.calls == [(method, "/redirect", None)]


@pytest.mark.parametrize("payload", [b"broken json", b"\xff"])
def test_invalid_response_encoding_blocks_writes_without_retry(backend, payload):
    backend.overrides["/invalid"] = (200, payload)
    client = RobosimClient(backend.cfg)
    with pytest.raises(RobosimError) as exc:
        client.request("POST", "/invalid")
    assert exc.value.code == "robosim_protocol_error"
    assert exc.value.outcome_unknown is True
    with pytest.raises(RobosimError, match="Previous command"):
        client.request("POST", "/test")
    assert backend.calls == [("POST", "/invalid", None)]


@pytest.mark.parametrize("action", ["joint", "tcp"])
def test_post_write_readback_failure_blocks_recovery_and_later_commands(backend, monkeypatch, action):
    from urllib.error import URLError

    workspace = RobosimWorkspace(backend.cfg)
    session, _ = bind_robosim_session(workspace, workspace.load_robot())
    session.connect()
    opener = workspace.client._opener.open
    sent = False

    def fail_readback(request, **kwargs):
        nonlocal sent
        if sent and "/symbiosis/observation" in request.full_url:
            raise URLError("readback unavailable")
        response = opener(request, **kwargs)
        if request.get_method() == "POST":
            sent = True
        return response

    monkeypatch.setattr(workspace.client._opener, "open", fail_readback)
    with pytest.raises(RobosimError) as exc:
        if action == "joint":
            session.api.move_joint({"wrist": 0.1})
        else:
            session.api.goto_pose({"x": 300, "y": 0, "z": 400, "rx": 0, "ry": 0, "rz": 0})
    assert exc.value.outcome_unknown and exc.value.skip_recovery
    assert workspace.client.commands.snapshot()["write_blocked"]
    writes = [c for c in backend.calls if c[0] == "POST"]
    with pytest.raises(RobosimError, match="Previous command"):
        session.api.open_gripper()
    assert [c for c in backend.calls if c[0] == "POST"] == writes
    session.disconnect()
    workspace.close()


@pytest.mark.parametrize("first_connect_fails", [False, True])
def test_bound_session_keeps_identity_across_reconnection(backend, first_connect_fails):
    workspace = RobosimWorkspace(backend.cfg)
    session, _ = bind_robosim_session(workspace, workspace.load_robot())
    if first_connect_fails:
        backend.robot = object()
        with pytest.raises(RobosimError, match="stale"):
            session.connect()
    else:
        session.connect()
        session.disconnect()
        backend.robot = object()
    writes = [c for c in backend.calls if c[0] == "POST"]
    for _ in range(2):
        with pytest.raises(RobosimError, match="stale"):
            session.connect()
    assert [c for c in backend.calls if c[0] == "POST"] == writes
    assert workspace.refresh()["robot_state"]["robot_type"] == "moz1"
    workspace.close()


def test_bound_session_reconnect_same_identity_keeps_borrowed_client(backend):
    workspace = RobosimWorkspace(backend.cfg)
    session, _ = bind_robosim_session(workspace, workspace.load_robot())
    session.connect()
    session.disconnect()
    session.connect()
    assert session.env.low_level.client is workspace.client
    session.disconnect()
    workspace.refresh()
    workspace.close()
    with pytest.raises(RobosimError, match="closed"):
        session.connect()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"success": True},
        {"success": True, "data": None},
        {"success": True, "data": {}},
        {"success": True, "data": {"robot_type": "moz1"}},
    ],
)
async def test_malformed_gripper_result_never_commits_release(backend, payload):
    from jiuwensymbiosis.tools.robot_control_tool import RobotControlTool

    workspace = RobosimWorkspace(backend.cfg)
    session, _ = bind_robosim_session(workspace, workspace.load_robot())
    session.connect()
    backend.overrides["/robot/gripper/open"] = (200, payload)
    output = await RobotControlTool(session.api, env=session.env).invoke({"action": "open_gripper"})
    assert not output.success
    assert output.data["outcome_unknown"] and output.data["skip_recovery"]
    assert workspace.client.commands.snapshot()["write_blocked"]
    session.disconnect()
    workspace.close()


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"arms_reset": ["right"]},
        {"arms_reset": ["left"], "partial_success": True, "gripper_released": False, "failed_arms": {}},
    ],
)
def test_home_requires_selected_arm_completion_evidence(backend, data):
    driver = RobosimLowLevel(backend.cfg)
    backend.overrides["/skills/reset_to_home"] = (200, {"success": True, "data": data})
    with pytest.raises(RobosimError) as exc:
        driver.home()
    assert exc.value.outcome_unknown
    assert driver.client.commands.snapshot()["write_blocked"]
    driver.close()


def test_real_readback_error_stops_runner_without_release_or_home(backend, monkeypatch):
    from urllib.error import URLError

    from jiuwensymbiosis.agent.fast.runner import run_sequence
    from jiuwensymbiosis.agent.fast.sequence import parse_sequence
    from jiuwensymbiosis.tools.robot_control_tool import _build_action_index

    workspace = RobosimWorkspace(backend.cfg)
    session, _ = bind_robosim_session(workspace, workspace.load_robot())
    session.connect()
    backend.calls.clear()
    opener = workspace.client._opener.open

    def fail_readback(request, **kwargs):
        if any(method == "POST" for method, *_ in backend.calls) and "/symbiosis/observation" in request.full_url:
            raise URLError("readback unavailable")
        return opener(request, **kwargs)

    monkeypatch.setattr(workspace.client._opener, "open", fail_readback)
    index = _build_action_index(session.api, env=session.env, planner_only=True)
    steps = parse_sequence(
        [{"op": "move_joint", "params": {"targets": {"wrist": 0.1}}}, {"op": "open_gripper"}], allowed_ops=index
    )
    result = run_sequence(session, steps, action_index=index)
    assert result["outcome_unknown"] and not result["ok"]
    assert result["steps_done"] == 1
    assert [path for method, path, _ in backend.calls if method == "POST"] == ["/robot/joints"]
    assert workspace.client.commands.snapshot()["write_blocked"]
    session.disconnect()
    workspace.close()


def test_motion_result_uses_readback_inside_command_scope(backend):
    workspace = RobosimWorkspace(backend.cfg)
    session, _ = bind_robosim_session(workspace, workspace.load_robot())
    session.connect()
    backend.calls.clear()
    result = session.api.move_joint({"wrist": 0.0})
    assert result == {"ok": True, "joint_positions": {"wrist": 0.0}}
    reads = [p for m, p, _ in backend.calls if p.startswith("/symbiosis/observation")]
    assert len(reads) == 2  # Preflight and completion, no unguarded API read after completion.
    session.disconnect()
    workspace.close()


def test_valid_native_gripper_and_home_evidence_is_accepted(backend):
    workspace = RobosimWorkspace(backend.cfg)
    session, _ = bind_robosim_session(workspace, workspace.load_robot())
    session.connect()
    session.env.home()
    assert workspace.client.commands.snapshot()["command"]["status"] == "completed"
    assert session.api.open_gripper() == {"ok": True}
    backend.overrides["/robot/gripper/close"] = (
        200,
        {
            "success": True,
            "data": {"robot_type": "moz1", "gripper_open": False, "holding": "cube", "holding_arm_id": "left"},
        },
    )
    assert session.api.close_gripper() == {"ok": True, "holding": "cube"}
    assert not workspace.client.commands.snapshot()["write_blocked"]
    session.disconnect()
    workspace.close()
