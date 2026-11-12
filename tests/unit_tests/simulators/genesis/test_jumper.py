"""Jumper asset registration, bundle boundaries and shared motion capability."""

import hashlib
from pathlib import Path

import pytest

from examples.jumper_sim import register
from jiuwensymbiosis.simulators.assets import AssetError, AssetLibrary, load_config
from jiuwensymbiosis.simulators.assets.config import RootSpec
from jiuwensymbiosis.simulators.assets.scanners import scan_root
from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings
from jiuwensymbiosis.simulators.genesis.jumper_policy import JumperPolicy


def checkout(root):
    for name in (
        "assets/jumper/jumper.xml",
        "assets/jumper/motor/motor_config.yaml",
        "out/bundle_example/jumper/bundle.json",
    ):
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("fixture")
    return root


def test_registration_is_idempotent_and_resolves_original_files(tmp_path):
    source = checkout(tmp_path / "upstream")
    home = tmp_path / "home"
    result = register(source, home)
    assert register(source, home) == result
    config = load_config(Path(result["config"]))
    assert len(config.spec.roots) == 1
    resolved = AssetLibrary(config.index).resolve(result["scene_id"])
    assert resolved.asset.preview_mode == "jumper"
    assert resolved.preview_path == source / "assets/jumper/jumper.xml"
    assert resolved.paths["policy_bundle"] == source / "out/bundle_example/jumper/bundle.json"
    assert PreviewSettings(scene_id=result["scene_id"], robot_mode="jumper").robot_id == ""
    with pytest.raises(ValueError, match="elsewhere"):
        register(checkout(tmp_path / "other"), home)
    assert load_config(Path(result["config"])).roots["jumper"] == source


def test_scan_refuses_missing_and_escaping_assets(tmp_path):
    spec = RootSpec(id="jumper", scanner="jumper", path=str(tmp_path))
    with pytest.raises(AssetError, match="缺失"):
        scan_root(spec, tmp_path)
    source = checkout(tmp_path)
    model = source / "assets/jumper/jumper.xml"
    model.unlink()
    model.symlink_to(Path(__file__).resolve())
    with pytest.raises(AssetError, match="离开根目录"):
        scan_root(spec, source)


def test_bundle_file_digest_and_path_boundary(tmp_path):
    policy = object.__new__(JumperPolicy)
    policy.root = tmp_path
    data = b"original-model"
    (tmp_path / "model.onnx").write_bytes(data)
    policy.manifest = {"files": {"model.onnx": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}
    assert policy.file("model.onnx") == tmp_path / "model.onnx"
    (tmp_path / "model.onnx").write_bytes(b"tampered-model")
    with pytest.raises(ValueError, match="digest"):
        policy.file("model.onnx")
    with pytest.raises(ValueError, match="outside"):
        policy.file("../outside")


def test_shared_agent_accepts_policy_locomotion_without_video(tmp_path, monkeypatch):
    import time

    from jiuwensymbiosis.simulators.genesis.agent import make_session
    from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview
    from jiuwensymbiosis.tools.builder import list_tool_meta

    preview = GenesisPreview(state_dir=tmp_path)
    state = {
        "state": "ready",
        "measured_at": time.monotonic(),
        "scene": {"instance_id": "jumper"},
        "robot": {"control_mode": "policy_locomotion"},
    }
    monkeypatch.setattr(preview, "status", lambda: state)
    with make_session(preview) as session:
        assert {tool["name"] for tool in list_tool_meta(session.api, env=session.env)} >= {
            "navigate_relative",
            "rotate_base",
        }
        assert session.env.supports_lateral
        assert session.env.capabilities == {"motion.base"}
        assert preview._motion_owner
    assert preview._motion_owner is None


@pytest.mark.parametrize("enabled", [False, True])
def test_policy_actions_are_capability_gated_and_do_not_claim_payload(tmp_path, monkeypatch, enabled):
    import time

    from jiuwensymbiosis.api.actions import ACTIONS
    from jiuwensymbiosis.simulators.genesis.agent import make_session
    from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview
    from jiuwensymbiosis.tools.builder import list_tool_meta

    preview = GenesisPreview(state_dir=tmp_path)
    state = {
        "state": "ready",
        "measured_at": time.monotonic(),
        "scene": {"instance_id": "jumper"},
        "robot": {"control_mode": "policy_locomotion", "policy_actions": ["jump", "set_gripper"]},
    }
    monkeypatch.setattr(preview, "status", lambda: state)
    with make_session(preview, actions_enabled=enabled) as session:
        names = {tool["name"] for tool in list_tool_meta(session.api, env=session.env)}
        assert ({"jump", "open_gripper", "close_gripper"} <= names) is enabled
        assert "set_gripper" not in names
        assert ("motion.jump" in session.env.capabilities) is enabled
    assert "set_gripper" not in ACTIONS
    assert ACTIONS["jump"].invalidates_locations


@pytest.mark.parametrize(("action", "closed"), [("open_gripper", False), ("close_gripper", True)])
@pytest.mark.parametrize("side", [None, "left", "right"])
@pytest.mark.parametrize("ok", [False, True])
async def test_shared_gripper_dispatch_preserves_side_and_never_invents_payload(action, closed, side, ok):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from jiuwensymbiosis.simulators.genesis.agent import JumperApi, JumperEnv
    from jiuwensymbiosis.tools.robot_control_tool import RobotControlTool, read_grasp_confirmation

    driver = SimpleNamespace(
        actions_enabled=True,
        observation_mode="visual",
        policy_action=Mock(return_value={"ok": ok, "settled": ok, "grasp_verified": False}),
    )
    env = JumperEnv(driver)
    api = JumperApi(env)
    tool = RobotControlTool(api, env=env)
    assert "set_gripper" not in tool.available_actions
    meta = getattr(api, action).__tool_meta__
    assert meta.input_params["properties"]["side"]["enum"] == ["left", "right"]
    assert meta.input_params["properties"]["side"]["default"] == "left"
    # Width/force are accepted hints; no force/width controller is invented.
    params = {"width_mm": 42.0} if not closed else {"force_n": 5.0}
    if side is not None:
        params["side"] = side
    result = await tool.invoke({"action": action, "params": params})
    assert result.success
    returned = result.data["result"]
    assert returned["ok"] is ok
    assert returned["state"] == (("closed" if closed else "open") if ok else "unconfirmed")
    driver.policy_action.assert_called_once_with("set_gripper", side=side or "left", closed=closed)
    assert "payload.held" not in api.memory.self_state
    if closed:
        assert read_grasp_confirmation(api, returned) is None
        assert not api.trust_command_grasp
    elif ok:
        assert "payload.clear" in api.memory.self_state


@pytest.mark.parametrize("action", ["open_gripper", "close_gripper"])
@pytest.mark.parametrize("side", ["default", "middle", None])
def test_jumper_gripper_rejects_invalid_side_before_dispatch(action, side):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from jiuwensymbiosis.simulators.genesis.agent import JumperApi, JumperEnv

    driver = SimpleNamespace(actions_enabled=True, observation_mode="visual", policy_action=Mock())
    api = JumperApi(JumperEnv(driver))
    with pytest.raises(ValueError, match="left/right"):
        getattr(api, action)(side=side)
    driver.policy_action.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "state"),
    [
        ("ground_truth", {"scene": {"engine": "MuJoCo"}}),
        ("perception", {"scene": {"engine": "MuJoCo"}}),
        ("perception", {"renderer": "Genesis Rasterizer"}),
    ],
)
def test_unimplemented_observation_backends_fail_before_exposing_tools(tmp_path, monkeypatch, mode, state):
    from jiuwensymbiosis.simulators.genesis.agent import make_session
    from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview

    preview = GenesisPreview(state_dir=tmp_path)
    monkeypatch.setattr(preview, "status", lambda: state)
    with pytest.raises(ValueError, match="尚未接入"):
        make_session(preview, observation_mode=mode)
    assert preview._motion_owner is None


@pytest.fixture
def policy_motion():
    from collections import deque
    from types import SimpleNamespace as NS
    from unittest.mock import Mock

    import numpy as np

    from jiuwensymbiosis.simulators.genesis.jumper_motion import JumperMotion

    fsm = NS(mode="locomotion", running_policy=True)
    events = deque()
    policy = NS(
        fsm=fsm,
        wire=["LF_J4_joint", "RF_J4_joint"],
        dt=0.001,
        key_events=events,
        operators=False,
        use_operators=Mock(),
        tap_mode=Mock(side_effect=lambda mode: events.extend([(mode, True), (mode, False)])),
    )
    policy.use_operators.side_effect = lambda enabled: setattr(policy, "operators", enabled)
    contact = NS(geom1=0, geom2=1)
    robot = NS(
        policy=policy,
        stop=Mock(),
        model=NS(geom_bodyid=np.asarray([0, 1])),
        data=NS(qpos=np.asarray([0.0, 0.0, 0.105, 0.0, 0.0]), qvel=np.zeros(2), contact=[contact] * 3),
        qids=np.asarray([3, 4]),
        vids=np.asarray([0, 1]),
        targets=np.zeros(2),
    )
    robot.observe_base = lambda: {
        "position_m": robot.data.qpos[:3].tolist(),
        "linear_velocity_m_s": [0.0, 0.0, 0.0],
        "angular_velocity_rad_s": [0.0, 0.0, 0.0],
    }
    return JumperMotion(robot, NS(active=False)), robot


def jump_request(**changes):
    return {
        "id": "one",
        "scene_instance_id": "scene",
        "config_revision": 0,
        "lease_until": 10.0,
        "cancelled": False,
        "kind": "jump",
        **changes,
    }


def accept(motion, request):
    motion.accept_request(request, scene_id="scene", revision=0, now=1.0, step=0)


@pytest.mark.parametrize("changes", [{"cancelled": True}, {"lease_until": 0.0}, {"config_revision": 1}])
def test_expired_or_foreign_policy_request_never_triggers_controller(policy_motion, changes):
    motion, robot = policy_motion
    accept(motion, jump_request(**changes))
    motion.tick(0.01, 1)
    robot.policy.tap_mode.assert_not_called()
    assert not motion.active


def test_cancellation_before_native_tick_clears_queued_jump(policy_motion):
    motion, robot = policy_motion
    accept(motion, jump_request())
    motion.tick(0.01, 1)
    assert robot.policy.key_events
    accept(motion, jump_request(cancelled=True))
    assert not robot.policy.key_events
    for step in range(120):
        motion.tick(0.01, step)
    assert motion.result["settled"] and not motion.result["ok"]


def test_cancelled_unstarted_closure_does_not_leave_false_closed_state(policy_motion):
    from unittest.mock import Mock

    motion, robot = policy_motion
    robot.policy.claw = Mock(side_effect=lambda side, closed: robot.policy.key_events.append(("space", closed)))
    request = jump_request(kind="set_gripper", side="left", closed=True)
    accept(motion, request)
    motion.tick(0.01, 1)
    assert motion.claw_closed and robot.policy.key_events
    accept(motion, {**request, "cancelled": True})
    assert not robot.policy.key_events
    assert motion.claw_side is None and not motion.claw_closed
    for step in range(300):
        motion.tick(0.01, step)
    assert motion.result["settled"] and not robot.policy.operators


@pytest.mark.parametrize("flight", [False, True])
def test_jump_requires_measured_flight_and_landing_and_deduplicates(policy_motion, flight):
    motion, robot = policy_motion
    request = jump_request()
    accept(motion, request)
    motion.tick(0.01, 1)
    robot.policy.key_events.clear()
    robot.policy.fsm.mode = "jump"
    if flight:
        robot.data.qpos[2] = 0.22
        robot.data.contact = []
    for _ in range(100):
        motion.sample_physics()
    assert motion.active
    robot.data.qpos[2] = 0.105
    from types import SimpleNamespace

    robot.data.contact = [SimpleNamespace(geom1=0, geom2=1)] * 3
    robot.policy.fsm.mode = "locomotion"
    for step in range(120):
        motion.tick(0.01, step)
    assert motion.result["ok"] is flight
    assert motion.result["settled"]
    accept(motion, request)
    assert not motion.active
    robot.policy.tap_mode.assert_called_once_with("jump")


def test_closed_claw_blocks_jump_without_releasing_it(policy_motion):
    motion, robot = policy_motion
    motion.claw_side, motion.claw_closed = "left", True
    accept(motion, jump_request())
    for step in range(120):
        motion.tick(0.01, step)
    assert motion.result["reason"] == "gripper_requires_open" and not motion.result["ok"]
    assert motion.claw_closed
    robot.policy.tap_mode.assert_not_called()


def test_policy_feedback_keeps_physical_measurements_private():
    from jiuwensymbiosis.simulators.genesis.visual_policy import policy_feedback

    result = policy_feedback(
        {
            "kind": "jump",
            "ok": True,
            "settled": True,
            "flight_s": 0.2,
            "max_height_m": 123.456,
            "after": {"position_m": [1, 2, 3]},
        }
    )
    assert result["airborne"]
    assert "123.456" not in str(result) and "position_m" not in str(result)
