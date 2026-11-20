"""Model service, camera calibration and physical-success boundaries for a configured pick."""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.common.observation_policy import allowed_tools
from jiuwensymbiosis.simulators.genesis import grasp_pose
from jiuwensymbiosis.simulators.genesis.grasp_pick import pick_target
from jiuwensymbiosis.simulators.genesis.pick_lift import PickLift

CONFIG = Path(__file__).resolve().parents[4] / "configs/simulators/grasp"
REACHED = {"ok": True, "settled": True, "goal_reached": True, "stop_confirmation": "measured"}


def test_rgbd_reprojects_only_valid_masked_depth_in_metres():
    depth = np.array([[np.nan, 2.0], [1.0, 0.0]])
    mask = np.array([[True, True], [False, True]])
    frame = np.eye(4)
    frame[:3, 3] = [1, 2, 3]
    points = grasp_pose.rgbd_points(depth, mask, np.diag([2.0, 3.0, 1.0]), frame)
    np.testing.assert_allclose(points, [[2, 2, 5]])
    with pytest.raises(ValueError, match="mask"):
        grasp_pose.rgbd_points(depth, mask.astype(int), np.eye(3), frame)


@pytest.mark.parametrize("source", ["rgbd", "ground_truth"])
def test_pick_uses_only_configured_sensor_and_inferred_rotation(source, monkeypatch):
    b = grasp_pose.GraspPoseBackend.load(CONFIG / f"moz1-right-{source}.json")
    m = np.eye(4)
    m[:3, :3] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    m[:3, 3] = [0.2, 0.3, 0.4]

    def infer(endpoint, request):
        assert request["point_cloud"].shape == (120, 3)
        return {"grasps": m[None], "confidences": [0.9]}

    monkeypatch.setattr(grasp_pose, "infer", infer)
    state = {
        "state": "ready",
        "scene": {"instance_id": "scene"},
        "config_revision": 1,
        "effectors": {"right": {"object_id": 42}},
    }
    commands = []

    def query(op, params, **kw):
        assert op == "grasp_ik"
        assert not params.get("direct_ik", False)
        np.testing.assert_allclose(params["poses"][1], m @ b.tool_offset)
        assert params["poses"][0][2][3] == pytest.approx((m @ b.tool_offset)[2, 3] + 0.08)
        assert params["poses"][2][2][3] == pytest.approx((m @ b.tool_offset)[2, 3] + 0.08)
        return {"scene_instance_id": "scene", "targets": [{"q": 1.0}, {"q": 2.0}, {"q": 3.0}]}

    def move(**kw):
        commands.append(kw)
        return REACHED | {"holding_payload": True, "grasp_confirmed": True}

    def truth(op, params):
        assert source == "ground_truth" and op == "grasp_cloud"
        return {"body_idx": 42, "captured_at": time.monotonic(), "points_m": np.zeros((120, 3))}

    def capture():
        assert source == "rgbd"
        return SimpleNamespace(
            rgb=np.zeros((10, 12, 3)),
            depth_m=np.ones((10, 12)),
            intrinsics=np.eye(3),
            frame_id="sensor",
            captured_monotonic_s=time.monotonic(),
            metadata={"T_planning_base_camera_m": np.eye(4)},
        )

    driver = SimpleNamespace(
        preview=SimpleNamespace(status=lambda: state, query_scene=query),
        cancelled=None,
        query_truth=truth,
        move_arm=move,
        require_motion=lambda: None,
    )
    env = SimpleNamespace(
        driver=driver,
        holding_payload=False,
        set_gripper=lambda opened, name: REACHED if opened else pytest.fail("separate close"),
        grab_calibrated_frame=capture,
        validate_frame=lambda frame: None,
    )
    result = pick_target(env, b, "object", lambda *a, **kw: [{"mask": np.ones((10, 12), bool)}])
    assert result["ok"] and len(commands) == 3 and "pick_lift" in commands[-1]
    assert all(not command.get("direct_ik", False) for command in commands)
    assert commands[0]["targets"] == {"q": 1.0}
    assert commands[1]["targets"] == {"q": 2.0}

    def empty(**kw):
        return REACHED | {"grasp_confirmed": False, "holding_payload": False}

    driver.move_arm = empty
    assert not pick_target(env, b, "object", lambda *a, **kw: [{"mask": np.ones((10, 12), bool)}])["ok"]


def test_model_fails_on_non_rigid_pose_and_point_cloud_failure_never_uses_truth(monkeypatch):
    b = grasp_pose.GraspPoseBackend.load(CONFIG / "moz1-right-rgbd.json")
    with pytest.raises(ValueError, match="insufficient"):
        b.point_cloud(rgbd=lambda: [], ground_truth=lambda: pytest.fail("unexpected truth"))
    bad = np.eye(4)
    bad[0, 0] = -1
    monkeypatch.setattr(grasp_pose, "infer", lambda *args: {"grasps": bad[None], "confidences": [0.8]})
    with pytest.raises(ValueError, match="rotation"):
        b.propose(np.zeros((120, 3)))


def test_pick_requires_actions_and_matching_observation_policy():
    assert "pick_target" not in allowed_tools("visual", True, pick_enabled=True)
    assert "pick_target" not in allowed_tools("perception", True, pick_enabled=True)
    assert "pick_target" not in allowed_tools("ground_truth", False, pick_enabled=True)
    assert "pick_target" in allowed_tools("perception", True, detector_enabled=True, pick_enabled=True)


def test_short_hold_cannot_promote_payload():
    pick = PickLift.__new__(PickLift)
    pick.stable_s = 0.1
    assert not pick.finish()


def test_degenerate_or_reflected_model_outputs_fail_closed(monkeypatch):
    b = grasp_pose.GraspPoseBackend.load(CONFIG / "moz1-right-rgbd.json")
    for scores in ([np.nan], [1.1], [], [[0.9]]):
        monkeypatch.setattr(
            grasp_pose, "infer", lambda *args, scores=scores: {"grasps": np.eye(4)[None], "confidences": scores}
        )
        with pytest.raises(ValueError):
            b.propose(np.zeros((100, 3)))


def test_trial_lift_requires_unsupported_target_bilateral_contact_and_height():
    from jiuwensymbiosis.simulators.genesis.grasp_evidence import GraspEvidence

    body = SimpleNamespace(idx=3, links=["payload"])
    pick = PickLift.__new__(PickLift)
    pick.name = "hand"
    pick.body = body
    pick.tool = "tool"
    pick.initial_z = 0.0
    pick.stable_s = 0.0
    pick.reference = None
    pick.payload = {3: {"support_geoms": []}}
    obj = np.eye(4)
    pick.pose = lambda link: obj if link == "payload" else np.eye(4)
    supported = [True]
    forces = np.array([1.0, 1.0])
    pick.ee = SimpleNamespace(
        groups={"hand": [{1}, {2}]},
        entity=SimpleNamespace(links=[SimpleNamespace(idx=i) for i in (1, 2, 4)]),
        evidence={"hand": GraspEvidence()},
        support_geoms={},
        bodies=lambda: {3: body},
        supported=lambda _: supported[0],
        contact_forces=lambda _: (np.array([1, 2]), np.array([9, 9]), forces),
    )
    for _ in range(20):
        pick.measure(0.1)
    assert not pick.finish()
    supported[0] = False
    obj[2, 3] = 0.08
    forces[0] = 0.0
    for _ in range(20):
        pick.measure(0.1)
    assert not pick.finish()
    forces[0] = 1.0
    for _ in range(20):
        pick.measure(0.1)
    assert pick.finish()
    assert pick.ee.evidence["hand"].report()["holding_payload"] is True
    assert pick.ee.evidence["hand"].object_id == 3


def test_session_binds_only_the_current_scenes_configured_model():
    import json

    from jiuwensymbiosis.simulators.common.agent import make_session

    scene = {
        "instance_id": "one",
        "arm_planning_ready": True,
        "effectors_ready": True,
        "grasp_pose_backends": ["ground_truth"],
    }
    config = json.loads((CONFIG / "moz1-right-ground_truth.json").read_text())
    preview = SimpleNamespace(
        status=lambda: {"state": "ready", "scene": scene},
        query_scene=lambda *args, **kw: {"scene_instance_id": "one", "config": config},
    )
    session = make_session(preview, observation_mode="ground_truth")
    assert session.env.grasp_backend.source == "ground_truth"
    from jiuwensymbiosis.tools.builder import list_tool_meta

    actions = {meta["name"] for meta in list_tool_meta(session.api)}
    assert {"pick_target", "open_gripper", "close_gripper"} <= actions
    assert "set_gripper" not in actions
    assert not hasattr(make_session(preview, observation_mode="visual").api, "pick_target")
    preview.query_scene = lambda *args, **kw: {"scene_instance_id": "two", "config": config}
    with pytest.raises(RuntimeError, match="Scene changed"):
        make_session(preview, observation_mode="ground_truth")
