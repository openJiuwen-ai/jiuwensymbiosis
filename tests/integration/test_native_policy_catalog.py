"""Run every released control through the same lease and physics loop as the worker."""

import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("JIUWEN_POLICY_CATALOG_INTEGRATION") != "1", reason="Needs complete official assets"
    ),
]


def run_control(robot, identifier, name, params):
    request = {
        "id": identifier,
        "kind": name,
        "params": params,
        "scene_instance_id": robot.info["instance_id"],
        "config_revision": 0,
        "cancelled": False,
        "lease_until": 1.0,
    }
    robot.motion.accept_request(request, scene_id=robot.info["instance_id"], revision=0, now=0, step=robot.steps)
    initial = robot.data.qpos[robot.qids].copy()
    peak = 0.0
    stop_at = robot.data.time + 300
    while robot.motion.active and robot.data.time < stop_at:
        robot.step()
        peak = max(peak, float(np.max(np.abs(robot.data.qpos[robot.qids] - initial))))
    return dict(robot.motion.result, params=params, max_joint_change_rad=peak)


def arrange_kick_ball(robot, name, *, distance=0.09, speed=0.0):
    # Arrange the fixture before requesting any action.
    import mujoco as mj

    pose = robot.observe_base()
    x, y = pose["position_m"][:2]
    yaw = pose["body_yaw_rad"]
    lateral = 0.042 if name == "kick_left" else -0.042
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    pos = robot.policy.ball_qpos_adr
    vel = robot.policy.ball_qvel_adr
    robot.data.qpos[pos : pos + 7] = [
        x + cos_yaw * distance - sin_yaw * lateral,
        y + sin_yaw * distance + cos_yaw * lateral,
        0.035,
        1,
        0,
        0,
        0,
    ]
    robot.data.qvel[vel : vel + 6] = [speed, 0, 0, 0, speed / 0.035, 0]
    mj.mj_forward(robot.model, robot.data)


def run_kick(robot, identifier, name, params):
    trigger = robot.policy.trigger_behavior
    pos = robot.policy.ball_qpos_adr
    vel = robot.policy.ball_qvel_adr
    evidence = {}

    def check_ball_at_policy_start(kind):
        before_pos = robot.data.qpos[pos : pos + 7].copy()
        before_vel = robot.data.qvel[vel : vel + 6].copy()
        trigger(kind)
        np.testing.assert_array_equal(robot.data.qpos[pos : pos + 7], before_pos)
        np.testing.assert_array_equal(robot.data.qvel[vel : vel + 6], before_vel)
        evidence.update(
            ball_position_jump_at_trigger_m=0.0,
            ball_linear_velocity_before_trigger_m_s=before_vel[:3].tolist(),
        )

    with patch.object(robot.policy, "trigger_behavior", check_ball_at_policy_start):
        result = run_control(robot, identifier, name, params)
    assert evidence, "Kick policy never started"
    return {**result, **evidence}


@pytest.mark.parametrize("variant", ["jumper", "microduck", "rollers"])
def test_complete_official_motion_catalog(tmp_path, monkeypatch, variant):
    import mujoco as mj

    from jiuwensymbiosis.simulators.assets import AssetLibrary, load_config
    from jiuwensymbiosis.simulators.assets.robot_assets import register_robot
    from jiuwensymbiosis.simulators.common.camera import PreviewSettings
    from jiuwensymbiosis.simulators.mujoco import jumper_runtime, microduck_runtime
    from jiuwensymbiosis.simulators.mujoco.policy_actions import RECORDINGS

    body = "jumper" if variant == "jumper" else "microduck"
    source = Path(os.environ["JIUWEN_" + body.upper() + "_SOURCE"])
    registered = register_robot(body, source, tmp_path)
    library = AssetLibrary(load_config(registered["config"]).index)
    module = jumper_runtime if body == "jumper" else microduck_runtime
    monkeypatch.setattr(module, "resolve_scene", library.resolve)
    scene = registered["scene_id"]
    if variant == "rollers":
        scene = next(a.id for a in library.search(root_id=body).items if a.options.get("rollers"))
    settings = PreviewSettings(scene_id=scene, robot_mode=body, backend="mujoco")
    robot = (jumper_runtime.JumperRuntime if body == "jumper" else microduck_runtime.MicroduckRuntime)(
        settings, tmp_path
    )
    cases = {
        "jumper": [
            ("ref_free_jump", {}),
            ("jump", {}),
            ("jumper_posture", {"pitch": 0.12}),
            ("jumper_posture", {}),
            ("claw_pose", {"side": "right", "pose": "web_up"}),
            ("claw_step", {"side": "right", "direction": "forward"}),
            ("stow_claw", {}),
            ("set_gripper", {"closed": True}),
            ("set_gripper", {"closed": False}),
            ("claw_aperture", {"side": "left", "opening": 0.5}),
            ("claw_aperture", {"side": "left", "opening": 1.0}),
            ("claw_aperture", {"side": "right", "opening": 0.5}),
            ("claw_aperture", {"side": "right", "opening": 1.0}),
            ("stow_claw", {}),
            *((name, {}) for name in RECORDINGS),
        ],
        "microduck": [
            ("head_pose", {"head_yaw": 0.5}),
            ("body_pose", {"pitch": 0.1}),
            ("mouth", {"opening": 1.0}),
            ("mouth", {"opening": 0.0}),
            *(
                (name, {})
                for name in (
                    "sit",
                    "stand",
                    "ground_pick",
                    "kick_left",
                    "kick_right",
                    "roulade",
                    "classic_gait",
                    "velstand_gait",
                    "got_a_talent",
                    "sprint_2m",
                    "hill_climb_1m5",
                    "sock_heist_6m",
                    "velstand_gait",
                    "recover_stance",
                )
            ),
            ("roulade", {"repetitions": 2}),
        ],
        "rollers": [
            ("mouth", {"opening": 1.0}),
            ("mouth", {"opening": 0.0}),
            *(
                (name, {})
                for name in (
                    "roller_skate",
                    "roller_crouch",
                    "roller_sprint_2m",
                    "roller_skate",
                    "roller_crouch",
                    "roller_gait",
                    "sit",
                    "stand",
                    "kick_left",
                    "kick_right",
                )
            ),
        ],
    }[variant]
    evidence = Path(os.environ.get("JIUWEN_POLICY_CATALOG_EVIDENCE", tmp_path))
    evidence.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        assert set(robot.snapshot()["policy_actions"]) == {name for name, _ in cases}
        if body == "microduck":
            arrange_kick_ball(robot, "kick_left", distance=1, speed=0.2)
            miss = run_kick(robot, "out-of-reach", "kick_left", {})
            results.append(miss)
            assert np.linalg.norm(miss["ball_linear_velocity_before_trigger_m_s"]) > 0.01
            assert miss["settled"] and miss["reason"] == "kick_unconfirmed", miss
            assert not miss["ok"] and not miss["foot_ball_contact"], miss
            robot.close()
            robot = module.MicroduckRuntime(settings, tmp_path)
        for index, (name, params) in enumerate(cases):
            if name == "recover_stance":
                # Test fixture: start on its side. Only the running policy may right the robot.
                robot.data.qpos[3:7] = [math.sqrt(0.5), math.sqrt(0.5), 0, 0]
                mj.mj_forward(robot.model, robot.data)
                assert not robot.upright()
            if variant == "rollers" and name.startswith("kick_"):
                if name == "kick_left":
                    # Also exercise an unassisted kick from the last skating stance. A real miss
                    # is part of the evidence and must never be mistaken for a successful kick.
                    trial = run_kick(robot, "rolling-stance", name, params)
                    trial["entry"] = "after_skating_and_standing"
                    results.append(trial)
                    assert trial["settled"]
                    assert trial["ok"] == (trial["foot_ball_contact"] and trial["ball_displacement_m"] >= 0.01)
                robot.close()
                robot = module.MicroduckRuntime(settings, tmp_path)
            if name.startswith("kick_"):
                arrange_kick_ball(robot, name)
                result = run_kick(robot, str(index), name, params)
            else:
                result = run_control(robot, str(index), name, params)
            peak = result["max_joint_change_rad"]
            if name.startswith("kick_"):
                assert result["foot_ball_contact"] and result["ball_displacement_m"] >= 0.01
                if variant == "rollers":
                    result["entry"] = "factory_standing_pose"
            results.append(result)
            assert result["status"] == "completed" and result["settled"], result
            if name == "ref_free_jump":
                assert result["ok"] == (result["flight_s"] >= 0.02)
                assert result["reason"] in {"reached", "flight_unconfirmed"}
                assert robot.policy.fsm.mode == "locomotion"
            else:
                assert result["ok"], result
            if name in RECORDINGS:
                assert result["duration_sim_s"] >= robot.policy.durations[name] and peak > 0.1
            if name == "mouth":
                assert abs(result["joint_angle_rad"] - result["joint_target_rad"]) < 0.05
            if name == "claw_aperture" and params["opening"] == 0.5:
                sign = 1 if params["side"] == "left" else -1
                assert abs(sign * result["joint_angle_rad"] - (0.10 - math.pi / 6)) < 0.07
                assert robot.snapshot()["gripper"]["opening_command"] == 0.5
            if name == "recover_stance":
                assert robot.upright() and peak > 0.5
            if name == "sprint_2m":
                x0, y0 = result["before"]["position_m"][:2]
                x1, y1 = result["after"]["position_m"][:2]
                yaw = result["before"]["body_yaw_rad"]
                forward = (x1 - x0) * math.cos(yaw) + (y1 - y0) * math.sin(yaw)
                assert result["forward_displacement_m"] == pytest.approx(forward)
                assert 2.0 <= forward < 2.3 and robot._command == (0, 0, 0), result
                assert robot.gait == "velstand_gait"
                robot.motion.start(
                    {"id": "turn-after-sprint", "kind": "navigate", "dx": 0, "dy": 0, "dyaw": math.pi},
                    robot.steps,
                )
                while robot.motion.active:
                    robot.step()
                turn = robot.motion.result
                results.append(turn)
                assert turn["ok"] and turn["settled"] and turn["stop_confirmation"] == "measured", turn
            if name in {"hill_climb_1m5", "sock_heist_6m", "classic_gait"}:
                start = robot.data.qpos[:2].copy()
                robot.command(0.4, 0, 0, duration=0.8)
                for _ in range(round(1.8 / settings.dt)):
                    robot.step()
                result["gait_displacement_m"] = math.dist(start, robot.data.qpos[:2])
                assert result["gait_displacement_m"] > 0.01 and robot.upright(), result
    finally:
        robot.close()
        (evidence / f"{variant}-catalog.json").write_text(json.dumps(results, indent=2))
