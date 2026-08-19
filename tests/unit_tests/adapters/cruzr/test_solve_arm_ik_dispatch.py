# coding: utf-8
"""solve_arm_ik dispatches to pinocchio when available, else the legacy DLS."""

from jiuwensymbiosis.adapters.cruzr.geometry import ARM_JOINTS, ArmTarget, solve_arm_ik
from jiuwensymbiosis.kinematics.ik import IKResult
from jiuwensymbiosis.kinematics.urdf_chain import Chain


def _tgt():
    return ArmTarget("left", (0.40, 0.20, 0.65), (1, 0, 0), (0, 1, 0), (-0.09, 0.0, 0.0))


def _fixed():
    return {"lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
            "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0}


def test_dispatch_uses_pinocchio_when_available(monkeypatch):
    import jiuwensymbiosis.kinematics.ik_pinocchio as pik
    called = {}

    def _fake_pin(urdf_path, arm_joints, leaf_link, limits, *a, **k):
        called["urdf"] = urdf_path
        called["leaf"] = leaf_link
        return IKResult(q={j: 0.1 for j in arm_joints}, converged=True, pos_err_m=0.001, normal_err=0.001, iters=3)

    monkeypatch.setattr(pik, "pin_available", lambda: True)
    monkeypatch.setattr(pik, "solve_pose_ik_pin", _fake_pin)
    chain = Chain(joints=[], urdf_path="/some.urdf", leaf_link="L_sixforce_link")
    res = solve_arm_ik(chain, _fixed(), "left", _tgt())
    assert called["urdf"] == "/some.urdf" and called["leaf"] == "L_sixforce_link"
    assert res.converged and set(res.q) == set(ARM_JOINTS["left"])


def test_dispatch_falls_back_to_dls_when_pin_unavailable(monkeypatch):
    import jiuwensymbiosis.kinematics.ik_pinocchio as pik
    import jiuwensymbiosis.adapters.cruzr.geometry as gp
    legacy = {}

    def _fake_dls(chain, q_fixed, arm_joints, *a, **k):
        legacy["hit"] = True
        return IKResult(q={j: 0.0 for j in arm_joints}, converged=True, pos_err_m=0.0, normal_err=0.0, iters=1)

    monkeypatch.setattr(pik, "pin_available", lambda: False)
    monkeypatch.setattr(gp, "ik_solve_pose", _fake_dls)
    chain = Chain(joints=[], urdf_path="/some.urdf", leaf_link="L_sixforce_link")
    res = solve_arm_ik(chain, _fixed(), "left", _tgt())
    assert legacy.get("hit") is True and res.converged


def test_dispatch_forwards_check_collision_and_package_dir(monkeypatch):
    import jiuwensymbiosis.kinematics.ik_pinocchio as pik
    seen = {}

    def _fake_pin(urdf_path, arm_joints, leaf_link, limits, *a, **k):
        seen["check_collision"] = k.get("check_collision")
        seen["package_dir"] = k.get("package_dir")
        return IKResult(q={j: 0.0 for j in arm_joints}, converged=True, pos_err_m=0.0, normal_err=0.0, iters=1)

    monkeypatch.setattr(pik, "pin_available", lambda: True)
    monkeypatch.setattr(pik, "solve_pose_ik_pin", _fake_pin)
    chain = Chain(joints=[], urdf_path="/some.urdf", leaf_link="L_sixforce_link")
    solve_arm_ik(chain, _fixed(), "left", _tgt(), check_collision=True, package_dir="/pkg")
    assert seen["check_collision"] is True and seen["package_dir"] == "/pkg"
