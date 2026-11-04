"""False positives and lost-payload state must not become planning facts."""

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.grasp_evidence import GraspEvidence


def observe(evidence, seconds=0.3, **changes):
    sample = {
        "object_id": 7,
        "relative": np.eye(4),
        "height": 0.5,
        "contact": True,
        "supported": True,
        "speed": 0.0,
        "angular_speed": 0.0,
        "opened": False,
    }
    sample.update(changes)
    for _ in range(round(seconds / 0.01)):
        evidence.observe(dt=0.01, **sample)


def test_contact_alone_does_not_confirm_payload():
    evidence = GraspEvidence()
    observe(evidence)
    assert evidence.state == "contact"
    assert evidence.holding is None
    assert evidence.report()["contact_confirmed"]
    assert not evidence.report()["grasp_confirmed"]


def test_trial_lift_then_supported_release():
    evidence = GraspEvidence()
    observe(evidence)
    observe(evidence, height=0.52, supported=False)
    assert evidence.holding is True
    observe(evidence, opened=True, contact=False, supported=False)
    assert evidence.holding is None
    assert not evidence.report()["placement_confirmed"]
    observe(evidence, opened=True, contact=False, supported=True)
    assert evidence.holding is False
    assert evidence.report()["placement_confirmed"]


@pytest.mark.parametrize("changes", [{"contact": False}, {"object_id": 8}, {"relative": np.diag([1.0, 1.0, 1.0, 2.0])}])
def test_loss_or_identity_change_never_reconfirms_without_release(changes):
    evidence = GraspEvidence()
    observe(evidence)
    observe(evidence, height=0.52, supported=False)
    if "relative" in changes:
        changes = {"relative": np.eye(4)}
        changes["relative"][0, 3] = 0.01
    observe(evidence, **changes)
    assert evidence.holding is None
    observe(evidence, height=0.52, supported=False)
    assert evidence.holding is None


@pytest.mark.parametrize("changes", [{"speed": 0.1}, {"angular_speed": 0.2}, {"supported": False}])
def test_opening_on_unstable_support_is_not_place(changes):
    evidence = GraspEvidence()
    observe(evidence)
    observe(evidence, opened=True, contact=False, **changes)
    assert evidence.holding is None
    assert not evidence.report()["placement_confirmed"]


def test_nonfinite_measurement_invalidates_previous_confirmation():
    evidence = GraspEvidence()
    observe(evidence)
    observe(evidence, height=0.52, supported=False)
    with pytest.raises(ValueError, match="invalid grasp observation"):
        observe(evidence, height=float("nan"))
    assert evidence.holding is None


def test_single_sided_or_empty_contact_remains_clear():
    evidence = GraspEvidence()
    observe(evidence, contact=False)
    assert evidence.holding is False
    assert not evidence.report()["grasp_confirmed"]


def test_fall_after_release_revokes_place_but_hand_stays_clear():
    evidence = GraspEvidence()
    observe(evidence)
    observe(evidence, opened=True, contact=False)
    assert evidence.report()["placement_confirmed"]
    observe(evidence, opened=True, contact=False, supported=False, speed=0.1)
    assert evidence.holding is False
    assert not evidence.report()["placement_confirmed"]


def test_midair_contact_does_not_create_a_verified_grasp():
    evidence = GraspEvidence()
    observe(evidence, supported=False)
    observe(evidence, height=0.52, supported=False)
    assert evidence.holding is None


def test_close_ramps_to_endpoint_through_contact_completion_and_idle():
    from types import SimpleNamespace

    from jiuwensymbiosis.simulators.genesis.end_effectors import EndEffectors

    writes = []
    controller = EndEffectors.__new__(EndEffectors)
    controller.entity = SimpleNamespace(
        get_dofs_position=lambda ids: np.array([[0.02]]),
        get_dofs_velocity=lambda ids: np.array([[0.0]]),
        control_dofs_position=lambda q, ids: writes.append(q.copy()),
    )
    controller.configurations = {
        "arbitrary": {
            "joints": {
                "slider": {
                    "open": 0.04,
                    "close": 0.0,
                    "velocity": 0.03,
                    "tolerance": 0.002,
                }
            }
        }
    }
    controller.evidence = {"arbitrary": GraspEvidence(state="held", object_id=4, contact_time=0.3)}
    controller.ids = {"arbitrary": [0]}
    controller.targets = {"arbitrary": np.array([0.018])}
    controller.closing = set()
    controller.opening = {}
    controller.measure = lambda dt: None
    for _ in range(3):
        controller.start("arbitrary", opened=False)
        controller.tick(0.01)
        controller.cancel()
    assert np.allclose(writes, [[0.0177], [0.0174], [0.0171]])
    controller.start("arbitrary", opened=False)
    for _ in range(100):
        controller.tick(0.01)
    assert controller.result["ok"]
    assert controller.command is None
    assert not controller.closing
    assert writes[-1][0] == 0.0
    assert all(after <= before for before, after in zip(writes[:-1], writes[1:], strict=True))

    with pytest.raises(ValueError, match="stable support"):
        controller.start("arbitrary", opened=True)


@pytest.mark.parametrize("opened,state,object_id", [(False, "unknown", 1), (True, "clear", 1)])
def test_confirmation_lost_while_stopping_fails_command(tmp_path, opened, state, object_id):
    from types import SimpleNamespace

    from jiuwensymbiosis.simulators.genesis.arm_motion import ArmMotion

    motion = ArmMotion.__new__(ArmMotion)
    evidence = GraspEvidence(state=state, object_id=object_id)
    motion.effectors = SimpleNamespace(tick=lambda dt: None, evidence={"hand": evidence})
    motion.runtime = SimpleNamespace(robot=SimpleNamespace(stop=lambda: None))
    motion.follower = SimpleNamespace(tick=lambda dt: None, terminal=False)
    motion.stopped = lambda: True
    motion.check_tool_goal = lambda: None
    motion.phase = "confirming"
    motion.captured = None
    motion.elapsed = 0.0
    motion.stable = 0.25
    motion.reason = None
    motion.gripper_request = ("hand", opened)
    motion.result = {}
    motion.execution_path = tmp_path / "result.json"
    motion.tick(0.01)
    assert motion.result["settled"]
    assert not motion.result["ok"]
    assert motion.result["reason"] == "grasp_or_release_confirmation_lost"
