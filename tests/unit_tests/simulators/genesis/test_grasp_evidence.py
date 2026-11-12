"""False positives and lost-payload state must not become planning facts."""

from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.arm_motion import ArmMotion
from jiuwensymbiosis.simulators.genesis.end_effectors import EndEffectors
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


@pytest.mark.parametrize(
    "sequence,holding,placement",
    [
        ("contact", None, False),
        ("empty", False, False),
        ("contact lift", True, False),
        ("contact lift drop", None, False),
        ("contact lift release", False, True),
        ("contact release fall", False, False),
        ("midair lift", None, False),
    ],
)
def test_contact_lift_release_and_fall_evidence(sequence, holding, placement):
    samples = {
        "contact": {},
        "empty": {"contact": False},
        "midair": {"supported": False},
        "lift": {"height": 0.52, "supported": False},
        "release": {"opened": True, "contact": False},
        "drop": {"opened": True, "contact": False, "supported": False},
        "fall": {"opened": True, "contact": False, "supported": False, "speed": 0.1},
    }
    evidence = GraspEvidence()
    for stage in sequence.split():
        observe(evidence, **samples[stage])
    assert evidence.holding is holding
    assert evidence.report()["placement_confirmed"] is placement
    assert evidence.report()["grasp_confirmed"] is (holding is True)
    if sequence == "contact":
        assert evidence.state == "contact" and evidence.report()["contact_confirmed"]


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


def test_retry_and_cancel_preserve_latched_grip_target():
    writes = []
    controller = EndEffectors.__new__(EndEffectors)
    controller.entity = SimpleNamespace(
        get_dofs_position=lambda ids: np.array([[0.02]]),
        get_dofs_velocity=lambda ids: np.array([[0.0]]),
        control_dofs_position=lambda q, ids: writes.append(q.copy()),
    )
    motor = {"open": 0.04, "close": 0.0, "velocity": 0.03, "tolerance": 0.002, "preload": 0.002}
    controller.configurations = {"arbitrary": {"joints": {"slider": motor}}}
    controller.evidence = {"arbitrary": GraspEvidence(state="held", object_id=4, contact_time=0.3)}
    controller.ids = {"arbitrary": [0]}
    controller.targets = {"arbitrary": np.array([0.018])}
    controller.latched = {"arbitrary": True}
    controller.opening = {}
    controller.measure = lambda dt: None
    for _ in range(3):
        controller.start("arbitrary", opened=False)
        controller.tick(0.01)
        controller.cancel()
    assert np.allclose(writes, [[0.018]] * 3)
    with pytest.raises(ValueError, match="stable support"):
        controller.start("arbitrary", opened=True)


@pytest.mark.parametrize(
    "opened,state,object_id,ok",
    [(False, "unknown", 1, False), (True, "clear", 1, False), (True, "clear", None, True), (False, "contact", 1, True)],
)
@pytest.mark.parametrize("unwritable", [False, True])
def test_selected_gripper_confirmation_while_other_hand_holds(tmp_path, opened, state, object_id, ok, unwritable):
    motion = ArmMotion.__new__(ArmMotion)
    evidence = GraspEvidence(state=state, object_id=object_id, contact_time=0.3)
    motion.effectors = SimpleNamespace(
        tick=lambda dt: None, evidence={"hand": evidence, "other": GraspEvidence(state="held")}
    )
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
    motion.execution_path = tmp_path if unwritable else tmp_path / "result.json"
    motion.tick(0.01)
    assert motion.result["settled"]
    assert motion.result["ok"] is (ok and not unwritable)
    assert motion.result["holding_payload"] is evidence.holding
    reason = None if ok else "grasp_or_release_confirmation_lost"
    assert motion.result["reason"] == ("evidence_unwritable" if unwritable else reason)
    assert not motion.follower.terminal
