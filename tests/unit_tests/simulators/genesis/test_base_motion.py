"""Action semantics and physical completion, independent of the simulator SDK."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jiuwensymbiosis.simulators.genesis import base_motion as bm


@pytest.fixture
def controller(monkeypatch):
    state = {
        "position_m": [0.0, 0.0, 0.1],
        "body_yaw_rad": 0.0,
        "quaternion_wxyz": [1, 0, 0, 0],
        "linear_velocity_m_s": [0.0, 0.0, 0.0],
        "angular_velocity_rad_s": [0.0, 0.0, 0.0],
    }
    monkeypatch.setattr(bm, "measure", lambda _: {k: v.copy() if isinstance(v, list) else v for k, v in state.items()})
    robot = SimpleNamespace(command=Mock(), stop=Mock())
    return bm.RelativeMotion(robot), state


def test_turn_then_translate_uses_new_body_frame(controller):
    motion, state = controller
    motion.start("a", 0.3, 0.2, math.pi / 2, step=8)
    assert motion.target == pytest.approx([-0.2, 0.3, math.pi / 2])
    motion.tick(0.05, 9)
    assert motion.robot.command.call_args.args[:2] == (0.0, 0.0)
    state["body_yaw_rad"] = math.pi / 2
    motion.tick(0.05, 10)
    vx, vy, wz, _ = motion.robot.command.call_args.args
    assert vx > 0 and vy > 0 and wz == 0


def test_completion_requires_physical_stop_not_zero_command(controller):
    motion, state = controller
    motion.start("a", 0, 0.3, 0, step=0)
    state["position_m"][1] = 0.3
    state["linear_velocity_m_s"][1] = 0.1
    for i in range(20):
        motion.tick(0.05, i)
    assert motion.active and not motion.result["ok"]
    state["linear_velocity_m_s"][1] = 0
    for i in range(13):
        motion.tick(0.05, 20 + i)
    assert motion.result["ok"] and motion.result["settled"]
    assert motion.result["after"]["position_m"][1] == 0.3
    assert motion.result["end_step"] > motion.result["start_step"]


def test_navigation_runs_at_60hz_but_braking_observes_every_physics_tick(controller, monkeypatch):
    motion, state = controller
    motion.start("a", 0.2, 0, 0, step=0)
    reads = Mock(return_value=state)
    monkeypatch.setattr(bm, "measure", reads)
    for i in range(12):
        motion.tick(1 / 120, i)
    assert reads.call_count == 6
    assert motion.robot.command.call_count == 6
    assert motion.elapsed == pytest.approx(0.1)
    motion.cancel()
    reads.reset_mock()
    for i in range(12):
        motion.tick(1 / 120, 12 + i)
    assert reads.call_count == 12
    assert motion.active  # Still owes the entire 0.6-second stability window.


def test_drift_reenters_position_control(controller):
    motion, state = controller
    motion.start("a", 0.3, 0, 0, step=0)
    state["position_m"][0] = 0.295
    motion.tick(0.05, 1)
    assert motion.phase == "settle"
    state["position_m"][0] = 0.27
    motion.tick(0.05, 2)
    assert motion.phase == "translate" and motion.stable == 0


@pytest.mark.parametrize("values", [(float("nan"), 0, 0), (True, 0, 0), (0, float("inf"), 0), (0.8, 0.8, 0), (0, 0, 4)])
def test_invalid_move_is_rejected_before_mutation(controller, values):
    motion, _ = controller
    with pytest.raises(ValueError):
        motion.start("bad", *values, step=0)
    assert not motion.active and motion.result == {}
    motion.robot.command.assert_not_called()


def request(**kwargs):
    return {
        "id": "a",
        "dx": 0.3,
        "dy": 0,
        "dyaw": 0,
        "scene_instance_id": "scene",
        "config_revision": 2,
        "lease_until": 100,
        "cancelled": False,
        **kwargs,
    }


def test_cancel_before_worker_acceptance_and_no_duplicate_replay(controller):
    motion, _ = controller
    req = request(cancelled=True)
    motion.accept_request(req, scene_id="scene", revision=2, now=10, step=0)
    for i in range(15):
        motion.tick(0.05, i)
    assert not motion.result["ok"] and not motion.result["settled"]
    assert motion.result["status"] == "completed"
    motion.robot.command.assert_not_called()
    motion.accept_request(request(), scene_id="scene", revision=2, now=10, step=20)
    assert not motion.active


def test_stale_scene_cannot_move_and_expiry_cancels_active_motion(controller):
    motion, _ = controller
    motion.accept_request(request(), scene_id="other", revision=2, now=10, step=0)
    assert not motion.active
    motion.accept_request(request(), scene_id="scene", revision=2, now=10, step=0)
    motion.tick(0.05, 1)
    motion.accept_request(request(), scene_id="scene", revision=2, now=101, step=2)
    assert motion.phase == "brake"
    for i in range(15):
        motion.tick(0.05, 3 + i)
    assert not motion.result["ok"] and motion.result["reason"] == "cancelled_or_lease_expired"


def test_rotation_crosses_pi_by_shortest_error(controller):
    motion, state = controller
    state["body_yaw_rad"] = math.pi - 0.1
    motion.start("a", 0, 0, 0.2, step=0)
    motion.tick(0.05, 1)
    assert motion.robot.command.call_args.args[2] > 0


def test_root_minus_y_is_measured_as_body_front(monkeypatch):
    torch = pytest.importorskip("torch")
    transfers = []
    original = torch.Tensor.cpu

    def readback(tensor, *args, **kwargs):
        transfers.append(tensor.numel())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", readback)

    def tensor(values):
        return torch.tensor(values, dtype=torch.float64)

    entity = SimpleNamespace(
        get_pos=lambda: [tensor([0, 0, 0.1])],
        get_quat=lambda: [tensor([1, 0, 0, 0])],
        get_vel=lambda: [tensor([0.1, 0.2, 0.3])],
        get_ang=lambda: [tensor([0.4, 0.5, 0.6])],
    )
    state = bm.measure(SimpleNamespace(entity=entity))
    assert state["body_yaw_rad"] == pytest.approx(-math.pi / 2)
    assert state["position_m"] == [0, 0, 0.1]
    assert state["quaternion_wxyz"] == [1, 0, 0, 0]
    assert state["linear_velocity_m_s"] == [0.1, 0.2, 0.3]
    assert state["angular_velocity_rad_s"] == [0.4, 0.5, 0.6]
    assert transfers == [13]


@pytest.mark.parametrize("inactive", [{"cancelled": True}, {"lease_until": 10}, {"lease_until": 9}])
@pytest.mark.parametrize("dx", [0.3, 1.5, float("nan")])
def test_inactive_request_does_not_validate_measure_or_actuate(controller, monkeypatch, inactive, dx):
    motion, _ = controller
    reads = Mock(side_effect=RuntimeError("Non-finite physical base observation"))
    monkeypatch.setattr(bm, "measure", reads)
    motion.accept_request(request(dx=dx, **inactive), scene_id="scene", revision=2, now=10, step=7)
    assert not motion.active
    assert motion.result["id"] == "a"
    assert motion.result["status"] == "completed"
    assert motion.result["reason"] == "cancelled_or_lease_expired"
    assert motion.result["ok"] is False
    assert motion.result["settled"] is False  # No physical observation was taken.
    assert motion.result["end_step"] == 7
    motion.accept_request(request(), scene_id="scene", revision=2, now=10, step=8)
    motion.tick(0.05, 9)
    reads.assert_not_called()
    motion.robot.command.assert_not_called()
    motion.robot.stop.assert_not_called()


def test_live_request_after_rejected_id_still_starts(controller):
    motion, _ = controller
    motion.accept_request(request(cancelled=True), scene_id="scene", revision=2, now=10, step=0)
    motion.accept_request(request(id="b"), scene_id="scene", revision=2, now=10, step=1)
    motion.tick(0.05, 2)
    assert motion.active and motion.result["id"] == "b"
    motion.robot.command.assert_called_once()
