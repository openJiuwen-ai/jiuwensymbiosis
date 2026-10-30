"""Fixed-step physics keeps wall time while expensive frames run serially."""

from itertools import pairwise

import pytest

from jiuwensymbiosis.simulators.genesis.scheduling import PhysicsClock


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_physics_catches_up_after_a_slow_render_across_control_batches():
    wall = FakeTime()
    clock = PhysicsClock(1 / 120, monotonic=wall)
    wall.advance(0.100)  # One reference-quality frame blocks the sole GPU owner.
    batches = []
    while clock.next_step <= wall():
        batches.append(clock.run_batch(lambda: wall.advance(0.001)))
        if len(batches) == 1:
            assert not clock.render_ready()
            assert wall() < 0.110  # Control/stop gets serviced between batches.

    assert len(batches) > 1
    assert sum(batches) == 13  # The former eight-step cap silently lost this debt.
    assert clock.render_ready()
    assert clock.dt == 1 / 120
    assert clock.snapshot()["dropped_wall_time_s"] == 0


def test_extreme_stall_is_reported_instead_of_hidden_as_realtime_progress():
    wall = FakeTime()
    clock = PhysicsClock(1 / 120, monotonic=wall)
    wall.advance(2.0)  # A stalled driver or GPU exceeds the explicit 0.5 s limit.
    clock.run_batch(lambda: wall.advance(0.001))
    status = clock.snapshot()

    assert status["overload_events"] == 1
    assert 1.4 < status["dropped_wall_time_s"] < 1.6
    assert status["lag_s"] <= 0.5
    assert status["real_time_factor"] < 0.1
    assert clock.dt == 1 / 120


def test_slow_frames_retain_120_hz_physics_without_reducing_the_timestep():
    wall = FakeTime()
    clock = PhysicsClock(1 / 120, monotonic=wall)
    frames = 0
    while frames < 100:
        clock.run_batch(lambda: wall.advance(0.001))
        if clock.render_ready():
            frames += 1
            wall.advance(0.100)
            clock.mark_render()
    while clock.next_step <= wall():
        clock.run_batch(lambda: wall.advance(0.001))

    status = clock.snapshot()
    assert 119 <= status["actual_hz"] <= 120
    assert 0.99 <= status["real_time_factor"] <= 1
    assert status["dropped_wall_time_s"] == 0
    assert status["overload_events"] == 0
    assert status["dt_s"] == 1 / 120


def test_stop_can_interrupt_a_batch_and_debt_survives_between_batches():
    wall = FakeTime()
    clock = PhysicsClock(1 / 120, monotonic=wall)
    wall.advance(0.100)
    steps = 0

    def step():
        nonlocal steps
        steps += 1
        wall.advance(0.001)

    assert clock.run_batch(step, stopping=lambda: steps == 3) == 3
    assert not clock.render_ready()
    assert clock.snapshot()["simulated_s"] == pytest.approx(0.025)
    assert clock.run_batch(step, stopping=lambda: True) == 0
    assert steps == 3
    while clock.next_step <= wall():
        clock.run_batch(step)
    assert steps == 13


def test_fast_steps_still_yield_control_at_the_hard_batch_cap():
    wall = FakeTime()
    clock = PhysicsClock(0.001, monotonic=wall)
    wall.advance(0.400)

    assert clock.run_batch(lambda: None) == 64
    assert not clock.render_ready()
    assert clock.snapshot()["dropped_wall_time_s"] == 0


def test_new_clock_after_configuration_pause_does_not_fake_catchup_steps():
    wall = FakeTime()
    clock = PhysicsClock(1 / 120, monotonic=wall)
    wall.advance(0.050)
    clock.run_batch(lambda: None)
    wall.advance(10)  # Scene/renderer replacement intentionally pauses physics.
    clock = PhysicsClock(1 / 120, monotonic=wall)

    assert clock.run_batch(lambda: pytest.fail("No step is due yet")) == 0
    assert clock.snapshot()["simulated_s"] == 0
    assert clock.snapshot()["dropped_wall_time_s"] == 0
    wall.advance(1 / 120)
    assert clock.run_batch(lambda: None) == 1


@pytest.mark.parametrize("step_cost", [1 / 120, 0.0083])
def test_physics_near_capacity_keeps_video_alive_without_a_dropped_time_event(step_cost):
    wall = FakeTime()
    clock = PhysicsClock(1 / 120, monotonic=wall)
    wall.advance(0.100)
    clock.mark_render()
    frames = []
    while wall() < 2:
        clock.run_batch(lambda: wall.advance(step_cost))
        if clock.render_ready():
            # Near-capacity physics can keep stable debt below the drop limit.
            assert clock.snapshot()["overload_events"] == 0
            frames.append(wall())
            wall.advance(0.100)
            clock.mark_render()
            assert not clock.render_ready()

    assert len(frames) >= 3
    assert all(0.5 <= b - a <= 0.7 for a, b in pairwise(frames))
    assert clock.snapshot()["dropped_wall_time_s"] == 0


def test_sustained_slow_physics_keeps_a_bounded_degraded_picture_and_stop():
    wall = FakeTime()
    clock = PhysicsClock(1 / 120, monotonic=wall)
    wall.advance(1 / 120)
    frames = []
    while wall() < 10:
        batch_started = wall()
        assert clock.run_batch(lambda: wall.advance(0.010)) == 1
        assert wall() - batch_started == pytest.approx(0.010)
        if clock.render_ready():
            frames.append(wall())
            wall.advance(0.100)
            clock.mark_render()
            assert not clock.render_ready()  # Capture and video share one budget.

    late_frames = [t for t in frames if t > 5]
    assert len(late_frames) >= 5  # Cannot freeze permanently when physics is slow.
    assert all(0.5 <= b - a <= 0.7 for a, b in pairwise(late_frames))
    assert wall() - late_frames[-1] < 0.7
    status = clock.snapshot()
    assert status["degraded"] is True
    assert status["dropped_wall_time_s"] > 0
    assert status["real_time_factor"] < 1
    assert status["dt_s"] == 1 / 120
    assert clock.run_batch(lambda: pytest.fail("Stop must prevent another step"), stopping=lambda: True) == 0
