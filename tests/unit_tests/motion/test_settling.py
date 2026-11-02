"""Stopping needs a sustained stream of measured low velocities, not zero commands."""

import pytest

from jiuwensymbiosis.motion.settling import StopMonitor


def monitor():
    return StopMonitor(linear_limit=0.02, angular_limit=0.03, hold_s=0.6, max_age_s=0.25)


def sample(m, t, *, stamp=None, linear=0.001, angular=0.002, now=None):
    return m.update(
        now=t if now is None else now,
        sample_at=t,
        stamp=int(t * 1e9) if stamp is None else stamp,
        linear_speed=linear,
        angular_speed=angular,
    )


def test_only_sustained_fresh_low_velocity_confirms_stopping():
    m = monitor()
    assert not sample(m, 1.0, linear=0.3)
    for t in (1.1, 1.2, 1.3, 1.4, 1.5, 1.6):
        assert not sample(m, t)
    assert sample(m, 1.71)


@pytest.mark.parametrize("failure", ["linear", "angular", "nan", "gap", "rollback", "stale"])
def test_motion_bad_samples_and_gaps_restart_hold(failure):
    m = monitor()
    for t in (1.0, 1.2, 1.4):
        assert not sample(m, t)
    if failure == "linear":
        assert not sample(m, 1.5, linear=0.02)
    elif failure == "angular":
        assert not sample(m, 1.5, angular=-0.03)
    elif failure == "nan":
        assert not sample(m, 1.5, linear=float("nan"))
    elif failure == "gap":
        assert not sample(m, 1.8)
    elif failure == "rollback":
        assert not sample(m, 1.5, stamp=1)
    else:
        assert not sample(m, 1.4, now=1.7)
    assert not sample(m, 1.9)
    assert not sample(m, 2.1)
    assert not sample(m, 2.3)
    assert sample(m, 2.51)


def test_repeated_odom_never_counts_as_a_hold_even_with_new_receipt_times():
    m = monitor()
    for t in (1.0, 1.2, 1.4, 1.6, 1.8):
        assert not sample(m, t, stamp=100)
    assert not sample(m, 1.9, stamp=101)


def test_absent_and_future_feedback_cannot_confirm_stop():
    m = monitor()
    assert not m.update(now=1.0, sample_at=None, stamp=None, linear_speed=None, angular_speed=None)
    assert not sample(m, 2.0, now=1.0)


@pytest.mark.parametrize("bad", [True, 0, -0.1, float("nan"), float("inf")])
def test_invalid_thresholds_rejected(bad):
    with pytest.raises(ValueError):
        StopMonitor(linear_limit=bad, angular_limit=0.03, hold_s=0.6, max_age_s=0.25)
