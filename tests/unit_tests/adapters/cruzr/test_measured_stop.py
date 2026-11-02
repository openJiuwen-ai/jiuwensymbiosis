"""Worker publication/measurement loop and driver uncertainty latch, without ROS."""

import pytest

from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrNav
from jiuwensymbiosis.adapters.cruzr.ros2.wheel_worker import _build_parser, _confirm_stop
from jiuwensymbiosis.agent.lifecycle import HardwareCleanupError


@pytest.mark.parametrize("feedback", ["settles", "moving", "duplicate", "absent"])
def test_stop_keeps_publishing_zero_and_requires_fresh_velocity(feedback):
    clock = [1.0]
    state = {}
    commands = []
    args = _build_parser().parse_args([])

    def pump():
        if feedback == "absent":
            return
        state.update(
            stamp_ns=1 if feedback == "duplicate" else int(clock[0] * 1e9),
            received_at=clock[0],
            angular_speed_rad_s=0.0,
            linear_speed_m_s=0.1 if feedback == "moving" or clock[0] < 1.2 else 0.001,
        )

    def sleep(dt):
        clock[0] += dt

    out = _confirm_stop(lambda *v: commands.append(v), pump, state, args, clock=lambda: clock[0], sleep=sleep)
    assert all(v == (0.0, 0.0) for v in commands)
    assert out["settled"] is (feedback == "settles")
    assert out["stop_confirmation"] == ("measured" if feedback == "settles" else "unknown")
    assert 0.8 <= out["stop_duration_s"] <= 2.03


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize(
    "failed",
    [
        {"ok": False, "settled": False, "stop_confirmation": "unknown"},
        {"ok": True},
        {"ok": True, "settled": True},
        {"ok": True, "stop_confirmation": "measured"},
        {"ok": False, "reason": "wheel_bad_output"},
    ],
)
def test_unconfirmed_stop_blocks_all_subsequent_driver_commands(monkeypatch, resident, failed):
    from jiuwensymbiosis.adapters.cruzr import lowlevel

    nav = CruzrNav(CruzrConfig(resident_workers=resident))
    if resident:
        monkeypatch.setattr(nav._move, "request_json", lambda *a, **k: failed)
    else:
        monkeypatch.setattr(lowlevel.ros2_worker, "run_once", lambda *a, **k: failed)
    with pytest.raises(HardwareCleanupError):
        nav.navigate_relative(0.0)
    with pytest.raises(HardwareCleanupError):
        nav.start_spin()
    with pytest.raises(HardwareCleanupError):
        nav.navigate_relative(0.1)


def test_resident_confirmed_clean_fallback_can_use_one_shot(monkeypatch):
    from jiuwensymbiosis.adapters.cruzr import lowlevel

    nav = CruzrNav(CruzrConfig(resident_workers=True))
    monkeypatch.setattr(nav._move, "request_json", lambda *a, **k: None)
    calls = []
    stopped = {"ok": True, "settled": True, "stop_confirmation": "measured"}
    monkeypatch.setattr(lowlevel.ros2_worker, "run_once", lambda *a, **k: calls.append(a) or stopped)
    assert nav.navigate_relative(0.1) == stopped
    assert len(calls) == 1 and not nav._cleanup_errors


def test_body_stop_configuration_reaches_all_worker_modes():
    nav = CruzrNav(CruzrConfig(base_stop_linear_m_s=0.012, base_stop_hold_s=0.8))
    for cmd in (nav._serve_cmd(), nav._spin_cmd(1.0), nav._forward_cmd()):
        args = _build_parser().parse_args(cmd[2:])
        assert args.stop_linear_m_s == 0.012 and args.stop_hold_s == 0.8


@pytest.mark.parametrize(
    "config",
    [
        {"base_stop_hold_s": 0},
        {"base_stop_timeout_s": 0.3},
        {"base_stop_linear_m_s": float("nan")},
        {"base_odom_max_age_s": -1},
    ],
)
def test_bad_stop_configuration_rejected(config):
    with pytest.raises(ValueError):
        CruzrConfig(**config)


@pytest.mark.parametrize(
    "final",
    [
        None,
        {"ok": True},
        {"ok": True, "reason": "reached", "settled": True, "stop_confirmation": "measured"},
        {"ok": False, "reason": "worker_stopped", "settled": False, "stop_confirmation": "unknown"},
        {"ok": True, "reason": "worker_stopped", "settled": True, "stop_confirmation": "measured"},
    ],
)
def test_resident_shutdown_requires_a_final_measured_stop_record(final):
    import json
    import sys

    from jiuwensymbiosis.ros2.worker import ResidentWorker

    final_line = json.dumps(final) if final is not None else ""
    # Actual subprocess/pipe protocol, without ROS or hardware. Every request
    # succeeds; only the final stop record varies, including no final reply.
    body = (
        "import sys\nfor line in sys.stdin:\n"
        "    if line.strip() == 'stop':\n"
        f"        print({final_line!r}, flush=True)\n        break\n"
        "    print('{\"ok\":true}', flush=True)\n"
    )
    worker = ResidentWorker(
        lambda: [sys.executable, "-u", "-c", body], label="[stop-test]", validate_stop=CruzrNav._validate_resident_stop
    )
    assert worker.request_json("move", 2.0, bad_output_reason="bad") == {"ok": True}
    if final and final.get("reason") == "worker_stopped" and final.get("settled"):
        worker.stop()
        worker.stop()
    else:
        with pytest.raises(HardwareCleanupError, match="resident shutdown"):
            worker.stop()
        assert worker.proc is None
        with pytest.raises(RuntimeError, match="shutdown remains unconfirmed"):
            worker.request_json("another move", 2.0, bad_output_reason="bad")
        with pytest.raises(RuntimeError, match="shutdown remains unconfirmed"):
            worker.stop()
