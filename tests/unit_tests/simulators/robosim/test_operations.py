"""Command ownership and ambiguity survive client/thread lifecycle boundaries."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from jiuwensymbiosis.simulators.robosim.operations import CommandCoordinator, RobosimError, coordinator_for


def test_loopback_aliases_share_one_owner_and_threads_cannot_overlap():
    coordinator = coordinator_for("http://127.0.0.1:8001")
    assert coordinator_for("http://localhost:8001/") is coordinator
    assert coordinator_for("http://[::1]:8001") is coordinator
    assert coordinator_for("http://localhost:8002") is not coordinator

    def compete():
        with coordinator.command("other"):
            pytest.fail("Must not enter")

    with coordinator.command("move"), ThreadPoolExecutor(max_workers=1) as pool:
        with coordinator.command("nested HTTP"):
            assert coordinator.snapshot()["busy"]
        with pytest.raises(RobosimError) as exc:
            pool.submit(compete).result()
        assert exc.value.code == "robosim_busy"
    assert coordinator.snapshot()["command"]["status"] == "completed"


def test_catching_nested_timeout_does_not_clear_ambiguity():
    coordinator = CommandCoordinator()
    with coordinator.command("outer"):
        with pytest.raises(RobosimError), coordinator.command("inner"):
            raise RobosimError("timeout", code="transport", outcome_unknown=True)
    assert coordinator.snapshot()["write_blocked"]
    coordinator.observe_service("first-observed")
    assert coordinator.snapshot()["write_blocked"]
    coordinator.observe_service("first-observed")
    assert coordinator.snapshot()["write_blocked"]
    coordinator.observe_service("new-service")
    assert not coordinator.snapshot()["write_blocked"]


def test_restart_during_command_is_not_accepted_as_completion():
    coordinator = CommandCoordinator()
    coordinator.observe_service("original")
    with pytest.raises(RobosimError), coordinator.command("move"):
        coordinator.observe_service("restarted")
    assert coordinator.snapshot()["write_blocked"]


def test_interruption_is_reraised_and_blocks_later_writes():
    coordinator = CommandCoordinator()
    with pytest.raises(KeyboardInterrupt), coordinator.command("move"):
        raise KeyboardInterrupt
    assert coordinator.snapshot()["write_blocked"]
    assert not coordinator.snapshot()["busy"]


def test_readback_failure_after_accepted_write_is_unknown_even_when_nested():
    coordinator = CommandCoordinator()
    with pytest.raises(RobosimError) as exc:
        with coordinator.command("motion"):
            with coordinator.command("POST"):
                coordinator.accept_write()
            with coordinator.command("readback"):
                raise ValueError("missing measurement")
    assert exc.value.outcome_unknown and exc.value.skip_recovery
    assert isinstance(exc.value.__cause__, ValueError)
    assert coordinator.snapshot()["write_blocked"]


def test_completed_write_does_not_poison_next_commands_preflight_failure():
    coordinator = CommandCoordinator()
    with coordinator.command("completed motion"):
        coordinator.accept_write()
    with pytest.raises(ValueError), coordinator.command("invalid next target"):
        raise ValueError("invalid target")
    assert not coordinator.snapshot()["write_blocked"]
    with coordinator.command("next valid command"):
        coordinator.accept_write()
    assert coordinator.snapshot()["command"]["status"] == "completed"
