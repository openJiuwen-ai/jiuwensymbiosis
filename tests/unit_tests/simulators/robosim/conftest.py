"""Each test represents a fresh application process, except within that test."""

import pytest

from jiuwensymbiosis.simulators.robosim import operations


@pytest.fixture(autouse=True)
def isolate_command_registry(monkeypatch):
    monkeypatch.setattr(operations, "_coordinators", {})
