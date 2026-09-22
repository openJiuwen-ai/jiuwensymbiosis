# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared maintenance admission and event-queue contracts (no hardware)."""

from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.runtime.resources import ResourceBusyError, ResourceManager
from jiuwensymbiosis_gui.workbench.maintenance import (
    BoundedEventQueue,
    MaintenanceOwner,
    cleanup_report_for,
)


class _RuntimeWithResources:
    def __init__(self, resources: ResourceManager) -> None:
        self.resources = resources
        self.stop_callbacks: list[Any] = []

    def acquire_maintenance(self, binding, operation, *, stop=None):
        self.stop_callbacks.append(stop)
        return self.resources.acquire(binding.resources, operation=operation)

    def finish_maintenance(self, lease, report):
        return self.resources.finish(lease, report)


def test_bounded_event_queue_reports_dropped_old_events() -> None:
    events = BoundedEventQueue(maxsize=2)
    events.put(("frame", 1))
    events.put(("frame", 2))
    events.put(("frame", 3))
    events.put(("frame", 4))

    assert events.get_nowait() == ("gap", {"dropped": 2})
    assert events.get_nowait() == ("frame", 3)
    assert events.get_nowait() == ("frame", 4)
    with pytest.raises(queue.Empty):
        events.get_nowait()


def test_bounded_event_queue_validates_capacity() -> None:
    with pytest.raises(ValueError, match="positive"):
        BoundedEventQueue(maxsize=0)


def test_maintenance_owner_uses_selected_config_and_excludes_sidecars(tmp_path: Path, monkeypatch) -> None:
    import jiuwensymbiosis.runtime.bindings as bindings_module

    source = tmp_path / "selected.yaml"
    source.write_text("adapter: fake\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    resources = ResourceManager(tmp_path / "leases")
    runtime = _RuntimeWithResources(resources)
    binding = SimpleNamespace(resources=("fake-device:1",))
    calls: list[dict[str, Any]] = []

    def prepare(config_source, **kwargs):
        calls.append({"source": config_source, **kwargs})
        return binding

    monkeypatch.setattr(bindings_module, "prepare_binding", prepare)
    owner = MaintenanceOwner(runtime, source, workspace)

    def stop() -> None:
        pass

    prepared, lease = owner.acquire({"adapter": "fake"}, "perception_preview", stop=stop)
    assert prepared is binding
    assert calls == [
        {
            "source": source.resolve(),
            "config_snapshot": {"adapter": "fake"},
            "workspace": workspace.resolve(),
            "include_sidecars": False,
        }
    ]
    assert runtime.stop_callbacks == [stop]
    assert owner.finish(lease, CleanupReport()) is True


def test_running_task_lease_rejects_maintenance_before_session_creation(tmp_path: Path, monkeypatch) -> None:
    import jiuwensymbiosis.runtime.bindings as bindings_module

    source = tmp_path / "selected.yaml"
    source.write_text("adapter: fake\n", encoding="utf-8")
    manager = ResourceManager(tmp_path / "leases")
    runtime = _RuntimeWithResources(manager)
    binding = SimpleNamespace(resources=("camera:serial-1", "robot:bus-1"))
    monkeypatch.setattr(bindings_module, "prepare_binding", lambda *_args, **_kwargs: binding)
    owner = MaintenanceOwner(runtime, source, tmp_path / "workspace")
    task_lease = manager.acquire(binding.resources, operation="task")
    try:
        with pytest.raises(ResourceBusyError):
            owner.acquire({"adapter": "fake"}, "hardware_guiding")
    finally:
        manager.finish(task_lease, CleanupReport())


def test_selected_body_supplies_legacy_adapter_and_rejects_mismatch(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from jiuwensymbiosis.runtime import bindings

    prepare = Mock(return_value=SimpleNamespace(resources=("device:fake",)))
    monkeypatch.setattr(bindings, "prepare_binding", prepare)
    runtime = Mock()
    owner = MaintenanceOwner(runtime, tmp_path / "old.yaml", tmp_path, adapter="so101")
    original = {"env": {}}
    owner.acquire(original, "test")
    assert prepare.call_args.kwargs["config_snapshot"]["adapter"] == "so101"
    assert "adapter" not in original
    with pytest.raises(ValueError, match="adapter"):
        owner.acquire({"adapter": "piper"}, "test")
    assert prepare.call_count == 1


def test_missing_public_cleanup_report_keeps_session_blocked() -> None:
    session = SimpleNamespace(disconnect=lambda: None)

    report = cleanup_report_for(session)

    assert not report.released
    assert report.connected is True
    assert "no public cleanup_report" in report.errors[0]
