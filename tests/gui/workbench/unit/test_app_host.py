import gc
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from jiuwensymbiosis.agent.lifecycle import HardwareCleanupError
from jiuwensymbiosis.runtime import CleanupReport, Runtime
from jiuwensymbiosis_gui.workbench.app_host import AppHost
from jiuwensymbiosis_gui.workbench.app_state import AppState


def test_client_refresh_retains_owner_and_uses_new_subscription():
    host = AppHost()
    first, refreshed = AppState(host=host), AppState(host=host)
    engine = Mock()
    first.engine = engine
    assert first.engine is engine
    assert refreshed.engine is engine.subscribe.return_value
    assert host.current_engine is engine


def test_host_retains_tool_and_waits_for_real_completion():
    host = AppHost()
    tool = Mock()
    tool.is_running.return_value = True
    tool.can_dispose.return_value = False
    host.start_tool(tool)
    assert not host.close(timeout=0)["closed"]
    tool.stop.assert_called()
    tool.is_running.return_value = False
    tool.can_dispose.return_value = True
    assert host.close(timeout=0)["closed"]


def test_return_to_start_uses_driver_target_and_admission(tmp_path):
    runtime = Runtime(tmp_path / "ws", resource_directory=tmp_path / "locks")
    host = AppHost()
    session = MagicMock()
    session.cleanup_report.return_value = CleanupReport()
    binding = SimpleNamespace(resources=("device:fake",), build_session=lambda: session)
    events = []
    operation = host.return_to(runtime, binding, [1.0, 2.0], lambda kind, data: events.append((kind, data)))
    operation.join(2)
    session.env.low_level.move_joint_blocking.assert_called_once_with([1.0, 2.0])
    session.env.low_level.home.assert_not_called()
    assert events[-1] == ("pose_return_finished", {"ok": True})
    assert runtime.close()["closed"]
    runtime.store.close()


def test_completed_tools_are_collectable_after_host_poll(tmp_path):
    runtime = Runtime(tmp_path / "ws", resource_directory=tmp_path / "locks")
    host = AppHost()
    session = MagicMock()
    session.cleanup_report.return_value = CleanupReport()
    binding = SimpleNamespace(resources=("device:fake",), build_session=lambda: session)
    references = []
    try:
        for _ in range(3):
            operation = host.return_to(runtime, binding, [1.0], lambda *_: None)
            operation.join(2)
            assert not operation.is_running()
            references.append(weakref.ref(operation))
            del operation
            assert not host.is_busy()
        gc.collect()
        assert all(reference() is None for reference in references)
        assert host.close()["closed"]
    finally:
        runtime.close()
        runtime.store.close()


def test_return_construction_with_failed_rollback_keeps_resource_blocked(tmp_path):
    runtime = Runtime(tmp_path / "ws", resource_directory=tmp_path / "locks")
    host = AppHost()
    binding = SimpleNamespace(resources=("device:fake",), build_session=Mock(side_effect=HardwareCleanupError("open")))
    operation = host.return_to(runtime, binding, [1.0], lambda *_: None)
    operation.join(2)
    lease = operation._lease
    try:
        assert runtime.get_runtime_state()["blocked"]
        assert not runtime.close(timeout=0)["closed"]
        reference = weakref.ref(operation)
        del operation
        assert host.is_busy()
        assert not host.close(timeout=0)["closed"]
        gc.collect()
        assert reference() is not None  # an exited but blocked owner must survive
    finally:
        # The fake constructor acquired no hardware; restore the process slot.
        runtime.finish_maintenance(lease, CleanupReport())
        assert host.close()["closed"]
        runtime.store.close()
