"""Startup evidence comes from the active SDK config, without requiring a GPU."""

import json
import logging
import subprocess
import sys
import textwrap
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jiuwensymbiosis.simulators.genesis import startup_diagnostics as diagnostics
from jiuwensymbiosis.simulators.genesis.preview import _WORKER


@pytest.fixture
def cache_config():
    return SimpleNamespace(
        offline_cache=True,
        offline_cache_file_path="/actual/quadrants/cache",
        offline_cache_max_size_of_files=104857600,
        offline_cache_cleaning_policy="lru",
        offline_cache_cleaning_factor=0.25,
    )


def quadrants(cfg):
    return SimpleNamespace(lang=SimpleNamespace(impl=SimpleNamespace(current_cfg=Mock(return_value=cfg))))


def test_cache_snapshot_reads_active_config_instead_of_requested_environment(cache_config, monkeypatch):
    monkeypatch.setenv("QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES", "2147483647")
    monkeypatch.setenv("QD_OFFLINE_CACHE_FILE_PATH", "/requested/cache")
    sdk = quadrants(cache_config)
    result = diagnostics.quadrants_cache_config(sdk)
    sdk.lang.impl.current_cfg.assert_called_once_with()
    assert result == {"available": True, "source": "quadrants.lang.impl.current_cfg", **vars(cache_config)}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "failure", [AttributeError("missing"), RuntimeError("not initialized"), OSError("SDK readback")]
)
def test_cache_snapshot_reports_unavailable_instead_of_inventing_values(failure):
    sdk = quadrants(None)
    sdk.lang.impl.current_cfg.side_effect = failure
    assert diagnostics.quadrants_cache_config(sdk) == {"available": False, "reason": type(failure).__name__}


def test_package_versions_reports_missing_distribution_without_importing_it(monkeypatch):
    def version(name):
        if name == "quadrants":
            raise PackageNotFoundError(name)
        return "test-version"

    monkeypatch.setattr(diagnostics, "version", version)
    assert diagnostics.package_versions() == {
        "genesis-world": "test-version",
        "quadrants": None,
        "gs-nyx": "test-version",
        "gs-nyx-plugin": "test-version",
        "torch": "test-version",
    }


def test_unreadable_distribution_metadata_does_not_prevent_startup(monkeypatch):
    def version(name):
        if name == "quadrants":
            raise OSError("metadata is temporarily unreadable")
        return "test-version"

    monkeypatch.setattr(diagnostics, "version", version)
    startup = diagnostics.StartupDiagnostics(logger=Mock())
    result = startup.snapshot()
    assert result["versions"]["quadrants"] is None
    assert result["versions"]["genesis-world"] == "test-version"
    assert result["state"] == "running"


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(1)])
def test_optional_diagnostics_preserve_process_interruption(error, monkeypatch):
    sdk = quadrants(None)
    sdk.lang.impl.current_cfg.side_effect = error
    with pytest.raises(type(error)):
        diagnostics.quadrants_cache_config(sdk)
    monkeypatch.setattr(diagnostics, "version", Mock(side_effect=error))
    with pytest.raises(type(error)):
        diagnostics.package_versions()


def test_worker_startup_logs_survive_genesis_sdk_warning_level(monkeypatch, caplog):
    monkeypatch.syspath_prepend(str(_WORKER.parent))
    import worker

    # The real SDK changes its own namespace's level in gs.init. Leave the
    # worker logger inherited, so placing it under genesis would hide INFO.
    monkeypatch.setattr(logging.getLogger("genesis"), "level", logging.WARNING)
    for name in ("jiuwensymbiosis", "jiuwensymbiosis.simulators", "jiuwensymbiosis.simulators.genesis"):
        monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)
        monkeypatch.setattr(logging.getLogger(name), "propagate", True)
    caplog.set_level(logging.INFO)
    startup = worker.StartupDiagnostics(logger=worker.logger)
    startup.finish()
    assert any(
        record.name == "jiuwensymbiosis.simulators.genesis.worker" and "Genesis startup summary" in record.message
        for record in caplog.records
    )


def test_phases_publish_progress_preserve_nested_timings_and_freeze_at_completion(cache_config, monkeypatch, caplog):
    versions = Mock(return_value={"genesis-world": "test-version"})
    monkeypatch.setattr(diagnostics, "package_versions", versions)
    clock = [10.0]
    updates = []
    logger = logging.getLogger("startup_diagnostics_test")
    caplog.set_level(logging.INFO, logger=logger.name)
    startup = diagnostics.StartupDiagnostics(logger=logger, publish=updates.append, clock=lambda: clock[0])
    sdk = quadrants(cache_config)
    startup.collect_runtime(sdk)
    with startup.phase("runtime_scene"):
        clock[0] += 1
        with startup.phase("scene_build"):
            assert updates[-1]["current_phase"] == "scene_build"
            clock[0] += 2
        assert updates[-1]["current_phase"] == "runtime_scene"
        clock[0] += 1
    result = startup.finish()
    assert result["state"] == "complete"
    assert result["current_phase"] is None
    assert result["phases_ms"] == {"scene_build": 2000.0, "runtime_scene": 4000.0}
    assert result["total_ms"] == 4000.0
    assert result["quadrants_cache"]["offline_cache_max_size_of_files"] == 104857600
    clock[0] += 100
    assert startup.finish() == result
    assert startup.snapshot() == result
    result["phases_ms"]["scene_build"] = -1
    assert startup.snapshot()["phases_ms"]["scene_build"] == 2000.0
    versions.assert_called_once()
    sdk.lang.impl.current_cfg.assert_called_once()
    assert "Genesis startup phase scene_build: 2000.000 ms" in caplog.text
    json.dumps(startup.snapshot(), allow_nan=False)


def test_failed_nested_phase_retains_partial_timing_and_original_exception(monkeypatch):
    monkeypatch.setattr(diagnostics, "package_versions", lambda: {})
    clock = [0.0]
    startup = diagnostics.StartupDiagnostics(logger=Mock(), clock=lambda: clock[0])
    error = RuntimeError("scene failure")
    with pytest.raises(RuntimeError) as caught, startup.phase("runtime_scene"), startup.phase("scene_build"):
        clock[0] = 3
        raise error
    assert caught.value is error
    result = startup.finish()
    assert result["state"] == "failed"
    assert result["failed_phase"] == "scene_build"
    assert result["phases_ms"] == {"scene_build": 3000.0, "runtime_scene": 3000.0}
    assert result["quadrants_cache"] == {"available": False, "reason": "not_initialized"}


@pytest.mark.parametrize("already_ready", [False, True])
def test_unscoped_failure_only_marks_an_unfinished_startup_failed(monkeypatch, already_ready):
    monkeypatch.setattr(diagnostics, "package_versions", lambda: {})
    startup = diagnostics.StartupDiagnostics(logger=Mock())
    if already_ready:
        previous = startup.finish()
    result = startup.finish(error=True)
    assert result["state"] == ("complete" if already_ready else "failed")
    assert result["failed_phase"] == (None if already_ready else "unscoped")
    if already_ready:
        assert result == previous


def test_first_rgb_and_completed_total_are_stable_without_republishing_starting(monkeypatch):
    monkeypatch.setattr(diagnostics, "package_versions", lambda: {})
    clock = [10.0]
    updates = []
    startup = diagnostics.StartupDiagnostics(logger=Mock(), publish=updates.append, clock=lambda: clock[0])
    assert startup.snapshot()["first_frame_ms"] is None
    clock[0] = 15
    startup.mark_first_frame()
    ready = startup.finish()
    assert ready["start_boundary"] == "run_preview_before_sdk_imports"
    assert ready["end_boundary"] == "first_rgb_frame_completed"
    result = startup.snapshot()
    assert result["first_frame_ms"] == 5000.0
    assert result["total_ms"] == 5000.0
    assert result["state"] == "complete"
    assert updates == []
    clock[0] = 20
    startup.mark_first_frame()
    assert startup.snapshot() == result


@pytest.mark.parametrize(
    "mode",
    ["ready", "build_failure", "quality_change", "unscoped_before_ready", "unscoped_after_ready", "delayed_frame"],
)
def test_worker_publishes_startup_evidence_without_gpu(tmp_path, mode):
    # Exercise the real worker loop in an isolated interpreter. Fake only the
    # GPU-owned resources so optional imports cannot initialize a real device.
    code = f"import sys\nsys.path.insert(0, {str(_WORKER.parent)!r})\nMODE = {mode!r}\n"
    code += textwrap.dedent(
        """
        import json
        import logging
        from dataclasses import asdict
        from pathlib import Path
        from types import ModuleType, SimpleNamespace as NS
        import worker
        from camera import OrbitCamera, PreviewSettings
        logging.basicConfig(level=logging.INFO)

        cfg = NS(offline_cache=True, offline_cache_file_path='/actual/cache',
                 offline_cache_max_size_of_files=123456, offline_cache_cleaning_policy='lru',
                 offline_cache_cleaning_factor=0.25)
        calls = []
        def init(**kwargs):
            calls.append('init')
            logging.getLogger('genesis').setLevel(logging.WARNING)
        sys.modules['genesis'] = NS(cuda='fake', init=init,
                                    destroy=lambda: calls.append('destroy'))
        nyx = ModuleType('gs_nyx')
        nyx.nyx_py_sdk = NS(startup=lambda: calls.append('nyx_startup'),
                            shutdown=lambda: calls.append('nyx_shutdown'))
        sys.modules['gs_nyx'] = nyx
        sys.modules['gs_nyx.nyx_py_sdk'] = nyx.nyx_py_sdk
        def current_cfg():
            calls.append('current_cfg')
            return cfg
        sys.modules['quadrants'] = NS(lang=NS(impl=NS(current_cfg=current_cfg)))
        sys.modules['torch'] = NS(cuda=NS(current_stream=lambda: NS(synchronize=lambda: None)))

        class Runtime:
            def __init__(self, settings, directory, *, startup):
                with startup.phase('scene_build'):
                    if MODE == 'build_failure':
                        raise RuntimeError('test build failure')
                self.settings = settings
                self.info = {'mode': 'static_background', 'instance_id': 'test'}
                self.pose = self.initial_pose = OrbitCamera().snapshot()
                self.renderer = NS(read_rgb=lambda name: None, fov_y=60, render_counts={})
                self.robot = self.motion = self.indoor = None
                self.cameras = {}
                self.bodies = []
                self.steps = self.cycle = 0
            def frame_size(self, name): return (320, 240)
            def step(self): self.steps += 1
            def selection_snapshot(self): return None
            def close(self): calls.append('runtime_close')
            def close_renderer(self): calls.append('renderer_close')
            def build_renderer(self, settings): calls.append('renderer_build')

        class Mailbox:
            def __init__(self): self.states = []
            def read_control(self):
                ready = any(s['state'] == 'ready' for s in self.states)
                settings = asdict(PreviewSettings())
                revision = 0
                if MODE == 'quality_change' and ready:
                    settings['spp'] = 2
                    revision = 1
                return {'settings': settings, 'camera': OrbitCamera().snapshot(),
                        'stream_epoch': 0, 'streaming': False, 'keyframe_request': 0,
                        'config_revision': revision}
            def write_status(self, state): self.states.append(state)
            def stopping(self):
                ready_count = sum(s['state'] == 'ready' for s in self.states)
                return (any(s['state'] == 'failed' for s in self.states)
                        or ready_count >= (2 if MODE in {'quality_change', 'unscoped_after_ready', 'delayed_frame'} else 1))

        class Clock(worker.PhysicsClock):
            def run_batch(self, *args, **kwargs):
                ready = any(s['state'] == 'ready' for s in mailbox.states)
                if MODE == 'unscoped_before_ready' or (MODE == 'unscoped_after_ready' and ready):
                    raise RuntimeError('test failure outside startup phases')
                return super().run_batch(*args, **kwargs)
            def render_ready(self):
                if MODE == 'delayed_frame' and not any(s.get('rendered_frames') == 0 for s in mailbox.states):
                    return False
                return super().render_ready()

        worker.RuntimeScene = Runtime
        worker.NvEncoder = lambda *a, **kw: NS(close=lambda: calls.append('encoder_close'))
        worker.PhysicsClock = Clock
        mailbox = Mailbox()
        worker.run_preview(mailbox, Path.cwd())
        assert 'destroy' in calls and 'nyx_shutdown' in calls
        assert calls.count('current_cfg') == 1
        assert any(s.get('startup', {}).get('current_phase') == 'scene_build' for s in mailbox.states)
        if MODE == 'quality_change':
            ready = [s for s in mailbox.states if s['state'] == 'ready']
            assert ready[0]['startup'] == ready[-1]['startup']
            assert ready[-1]['config_revision'] == 1
            assert calls.count('renderer_build') == 1
        if MODE == 'delayed_frame':
            ready = [s for s in mailbox.states if s['state'] == 'ready']
            waiting = [s for s in mailbox.states if s.get('rendered_frames') == 0]
            assert waiting and all(s['state'] == 'starting' for s in waiting)
            assert all(s['startup']['state'] == 'running' for s in waiting)
            assert ready[0]['startup']['first_frame_ms'] is not None
            assert ready[0]['startup']['total_ms'] >= ready[0]['startup']['first_frame_ms']
            assert ready[0]['startup']['total_ms'] == ready[-1]['startup']['total_ms']
            assert not any(s['state'] == 'starting' for s in mailbox.states[mailbox.states.index(ready[0]) + 1:])
        print(json.dumps(mailbox.states[-1], allow_nan=False))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True, timeout=30
    )
    state = json.loads(result.stdout)
    assert "jiuwensymbiosis.simulators.genesis.worker:Genesis startup summary" in result.stderr
    failed = mode in {"build_failure", "unscoped_before_ready", "unscoped_after_ready"}
    assert state["state"] == ("failed" if failed else "ready")
    startup = state["startup"]
    assert startup["state"] == ("failed" if mode in {"build_failure", "unscoped_before_ready"} else "complete")
    assert startup["failed_phase"] == {"build_failure": "scene_build", "unscoped_before_ready": "unscoped"}.get(mode)
    assert startup["quadrants_cache"]["offline_cache_max_size_of_files"] == 123456
    assert {"imports", "genesis_init", "nyx_startup", "runtime_scene", "scene_build"} <= startup["phases_ms"].keys()
    assert all(ms >= 0 for ms in startup["phases_ms"].values())
    assert "genesis-world" in startup["versions"]
