"""Simulator discovery follows the workbench plugin boundary."""

import pytest

from jiuwensymbiosis.simulators.robosim import RobosimEnv
from jiuwensymbiosis_gui.workbench import registry


def test_robosim_registry_builds_without_connecting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Discovery must not connect to RoboSim")

    monkeypatch.setattr(RobosimEnv, "connect", forbidden)
    body = registry._body_from_dict({"key": "moz1", "backend": "robosim"})
    assert body is not None
    assert body.adapter is None
    assert body.backend == "robosim"
    assert body.target.module_name == "jiuwensymbiosis.simulators.robosim"
    session = body.build_real_session({"side": "right", "load_robot": False}, include_sidecars=False)
    assert isinstance(session.env, RobosimEnv)
    assert session.env.cfg.side == "right"
    assert session.env.cfg.load_robot is False


def test_missing_backend_is_unavailable():
    assert registry._body_from_dict({"key": "missing", "adapter": "does_not_exist"}) is None


def test_simulator_cannot_be_registered_as_a_hardware_adapter():
    assert registry._body_from_dict({"key": "moz1", "adapter": "robosim"}) is None
    assert registry._body_from_dict({"key": "moz1", "adapter": "piper", "backend": "robosim"}) is None


def test_workbench_run_config_keeps_backend_and_rejects_hardware_conflicts(monkeypatch):
    from jiuwensymbiosis_gui.workbench.config_model import ConfigModel
    from jiuwensymbiosis_gui.workbench.run_engine import RunEngine

    body = registry._body_from_dict({"key": "moz1", "backend": "robosim"})
    assert body is not None
    monkeypatch.setitem(registry._BODIES, "moz1", body)
    engine = RunEngine.__new__(RunEngine)
    engine._body_key = "moz1"
    engine._config = ConfigModel.from_dict({"backend": "robosim", "side": "right"})
    assert engine._real_session_config() == {"backend": "robosim", "side": "right"}
    with pytest.raises(ValueError):
        body.build_real_session({"adapter": "piper"})
    with pytest.raises(ValueError, match="硬件标定"):
        body.adapter_module()
    engine._config = ConfigModel.from_dict({"adapter": "robosim"})
    with pytest.raises(ValueError):
        engine._real_session_config()
