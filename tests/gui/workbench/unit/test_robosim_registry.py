"""Simulator discovery follows the workbench plugin boundary."""

from jiuwensymbiosis.simulators.robosim import RobosimEnv
from jiuwensymbiosis_gui.workbench import registry


def test_robosim_registry_builds_without_connecting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Discovery must not connect to RoboSim")

    monkeypatch.setattr(RobosimEnv, "connect", forbidden)
    body = registry._body_from_dict({"key": "moz1", "adapter": "robosim"})
    assert body is not None
    assert body.adapter_module() == "jiuwensymbiosis.simulators.robosim"
    session = body.build_real_session({"side": "right", "load_robot": False}, include_sidecars=False)
    assert isinstance(session.env, RobosimEnv)
    assert session.env.cfg.side == "right"
    assert session.env.cfg.load_robot is False


def test_missing_backend_is_unavailable():
    assert registry._body_from_dict({"key": "missing", "adapter": "does_not_exist"}) is None
