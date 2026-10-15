# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Config names survive the split between hardware and simulation packages."""

from __future__ import annotations

import importlib
import sys

import pytest

from jiuwensymbiosis.backends import adapter_module_name, backend_module_name, session_target


@pytest.mark.parametrize("name", ["piper", "so101", "cruzr", "third_party_robot"])
def test_hardware_names_keep_the_adapter_convention(name):
    assert adapter_module_name(name) == f"jiuwensymbiosis.adapters.{name}"


def test_simulator_discovery_does_not_import_its_client(monkeypatch):
    # Collection may already have imported this package. Remove the cache entry
    # so an accidental eager import cannot be hidden by another test module.
    name = "jiuwensymbiosis.simulators.robosim"
    monkeypatch.delitem(sys.modules, name, raising=False)
    assert backend_module_name("robosim") == name
    assert name not in sys.modules


def test_existing_robosim_yaml_builds_through_cli(monkeypatch, tmp_path):
    from jiuwensymbiosis.introspect import build_session
    from jiuwensymbiosis.simulators.robosim import RobosimApi, RobosimEnv, build_robosim_session

    def forbidden(*args, **kwargs):
        raise AssertionError("Discovery and session construction must not connect to RoboSim")

    monkeypatch.setattr(RobosimEnv, "connect", forbidden)
    config = tmp_path / "moz1.yaml"
    config.write_text("backend: robosim\nside: right\nload_robot: false\n", encoding="utf-8")
    sessions = [
        build_session(str(config)),
        build_robosim_session.from_yaml(config),
    ]
    for session in sessions:
        assert isinstance(session.api, RobosimApi)
        assert isinstance(session.env, RobosimEnv)
        assert session.env.cfg.side == "right"
        assert session.env.cfg.load_robot is False


def test_unknown_backend_remains_unavailable():

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(backend_module_name("does_not_exist"))


def test_robosim_is_no_longer_in_the_hardware_namespace():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("jiuwensymbiosis.adapters.robosim")


@pytest.mark.parametrize("url", ["http://localhost:8001/", "http://127.0.0.1:8001", "http://[::1]:8001"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_runtime_binding_serializes_shared_service(url, side, tmp_path):
    from jiuwensymbiosis.runtime.bindings import prepare_binding

    config = tmp_path / "robosim.yaml"
    config.write_text(f"backend: robosim\nservice_url: '{url}'\nside: {side}\n")
    binding = prepare_binding(config)
    assert binding.resources == ("robosim:local:8001",)
    config.write_text("adapter: does_not_exist\n")
    assert binding.build_session().env.cfg.side == side


def test_shipped_robosim_config_uses_grouped_agent_schema():
    from pathlib import Path

    import yaml

    from jiuwensymbiosis.agent.config import RobotAgentConfig
    from jiuwensymbiosis.runtime.bindings import prepare_binding

    source = Path(__file__).resolve().parents[2] / "configs/simulators/moz1.yaml"
    data = yaml.safe_load(source.read_text())
    cfg = RobotAgentConfig.from_dict(data["agent"])
    assert not cfg.modules.visual_feedback.enabled
    assert prepare_binding(source).resources == ("robosim:local:8001",)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"adapter": "piper", "backend": "robosim"},
        {"backend": None},
        {"backend": ""},
        {"backend": "../piper"},
        {"adapter": "robosim.client"},
    ],
)
def test_session_selector_rejects_missing_ambiguous_or_invalid_values(data):
    with pytest.raises(ValueError):
        session_target(data)


@pytest.mark.parametrize("data", [{"adapter": "robosim"}, {"backend": "piper"}])
def test_session_selector_never_falls_back_to_the_other_namespace(tmp_path, data):
    import yaml

    from jiuwensymbiosis.runtime.bindings import prepare_binding

    source = tmp_path / "wrong-namespace.yaml"
    source.write_text(yaml.safe_dump(data))
    with pytest.raises(ModuleNotFoundError):
        prepare_binding(source)


def test_simulation_binding_preserves_backend_identity_and_snapshot(tmp_path):
    from jiuwensymbiosis.runtime.bindings import prepare_binding

    source = tmp_path / "simulation.yaml"
    source.write_text("backend: robosim\nside: right\nload_robot: false\nname: custom_sim\n")
    binding = prepare_binding(source)
    source.write_text("adapter: piper\n")
    maintenance = binding.without_sidecars()
    for frozen in [binding, maintenance]:
        assert frozen.backend == "robosim"
        assert frozen.adapter is None
        assert frozen.target.module_name == "jiuwensymbiosis.simulators.robosim"
        assert "adapter" not in frozen.config_data()
        session = frozen.build_session()
        assert session.name == "custom_sim"
        assert session.env.cfg.side == "right"
        assert not session.env.cfg.load_robot
    assert maintenance.resources == binding.resources == ("robosim:local:8001",)
