# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Config names survive the split between hardware and simulation packages."""

from __future__ import annotations

import importlib
import sys

import pytest

from jiuwensymbiosis.backends import backend_module_name


@pytest.mark.parametrize("name", ["piper", "so101", "cruzr", "third_party_robot"])
def test_hardware_names_keep_the_adapter_convention(name):
    assert backend_module_name(name) == f"jiuwensymbiosis.adapters.{name}"


def test_simulator_discovery_does_not_import_its_client(monkeypatch):
    # Collection may already have imported this package. Remove the cache entry
    # so an accidental eager import cannot be hidden by another test module.
    name = "jiuwensymbiosis.simulators.robosim"
    monkeypatch.delitem(sys.modules, name, raising=False)
    assert backend_module_name("robosim") == name
    assert name not in sys.modules


def test_existing_robosim_yaml_builds_through_cli_and_gui(monkeypatch, tmp_path):
    from jiuwensymbiosis.gui import registry
    from jiuwensymbiosis.introspect import build_session
    from jiuwensymbiosis.simulators.robosim import RobosimApi, RobosimEnv, build_robosim_session

    def forbidden(*args, **kwargs):
        raise AssertionError("Discovery and session construction must not connect to RoboSim")

    monkeypatch.setattr(RobosimEnv, "connect", forbidden)
    config = tmp_path / "moz1.yaml"
    config.write_text("adapter: robosim\nside: right\nload_robot: false\n", encoding="utf-8")
    body = registry._body_from_dict({"key": "moz1", "adapter": "robosim"})
    assert body is not None
    assert body.adapter_module() == "jiuwensymbiosis.simulators.robosim"
    sessions = [
        build_session(str(config)),
        body.build_real_session({"adapter": "robosim", "side": "right", "load_robot": False}, include_sidecars=False),
        build_robosim_session.from_yaml(config),
    ]
    for session in sessions:
        assert isinstance(session.api, RobosimApi)
        assert isinstance(session.env, RobosimEnv)
        assert session.env.cfg.side == "right"
        assert session.env.cfg.load_robot is False


def test_unknown_backend_remains_unavailable():
    from jiuwensymbiosis.gui import registry

    assert registry._body_from_dict({"key": "missing", "adapter": "does_not_exist"}) is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(backend_module_name("does_not_exist"))


def test_robosim_is_no_longer_in_the_hardware_namespace():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("jiuwensymbiosis.adapters.robosim")
