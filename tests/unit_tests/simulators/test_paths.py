"""Source and installed simulator defaults never depend on the launch directory."""

from pathlib import Path

import pytest

from jiuwensymbiosis.simulators import _paths


def test_installed_defaults_use_home_without_creating_files(tmp_path, monkeypatch):
    monkeypatch.delenv("JIUWENSYMBIOSIS_SIMULATOR_HOME", raising=False)
    monkeypatch.setattr(_paths, "__file__", str(tmp_path / "site-packages/jiuwensymbiosis/simulators/_paths.py"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user"))
    monkeypatch.chdir(tmp_path)
    assert _paths.simulator_root() == tmp_path / "user/.jiuwensymbiosis/simulators"
    assert not (tmp_path / "user").exists()


def test_source_checkout_preserves_existing_data_locations(monkeypatch, tmp_path):
    monkeypatch.delenv("JIUWENSYMBIOSIS_SIMULATOR_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    assert _paths.simulator_root() == Path(__file__).resolve().parents[3]


def test_explicit_root_is_shared_across_launch_directories(tmp_path, monkeypatch):
    root = tmp_path / "custom"
    monkeypatch.setenv("JIUWENSYMBIOSIS_SIMULATOR_HOME", str(root))
    monkeypatch.chdir(tmp_path)
    assert _paths.simulator_root() == root
    assert not root.exists()
    monkeypatch.setenv("JIUWENSYMBIOSIS_SIMULATOR_HOME", "relative")
    with pytest.raises(ValueError, match="absolute path"):
        _paths.simulator_root()
