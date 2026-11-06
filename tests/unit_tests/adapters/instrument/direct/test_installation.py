# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under Apache-2.0.

"""Instrument modules install in the root wheel without a second contract or planner."""

from __future__ import annotations

import configparser
import email
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_REPO = Path(__file__).resolve().parents[5]
_LEGACY_DISTRIBUTIONS = {"symbiosis-core", "scientific-instruments", "instrument-onboarding"}
_ROBOT_CONSOLE_SCRIPTS = {
    "jiuwensymbiosis-run": "jiuwensymbiosis.cli:run_task",
    "piper-pick-demo": "jiuwensymbiosis.cli:piper_pick_demo",
    "jiuwensymbiosis-replay": "jiuwensymbiosis.cli:replay_main",
    "jiuwensymbiosis-gui": "jiuwensymbiosis_gui.__main__:main",
    "jiuwensymbiosis-calibrate-hand-eye": "scripts.calibrate.hand_eye_calib:main",
    "jiuwensymbiosis-calibrate-eye-in-hand": "scripts.calibrate.eye_in_hand_calib:main",
    "jiuwensymbiosis-calibrate-eye-to-hand": "scripts.calibrate.eye_to_hand_calib:main",
    "jiuwensymbiosis-actions": "jiuwensymbiosis.cli:actions_main",
    "jiuwensymbiosis-skills": "jiuwensymbiosis.cli:skills_main",
    "jiuwensymbiosis-state": "jiuwensymbiosis.cli:state_main",
}


def _run(arguments, *, cwd, environment):
    result = subprocess.run(
        arguments, cwd=cwd, env=environment, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def installed_root_wheel(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("installed-root-wheel")
    source = temporary / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(_REPO / name, source / name)
    for name in ("jiuwensymbiosis", "jiuwensymbiosis_gui", "examples", "scripts"):
        shutil.copytree(
            _REPO / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
        )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--wheel-dir",
            str(temporary / "dist"),
            str(source),
        ],
        cwd=temporary,
        environment=environment,
    )
    (wheel,) = (temporary / "dist").glob("*.whl")
    installed = temporary / "installed"
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--ignore-installed",
            "--target",
            str(installed),
            str(wheel),
        ],
        cwd=temporary,
        environment=environment,
    )
    # The installed modules must not resolve from an editable source tree.
    shutil.rmtree(source)
    return temporary, wheel, installed, environment


def test_root_wheel_has_one_distribution_and_instruments_extra(installed_root_wheel):
    _, wheel, _, _ = installed_root_wheel
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        distributions = {name.split("/", 1)[0] for name in names if ".dist-info/" in name}
        assert len(distributions) == 1
        distribution = distributions.pop()
        metadata = email.message_from_bytes(archive.read(distribution + "/METADATA"))
        assert metadata["Name"] == "jiuwensymbiosis"
        for module in ("common", "direct", "mcp", "direct/onboarding"):
            assert f"jiuwensymbiosis/adapters/instrument/{module}/__init__.py" in names
        for prefix in (
            "symbiosis_core/",
            "scientific_instruments/",
            "instrument_onboarding/",
            "jiuwensymbiosis/action_core/",
            "jiuwensymbiosis/instruments/",
            "jiuwensymbiosis/onboarding/",
            "jiuwensymbiosis/adapters/instrument_mcp/",
            "jiuwensymbiosis/adapters/mixed/",
        ):
            assert not any(name.startswith(prefix) for name in names)
        requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
        assert not ({canonicalize_name(requirement.name) for requirement in requirements} & _LEGACY_DISTRIBUTIONS)
        assert "instruments" in metadata.get_all("Provides-Extra", [])
        optional = {
            canonicalize_name(requirement.name): requirement
            for requirement in requirements
            if requirement.marker is not None
            and requirement.marker.evaluate({"extra": "instruments"})
            and not requirement.marker.evaluate({"extra": ""})
        }
        assert {"jsonschema", "pyserial", "pydantic"} <= optional.keys()
        assert "2.9" not in optional["pydantic"].specifier
        assert "2.10" in optional["pydantic"].specifier
        project = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        base = {str(requirement) for requirement in requirements if requirement.marker is None}
        assert base == {str(Requirement(value)) for value in project["dependencies"]}
        entries = configparser.ConfigParser()
        entries.read_string(archive.read(distribution + "/entry_points.txt").decode())
        assert dict(entries["console_scripts"]) == _ROBOT_CONSOLE_SCRIPTS


def test_installed_layers_and_old_robot_exports_resolve_from_root_wheel(installed_root_wheel):
    temporary, _, installed, environment = installed_root_wheel
    probe = r"""
import importlib.abc
import importlib.metadata
from importlib.machinery import PathFinder
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
Path.home = classmethod(lambda cls: root.parent / "user")
class RejectLegacyNamespaces(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split(".", 1)[0] not in {"symbiosis_core", "scientific_instruments", "instrument_onboarding"}, fullname
sys.meta_path.insert(0, RejectLegacyNamespaces())
import jiuwensymbiosis
from jiuwensymbiosis.adapters.instrument import build_instrument_session
from jiuwensymbiosis.adapters.instrument import direct
from jiuwensymbiosis.adapters.instrument.direct import onboarding
from jiuwensymbiosis.adapters.instrument.direct import action_registry
from jiuwensymbiosis.adapters.instrument.direct.action_registry import InstrumentOperation
from jiuwensymbiosis.api.actions import implements as robot_implements
from jiuwensymbiosis.api.decorators import ActionSpec as RobotActionSpec
from jiuwensymbiosis.adapters.instrument.direct.session import InstrumentSession
from jiuwensymbiosis.adapters.instrument.direct.onboarding import prepare_candidate
for module in (jiuwensymbiosis, direct, onboarding, action_registry):
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
assert jiuwensymbiosis.ActionSpec is RobotActionSpec
assert jiuwensymbiosis.implements is robot_implements
assert not hasattr(action_registry, "ActionSpec")
assert not hasattr(direct, "ActionSpec")
assert InstrumentOperation.__module__ == "jiuwensymbiosis.adapters.instrument.direct.action_registry"
assert PathFinder.find_spec("jiuwensymbiosis.action_core", jiuwensymbiosis.__path__) is None
assert PathFinder.find_spec("jiuwensymbiosis.adapters.instrument.direct.workflows", direct.__path__) is None
assert callable(InstrumentSession) and callable(prepare_candidate)
assert callable(build_instrument_session.config_factory.from_dict)
for removed in ("instruments", "onboarding", "adapters.instrument_mcp", "adapters.mixed", "agent.mixed_session"):
    import importlib.util
    assert importlib.util.find_spec("jiuwensymbiosis." + removed) is None
dist = importlib.metadata.distribution("jiuwensymbiosis")
assert Path(dist.locate_file("jiuwensymbiosis/adapters/instrument/__init__.py")).is_relative_to(root)
"""
    _run([sys.executable, "-I", "-c", probe, str(installed)], cwd=temporary, environment=environment)
