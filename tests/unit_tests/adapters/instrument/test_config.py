"""Single access selection, offline assembly and frozen source paths."""

import copy
import json
import subprocess
import sys

import pytest
import yaml

from jiuwensymbiosis.adapters.instrument import InstrumentConfig, build_instrument_session
from jiuwensymbiosis.adapters.instrument.direct.adapter_config import DirectConfig
from jiuwensymbiosis.runtime import prepare_binding
from tests.unit_tests.runtime.test_instrument_assembly import deployment


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["instrument"].pop("access"),
        lambda d: d["instrument"].update(access="auto"),
        lambda d: d["instrument"].update(transport="http_json"),
        lambda d: d["instrument"].update(authority="another-owner"),
        lambda d: d.update(instruments={}),
        lambda d: d.update(instrument_permissions={}),
        lambda d: d.update(instrument_mcp={}),
        lambda d: d.update(adapters=[]),
        lambda d: d.update(adapter="mixed"),
        lambda d: d.update(adapter="instrument_mcp"),
    ],
)
def test_invalid_configuration_rejected_before_construction(tmp_path, monkeypatch, change):
    data, _ = deployment()
    change(data)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(DirectConfig, "from_dict", lambda *_: pytest.fail("invalid branch was parsed"))
    with pytest.raises(ValueError):
        prepare_binding(path, workspace=tmp_path / "work")
    with pytest.raises(ValueError):
        InstrumentConfig.from_dict(data)


@pytest.mark.parametrize("selector", [None, "piper", "instrument"])
def test_adapter_list_is_rejected_even_alongside_a_single_selector(tmp_path, selector):
    data = {"adapters": [{"adapter": "instrument"}]}
    if selector is not None:
        data["adapter"] = selector
    path = tmp_path / "list.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="multiple adapters"):
        prepare_binding(path)


def test_yaml_and_dict_have_identical_defaults_and_resource_identity(tmp_path):
    data, _ = deployment()
    path = tmp_path / "instrument.yaml"
    path.write_text(yaml.safe_dump(data))
    assert InstrumentConfig.from_yaml(path) == InstrumentConfig.from_dict(data)
    config = InstrumentConfig.from_dict(data)
    assert build_instrument_session.resource_keys(config) == ("physical:synthetic-meter-001",)
    assert not hasattr(build_instrument_session(config), "env")


def test_duplicate_generated_action_names_are_rejected(monkeypatch):
    data, _ = deployment()
    monkeypatch.setattr("jiuwensymbiosis.adapters.instrument.common.naming.tool_name", lambda *_: "collision")
    with pytest.raises(ValueError, match="collision"):
        InstrumentConfig.from_dict(data)


def test_selected_direct_paths_are_normalized_once_and_frozen(tmp_path, monkeypatch):
    data, _ = deployment()
    from tests.unit_tests.adapters.instrument.direct.test_contracts_mapping import config

    installation = config("./serial/controller", transport="serial_line")
    installation["devices"][0]["physical_device_id"] = "synthetic-meter-001"
    data["instrument"]["installation"] = installation
    source = tmp_path / "instrument.yaml"
    source.write_text(yaml.safe_dump(data))
    original = DirectConfig.from_dict
    calls = []

    def counted(value):
        calls.append(copy.deepcopy(value))
        return original(value)

    monkeypatch.setattr(DirectConfig, "from_dict", counted)
    binding = prepare_binding(source, workspace=tmp_path / "work")
    source.write_text("not the frozen config")
    session = binding.build_session()
    binding.without_sidecars().build_session()
    assert len(calls) == 1
    assert session._config["devices"][0]["connection"]["port"] == str(tmp_path / "serial/controller")
    assert binding.resources == ("physical:synthetic-meter-001",)


def test_direct_never_imports_project_mcp_access(tmp_path):
    data, _ = deployment()
    probe = """
import importlib.abc, json, sys
class RejectMCP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("jiuwensymbiosis.adapters.instrument.mcp"):
            raise AssertionError("direct imported MCP access: " + fullname)
sys.meta_path.insert(0, RejectMCP())
from jiuwensymbiosis.adapters.instrument import build_instrument_session
factory = build_instrument_session.config_factory
cfg = factory.from_dict(json.loads(sys.argv[1]))
assert cfg.access == "direct"
assert build_instrument_session.resource_keys(cfg) == ("physical:synthetic-meter-001",)
assert not hasattr(build_instrument_session(cfg), "env")
"""
    run = subprocess.run([sys.executable, "-c", probe, json.dumps(data)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr


def test_mcp_isolated_offline_factory_is_parsed_once_and_frozen(tmp_path):
    data = {
        "adapter": "instrument",
        "instrument": {
            "access": "mcp",
            "physical_device_id": "serial-42",
            "authority": "fixture-host",
            "transport": "stdio",
            "command": sys.executable,
            "tools": [{"name": "read", "read_only": True}],
        },
    }
    source = tmp_path / "remote.yaml"
    source.write_text(yaml.safe_dump(data))
    probe = """
import importlib.abc, json, sys
from pathlib import Path
class RejectDirect(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("jiuwensymbiosis.adapters.instrument.direct"):
            raise AssertionError("MCP imported direct access: " + fullname)
sys.meta_path.insert(0, RejectDirect())
from jiuwensymbiosis.adapters.instrument import build_instrument_session
from jiuwensymbiosis.adapters.instrument.mcp.config import InstrumentMCPConfig
from jiuwensymbiosis.adapters.instrument.mcp.client import MCPConnection
from jiuwensymbiosis.runtime import prepare_binding
def no_connect(self):
    raise AssertionError("offline assembly connected to a device")
MCPConnection.connect = no_connect
source = Path(sys.argv[1])
data = json.loads(sys.argv[2])
factory = build_instrument_session.config_factory
expected = factory.from_dict(data)
assert factory.from_yaml(source) == expected
calls = []
original = InstrumentMCPConfig.from_dict
def counted(cls, value):
    calls.append(value)
    return original(value)
InstrumentMCPConfig.from_dict = classmethod(counted)
binding = prepare_binding(source, workspace=source.parent / "work")
source.write_text("configuration changed after freezing")
assert binding.build_session().config == expected.selected
assert binding.without_sidecars().build_session().config == expected.selected
assert binding.resources == ("remote-owner:fixture-host:physical:serial-42",)
assert len(calls) == 1
"""
    run = subprocess.run(
        [sys.executable, "-c", probe, str(source), json.dumps(data)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize("extra", [{"installation": {}}, {"permissions": {}}, {"physical_device_id": "root"}])
def test_mcp_rejects_direct_fields_and_root_physical_identity(extra):
    data = {"adapter": "instrument", "instrument": {"access": "mcp"}}
    if "physical_device_id" in extra:
        data.update(extra)
    else:
        data["instrument"].update(extra)
    with pytest.raises(ValueError):
        InstrumentConfig.from_dict(data)


def test_mcp_command_resolves_from_source_without_losing_venv_symlink(tmp_path):
    executable = tmp_path / "python-link"
    executable.symlink_to(sys.executable)
    data = {
        "adapter": "instrument",
        "instrument": {
            "access": "mcp",
            "physical_device_id": "serial-42",
            "authority": "fixture-host",
            "transport": "stdio",
            "command": "./python-link",
            "tools": [{"name": "read", "read_only": True}],
        },
    }
    source = tmp_path / "remote.yaml"
    source.write_text(yaml.safe_dump(data))
    config = InstrumentConfig.from_yaml(source)
    assert config.selected.command == str(executable)
    binding = prepare_binding(source, workspace=tmp_path / "work")
    assert binding.resources == ("remote-owner:fixture-host:physical:serial-42",)
    assert binding._effective_config.selected.command == str(executable)
