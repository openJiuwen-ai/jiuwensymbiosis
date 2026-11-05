"""Contract-derived offline exports without project-specific source tables."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from jiuwensymbiosis.adapters.instrument.direct.device_skills import build_device_catalogue, export_device_skills
from jiuwensymbiosis.adapters.instrument.direct.skill_verification import verify_device_skills


@pytest.fixture
def config():
    # Source metadata is synthetic and intentionally unrelated to any deployment.
    action = {
        "name": "sensor_read",
        "description": "Read synthetic sensor",
        "read_only": True,
        "parameters_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "source": {"document": "synthetic.md", "section": "Read", "excerpt": "read returns value", "sha256": "0" * 64},
        "completion_rule": "protocol_response_verified",
        "mapping": {
            "transport": "http_json",
            "method": "GET",
            "path": "/value",
            "response": {"checks": [{"path": ["value"], "type": "number"}]},
        },
    }
    return {
        "config_version": "1",
        "devices": [
            {
                "device_id": "custom sensor/一",
                "model": "synthetic",
                "adapter": "mapped_protocol",
                "operation_id": "sensor_read",
                "operation_ids": ["sensor_read"],
                "connection": {"base_url": "http://127.0.0.1:1", "timeout_s": 1},
                "configured": False,
                "configuration_evidence": "",
            }
        ],
        "extensions": [{"extension_id": "synthetic_sensor", "version": "1", "actions": [action]}],
    }


def test_export_is_deterministic_and_does_not_expose_connection(config, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    one = export_device_skills(config, first)
    two = export_device_skills(config, second)
    assert one == two and one["device_count"] == one["action_count"] == 1
    for name in [*one["files"], "skill-manifest.json"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert "http://127.0.0.1:1" not in (first / name).read_text()
    assert verify_device_skills(config, first)["status"] == "matched"
    (first / "personal-notes.md").write_text("preserve")
    export_device_skills(config, first)
    assert (first / "personal-notes.md").read_text() == "preserve"


def test_metadata_is_explicit_and_cannot_replace_contracts(config, tmp_path):
    calls = []

    def metadata(profile, actions):
        calls.append(profile)
        profile["model"] = "attempted modification"
        actions[0]["description"] = "attempted modification"
        return {"title": "Host supplied description", "document_review": "synthetic only"}

    catalogue = build_device_catalogue(config, metadata_provider=metadata)
    device = catalogue["devices"][0]
    assert "connection" not in calls[0]
    assert device["model"] == "synthetic" and device["actions"][0]["description"] == "Read synthetic sensor"
    assert device["metadata"]["title"] == "Host supplied description"
    export_device_skills(config, tmp_path, metadata_provider=metadata)
    assert verify_device_skills(config, tmp_path, metadata_provider=metadata)["status"] == "matched"
    assert verify_device_skills(config, tmp_path)["status"] == "mismatch"


def test_exported_schemas_are_the_managed_binding_views(config, tmp_path):
    from jiuwensymbiosis.adapters.instrument.direct.action_registry import ActionRegistry
    from jiuwensymbiosis.adapters.instrument.direct.session import InstrumentContractView, tool_name

    export_device_skills(config, tmp_path)
    directory = next((tmp_path / "devices").iterdir())
    schemas = json.loads((directory / "actions.schema.json").read_text())
    op = ActionRegistry(config["extensions"]).operations["sensor_read"]
    view = InstrumentContractView(op, tool_name(config["devices"][0]["device_id"], "sensor_read"))
    assert schemas[view.name] == {"inputSchema": view.input_schema, "outputSchema": view.output_schema}
    for schema in schemas[view.name].values():
        Draft202012Validator.check_schema(schema)
    assert not (tmp_path / "plan.schema.json").exists()


def test_manifest_detects_tamper_and_symlinks(config, tmp_path):
    export_device_skills(config, tmp_path)
    (tmp_path / "catalogue.json").write_text("{}")
    assert verify_device_skills(config, tmp_path)["status"] == "mismatch"
    export_device_skills(config, tmp_path)
    target = next((tmp_path / "devices").iterdir()) / "actions.schema.json"
    target.unlink()
    target.symlink_to(tmp_path / "catalogue.json")
    with pytest.raises(ValueError, match="Export path"):
        export_device_skills(config, tmp_path)
    assert verify_device_skills(config, tmp_path)["status"] == "mismatch"


def test_static_export_and_verify_without_runtime_state(config, tmp_path, monkeypatch):
    from jiuwensymbiosis.adapters.instrument.direct.providers import MappedAdapterProvider
    from jiuwensymbiosis.runtime.store import JobStore

    def reject(*args, **kwargs):
        raise AssertionError("offline export must not create adapters or Runtime storage")

    monkeypatch.setattr(MappedAdapterProvider, "create_adapter", reject)
    monkeypatch.setattr(JobStore, "__init__", reject)
    export_device_skills(config, tmp_path)
    assert verify_device_skills(config, tmp_path)["status"] == "matched"


@pytest.mark.parametrize("drift", ["schema", "version", "mutable_definition"])
def test_registry_cannot_override_installed_skill_declarations(tmp_path, drift):
    from jiuwensymbiosis.adapters.instrument.direct.action_registry import ActionRegistry
    from tests.unit_tests.adapters.instrument.direct.test_contracts_mapping import config as controller_config

    installed = controller_config()
    changed = deepcopy(installed["extensions"])
    if drift == "schema":
        changed[0]["actions"][0]["parameters_schema"]["properties"]["level"]["maximum"] = 200
    elif drift == "version":
        changed[0]["version"] = "different-contract"
    registry = ActionRegistry(changed)
    if drift == "mutable_definition":
        registry.definitions["controller_set"]["parameters_schema"]["properties"]["level"]["maximum"] = 200
    target = tmp_path / "skills"
    with pytest.raises(ValueError, match="registry.*installed"):
        export_device_skills(installed, target, registry=registry)
    assert not target.exists()


def test_matching_registry_keeps_export_and_verification_on_installed_contract(config, tmp_path):
    from jiuwensymbiosis.adapters.instrument.direct.action_registry import ActionRegistry

    registry = ActionRegistry(config["extensions"])
    assert build_device_catalogue(config, registry=registry) == build_device_catalogue(config)
    export_device_skills(config, tmp_path, registry=registry)
    assert verify_device_skills(config, tmp_path)["status"] == "matched"
