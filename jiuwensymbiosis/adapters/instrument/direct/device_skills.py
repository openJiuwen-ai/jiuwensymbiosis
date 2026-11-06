"""Offline Skill export of the same declarations consumed by ActionBinding."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from jiuwensymbiosis.adapters.instrument.direct.action_registry import ActionRegistry
from jiuwensymbiosis.adapters.instrument.direct.config import action_ids, action_ready, validate_config
from jiuwensymbiosis.adapters.instrument.direct.session import InstrumentContractView, tool_name

MetadataProvider = Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], Mapping[str, Any]]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)[:50] + "_" + hashlib.sha256(value.encode()).hexdigest()[:8]


def build_device_catalogue(
    config: dict[str, Any],
    *,
    registry: Any = None,
    providers: Mapping | None = None,
    metadata_provider: MetadataProvider | None = None,
) -> dict[str, Any]:
    """Describe schemas, provenance and deployment ranges without opening devices."""
    config = validate_config(config, providers=providers)
    installed_registry = ActionRegistry(config["extensions"])
    if registry is not None and (
        not isinstance(registry, ActionRegistry)
        or any(
            getattr(registry, field) != getattr(installed_registry, field)
            for field in ("operations", "definitions", "documented_commands", "action_extensions", "read_only")
        )
    ):
        raise ValueError("Skill registry must match the installed declarations")
    # The optional registry can confirm a matching installation, but cannot
    # replace its action truth or mutable provenance during export.
    registry = installed_registry
    devices = []
    for profile in sorted(config["devices"], key=lambda p: p["device_id"]):
        actions = []
        for name in sorted(action_ids(profile)):
            operation = registry.operations[name]
            view = InstrumentContractView(operation, tool_name(profile["device_id"], name))
            actions.append(
                {
                    "operation_id": name,
                    "function": view.name,
                    "version": operation.version,
                    "description": view.description,
                    "input_schema": view.input_schema,
                    "output_schema": view.output_schema,
                    "required_capabilities": sorted(view.required_capabilities),
                    "required_conditions": list(operation.required_conditions),
                    "read_only": name in registry.read_only,
                    "completion_rule": operation.completion_rule,
                    "configured": action_ready(profile, name),
                    "source": operation.definition["source"],
                }
            )
        public = {key: deepcopy(profile[key]) for key in ("device_id", "model", "limits", "configuration_evidence")}
        metadata = dict(metadata_provider(deepcopy(public), deepcopy(actions))) if metadata_provider else {}
        coverage = [
            deepcopy(row)
            for extension_id in sorted({registry.action_extensions[n] for n in action_ids(profile)})
            for row in registry.documented_commands[extension_id]
        ]
        devices.append(
            {
                **public,
                "directory": (PurePosixPath("devices") / _safe_name(profile["device_id"])).as_posix(),
                "actions": actions,
                "coverage": coverage,
                "metadata": metadata,
            }
        )
    catalogue = {"format_version": "2", "hardware_io": False, "model_calls": False, "devices": devices}
    catalogue["catalogue_digest"] = _digest(catalogue)
    return catalogue


def export_device_skills(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    registry: Any = None,
    providers: Mapping | None = None,
    metadata_provider: MetadataProvider | None = None,
) -> dict[str, Any]:
    """Publish static schemas and review provenance, never credentials or grants."""
    catalogue = build_device_catalogue(
        config, registry=registry, providers=providers, metadata_provider=metadata_provider
    )
    files = {"catalogue.json": _json(catalogue)}
    for device in catalogue["devices"]:
        directory = PurePosixPath(device["directory"])
        lines = [
            f"# {device['model']} ({device['device_id']})",
            "",
            "Tools are generated from reviewed instrument declarations. Use the managed Runtime entry point.",
            "Runtime owns job/invocation IDs, request deduplication, admission and feedback events.",
            "Provide a stable request key. Unknown outcomes must not be resent or declared complete by a model.",
            "Protocol response verification does not establish physical completion.",
            "",
        ]
        for action in device["actions"]:
            lines += [
                f"## {action['function']}",
                "",
                action["description"],
                "",
                f"Contract version: {action['version']}; completion: {action['completion_rule']}.",
                "Source: " + _json(action["source"]).strip(),
                "",
                "```json",
                _json(action["input_schema"]).strip(),
                "```",
                "",
            ]
        files.update(
            {
                (directory / "SKILL.md").as_posix(): "\n".join(lines),
                (directory / "device.json").as_posix(): _json(device),
                (directory / "actions.schema.json").as_posix(): _json(
                    {
                        a["function"]: {"inputSchema": a["input_schema"], "outputSchema": a["output_schema"]}
                        for a in device["actions"]
                    }
                ),
                (directory / "coverage.json").as_posix(): _json(device["coverage"]),
            }
        )
    for extension in config.get("extensions", []):
        if "source_text" in extension:
            source_path = PurePosixPath("onboarding_sources") / extension["extension_id"] / "source.md"
            files[source_path.as_posix()] = extension["source_text"]
    manifest = {
        "format_version": "2",
        "catalogue_digest": catalogue["catalogue_digest"],
        "hardware_io": False,
        "model_calls": False,
        "device_count": len(catalogue["devices"]),
        "action_count": sum(len(d["actions"]) for d in catalogue["devices"]),
        "files": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in sorted(files.items())},
    }
    files["skill-manifest.json"] = _json(manifest)
    root = Path(output_dir).expanduser().resolve()
    for name in files:
        target = root / name
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError(f"Export path must stay inside output directory: {name}")
        if target.exists() and not target.is_file():
            raise ValueError(f"Export target is not a regular file: {name}")
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return manifest
