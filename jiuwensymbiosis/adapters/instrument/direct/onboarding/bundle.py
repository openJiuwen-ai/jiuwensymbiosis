"""Import reviewed extraction data, bind source evidence and install deterministically."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jiuwensymbiosis.adapters.instrument.direct.action_registry import ActionRegistry
from jiuwensymbiosis.adapters.instrument.direct.config import action_ids, action_ready, load_config, validate_config


def _decode_json(text: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("input must be a JSON object")
    json.dumps(value, allow_nan=False)
    return value


def _read_json(path: str | Path) -> dict:
    return _decode_json(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _document(path: str | Path) -> tuple[bytes, str, str]:
    data = Path(path).read_bytes()
    if not data or len(data) > 5_000_000:
        raise ValueError("source document must be nonempty and at most 5 MB")
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("supply UTF-8 Markdown/text extracted from the original document") from exc
    return data, text, hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class _BundleSnapshot:
    source_bytes: bytes
    source_text: str
    source_digest: str
    candidate_json: str

    @classmethod
    def read(cls, bundle: str | Path) -> _BundleSnapshot:
        bundle = Path(bundle)
        data, text, digest = _document(bundle / "source.md")
        candidate_json = (bundle / "candidate.json").read_text(encoding="utf-8")
        return cls(data, text, digest, candidate_json)

    def candidate(self) -> dict:
        return _decode_json(self.candidate_json)


def prepare_candidate(document: str | Path, output: str | Path, *, device_id: str, model: str) -> dict:
    """Create an explicitly incomplete extraction form; do not guess a protocol."""
    _, _, digest = _document(document)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("candidate output already exists")
    candidate = {
        "candidate_version": "1",
        "source_sha256": digest,
        "review": {"reviewed_by": "", "reviewed_at": ""},
        "extension": {"extension_id": device_id.replace("-", "_"), "version": "1", "actions": []},
        "device": {
            "device_id": device_id,
            "model": model,
            "adapter": "mapped_protocol",
            "operation_id": "FILL_DOCUMENTED_ACTION",
            "operation_ids": [],
            "connection": {},
            "configured": False,
            "configuration_evidence": "",
        },
        "coverage": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, candidate)
    return {
        "status": "candidate_requires_extraction",
        "candidate": str(output),
        "source_sha256": digest,
        "hardware_io": False,
        "model_used": False,
    }


def _candidate_shape(candidate: dict) -> None:
    extension = candidate.get("extension")
    if not isinstance(extension, dict) or not isinstance(extension.get("actions"), list):
        raise ValueError("candidate extension requires an actions array")
    if any(
        not isinstance(action, dict) or not isinstance(action.get("source"), dict) for action in extension["actions"]
    ):
        raise ValueError("each candidate action requires a source object")
    if not isinstance(candidate.get("device"), dict):
        raise ValueError("candidate device must be an object")


def _validate(candidate: dict, source_text: str, digest: str) -> dict:
    _candidate_shape(candidate)
    if set(candidate) != {"candidate_version", "source_sha256", "review", "extension", "device", "coverage"}:
        raise ValueError("candidate contains missing or unknown fields")
    if candidate["candidate_version"] != "1" or candidate["source_sha256"] != digest:
        raise ValueError("candidate source digest or version mismatch")
    review = candidate["review"]
    if not isinstance(review, dict) or set(review) != {"reviewed_by", "reviewed_at"}:
        raise ValueError("review requires reviewed_by/reviewed_at")
    if any(not isinstance(v, str) or not v.strip() for v in review.values()):
        raise ValueError("review identity and timestamp are required before publication")
    if datetime.fromisoformat(review["reviewed_at"]).tzinfo is None:
        raise ValueError("review timestamp must include a timezone")
    registry = ActionRegistry([candidate["extension"]])
    for definition in registry.definitions.values():
        evidence = definition["source"]
        if evidence["sha256"] != digest or evidence["document"] != "source.md":
            raise ValueError("action source digest/name must match the bundled document")
        if evidence["excerpt"] not in source_text:
            raise ValueError(f"source excerpt for {definition['name']} does not occur in the document")
        if evidence["section"] not in source_text:
            raise ValueError(f"source section for {definition['name']} does not occur in the document")
    coverage = candidate["coverage"]
    if "coverage" in candidate["extension"] and candidate["extension"]["coverage"] != coverage:
        raise ValueError("extension and document coverage disagree")
    if not isinstance(coverage, list) or not coverage:
        raise ValueError("document command coverage is required")
    names = []
    for row in coverage:
        if not isinstance(row, dict) or set(row) != {"command", "action", "status", "evidence"}:
            raise ValueError("coverage requires command/action/status/evidence")
        if row["status"] not in {"implemented", "blocked"}:
            raise ValueError("coverage status must be implemented or blocked")
        if not isinstance(row["evidence"], str) or not row["evidence"].strip() or row["evidence"] not in source_text:
            raise ValueError("coverage evidence must occur in the source document")
        if row["status"] == "implemented" and row["action"] not in registry.definitions:
            raise ValueError("implemented coverage references an unregistered action")
        if row["status"] == "blocked" and row["action"] is not None:
            raise ValueError("blocked document commands must not advertise an executable action")
        names.append(row["command"])
    if len(set(names)) != len(names) or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("coverage command names must be nonempty and unique")
    covered = {row["action"] for row in coverage if row["status"] == "implemented"}
    if covered != set(registry.definitions):
        raise ValueError("every registered action requires a coverage row")
    config = validate_config(
        {
            "config_version": "1",
            "devices": [candidate["device"]],
            "object_types": [],
            "extensions": [candidate["extension"]],
        }
    )
    if set(action_ids(config["devices"][0])) != set(registry.definitions):
        raise ValueError("candidate device must declare all and only its extension actions")
    return {
        "status": "validated",
        "device_id": config["devices"][0]["device_id"],
        "actions": sorted(registry.definitions),
        "source_sha256": digest,
        "blocked_commands": [row["command"] for row in coverage if row["status"] == "blocked"],
        "hardware_io": False,
        "model_used": False,
        "physical_device_verified": False,
        "note": (
            "Mapping/schema/source checks passed; "
            "document interpretation is a reviewed input, not automatically proven."
        ),
    }


def build_bundle(document: str | Path, candidate_path: str | Path, output: str | Path) -> dict:
    """Seal a reviewed document extraction after all deterministic checks pass."""
    data, source_text, digest = _document(document)
    candidate = _read_json(candidate_path)
    _candidate_shape(candidate)
    # The tool seals provenance; it does not generate or modify protocol facts.
    for action in candidate.get("extension", {}).get("actions", []):
        action.setdefault("source", {}).update(document="source.md", sha256=digest)
    candidate["extension"]["coverage"] = deepcopy(candidate["coverage"])
    candidate["extension"]["source_text"] = source_text
    result = _validate(candidate, source_text, digest)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("bundle output already exists; select a new version directory")
    output.mkdir(parents=True)
    (output / "source.md").write_bytes(data)
    _write_json(output / "candidate.json", candidate)
    _write_json(output / "validation.json", result)
    (output / "README.md").write_text(
        "# 已验证的新设备映射候选\n\n"
        "validate 仅校验契约、协议映射和文档证据，不证明实机连接或物理效果。\n"
        "安装使用 Python API jiuwensymbiosis.adapters.instrument.direct.onboarding.install_bundle()，输出新配置；本目录不会在运行时调用模型。\n",
        encoding="utf-8",
    )
    return {**result, "bundle": str(output)}


def validate_bundle(bundle: str | Path) -> dict:
    """Revalidate bundled source and mapping; never trust a saved PASS report."""
    snapshot = _BundleSnapshot.read(bundle)
    return _validate(snapshot.candidate(), snapshot.source_text, snapshot.source_digest)


def install_bundle(
    bundle: str | Path,
    output: str | Path,
    *,
    base_config: str | Path | None = None,
    skills_dir: str | Path | None = None,
) -> dict:
    """Merge into a new deployment file; never touch an existing deployment or device."""
    snapshot = _BundleSnapshot.read(bundle)
    candidate = snapshot.candidate()
    report = _validate(candidate, snapshot.source_text, snapshot.source_digest)
    config = (
        load_config(base_config)
        if base_config
        else {"config_version": "1", "devices": [], "object_types": [], "extensions": []}
    )
    config = deepcopy(config)
    candidate["extension"]["coverage"] = deepcopy(candidate["coverage"])
    config["devices"].append(candidate["device"])
    config.setdefault("extensions", []).append(candidate["extension"])
    config = validate_config(config)
    installed_device = next(device for device in config["devices"] if device["device_id"] == report["device_id"])
    configured_actions = [name for name in action_ids(installed_device) if action_ready(installed_device, name)]
    unconfigured_actions = [name for name in action_ids(installed_device) if not action_ready(installed_device, name)]
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("installation never overwrites an existing deployment")
    if skills_dir and (Path(skills_dir).exists() or Path(skills_dir).is_symlink()):
        raise ValueError("skills output already exists; select a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = None
    created_config = False
    try:
        if skills_dir:
            from jiuwensymbiosis.adapters.instrument.direct.device_skills import export_device_skills

            destination = Path(skills_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged = Path(tempfile.mkdtemp(prefix=".onboarding-", dir=destination.parent))
            export_device_skills(config, staged)
            evidence_dir = staged / "onboarding_sources" / candidate["extension"]["extension_id"]
            evidence_dir.mkdir(parents=True, exist_ok=True)
            (evidence_dir / "source.md").write_bytes(snapshot.source_bytes)
            _write_json(evidence_dir / "candidate.json", candidate)
            _write_json(evidence_dir / "validation.json", report)
        with output.open("x", encoding="utf-8") as stream:
            created_config = True
            stream.write(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        if staged:
            if Path(skills_dir).exists() or Path(skills_dir).is_symlink():
                raise ValueError("skills destination appeared during installation")
            staged.rename(skills_dir)
            staged = None
    except Exception:
        if created_config:
            output.unlink(missing_ok=True)
        raise
    finally:
        if staged:
            shutil.rmtree(staged)
    return {
        **report,
        "status": "installed",
        "config": str(output),
        "skills_dir": str(skills_dir) if skills_dir else None,
        "activation": "configured"
        if not unconfigured_actions
        else "partially_configured"
        if configured_actions
        else "discovery_only",
        "configured_actions": configured_actions,
        "unconfigured_actions": unconfigured_actions,
    }


def add_device(document, candidate_path, output, *, base_config=None, skills_dir=None):
    """One offline entry point: seal reviewed extraction, install and export Skills.

    Interpretation is supplied by the onboarding Agent/reviewer; no model runs in
    production. Existing destinations remain untouched on validation failure.
    """
    with tempfile.TemporaryDirectory(prefix="instrument-add-") as directory:
        bundle = Path(directory) / "bundle"
        build_bundle(document, candidate_path, bundle)
        return install_bundle(bundle, output, base_config=base_config, skills_dir=skills_dir)
