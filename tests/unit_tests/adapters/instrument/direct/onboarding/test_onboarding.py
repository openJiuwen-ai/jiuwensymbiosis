"""Reviewed source import, transaction boundaries and local protocol proof."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jiuwensymbiosis.adapters.instrument.direct.config import load_config
from jiuwensymbiosis.adapters.instrument.direct.onboarding import (
    build_bundle,
    install_bundle,
    prepare_candidate,
    validate_bundle,
)
from jiuwensymbiosis.adapters.instrument.direct.skill_verification import verify_device_skills

from .fixtures import temperature_fixture as example

FIXTURES = Path(example.__file__).resolve().parent
SOURCE = FIXTURES / "temperature.source.md"


@pytest.fixture
def candidate():
    return json.loads((FIXTURES / "temperature.candidate.json").read_text(encoding="utf-8"))


def save(tmp_path, value, name="candidate.json"):
    path = tmp_path / name
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def bundle(tmp_path, candidate):
    path = tmp_path / "bundle"
    build_bundle(SOURCE, save(tmp_path, candidate), path)
    return path


def test_prepare_is_incomplete_and_cannot_activate(tmp_path):
    output = tmp_path / "candidate.json"
    result = prepare_candidate(SOURCE, output, device_id="new-temperature", model="synthetic")
    prepared = json.loads(output.read_text())
    assert result["model_used"] is False and result["hardware_io"] is False
    assert prepared["device"]["configured"] is False
    assert prepared["extension"]["actions"] == []
    with pytest.raises(ValueError):
        build_bundle(SOURCE, output, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
    with pytest.raises(ValueError, match="already exists"):
        prepare_candidate(SOURCE, output, device_id="another", model="synthetic")


def test_source_review_build_install_and_blocked_coverage(tmp_path, candidate):
    sealed = bundle(tmp_path, candidate)
    assert validate_bundle(sealed)["blocked_commands"] == ["OPAQUE"]
    deployment, skills = tmp_path / "deployment.json", tmp_path / "skills"
    result = install_bundle(sealed, deployment, skills_dir=skills)
    config = load_config(deployment)
    assert result["status"] == "installed" and result["activation"] == "discovery_only"
    assert result["physical_device_verified"] is False
    assert verify_device_skills(config, skills)["status"] == "matched"
    directory = next((skills / "devices").iterdir())
    coverage = json.loads((directory / "coverage.json").read_text())
    assert any(row["command"] == "OPAQUE" and row["status"] == "blocked" for row in coverage)
    assert (skills / "onboarding_sources/synthetic_temperature/source.md").read_bytes() == SOURCE.read_bytes()


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c["review"].update(reviewed_by=""),
        lambda c: c["review"].update(reviewed_at="2000-01-01T00:00:00"),
        lambda c: c.update(source_sha256="0" * 64),
        lambda c: c["extension"]["actions"][0]["source"].update(excerpt="not in this document"),
        lambda c: c["coverage"].pop(0),
        lambda c: c["coverage"][-1].update(action="temperature_read"),
        lambda c: c["extension"]["actions"][0].update(name="get_task"),
        lambda c: c["extension"]["actions"][2]["mapping"].update(python="print('forbidden')"),
    ],
)
def test_invalid_candidate_never_publishes(tmp_path, candidate, change):
    change(candidate)
    with pytest.raises((ValueError, TypeError)):
        build_bundle(SOURCE, save(tmp_path, candidate), tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_revalidation_rejects_source_tamper_even_with_saved_pass(tmp_path, candidate):
    sealed = bundle(tmp_path, candidate)
    (sealed / "source.md").write_text("Changed source", encoding="utf-8")
    assert json.loads((sealed / "validation.json").read_text())["status"] == "validated"
    with pytest.raises(ValueError):
        validate_bundle(sealed)
    with pytest.raises(ValueError):
        install_bundle(sealed, tmp_path / "deployment.json", skills_dir=tmp_path / "skills")
    assert not (tmp_path / "deployment.json").exists() and not (tmp_path / "skills").exists()


def test_existing_destinations_and_duplicate_devices_remain_intact(tmp_path, candidate):
    sealed = bundle(tmp_path, candidate)
    deployment = tmp_path / "deployment.json"
    install_bundle(sealed, deployment)
    original = deployment.read_bytes()
    with pytest.raises(ValueError, match="overwrites"):
        install_bundle(sealed, deployment)
    with pytest.raises(ValueError):
        install_bundle(sealed, tmp_path / "duplicate.json", base_config=deployment)
    assert deployment.read_bytes() == original and not (tmp_path / "duplicate.json").exists()
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "keep.txt").write_text("preserve")
    with pytest.raises(ValueError, match="skills output already exists"):
        install_bundle(sealed, tmp_path / "another.json", skills_dir=skills)
    assert (skills / "keep.txt").read_text() == "preserve"


def test_export_failure_rolls_back_new_installation(tmp_path, candidate, monkeypatch):
    import jiuwensymbiosis.adapters.instrument.direct.device_skills as skills

    sealed = bundle(tmp_path, candidate)

    def reject(*args, **kwargs):
        raise ValueError("injected export error")

    monkeypatch.setattr(skills, "export_device_skills", reject)
    with pytest.raises(ValueError, match="injected export error"):
        install_bundle(sealed, tmp_path / "new/deployment.json", skills_dir=tmp_path / "new/skills")
    assert not (tmp_path / "new/deployment.json").exists()
    assert not (tmp_path / "new/skills").exists()
    assert not list((tmp_path / "new").glob(".onboarding-*"))


def test_skill_publication_failure_removes_new_config(tmp_path, candidate, monkeypatch):
    sealed = bundle(tmp_path, candidate)
    rename = Path.rename

    def reject_staging(self, target):
        if self.name.startswith(".onboarding-"):
            raise OSError("injected publication failure")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", reject_staging)
    with pytest.raises(OSError, match="injected publication failure"):
        install_bundle(sealed, tmp_path / "deployment.json", skills_dir=tmp_path / "skills")
    assert not (tmp_path / "deployment.json").exists() and not (tmp_path / "skills").exists()
    assert not list(tmp_path.glob(".onboarding-*"))


@pytest.mark.parametrize("extension", [[], {"actions": None}, {"actions": [{"source": []}]}])
def test_malformed_candidate_shapes_are_rejected(tmp_path, candidate, extension):
    candidate["extension"] = extension
    path = save(tmp_path, candidate)
    with pytest.raises(ValueError):
        build_bundle(SOURCE, path, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_installation_uses_one_source_snapshot(tmp_path, candidate, monkeypatch):
    import jiuwensymbiosis.adapters.instrument.direct.onboarding.bundle as implementation

    sealed = bundle(tmp_path, candidate)
    validate = implementation._validate

    def mutate_after_validation(value, text, digest):
        result = validate(value, text, digest)
        (sealed / "source.md").write_text("changed after immutable read", encoding="utf-8")
        return result

    monkeypatch.setattr(implementation, "_validate", mutate_after_validation)
    skills = tmp_path / "skills"
    result = install_bundle(sealed, tmp_path / "deployment.json", skills_dir=skills)
    assert result["status"] == "installed"
    assert (skills / "onboarding_sources/synthetic_temperature/source.md").read_bytes() == SOURCE.read_bytes()


@pytest.mark.parametrize("inject_failure", [False, True])
def test_document_skill_runtime_sequence_and_feedback(tmp_path, inject_failure):
    report = example.verify(tmp_path / "proof", inject_failure=inject_failure)
    assert report["status"] == "PASS", report["result"]
    assert report["hardware_verified"] is False and report["skills"]["status"] == "matched"
    writes = [item for item in report["requests"] if item["method"] == "POST"]
    assert writes == [{"method": "POST", "path": "/setpoint", "body": {"temperature_c": 35.0}}]
    assert report["temperature_reads"] == (1 if inject_failure else 2)
    assert report["ordering_owner"] == "Runtime/fastagent"
    assert report["feedback"]["invocations"][-1]["status"] == ("unknown" if inject_failure else "succeeded")
    assert all(
        item["device_binding"]["physical_device_id"] == "synthetic-temperature-fixture"
        for item in report["feedback"]["invocations"]
    )
