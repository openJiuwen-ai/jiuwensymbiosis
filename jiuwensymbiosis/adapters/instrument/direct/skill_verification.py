"""Check a Skill bundle against the installed runtime and current configuration."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from jiuwensymbiosis.adapters.instrument.direct.device_skills import MetadataProvider, export_device_skills


def verify_device_skills(
    config: dict[str, Any],
    directory: str | Path,
    *,
    registry=None,
    providers=None,
    metadata_provider: MetadataProvider | None = None,
) -> dict[str, Any]:
    """Regenerate offline and compare every generated file, without overwriting.

    Hand-written integration notes outside the generated manifest are allowed.
    This is compatibility/integrity verification, not hardware readiness proof.
    """
    root = Path(directory).expanduser().resolve()
    mismatches = []
    with tempfile.TemporaryDirectory(prefix="verify-function-skills-") as temporary:
        manifest = export_device_skills(
            config, temporary, registry=registry, providers=providers, metadata_provider=metadata_provider
        )
        expected = Path(temporary)
        for name in [*manifest["files"], "skill-manifest.json"]:
            target = root / name
            if target.is_symlink() or not target.resolve().is_relative_to(root):
                mismatches.append({"file": name, "reason": "path_outside_bundle_or_symlink"})
            elif not target.is_file():
                mismatches.append({"file": name, "reason": "missing"})
            elif (
                hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256((expected / name).read_bytes()).digest()
            ):
                mismatches.append({"file": name, "reason": "content_or_configuration_changed"})
        # Report formerly generated devices/actions absent from this revision;
        # never silently remove their files or a user's independent documents.
        previous_path = root / "skill-manifest.json"
        if previous_path.is_file() and not previous_path.is_symlink():
            try:
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                previous_files = previous["files"]
                if not isinstance(previous_files, dict):
                    raise ValueError("manifest.files must be an object")
                for name in sorted(set(previous_files) - manifest["files"].keys()):
                    mismatches.append({"file": name, "reason": "retired_generated_file"})
            except (OSError, ValueError, KeyError, TypeError):
                mismatches.append({"file": "skill-manifest.json", "reason": "invalid_manifest"})
    return {
        "status": "matched" if not mismatches else "mismatch",
        "hardware_io": False,
        "model_calls": False,
        "catalogue_digest": manifest["catalogue_digest"],
        "device_count": manifest["device_count"],
        "mismatches": mismatches,
    }
