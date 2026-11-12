"""Load portable planning semantics and explicitly referenced sphere assets."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def load_profile(path: Path) -> dict:
    import yaml

    path = Path(path).resolve(strict=True)
    profile = yaml.safe_load(path.read_text())
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("unsupported planning profile schema")
    defaults = profile.pop("default_limits", {})
    names = {name for group in profile["groups"].values() for name in group["joints"]}
    overrides = profile.get("limits", {})
    if set(overrides) - names:
        raise ValueError("limits for joints outside configured groups")
    profile["limits"] = {name: defaults | overrides.get(name, {}) for name in sorted(names)}
    reference = profile.pop("collision_source", None)
    if reference:
        source = (path.parent / os.path.expandvars(reference["path"])).resolve(strict=True)
        raw = source.read_bytes()
        data = yaml.safe_load(raw)
        if reference["format"] == "xrdf":
            spheres = data["geometry"][data["world_collision"]["geometry"]]["spheres"]
            ignore = data["self_collision"]["ignore"]
        elif reference["format"] == "curobo":
            kin = data["robot_cfg"]["kinematics"]
            spheres, ignore = kin["collision_spheres"], kin["self_collision_ignore"]
        else:
            raise ValueError("unsupported collision source format")
        excluded = set(reference.get("exclude_links", []))
        if excluded - set(spheres):
            raise ValueError("unknown excluded sphere links")
        profile["collision_spheres"] = {k: v for k, v in spheres.items() if k not in excluded}
        profile["self_collision_ignore"] = {
            k: [v for v in values if v in profile["collision_spheres"]]
            for k, values in ignore.items()
            if k in profile["collision_spheres"]
        }
        profile["collision_provenance"] = {
            "path": str(source),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "ignore_rationale": reference["ignore_rationale"],
        }
    return profile
