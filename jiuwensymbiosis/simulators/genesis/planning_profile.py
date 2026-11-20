"""Load portable planning semantics and explicitly referenced sphere assets."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _apply_sphere_overrides(profile: dict, path: Path) -> None:
    """Replace fitted link geometry while retaining the external body's policy."""
    import yaml

    path = path.resolve(strict=True)
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    provenance = profile.get("collision_provenance", {})
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported sphere override schema")
    if not provenance or data.get("base_collision_sha256") != provenance["sha256"]:
        raise ValueError("sphere overrides require the matching collision source")
    model_hash = data.get("prepared_model_sha256", "")
    if not isinstance(model_hash, str) or len(model_hash) != 64 or any(c not in "0123456789abcdef" for c in model_hash):
        raise ValueError("sphere overrides require a prepared model hash")
    spheres = profile["collision_spheres"]
    replacements = data.get("collision_spheres")
    removed = data.get("remove_links", [])
    if (
        not isinstance(replacements, dict)
        or not replacements
        or set(replacements) - set(spheres)
        or any(not isinstance(entries, list) or not entries for entries in replacements.values())
        or not isinstance(removed, list)
        or any(not isinstance(link, str) or link not in spheres or link in replacements for link in removed)
    ):
        raise ValueError("sphere overrides must replace known links with nonempty sphere lists")
    profile["collision_spheres"] = {k: replacements.get(k, v) for k, v in spheres.items() if k not in removed}
    profile["self_collision_ignore"] = {
        k: [v for v in values if v not in removed]
        for k, values in profile["self_collision_ignore"].items()
        if k not in removed
    }
    provenance["sphere_overrides"] = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "prepared_model_sha256": model_hash,
        "replaced_links": sorted(replacements),
        "removed_links": removed,
    }


def load_profile(path: Path) -> dict:
    import yaml

    path = Path(path).resolve(strict=True)
    profile = yaml.safe_load(path.read_text())
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("unsupported planning profile schema")
    references = profile.pop("grasp_pose_configs", {})
    sphere_overrides = profile.pop("collision_sphere_overrides", None)
    if "arm_profile" in profile:
        if set(profile) - {
            "arm_profile",
            "schema_version",
            "tabletop_fixture",
            "fixture_posture",
            "robot_collision_policy",
        }:
            raise ValueError("unexpected fields in grasp planning profile")
        base = (path.parent / profile["arm_profile"]).resolve(strict=True)
        model = yaml.safe_load(base.read_text())
        if model.get("schema_version") != 1 or "arm_profile" in model or "grasp_pose_configs" in model:
            raise ValueError("expected a standalone base arm profile")
        model["tabletop_fixture"] = profile.get("tabletop_fixture", False)
        model["fixture_posture"] = profile.get("fixture_posture", {})
        if "robot_collision_policy" in profile:
            model["robot_collision_policy"] = yaml.safe_load(
                (path.parent / profile["robot_collision_policy"]).read_text()
            )
        profile = model
    else:
        base = path
    defaults = profile.pop("default_limits", {})
    names = {name for group in profile["groups"].values() for name in group["joints"]}
    overrides = profile.get("limits", {})
    if set(overrides) - names:
        raise ValueError("limits for joints outside configured groups")
    profile["limits"] = {name: defaults | overrides.get(name, {}) for name in sorted(names)}
    reference = profile.pop("collision_source", None)
    if reference:
        source = (base.parent / os.path.expandvars(reference["path"])).resolve(strict=True)
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
        # Asset-specific adjacent geometry belongs in the profile, not the executor.
        adjacent = reference.get("adjacent_self_collision_pairs", [])
        for pair in adjacent:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or pair[0] == pair[1]
                or any(not isinstance(link, str) or link not in profile["collision_spheres"] for link in pair)
            ):
                raise ValueError("adjacent collision pair requires two distinct sphere links")
            first, second = pair
            values = profile["self_collision_ignore"].setdefault(first, [])
            if second not in values:
                values.append(second)
        profile["collision_provenance"] = {
            "path": str(source),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "ignore_rationale": reference["ignore_rationale"],
            "adjacent_self_collision_pairs": adjacent,
        }
    if sphere_overrides is not None:
        _apply_sphere_overrides(profile, path.parent / os.path.expandvars(sphere_overrides))
    # Referenced configs are optional; ordinary motion profiles need no model service.
    backends = {}
    for mode, filename in references.items():
        config = yaml.safe_load((path.parent / filename).read_text())
        if (
            mode not in {"perception", "ground_truth"}
            or config["point_cloud_source"] != {"perception": "rgbd", "ground_truth": "ground_truth"}[mode]
        ):
            raise ValueError("grasp input does not match observation mode")
        effector = profile["effectors"][config["effector"]]
        if (
            config["backend"] != "graspgenx"
            or effector["tool"] != config["control_link"]
            or profile["groups"][config["group"]]["tools"] != [effector["tool"]]
        ):
            raise ValueError("grasp descriptor does not match configured single tool group")
        backends[mode] = config
    if backends:
        profile["grasp_pose_backends"] = backends
    return profile
