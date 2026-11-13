"""Explicit registration and pinned downloads for external policy-robot assets."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

import yaml

from .._paths import simulator_root
from .config import RootSpec, load_config
from .index import build_index
from .models import AssetError
from .scanners import scan_root

__all__ = ["fetch_assets", "register_robot"]

REVISIONS = {
    "jumper": ("https://github.com/KingKongRobotics/jumper.git", "198686eac0d188815d0943e9fee50de415edc80a"),
    "microduck": ("https://github.com/pollen-robotics/microduck_rl.git", "0daab160d3fb1c2cbaba16a4a19052f36709dd9e"),
}
POLICY_REVISION = "283771bcb3690a0a94a288c108365d9b630d21e4"
MICRODUCK_FILES = {
    "velstand.onnx": "ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a",
    "manifest.json": "5c6a73f95f388f131aaef570f6113f0d36b84a72d5755662cd57e67adf7fd24c",
}
JUMPER_MANIFEST = "88a1e061ac17853837063cc3f1b69737042f4372fd931177ee6b4464172a3948"


def verify_digest(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise AssetError(f"Asset SHA256 mismatch: {path.name}")


def fetch_assets(body: str, destination: Path) -> None:
    """Fetch official assets once; refuse to alter any existing destination."""
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Asset directory already exists; use --source without --fetch: {destination}")
    if body not in REVISIONS:
        raise AssetError(f"Unsupported policy robot: {body}")
    url, revision = REVISIONS[body]
    destination.parent.mkdir(parents=True, exist_ok=True)
    patterns = (
        ["/assets/jumper/", "/out/bundle_example/jumper/", "/LICENSE", "/NOTICE", "/licenses/"]
        if body == "jumper"
        else ["/src/mjlab_microduck/robot/microduck/", "/scripts/infer_policy.py", "/LICENSE", "/LICENSE-MODELS.md"]
    )
    with tempfile.TemporaryDirectory(prefix=".robot-assets-", dir=destination.parent) as temporary:
        checkout = Path(temporary) / "checkout"
        for command in (
            ["git", "init", str(checkout)],
            ["git", "-C", str(checkout), "remote", "add", "origin", url],
            ["git", "-C", str(checkout), "sparse-checkout", "set", "--no-cone", *patterns],
            ["git", "-C", str(checkout), "fetch", "--filter=blob:none", "--depth=1", "origin", revision],
            ["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"],
        ):
            subprocess.run(command, check=True, stdout=sys.stderr)
        actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise AssetError("Unexpected upstream revision")
        if body == "microduck":
            policies = checkout / "policies"
            policies.mkdir(exist_ok=True)
            for name, digest in MICRODUCK_FILES.items():
                address = f"https://huggingface.co/pollen-robotics/microduck-policies/resolve/{POLICY_REVISION}/{name}"
                with urlopen(address, timeout=120) as response:
                    (policies / name).write_bytes(response.read())
                verify_digest(policies / name, digest)
        else:
            bundle = checkout / "out/bundle_example/jumper"
            verify_digest(bundle / "bundle.json", JUMPER_MANIFEST)
            for relative, entry in json.loads((bundle / "bundle.json").read_text())["files"].items():
                path = (bundle / relative).resolve()
                if not path.is_relative_to(bundle.resolve()):
                    raise AssetError("Jumper bundle reference leaves its directory")
                verify_digest(path, entry["sha256"])
        # mkdir reserves the name, protecting a destination created during the download.
        destination.mkdir()
        checkout.rename(destination)


def register_robot(body: str, source: Path, home: Path | None = None) -> dict[str, str]:
    """Append a complete external robot checkout without replacing existing roots."""
    if body not in REVISIONS:
        raise AssetError(f"Unsupported policy robot: {body}")
    source = source.expanduser().resolve()
    home = (home or simulator_root()).expanduser().resolve()
    entry = {"id": body, "path": str(source), "scanner": body}
    asset = scan_root(RootSpec.model_validate(entry), source).assets[0]
    if body == "microduck" and asset.preview_mode != "microduck":
        raise AssetError("请先下载官方 policies/velstand.onnx 和 policies/manifest.json")
    config = home / "configs/simulators/assets.yaml"
    if config.is_file():
        load_config(config)
        data = yaml.safe_load(config.read_text())
        existing = next((root for root in data["roots"] if root["id"] == body), None)
        if existing is not None and existing != entry:
            raise AssetError(f"{body} root already points elsewhere; edit assets.yaml explicitly")
        if existing is None:
            data["roots"].append(entry)
    else:
        data = {"version": 1, "index": str(home / "outputs/simulator-assets/index.sqlite"), "roots": [entry]}
    config.parent.mkdir(parents=True, exist_ok=True)
    candidate = config.with_suffix(".candidate.yaml")
    try:
        candidate.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        load_config(candidate)
        candidate.replace(config)
    finally:
        candidate.unlink(missing_ok=True)
    build_index(load_config(config))
    return {"backend": "mujoco", "scene_id": asset.id, "robot_mode": body, "config": str(config)}
