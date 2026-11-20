"""Explicit registration and pinned downloads for external policy-robot assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

import yaml

from .._paths import simulator_root
from .config import RootSpec, load_config
from .index import build_index
from .models import AssetError, local_file
from .policy_files import MICRODUCK_ARENA, MICRODUCK_FILES, MICRODUCK_REVISION, jumper_export_files
from .scanners import scan_root

__all__ = ["complete_policies", "fetch_assets", "register_robot"]

REVISIONS = {
    "jumper": ("https://github.com/KingKongRobotics/jumper.git", "198686eac0d188815d0943e9fee50de415edc80a"),
    "microduck": ("https://github.com/pollen-robotics/microduck_rl.git", "0daab160d3fb1c2cbaba16a4a19052f36709dd9e"),
}
POLICY_REVISION = MICRODUCK_REVISION
JUMPER_MANIFEST = "88a1e061ac17853837063cc3f1b69737042f4372fd931177ee6b4464172a3948"


def verify_digest(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise AssetError(f"Asset SHA256 mismatch: {path.name}")


def _policy_sources(body: str):
    if body == "jumper":
        revision = REVISIONS[body][1]
        for _, _, relative, digest in jumper_export_files():
            yield relative, f"https://raw.githubusercontent.com/KingKongRobotics/jumper/{revision}/{relative}", digest
        return
    for name, digest in MICRODUCK_FILES.items():
        yield (
            f"policies/{name}",
            f"https://huggingface.co/pollen-robotics/microduck-policies/resolve/{POLICY_REVISION}/{name}",
            digest,
        )
    for name, release in MICRODUCK_ARENA.items():
        for filename, digest in release["files"].items():
            url = f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/{filename}"
            yield f"policies/arena/{name}/{filename}", url, digest


def complete_policies(body: str, root: Path) -> None:
    """Fill a local official model checkout. Existing files are verified and never overwritten."""
    root = root.expanduser().resolve(strict=True)
    entry = "assets/jumper/jumper.xml" if body == "jumper" else "src/mjlab_microduck/robot/microduck/scene.xml"
    if body not in REVISIONS or not local_file(root, entry).is_file():
        raise AssetError("An existing official robot model checkout is required")
    with tempfile.TemporaryDirectory(prefix=".policy-download-", dir=root) as temp:
        pending: list[tuple[Path, str, str]] = []
        for relative, url, digest in _policy_sources(body):
            target = local_file(root, relative)
            if target.exists():
                verify_digest(target, digest)
                continue
            path = Path(temp) / str(len(pending))
            with urlopen(url, timeout=120) as response, path.open("wb") as output:
                shutil.copyfileobj(response, output)
            verify_digest(path, digest)
            pending.append((path, relative, digest))
        for path, relative, digest in pending:
            target = local_file(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except FileExistsError:
                verify_digest(target, digest)


def fetch_assets(body: str, destination: Path) -> None:
    """Fetch official assets once; refuse to alter any existing destination."""
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Asset directory already exists; use --source without --fetch: {destination}")
    if body not in REVISIONS:
        raise AssetError(f"Unsupported policy robot: {body}")
    git_executable = shutil.which("git")
    if git_executable is None:
        raise AssetError("Git executable is required to fetch robot assets")
    url, revision = REVISIONS[body]
    destination.parent.mkdir(parents=True, exist_ok=True)
    patterns = (
        [
            "/assets/jumper/",
            "/out/bundle_example/jumper/",
            "/LICENSE",
            "/NOTICE",
            "/licenses/",
            *("/" + rel for _, _, rel, _ in jumper_export_files()),
        ]
        if body == "jumper"
        else ["/src/mjlab_microduck/robot/microduck/", "/scripts/infer_policy.py", "/LICENSE", "/LICENSE-MODELS.md"]
    )
    with tempfile.TemporaryDirectory(prefix=".robot-assets-", dir=destination.parent) as temporary:
        checkout = Path(temporary) / "checkout"
        for command in (
            [git_executable, "init", str(checkout)],
            [git_executable, "-C", str(checkout), "remote", "add", "origin", url],
            [git_executable, "-C", str(checkout), "sparse-checkout", "set", "--no-cone", *patterns],
            [git_executable, "-C", str(checkout), "fetch", "--filter=blob:none", "--depth=1", "origin", revision],
            [git_executable, "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"],
        ):
            subprocess.run(command, check=True, stdout=sys.stderr)
        actual = subprocess.check_output([git_executable, "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise AssetError("Unexpected upstream revision")
        if body == "microduck":
            complete_policies(body, checkout)
        else:
            bundle = checkout / "out/bundle_example/jumper"
            verify_digest(bundle / "bundle.json", JUMPER_MANIFEST)
            for relative, entry in json.loads((bundle / "bundle.json").read_text())["files"].items():
                path = (bundle / relative).resolve()
                if not path.is_relative_to(bundle.resolve()):
                    raise AssetError("Jumper bundle reference leaves its directory")
                verify_digest(path, entry["sha256"])
            for _, _, relative, digest in jumper_export_files():
                verify_digest(checkout / relative, digest)
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
