"""Filesystem defaults shared by the host and the isolated simulator worker.

Standard library only: the assets directory entry point also imports this
module without initializing jiuwensymbiosis or the Agent SDK.
"""

from __future__ import annotations

import os
from pathlib import Path


def simulator_root() -> Path:
    """Use an explicit data root, a source checkout, or a writable user directory.

    Set JIUWENSYMBIOSIS_SIMULATOR_HOME before launching the host/asset CLI.
    Relative overrides are rejected so host and worker cannot disagree by cwd.
    Merely resolving a path never creates files or directories.
    """
    configured = os.environ.get("JIUWENSYMBIOSIS_SIMULATOR_HOME")
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ValueError("JIUWENSYMBIOSIS_SIMULATOR_HOME must be an absolute path")
        return root.resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "configs/simulators").is_dir():
        return checkout
    return Path.home() / ".jiuwensymbiosis" / "simulators"
