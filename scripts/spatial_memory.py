"""Spatial-memory CLI with JSON stdout, including during SDK initialization."""

from __future__ import annotations

import contextlib
import sys


def main() -> None:
    # The existing core package initializes third-party SDK loggers on import.
    # Keep their startup notices off the machine-readable result channel.
    with contextlib.redirect_stdout(sys.stderr):
        from jiuwensymbiosis.spatial_memory.__main__ import main as run

    run()


if __name__ == "__main__":
    main()
