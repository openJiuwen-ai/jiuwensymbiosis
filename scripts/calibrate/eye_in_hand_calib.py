#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Eye-in-hand console facade with an explicit legacy compatibility boundary.

The established eye-in-hand CLI owns options such as ``--selftest`` and
``--generate-board``.  New archive-oriented modes are dispatched to the
body-agnostic workflow, while a bare/boolean ``--auto`` remains legacy so its
meaning cannot change underneath existing scripts.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("jiuwensymbiosis.calibration.eye_in_hand_cli")

_LEGACY_NOTICE = (
    "[calib] compatibility mode: this invocation uses the legacy Piper eye-in-hand wizard. "
    "Use --collect-poses, --auto WAYPOINT_ARCHIVE, or --replay STATION_ARCHIVE "
    "for the unified calibration workflow."
)


def _use_unified(argv: list[str]) -> bool:
    """Return whether ``argv`` uses an unambiguous archive workflow mode."""
    if "--help" in argv:
        return False
    for index, token in enumerate(argv):
        if token == "--collect-poses" or token.startswith("--collect-poses="):
            return True
        if token == "--replay" or token.startswith("--replay="):
            return True
        if token.startswith("--auto="):
            return bool(token.partition("=")[2])
        if token == "--auto" and index + 1 < len(argv) and not argv[index + 1].startswith("-"):
            return True
    return False


def _run_unified(argv: list[str]) -> int:
    from scripts.calibrate.hand_eye_calib import main as unified_main

    return unified_main(argv, prog="eye_in_hand_calib.py", require_mount="eye_in_hand")


def _run_legacy(argv: list[str]) -> int:
    from scripts.calibrate.calibrate_hand_eye import main as legacy_main

    return legacy_main(argv)


def main(argv: list[str] | None = None) -> int:
    """Dispatch archive modes to unified workflow and all other modes to legacy CLI."""
    args = list(sys.argv[1:] if argv is None else argv)
    if _use_unified(args):
        return _run_unified(args)
    logger.warning(_LEGACY_NOTICE)
    return _run_legacy(args)


if __name__ == "__main__":
    raise SystemExit(main())
