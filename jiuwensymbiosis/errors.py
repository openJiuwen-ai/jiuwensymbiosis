# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Machine-readable failure codes + the exception types that carry them.

A ``code`` names **what kind of failure happened**, decided at the failure site
where that is still certain, and travels with the failure (exception attribute →
step dict → GUI payload) so the presentation layer can look it up in a table
instead of re-deriving it by grepping error text. Codes carry no user-facing
wording — the Chinese diagnosis cards live in ``jiuwensymbiosis_gui.workbench.diagnostics``.

This module imports only ``jiuwensymbiosis.contracts`` (itself dependency-free):
perception / rails / the fast runner / the GUI all depend on it, so it stays a leaf
to keep those imports acyclic.
"""

from __future__ import annotations

from jiuwensymbiosis.contracts import DETECTION_REASONS, DetectionReason

__all__ = [
    "DETECTION_REASONS",
    "DetectionReason",
    "ERROR_CODES",
    "SAFETY_REJECTED",
    "GRASP_NOT_CONFIRMED",
    "DETECTOR_START_TIMEOUT",
    "JiuwenSymbiosisError",
    "SafetyViolationError",
    "DetectionError",
    "GraspNotConfirmedError",
    "DetectorStartError",
    "error_code",
]

SAFETY_REJECTED = "safety_rejected"
GRASP_NOT_CONFIRMED = "grasp_not_confirmed"
DETECTOR_START_TIMEOUT = "detector_start_timeout"

# The codes this module defines. Not a whitelist: adapters carry their own stable
# codes on the same ``.code`` attribute (``CartesianServoError`` →
# ``cartesian_bounds_rejected`` etc., already forwarded through ``ServoResult``),
# and those must survive the trip too. A consumer that has no entry for a code
# simply falls back to its generic handling.
ERROR_CODES = DETECTION_REASONS | {SAFETY_REJECTED, GRASP_NOT_CONFIRMED, DETECTOR_START_TIMEOUT}


class JiuwenSymbiosisError(Exception):
    """Base framework error carrying a machine-readable :attr:`code`."""

    code: str = ""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class SafetyViolationError(JiuwenSymbiosisError, ValueError):
    """A rail refused a command.

    Stays a ``ValueError``: the LLM self-correction contract and every existing
    ``except ValueError`` call site in motion code depend on that type.
    """

    code = SAFETY_REJECTED


class DetectionError(JiuwenSymbiosisError, RuntimeError):
    """Perception produced no usable result; ``code`` is one of ``DETECTION_REASONS``."""


class GraspNotConfirmedError(JiuwenSymbiosisError, RuntimeError):
    """The gripper closed without object contact."""

    code = GRASP_NOT_CONFIRMED


class DetectorStartError(JiuwenSymbiosisError, RuntimeError):
    """The detection sidecar never became reachable."""

    code = DETECTOR_START_TIMEOUT


def error_code(exc: BaseException) -> str:
    """The failure's ``code``, or ``""`` when it carries none."""
    code = getattr(exc, "code", "")
    return code if isinstance(code, str) else ""
