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

from typing import Any

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
    "INFERENCE_ERROR_CODES",
    "InferenceServiceError",
    "error_code",
    "find_skip_recovery_exception",
    "is_inference_service_failure",
    "SAFETY_ERROR_CODE_KEY",
]

SAFETY_REJECTED = "safety_rejected"
# Set on the callback context when SafetyRail rejects a call, so the fast
# executor can still report the machine code for an aborted call (the framework
# discards the ToolOutput, and with it ``data["error_code"]``).
SAFETY_ERROR_CODE_KEY = "jiuwensymbiosis.safety_error_code"
GRASP_NOT_CONFIRMED = "grasp_not_confirmed"
DETECTOR_START_TIMEOUT = "detector_start_timeout"

# The codes this module defines. Not a whitelist: adapters carry their own stable
# codes on the same ``.code`` attribute (``CartesianServoError`` →
# ``cartesian_bounds_rejected`` etc., already forwarded through ``ServoResult``),
# and those must survive the trip too. A consumer that has no entry for a code
# simply falls back to its generic handling.
INFERENCE_ERROR_CODES = frozenset(
    {
        "inference_unavailable",
        "inference_timeout",
        "inference_busy",
        "inference_protocol_error",
        "inference_result_stale",
    }
)
ERROR_CODES = DETECTION_REASONS | {SAFETY_REJECTED, GRASP_NOT_CONFIRMED, DETECTOR_START_TIMEOUT} | INFERENCE_ERROR_CODES


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

    ``skip_recovery`` is the same opt-out marker adapter errors use. The marker
    says the *raiser* judged recovery unnecessary — typically a pre-dispatch
    envelope check where nothing moved — not that the arm provably never moved.
    Adapters that detect a violation after commands were sent must clear it
    (see SO-101's tracking violation) so ``RecoveryRail`` still releases/homes.
    The rail honours it through the ``AbortError(cause=...)`` wrapper the
    callback framework re-raises.
    """

    code = SAFETY_REJECTED
    skip_recovery = True


class DetectionError(JiuwenSymbiosisError, RuntimeError):
    """Perception produced no usable result; ``code`` is one of ``DETECTION_REASONS``."""


class GraspNotConfirmedError(JiuwenSymbiosisError, RuntimeError):
    """The gripper closed without object contact."""

    code = GRASP_NOT_CONFIRMED


class DetectorStartError(JiuwenSymbiosisError, RuntimeError):
    """The detection sidecar never became reachable."""

    code = DETECTOR_START_TIMEOUT


class InferenceServiceError(JiuwenSymbiosisError, RuntimeError):
    """A model service failed, distinct from a successful empty observation."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "inference_unavailable",
        service: str = "",
        request_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code)
        self.service = service
        self.request_id = request_id
        self.retryable = retryable


def error_code(exc: BaseException) -> str:
    """The failure's ``code``, or ``""`` when it carries none."""
    code = getattr(exc, "code", "")
    return code if isinstance(code, str) else ""


def is_inference_service_failure(result: Any) -> bool:
    """True when a result dict attributes its failure to the model service.

    Distinguishes a dead or unavailable inference service from a successful
    empty observation; callers raise :class:`InferenceServiceError` from it.
    """
    return (
        isinstance(result, dict)
        and not result.get("ok")
        and (
            result.get("reason") in {"detector_unavailable", "no_detector"}
            or str(result.get("error_code", "")).startswith("inference_")
        )
    )


def find_skip_recovery_exception(exception: Any) -> BaseException | None:
    """Find a ``skip_recovery`` opt-out marker through exception wrappers.

    OpenJiuwen wraps tool failures in ``AbilityExecutionError``; adapters wrap
    driver errors again. The marker therefore has to be searched through the
    explicit cause chain (``__cause__`` / ``cause`` / …), never through
    ``__context__``: an error raised while *handling* a marked error is a real
    failure and must keep its recovery.

    Returns an exception that explicitly declares the marker, or ``None``.
    The rule is monotonic and fail-safe in the recovery direction: any explicit
    ``skip_recovery = False`` in the chain wins over a ``True`` declared
    elsewhere, so a wrapper can clear an opt-out inherited from its cause with
    the usual ``raise X(...) from exc``. Unmarked wrappers are transparent.
    Inspection is fail-safe: a wrapper property that raises is treated like a
    missing attribute.
    """
    if not isinstance(exception, BaseException):
        return None
    pending: list[BaseException] = [exception]
    visited: set[int] = set()
    marked_candidate: BaseException | None = None
    wrapper_attrs = (
        "__cause__",
        "cause",
        "exception",
        "original_exception",
        "original_error",
    )
    while pending:
        current = pending.pop()
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)
        try:
            marker = current.skip_recovery
        except AttributeError:
            marker = None
        except Exception:  # noqa: BLE001 - marker lookup must be fail-safe
            marker = None
        if marker is not None:
            if not marker:
                # Post-dispatch failures declare False; recovery must win even
                # when an outer wrapper declares True.
                return None
            if marked_candidate is None:
                marked_candidate = current
        for attr in wrapper_attrs:
            try:
                nested = getattr(current, attr, None)
            except Exception:  # noqa: BLE001 - wrapper inspection must be fail-safe
                continue
            if isinstance(nested, BaseException) and id(nested) not in visited:
                pending.append(nested)
    return marked_candidate
