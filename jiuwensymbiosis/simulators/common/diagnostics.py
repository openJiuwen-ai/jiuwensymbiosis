"""Bounded failure diagnostics without exception payloads or traceback locals."""

from __future__ import annotations

from errno import EMFILE, ENFILE

from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)


def is_file_descriptor_limit(error: BaseException) -> bool:
    """Recognize the OS limit even when a local transport wraps the error."""
    causes: set[int] = set()
    cause: BaseException | None = error
    while cause is not None and id(cause) not in causes:
        if isinstance(cause, OSError) and cause.errno in {EMFILE, ENFILE}:
            return True
        causes.add(id(cause))
        cause = cause.__cause__
    return False


def report_failure(code: str, error: Exception) -> None:
    """Log a caller-owned event code and a fixed, payload-free error category.

    Vendor exception messages, reprs and even custom class names may contain
    credentials or camera content. Only builtin categories leave this boundary.
    """
    categories = (
        TimeoutError,
        PermissionError,
        FileNotFoundError,
        ConnectionError,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
    )
    category = next((kind.__name__ for kind in categories if isinstance(error, kind)), "Exception")
    logger.warning("%s (%s)", code, category)
