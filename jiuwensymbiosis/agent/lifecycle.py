"""Observable cleanup evidence shared by sessions and resource owners."""

from collections.abc import Sequence
from dataclasses import dataclass


class HardwareCleanupError(RuntimeError):
    """Hardware rollback failed and a released resource cannot be confirmed."""

    def __init__(
        self, operation: str, original_error: BaseException | None = None, cleanup_errors: Sequence[BaseException] = ()
    ) -> None:
        self.operation = operation
        self.original_error = original_error
        self.cleanup_errors = tuple(cleanup_errors)
        details = [f"{type(exc).__name__}: {exc}" for exc in self.cleanup_errors]
        if original_error is not None:
            details.insert(0, f"original {type(original_error).__name__}: {original_error}")
        super().__init__(f"{operation}: hardware cleanup unconfirmed" + ("; " + "; ".join(details) if details else ""))


@dataclass(frozen=True)
class CleanupReport:
    """A resource is reusable only after all work and cleanup are confirmed."""

    pending_work: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    connected: bool = False

    @property
    def released(self) -> bool:
        return not self.pending_work and not self.errors and not self.connected
