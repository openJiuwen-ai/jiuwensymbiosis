"""Direct protocol errors, including explicit proof of pre-dispatch rejection."""

from __future__ import annotations


class DriverError(RuntimeError):
    def __init__(self, code: str, message: str, *, uncertain: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.uncertain = uncertain

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "uncertain": self.uncertain}


class CommandNotSent(DriverError):
    """Explicit adapter proof that the requested action never reached wire I/O.

    Ordinary DriverError.uncertain=False is not this proof: vendor errors may
    occur after writes. Only the bounded pre-dispatch section raises this type.
    """
