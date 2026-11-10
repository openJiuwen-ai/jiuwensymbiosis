"""Load simulator modules by location without core hooks or SDK path shadowing."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

_PACKAGE = "_jiuwensymbiosis_simulators"


def load_genesis_module(name: str) -> ModuleType:
    """Load a Genesis submodule in the isolated simulator package namespace."""
    if _PACKAGE not in sys.modules:
        entry = Path(__file__).resolve().parents[3] / "jiuwensymbiosis/simulators/__init__.py"
        spec = importlib.util.spec_from_file_location(_PACKAGE, entry)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load simulator package from {entry}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PACKAGE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[_PACKAGE]
            raise
    return importlib.import_module(f"{_PACKAGE}.genesis.{name}")


def configure_script_logging(logger: logging.Logger) -> None:
    """Configure this standalone process logger without importing core or changing root."""
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
