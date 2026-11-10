"""Explicitly selected asset-specific material conversion rules.

A library root declares which recipe set applies to it (the ``material_rules``
key in ``assets.yaml``); ``rules_for`` resolves that declared key here. Adding a
second source means adding one entry to ``_RULES_MODULES`` — no generic module
changes. An unknown key warns loudly instead of silently rendering unconverted.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..material_repair import MaterialRules

logger = logging.getLogger("jiuwensymbiosis.simulators.genesis.material_rules")

# Declared key -> module providing ``RULES``. Lazy, so importing this package
# never pulls pxr or any vendor recipe set.
_RULES_MODULES: dict[str, str] = {"interioragent": "interioragent"}


def rules_for(key: str) -> MaterialRules | None:
    """Resolve a declared key to its rule set; unknown keys warn and convert nothing."""
    module_name = _RULES_MODULES.get(key)
    if module_name is None:
        logger.warning("Unknown material_rules key %r; known keys: %s", key, sorted(_RULES_MODULES))
        return None
    # Package import, or the isolated worker's top-level sibling-module import.
    module = import_module("." + module_name, __package__) if __package__ else import_module(module_name)
    return module.RULES
