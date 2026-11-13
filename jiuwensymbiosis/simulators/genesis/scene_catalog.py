"""Genesis material recipes selected by the declared asset source."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..assets import ResolvedAsset
    from .material_repair import MaterialRules


def material_rules_for_asset(asset: ResolvedAsset | None) -> MaterialRules | None:
    """Select recipes from the root's declared ``material_rules`` key, never paths or shader names."""
    if asset is None:
        return None
    key = getattr(asset.asset, "material_rules", None)
    if not key:
        return None
    if TYPE_CHECKING or __package__:
        from .material_rules import rules_for
    else:
        from material_rules import rules_for
    return rules_for(key)
