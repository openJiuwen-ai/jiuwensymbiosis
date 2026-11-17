"""Portable shader-family conversions applied to every USD asset.

No asset-root opt-in is required. Transparency alone does not imply glass:
only the explicitly recognized OmniGlass family requests native transmission.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING or (__package__ and __package__ != "material_rules"):
    from ..material_surface import constant
else:
    from material_surface import constant


def _omniglass(stage, preview, shader, family, report, directory) -> bool:
    """Keep the established thin-shell approximation and its fidelity warning."""

    def value(name, default):
        authored = shader.GetInput(name).Get()
        return default if authored is None else authored

    preview.GetPrim().SetCustomDataByKey("symbiosis:nativeGlass", True)
    constant(preview, "diffuseColor", tuple(float(v) for v in value("glass_color", (1, 1, 1))[:3]))
    constant(preview, "roughness", float(value("frosting_roughness", 0)))
    constant(preview, "ior", float(value("glass_ior", 1.5)))
    constant(preview, "opacity", 0.15)
    report["issues"].append(
        {
            "kind": "approximation",
            "shader": str(shader.GetPath()),
            "detail": "OmniGlass uses native double-sided thin transmission; volume absorption/thickness remain approximate",
        }
    )
    return True


RULES = {"OmniGlass": _omniglass}
