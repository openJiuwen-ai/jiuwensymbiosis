"""Write UsdPreviewSurface inputs as constants or as sampled (baked) textures.

The sampling graph is expressed as data: a node is a shader id plus a mapping
of input name to (value type name, value), and ``_author_node`` turns one such
description into a USD shader. The PreviewSurface wiring therefore lives in one
declarative place, independent of any asset template.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

# PreviewSurface inputs read three texture channels; all other targets read one.
_THREE_CHANNEL_TARGETS = frozenset({"diffuseColor", "normal", "emissiveColor", "specularColor"})
# Tangent-space normal maps are stored in [0, 1] and remapped to [-1, 1].
_NORMAL_REMAP: dict[str, tuple[str, Any]] = {
    "scale": ("Float4", (2, 2, 2, 1)),
    "bias": ("Float4", (-1, -1, -1, 0)),
}


def _value_type(name: str) -> Any:
    """Resolve an ``Sdf.ValueTypeNames`` member by name, e.g. ``"Token"``."""
    from pxr import Sdf

    return getattr(Sdf.ValueTypeNames, name)


def _author_node(stage: Any, path: Any, shader_id: str, values: dict[str, tuple[str, Any]]) -> Any:
    """Define one shader prim and set its constant inputs from ``values``."""
    from pxr import UsdShade

    node = UsdShade.Shader.Define(stage, path)
    node.CreateIdAttr(shader_id)
    for name, (type_name, value) in values.items():
        node.CreateInput(name, _value_type(type_name)).Set(value)
    return node


def texture(
    stage, preview, path: Path, target: str, report: dict, *, uv=None, color=False, scale=None, bias=None
) -> None:
    """Bake ``path`` if needed, then drive ``preview``'s ``target`` input from it."""
    from pxr import Sdf

    if TYPE_CHECKING or __package__:
        from .material_baking import bake_texture
    else:
        from material_baking import bake_texture

    path, color, baked = bake_texture(path, Path(report["texture_directory"]), target, uv, color, scale, bias)
    if baked:
        report["baked_textures"].append(baked)

    sampler_values: dict[str, tuple[str, Any]] = {
        "file": ("Asset", Sdf.AssetPath(str(path))),
        "sourceColorSpace": ("Token", "sRGB" if color else "raw"),
        "wrapS": ("Token", "repeat"),
        "wrapT": ("Token", "repeat"),
    }
    if target == "normal":
        sampler_values.update(_NORMAL_REMAP)
    sampler_path = preview.GetPath().AppendChild(target + "Texture")
    sampler = _author_node(stage, sampler_path, "UsdUVTexture", sampler_values)
    coordinates = _author_node(
        stage, sampler_path.AppendChild("UV"), "UsdPrimvarReader_float2", {"varname": ("Token", "st")}
    )
    st = coordinates.CreateOutput("result", _value_type("Float2"))
    sampler.CreateInput("st", _value_type("Float2")).ConnectToSource(st)

    channel, type_name = ("rgb", "Float3") if target in _THREE_CHANNEL_TARGETS else ("r", "Float")
    preview.GetInput(target).ConnectToSource(sampler.CreateOutput(channel, _value_type(type_name)))
    report["textures"] += 1


def constant(preview, target: str, value) -> None:
    """Set ``target`` to a scalar, or to a color/normal triple when ``value`` is a tuple."""
    if not isinstance(value, tuple):
        type_name = "Float"
    elif target == "normal":
        type_name = "Normal3f"
    else:
        type_name = "Color3f"
    preview.CreateInput(target, _value_type(type_name)).Set(value)
