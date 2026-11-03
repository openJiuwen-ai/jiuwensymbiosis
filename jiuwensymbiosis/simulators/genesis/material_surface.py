"""PreviewSurface authoring and texture baking, independent of asset templates."""

from pathlib import Path
from typing import TYPE_CHECKING


def texture(
    stage, preview, path: Path, target: str, report: dict, *, uv=None, color=False, scale=None, bias=None
) -> None:
    from pxr import Sdf, UsdShade

    if TYPE_CHECKING or __package__:
        from .material_baking import bake_texture
    else:
        from material_baking import bake_texture

    path, color, baked = bake_texture(path, Path(report["texture_directory"]), target, uv, color, scale, bias)
    if baked:
        report["baked_textures"].append(baked)
    tex = UsdShade.Shader.Define(stage, preview.GetPath().AppendChild(target + "Texture"))
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(path)))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if color else "raw")
    for axis in ("S", "T"):
        tex.CreateInput("wrap" + axis, Sdf.ValueTypeNames.Token).Set("repeat")
    if target == "normal":
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2, 2, 2, 1))
        tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1, -1, -1, 0))
    reader = UsdShade.Shader.Define(stage, tex.GetPath().AppendChild("UV"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    vector = target in {"diffuseColor", "normal", "emissiveColor", "specularColor"}
    channel = "rgb" if vector else "r"
    tex.CreateOutput(channel, Sdf.ValueTypeNames.Float3 if vector else Sdf.ValueTypeNames.Float)
    preview.GetInput(target).ConnectToSource(tex.ConnectableAPI(), channel)
    report["textures"] += 1


def constant(preview, target: str, value) -> None:
    from pxr import Sdf

    kind = Sdf.ValueTypeNames.Float
    if isinstance(value, tuple):
        kind = Sdf.ValueTypeNames.Normal3f if target == "normal" else Sdf.ValueTypeNames.Color3f
    preview.CreateInput(target, kind).Set(value)
