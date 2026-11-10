"""Known InteriorAgent MDL templates and asset-layout repairs.

These recipes approximate selected effects; they are not a general MDL compiler.
Only explicitly selected InteriorAgent assets should use RULES.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING or (__package__ and __package__ != "material_rules"):
    from ..material_surface import constant as _constant
    from ..material_surface import texture
else:
    from material_surface import constant as _constant
    from material_surface import texture

LEGACY = {"M_BaseMaterial", "M_BaseMaterial_twosided", "M_TranslucentMaterial"}
MODERN = {"M_BaseMaterial_v2", "M_BaseMaterial_v2_twosided", "M_TranslucentMaterial_v2"}
# Explicit recipes for the InteriorAgent MDL templates, not arbitrary MDL.
CLOTH = {
    "rongbu_Inst": ("3d66Mat-13918103-maps-4.png", "GeneratedName_16225", (0.5, 1.0)),
    "rongbu1": ("3d66Mat-13918078-maps-3.png", "GeneratedName_17050", (1.0, 1.0)),
    "rongbu2": ("3d66Mat-13918103-maps-5.png", "GeneratedName_11155", (1.0, 1.0)),
    "rongbu3": ("3d66Mat-13918103-maps-4.png", "GeneratedName_4743", (5.0, 5.0)),
}
EMISSION = "T_5f0d8e6ed849e100010ee906_emission_Mat"
MIRROR = "M_59268dbcbc974b45eaafd427"


def _value(shader, name, default):
    value = shader.GetInput(name).Get()
    return default if value is None else value


def _rgb(value):
    return tuple(float(v) for v in value[:3])


@lru_cache(maxsize=128)
def _texture_defaults(module: str, modified_ns: int) -> dict[str, str]:
    # Known templates may omit a USD input when its MDL default is intended.
    return dict(re.findall(r'uniform\s+texture_2d\s+(\w+)\s*=\s*texture_2d\("([^"]+)"', Path(module).read_text()))


def _module_path(shader, report):
    asset = shader.GetSourceAsset("mdl")
    if not asset:
        return None
    if asset.resolvedPath:
        return Path(asset.resolvedPath)
    parts = asset.path.replace("\\", "/").split("/")
    if "Materials" not in parts:
        return None
    start = parts.index("Materials")
    suffix = Path(*parts[start:])
    for spec in shader.GetPrim().GetAttribute("info:mdl:sourceAsset").GetPropertyStack():
        if not spec.layer.realPath:
            continue
        for parent in Path(spec.layer.realPath).parents:
            candidate = parent / suffix
            if candidate.is_file():
                report["issues"].append(
                    {"kind": "repaired_material_module_path", "original": asset.path, "resolved": str(candidate)}
                )
                return candidate
            if parent.name == "InteriorAgent":
                break
    return None


def _resolve_texture(shader, name, report):
    """Resolve at the authoring layer, never relative to the generated overlay."""
    value = shader.GetInput(name).Get()
    if value is None:
        path = _module_path(shader, report)
        if path:
            default = _texture_defaults(str(path), path.stat().st_mtime_ns).get(name)
            if default:
                candidate = path.parent / default
                if candidate.is_file():
                    return candidate.resolve()
    if not value or not value.path:
        report["issues"].append({"kind": "missing_texture", "shader": str(shader.GetPath()), "input": name})
        return None
    if value.resolvedPath and Path(value.resolvedPath).is_file():
        return Path(value.resolvedPath)
    # Some shipped assets spell ../Materials as .../Materials. Repair only an
    # exact Materials-relative suffix in this authoring asset's own ancestors.
    parts = value.path.replace("\\", "/").split("/")
    if "Materials" in parts:
        start = parts.index("Materials")
        suffix = Path(*parts[start:])
        for spec in shader.GetInput(name).GetAttr().GetPropertyStack():
            if not spec.layer.realPath:
                continue
            for parent in Path(spec.layer.realPath).parents:
                candidate = parent / suffix
                if candidate.is_file():
                    report["issues"].append(
                        {"kind": "repaired_texture_path", "original": value.path, "resolved": str(candidate)}
                    )
                    return candidate
                if parent.name == "InteriorAgent":
                    break
    report["issues"].append(
        {"kind": "missing_texture", "shader": str(shader.GetPath()), "input": name, "asset": value.path}
    )
    return None


def _uv_transform(shader, prefix, modern):
    """USD UV affine equivalent of the local MDL templates' two V flips."""
    uv = _value(shader, prefix + ("UVW" if modern else "_UVA"), (1, 1, 0, 0))
    sx, sy = float(uv[0]), float(uv[1])
    angle = -float(uv[2]) * 2 * math.pi if modern else float(uv[2])
    offset = _value(shader, prefix + "Offset", (0, 0, 0, 0)) if modern else (0, 0)
    x = float(offset[0]) - 0.5
    y = float(offset[1]) - 0.5 if modern else 0.5 - sy
    c, s = math.cos(angle), math.sin(angle)
    return (sx, sy), math.degrees(angle), (0.5 + c * x - s * y, 0.5 + s * x + c * y)


def _texture(stage, preview, source, name, target, report, *, path=None, **kwargs):
    path = path or _resolve_texture(source, name, report)
    if path is not None:
        texture(stage, preview, path, target, report, **kwargs)


def _legacy(stage, preview, shader, family, report):
    mappings = (
        ("BaseColor", "diffuseColor", (1, 1, 1, 1)),
        ("Metallic", "metallic", (0, 0, 0, 1)),
        ("Gloss", "roughness", (0.1, 0.1, 0.1, 1)),
        ("Emissive", "emissiveColor", (1, 1, 1, 1)),
    )
    for prefix, target, default in mappings:
        # Input names verified against the shipped templates (M_BaseMaterial.mdl
        # declares BaseColor_Color/Metallic_Color/...): an absent input is the
        # normal default-equals case in these assets, not a naming mismatch.
        value = _value(shader, prefix + "_Color", default)
        gain = float(_value(shader, "EmissiveIntensity", 0)) if prefix == "Emissive" else 1.0
        vector = target in {"diffuseColor", "emissiveColor"}
        value = tuple(v * gain for v in _rgb(value)) if vector else float(value[0])
        if prefix == "Gloss":
            value = 1 - value
        _constant(preview, target, value)
        if abs(float(_value(shader, "Is" + prefix + "Tex", 0)) - 1) < 1e-5 and gain != 0:
            scale, bias = ((-1, -1, -1, 1), (1, 1, 1, 0)) if prefix == "Gloss" else ((gain,) * 3 + (1,), None)
            _texture(
                stage,
                preview,
                shader,
                prefix + "_Tex",
                target,
                report,
                uv=_uv_transform(shader, prefix, False),
                color=True,
                scale=scale,
                bias=bias,
            )
    _constant(preview, "opacity", float(_value(shader, "Opacity", 1)))
    _constant(preview, "ior", float(_value(shader, "FresnelB", 1.5)))
    if float(_value(shader, "IsNormalTex", 0)) == 1:
        _constant(preview, "normal", (0.0, 0.0, 1.0))
        _texture(
            stage,
            preview,
            shader,
            "Normal_Tex",
            "normal",
            report,
            uv=_uv_transform(shader, "Normal", False),
            scale=(2, 2, 2, 1),
            bias=(-1, -1, -1, 0),
        )
    if "Translucent" in family:
        report["issues"].append(
            {
                "kind": "approximation",
                "shader": str(shader.GetPath()),
                "detail": "MDL transmission approximated by PreviewSurface opacity and IOR",
            }
        )


def _modern(stage, preview, shader, family, report):
    for prefix, target, default in (
        ("Color", "diffuseColor", (1.0, 1.0, 1.0)),
        ("Emission", "emissiveColor", (0.0, 0.0, 0.0)),
        ("Metallic", "metallic", 0.0),
        ("Roughness", "roughness", 0.0),
        ("Opacity", "opacity", 0.0 if "Translucent" in family else 1.0),
    ):
        vector = isinstance(default, tuple)
        # The doubled name is real: M_BaseMaterial_v2.mdl declares ``float4
        # ColorColor`` (v1 templates use ``X_Color`` instead), so this is a
        # verified template name, not a typo. An absent input is the normal
        # default-equals case in these assets, not a naming mismatch.
        value = _value(shader, prefix + ("Color" if vector else "Float"), default)
        mode = float(_value(shader, prefix + "IsTexType", 0))
        if mode >= 2:
            # PreviewSurface cannot express the MDL template's angle-dependent
            # two-lobe Fresnel blend. Retain its front-facing branch explicitly.
            mode = float(_value(shader, prefix + "FalloffValue1Type", 0))
            if mode >= 2:
                front = float(_value(shader, prefix + "FalloffFloat", 0))
                value = (front,) * 3 if vector else front
            report["issues"].append(
                {
                    "kind": "approximation",
                    "shader": str(shader.GetPath()),
                    "detail": prefix + " Fresnel blend uses front-facing branch",
                }
            )
        _constant(preview, target, _rgb(value) if vector else float(value))
        if abs(mode - 1) < 1e-5:
            _texture(
                stage,
                preview,
                shader,
                prefix + "Tex",
                target,
                report,
                uv=_uv_transform(shader, prefix, True),
                color=True,
            )
    if float(_value(shader, "IsNormal", 0)) == 1:
        gain = float(_value(shader, "NormalIntensity", 0))
        if gain:
            _constant(preview, "normal", (0.0, 0.0, 1.0))
            _texture(
                stage,
                preview,
                shader,
                "NormalTex",
                "normal",
                report,
                uv=_uv_transform(shader, "Normal", True),
                scale=(2 * gain, 2 * gain, 2, 1),
                bias=(-gain, -gain, -1, 0),
            )


def _mdl_texture(shader, filename, report):
    """Resolve texture literals in a known MDL template at that module's layer."""
    module = _module_path(shader, report)
    if module:
        path = module.parent / "Textures" / filename
        if path.is_file():
            return path
    report["issues"].append({"kind": "missing_texture", "shader": str(shader.GetPath()), "asset": filename})
    return None


def _world_grid(stage, preview, shader, report, directory) -> bool:
    """Bake the shipped grid template's three-frequency mask into one UV tile.

    This procedural template has a two-UV-unit period. Its algebra comes from
    WorldGridMaterial.mdl; the 1024px bake is explicitly a sampled approximation.
    """
    import hashlib

    import numpy as np
    from PIL import Image
    from scipy.ndimage import map_coordinates

    path = _mdl_texture(shader, "T_Default_Material_Grid_M.png", report)
    if path is None:
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    target = directory / "textures" / ("grid-" + digest)
    diffuse = target.with_name(target.name + "-color.png")
    roughness = target.with_name(target.name + "-rough.png")
    if not diffuse.is_file() or not roughness.is_file():
        mask = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
        size = 1024
        u, v = np.meshgrid((np.arange(size) + 0.5) / size * 2, (np.arange(size) + 0.5) / size * 2)

        def sample(frequency):
            coordinates = [(v * frequency % 1) * mask.shape[0] - 0.5, (u * frequency % 1) * mask.shape[1] - 0.5]
            return np.stack(
                [map_coordinates(mask[..., c], coordinates, order=1, mode="grid-wrap") for c in range(3)], axis=-1
            )

        fine, medium, coarse = sample(20), sample(5), sample(0.5)
        a = 0.4 + 0.6 * fine[..., 0]
        mixed = a * (1 - medium[..., 1]) + (1 - a) * medium[..., 1]
        color = mixed * (0.295 + 0.365 * (0.5 * (coarse[..., 1] + medium[..., 1]) + 0.25)) * 0.5
        rough = np.clip(0.5 * coarse[..., 1] + 0.7 + 0.3 * medium[..., 1], 0, 1)
        # Store color in sRGB; numeric roughness remains linear.
        srgb = np.where(color <= 0.0031308, 12.92 * color, 1.055 * color ** (1 / 2.4) - 0.055)
        diffuse.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(np.clip(srgb, 0, 1) * 255)).convert("RGB").save(diffuse)
        Image.fromarray(np.uint8(rough * 255)).convert("RGB").save(roughness)
    for name, path, color in (("diffuseColor", diffuse, True), ("roughness", roughness, False)):
        _constant(preview, name, (0.1, 0.1, 0.1) if color else 0.8)
        _texture(stage, preview, shader, "", name, report, path=path, color=color, uv=((0.5, 0.5), 0.0, (0.0, 0.0)))
    normal = _mdl_texture(shader, "T_Default_Material_Grid_N.png", report)
    if normal:
        _constant(preview, "normal", (0.0, 0.0, 1.0))
        _texture(
            stage,
            preview,
            shader,
            "",
            "normal",
            report,
            path=normal,
            uv=((10.0, 10.0), 0.0, (0.0, 0.0)),
            scale=(0.6, 0.6, 2, 1),
            bias=(-0.3, -0.3, -1, 0),
        )
    report["issues"].append(
        {
            "kind": "approximation",
            "shader": str(shader.GetPath()),
            "detail": "WorldGridMaterial procedural color and roughness baked to a 1024px periodic tile",
        }
    )
    return True


def _other(stage, preview, shader, family, report, directory) -> bool:
    if family == "BasicShapeMaterial":
        _constant(preview, "diffuseColor", _rgb(_value(shader, "Color", (0.9, 0.9, 0.9))))
        _constant(preview, "roughness", float(_value(shader, "Roughness", 0.6407)))
    elif family == "OmniGlass":
        _constant(preview, "diffuseColor", _rgb(_value(shader, "glass_color", (1, 1, 1))))
        _constant(preview, "roughness", float(_value(shader, "frosting_roughness", 0)))
        _constant(preview, "ior", float(_value(shader, "glass_ior", 1.5)))
        _constant(preview, "opacity", 0.15)
        report["issues"].append(
            {
                "kind": "approximation",
                "shader": str(shader.GetPath()),
                "detail": (
                    "OmniGlass volume absorption/transmission replaced by thin transparent PreviewSurface opacity=0.15"
                ),
            }
        )
    elif family == "OmniSurface":
        _constant(preview, "diffuseColor", _rgb(_value(shader, "diffuse_reflection_color", (0.8, 0.8, 0.8))))
        _constant(preview, "roughness", float(_value(shader, "specular_reflection_roughness", 0.5)))
        image = shader.GetInput("diffuse_reflection_color_image").Get()
        if image and image.path:
            _texture(stage, preview, shader, "diffuse_reflection_color_image", "diffuseColor", report, color=True)
        report["issues"].append(
            {
                "kind": "approximation",
                "shader": str(shader.GetPath()),
                "detail": "OmniSurface subsurface and diffuse transmission are not supported by PreviewSurface",
            }
        )
    elif family in CLOTH:
        diffuse_filename, roughness_input, uv_scale = CLOTH[family]
        gain = float(_value(shader, roughness_input, 0.2))
        for target, filename, color, scale, bias in (
            ("diffuseColor", diffuse_filename, True, None, None),
            ("roughness", "3d66Mat-10966853-maps-2.png", True, (gain, gain, gain, 1), None),
            ("normal", "22.png", False, (2, 2, 2, 1), (-1, -1, -1, 0)),
        ):
            _constant(
                preview,
                target,
                0.2 if target == "roughness" else (0.0, 0.0, 1.0) if target == "normal" else (0.5, 0.5, 0.5),
            )
            path = _mdl_texture(shader, filename, report)
            if path:
                _texture(
                    stage,
                    preview,
                    shader,
                    "",
                    target,
                    report,
                    path=path,
                    color=color,
                    scale=scale,
                    bias=bias,
                    uv=(uv_scale, 0.0, (0.0, 1 - uv_scale[1])),
                )
    elif family in {MIRROR, "boli"}:
        glass = family == "boli"
        _constant(
            preview,
            "diffuseColor",
            _rgb(
                _value(
                    shader,
                    "GeneratedName_26377" if glass else "GeneratedName_11172",
                    (0.008078, 0.041667, 0.023266) if glass else (1, 1, 1),
                )
            ),
        )
        _constant(preview, "metallic", float(_value(shader, "Metallic", 1)))
        _constant(
            preview,
            "roughness",
            float(_value(shader, "GeneratedName_18017" if glass else "GeneratedName_8822", 0.01 if glass else 0)),
        )
        if glass:
            _constant(preview, "opacity", min(1.0, 0.15 * float(_value(shader, "GeneratedName_2663", 1.5))))
            report["issues"].append(
                {
                    "kind": "approximation",
                    "shader": str(shader.GetPath()),
                    "detail": "boli view-dependent opacity uses its front-facing Fresnel value",
                }
            )
    elif family == EMISSION:
        path = _mdl_texture(shader, "T_5f0d8e6ed849e100010ee906_emission.png", report)
        for target, gain in (("diffuseColor", 1), ("emissiveColor", 2)):
            _constant(preview, target, (0.0, 0.0, 0.0))
            if path:
                _texture(stage, preview, shader, "", target, report, path=path, color=True, scale=(gain, gain, gain, 1))
    elif family == "WorldGridMaterial":
        return _world_grid(stage, preview, shader, report, directory)
    else:
        return False
    return True


def _convert(stage, preview, shader, family, report, directory) -> bool:
    if family in LEGACY:
        _legacy(stage, preview, shader, family, report)
    elif family in MODERN:
        _modern(stage, preview, shader, family, report)
    else:
        return _other(stage, preview, shader, family, report, directory)
    return True


RULES = dict.fromkeys(
    LEGACY
    | MODERN
    | set(CLOTH)
    | {"BasicShapeMaterial", "OmniGlass", "OmniSurface", "boli", MIRROR, EMISSION, "WorldGridMaterial"},
    _convert,
)
