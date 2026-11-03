"""Non-destructive USD material preparation using explicitly supplied rules.

No asset-specific rules are selected here. With no rules, source materials stay
unchanged and unsupported families are reported for the caller to inspect.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, TypedDict

logger = logging.getLogger("jiuwensymbiosis.simulators.genesis.material_repair")
SUPPORTED = {"UsdPreviewSurface", "OmniPBR"}


class MaterialReport(TypedDict):
    """What one ``prepare_materials`` run established; converters append to it."""

    source: str
    converted: int
    textures: int
    materials: list[dict[str, str]]
    remaining: list[dict[str, str]]
    issues: list[dict[str, str]]
    baked_textures: list[dict[str, str]]
    texture_directory: str


class MaterialConverter(Protocol):
    """Author the preview shader for one unsupported family.

    Returns False when the family cannot be converted, recording missing inputs
    and approximation details in ``report["issues"]``. Raising is allowed: the
    caller isolates the failure to this one material instead of the whole scene.
    """

    def __call__(
        self, stage: Any, preview: Any, shader: Any, family: str, report: MaterialReport, directory: Path
    ) -> bool:
        """Convert ``shader`` into ``preview``; see the class docstring for the contract."""


MaterialRules = Mapping[str, MaterialConverter]


def prepare_materials(source: Path, directory: Path, *, rules: MaterialRules | None = None) -> tuple[Path, dict]:
    """Repair unsupported surface families before Nyx substitutes default gray.

    Supported materials and original files stay unchanged. Unknown graphs remain
    visible in the report and are NOT counted as repaired. The returned layer
    carries only material overrides; source transforms/geometry are referenced.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    source = Path(source).resolve(strict=True)
    original = Usd.Stage.Open(str(source))
    if original is None:
        raise ValueError(f"Cannot inspect USD materials: {source}")
    report: dict[str, Any] = {
        "source": str(source),
        "converted": 0,
        "textures": 0,
        "materials": [],
        "remaining": [],
        "issues": [],
        "baked_textures": [],
        "texture_directory": str(Path(directory).resolve() / "textures"),
    }
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths = [str(source)]
    # Stage metadata does not compose from a sublayer. Without copying units,
    # metre-scale rooms become centimetre-scale and appear to vanish in Nyx.
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
    if original.GetDefaultPrim():
        stage.SetDefaultPrim(stage.GetPrimAtPath(original.GetDefaultPrim().GetPath()))
    # Work on this disposable composition, deinstancing only when authoring an
    # override inside an instance is necessary. Original assets stay instanceable.
    materials = [
        p.GetPath() for p in Usd.PrimRange.Stage(original, Usd.TraverseInstanceProxies()) if p.IsA(UsdShade.Material)
    ]
    for path in materials:
        material = UsdShade.Material(stage.GetPrimAtPath(path))
        shader, _, _ = material.ComputeSurfaceSource()
        if not shader:
            shader, _, _ = material.ComputeSurfaceSource("mdl")
        if not shader:
            continue
        family = str(shader.GetShaderId() or shader.GetSourceAssetSubIdentifier("mdl"))
        if family in SUPPORTED:
            continue
        convert = (rules or {}).get(family)
        if convert is None:
            report["remaining"].append({"path": str(path), "family": family})
            continue
        ancestors = []
        shader_path = shader.GetPath()
        parent = material.GetPrim().GetParent()
        while parent and not parent.IsPseudoRoot():
            ancestors.append(parent.GetPath())
            parent = parent.GetParent()
        for parent_path in reversed(ancestors):
            parent = stage.GetPrimAtPath(parent_path)
            if parent.IsInstance():
                stage.OverridePrim(parent_path).SetInstanceable(False)
        # Deinstancing invalidates handles to the old instance proxies.
        material = UsdShade.Material(stage.GetPrimAtPath(path))
        shader = UsdShade.Shader(stage.GetPrimAtPath(shader_path))
        preview = UsdShade.Shader.Define(stage, path.AppendChild("__NyxPreviewSurface"))
        preview.CreateIdAttr("UsdPreviewSurface")
        # One unreadable texture or template must not sink the whole scene: the
        # failure is isolated to this material and reported, never propagated.
        textures_before, baked_before = report["textures"], len(report["baked_textures"])
        try:
            converted = convert(stage, preview, shader, family, report, Path(directory).resolve())
        except Exception as exc:  # noqa: BLE001 - best-effort repair, not a hard scene failure
            converted = False
            report["issues"].append(
                {"kind": "conversion_error", "shader": str(shader_path), "detail": f"{type(exc).__name__}: {exc}"}
            )
        if not converted:
            stage.RemovePrim(preview.GetPath())
            report["textures"] = textures_before
            del report["baked_textures"][baked_before:]
            report["remaining"].append({"path": str(path), "family": family, "reason": "conversion_unavailable"})
            continue
        material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        report["converted"] += 1
        report["materials"].append({"path": str(path), "from": family, "to": "UsdPreviewSurface"})
    if report["remaining"]:
        logger.warning(
            "Nyx material repair: %d unknown surface shaders remain unsupported; see material report",
            len(report["remaining"]),
        )
    if report["converted"]:
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "nyx-materials.usda"
        if destination == source:
            raise ValueError("Material repair cannot overwrite its source")
        stage.GetRootLayer().Export(str(destination))
        (directory / "material-repair.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        logger.info(
            "Nyx material repair: %d converted, %d textures, %d unsupported",
            report["converted"],
            report["textures"],
            len(report["remaining"]),
        )
        return destination, report
    return source, report
