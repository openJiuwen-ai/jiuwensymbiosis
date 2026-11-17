"""Camera display settings; material conversion is decided by the root's declared rules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

PROFILE_PATH = Path(__file__).with_name("data") / "genesis_appearance.yaml"
TONE_MAPPERS = {"reinhard": "Reinhard", "neutral": "Neutral", "aces_1_3": "ACES_1_3"}


@dataclass(frozen=True)
class ColorManagement:
    """Display transform only; the source light rig and material values stay intact."""

    tone_mapper: str = "neutral"
    exposure_adjust_ev: float = 0.0

    def effective_parameters(self, base_exposure_ev: float) -> dict:
        """Report the exact camera exposure without changing lighting calibration."""
        base = _finite_number(base_exposure_ev, "基础曝光")
        effective = _finite_number(base + self.exposure_adjust_ev, "最终曝光")
        return {
            "tone_mapper": self.tone_mapper,
            "base_exposure_ev": base,
            "exposure_adjust_ev": self.exposure_adjust_ev,
            "effective_exposure_ev": effective,
        }


@dataclass(frozen=True)
class AppearanceProfile:
    """Shared camera color settings, independent of asset identity."""

    color_management: ColorManagement = field(default_factory=ColorManagement)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}必须是有限数值。")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label}必须是有限数值。") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}必须是有限数值。")
    return result


def load_appearance_profile() -> AppearanceProfile:
    """Read the shared display policy; scene-specific overrides are not accepted."""
    import yaml

    document = yaml.safe_load(PROFILE_PATH.read_text())
    if not isinstance(document, dict) or set(document) != {"color_management"}:
        raise ValueError("显示配置必须只包含全局 color_management。")
    raw = document
    color = raw.get("color_management", {})
    if not isinstance(color, dict) or set(color) - {"tone_mapper", "exposure_adjust_ev"}:
        raise ValueError("不支持的 color_management 配置。")
    tone_mapper = color.get("tone_mapper", "neutral")
    if not isinstance(tone_mapper, str) or tone_mapper not in TONE_MAPPERS:
        raise ValueError("tone_mapper 必须为 reinhard、neutral 或 aces_1_3。")
    adjustment = _finite_number(color.get("exposure_adjust_ev", 0.0), "曝光调整")
    return AppearanceProfile(ColorManagement(tone_mapper, adjustment))
