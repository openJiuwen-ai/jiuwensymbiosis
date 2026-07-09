# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""在本机标准缓存里定位已下好的视觉检测模型(纯逻辑,无 Qt / 无 nicegui)。

检测器要两个模型:开放词表检测的 GroundingDINO 与分割的 SAM2,任一缺失都起不来。
运行页「错误诊断」的「自动检测」按钮借此在 HF hub 各快照 + ModelScope 缓存里找一个
可直接加载的目录;指向本地目录还能顺带绕过「已缓存却仍联网校验」的坑。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

__all__ = [
    "GDINO_REPO",
    "SAM2_REPO",
    "HF_MIRROR",
    "looks_like_gdino_dir",
    "looks_like_sam2_dir",
    "cache_candidates",
    "detect_local_model",
]

HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
MODELSCOPE = Path.home() / ".cache" / "modelscope" / "hub" / "models"
GDINO_REPO = "IDEA-Research/grounding-dino-base"
SAM2_REPO = "facebook/sam2.1-hiera-large"
HF_MIRROR = "https://hf-mirror.com"


def _has_weights(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()


def looks_like_gdino_dir(path: Path) -> bool:
    """目录是否像一个可直接加载的 GroundingDINO 模型(有 config + 权重)。"""
    return (path / "config.json").is_file() and _has_weights(path)


def looks_like_sam2_dir(path: Path) -> bool:
    """目录是否像一个可直接加载的 SAM2 模型(有 config + processor_config + 权重)。"""
    return (path / "config.json").is_file() and (path / "processor_config.json").is_file() and _has_weights(path)


def cache_candidates(repo_id: str) -> list[Path]:
    """某 HF repo id 在本机标准缓存里可能的模型目录(HF hub 各快照 + ModelScope)。"""
    org, name = repo_id.split("/", 1)
    candidates: list[Path] = []
    snapshots = HF_HUB / f"models--{org}--{name}" / "snapshots"
    if snapshots.is_dir():
        candidates.extend(sorted(p for p in snapshots.iterdir() if p.is_dir()))
    candidates.append(MODELSCOPE / org / name)
    return candidates


def detect_local_model(repo_id: str, validator: Callable[[Path], bool]) -> Path | None:
    """在标准缓存里找一个通过 ``validator`` 的本地模型目录;没有则返回 ``None``。"""
    return next((p for p in cache_candidates(repo_id) if p.is_dir() and validator(p)), None)
