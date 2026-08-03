# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""随包 configs/so101/so101.yaml 能被 So101Config.from_dict 构建(GUI 真机会话经 from_dict)。

守住 GUI 真机会话与该配置文件的契约:env.cfg.low_level 嵌套结构 + upstream 字段命名
(place_z_offset_mm 等)可被当前 So101Config 直接加载。随包配置为真实左臂实验配置(带相机)。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from jiuwensymbiosis.adapters.so101.config import So101Config
from jiuwensymbiosis.adapters.so101.env import So101Env


def _repo_root() -> Path:
    import jiuwensymbiosis

    return Path(jiuwensymbiosis.__file__).resolve().parent.parent


def _load_shipped_config() -> dict:
    path = _repo_root() / "configs" / "so101" / "so101.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_shipped_config_builds_via_from_dict():
    cfg = So101Config.from_dict(_load_shipped_config())
    assert cfg.name == "so101"
    assert cfg.port  # 串口必填
    assert cfg.place_z_offset_mm is not None  # upstream 放置偏移字段可加载


def test_shipped_config_camera_enables_vision_capabilities():
    cfg = So101Config.from_dict(_load_shipped_config())
    caps = So101Env(cfg).capabilities
    assert "grasp.parallel" in caps
    assert any(c.startswith("vision.") for c in caps)  # 真实配置带相机 → 暴露视觉能力
