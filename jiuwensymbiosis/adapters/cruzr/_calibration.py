# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cruzr 相机标定加载。

JSON 形如::

    {"intrinsics": [[fx,0,ppx],[0,fy,ppy],[0,0,1]],
     "tf_base_cam": [[...4x4...]]}

两个字段都可缺省：``intrinsics`` 仅作 camera_info 兜底，``tf_base_cam`` 是 3D 输出
的唯一外参来源（必须在真实机器人上测量）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _as_array(value: Any, shape: tuple[int, ...]) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"calibration array expected {shape}, got {arr.shape}")
    return arr


def load_cruzr_camera_calib(path: str | Path) -> dict[str, Optional[np.ndarray]]:
    """Load ``{intrinsics(3x3)|None, tf_base_cam(4x4)|None}`` from a JSON file."""
    payload = json.loads(Path(path).read_text())
    return {
        "intrinsics": _as_array(payload.get("intrinsics"), (3, 3)),
        "tf_base_cam": _as_array(payload.get("tf_base_cam"), (4, 4)),
    }
