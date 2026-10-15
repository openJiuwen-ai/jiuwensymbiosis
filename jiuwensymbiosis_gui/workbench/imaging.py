# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""相机帧编码:``ndarray`` → JPEG data URI,供 NiceGUI ``interactive_image`` 的 source。

放在后台 worker 线程里编码(见 ``run_engine``),界面线程只做 ``set_source``,不占用
uvicorn 事件循环。无 Qt / 无 nicegui 依赖,可独立单测。
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

__all__ = ["ndarray_to_jpeg_bytes", "to_data_uri"]


def _to_pil(rgb: Any) -> Image.Image:
    """把 HxWx3 (RGB) / HxWx4 (RGBA) / HxW (灰度) 的 ``uint8`` ndarray 转成 PIL 图。"""
    arr = np.ascontiguousarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA")
    raise ValueError(f"unsupported frame shape {arr.shape}")


def ndarray_to_jpeg_bytes(rgb: Any, *, quality: int = 85) -> bytes:
    """把相机帧编码成 JPEG 字节。RGBA 会先合成到 RGB(JPEG 不支持透明通道)。"""
    img = _to_pil(rgb)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def to_data_uri(rgb: Any, *, quality: int = 85) -> str:
    """把相机帧编码成 ``data:image/jpeg;base64,…`` URI,可直接喂给 NiceGUI 图片控件。"""
    b64 = base64.b64encode(ndarray_to_jpeg_bytes(rgb, quality=quality)).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
