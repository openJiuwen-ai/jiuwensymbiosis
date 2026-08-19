# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pure ROS-image → numpy decoding — body-agnostic, any ``sensor_msgs/Image`` stream.

Deliberately dependency-free: **numpy + stdlib only, nothing from the parent
package**. A frame-grab worker (e.g. ``adapters/cruzr/ros2/camera_worker.py``) runs
under ``/usr/bin/python3`` (conda 3.11 cannot import rclpy) where the agent
stack's ``openjiuwen`` dependency is absent, so it loads this module by file
path (``importlib.util.spec_from_file_location``) to avoid running the package
root ``__init__``. Keep this module free of intra-package imports — the
``test_image_decode_isolation`` tests enforce that contract.

``camera.py`` re-exports ``decode_image_msg`` from here, so existing callers
(and ``from ...cruzr.camera import decode_image_msg``) are unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_COLOR_ENCODINGS = {"rgb8", "bgr8"}
_DEPTH_ENCODINGS = {"mono16", "16uc1"}
# Packed YUV 4:2:2, UYVY byte order (U Y0 V Y1), per ROS sensor_msgs "yuv422".
# The Cruzr head stereo cameras publish this (2 bytes/pixel). Chroma is shared
# across each horizontal pixel pair.
_YUV422_ENCODINGS = {"yuv422"}


def _default_step(width: int, encoding: str) -> int:
    """Bytes per row when the message omits ``step``."""
    if encoding in _COLOR_ENCODINGS:
        return width * 3
    if encoding in _DEPTH_ENCODINGS:
        return width * 2
    if encoding in _YUV422_ENCODINGS:
        return width * 2
    raise ValueError(f"unsupported encoding: {encoding}")


def _encoding_to_str(encoding: Any) -> str:
    """Normalize an image ``encoding`` field to a lowercase string.

    ``sensor_msgs/Image.encoding`` is a plain ``str``. ``shm_msgs/Image*`` instead
    nests it as a ``shm_msgs/String`` (``char[256] data`` + ``uint8 size``), where
    ``data`` arrives as a uint8 ndarray/bytes. Extract the ascii text either way.
    """
    if isinstance(encoding, str):
        return encoding.strip().lower()
    data = getattr(encoding, "data", None)
    if data is None:
        return str(encoding).strip().lower()
    raw = bytes(bytearray(data))
    size = getattr(encoding, "size", None)
    raw = raw[: int(size)] if size else raw.split(b"\x00", 1)[0]
    return raw.decode("ascii", "ignore").strip().lower()


def decode_image_msg(msg: Any, *, depth_scale: float = 0.001) -> np.ndarray:
    """Decode a ROS image message (``sensor_msgs/Image``-compatible fields) to numpy.

    Reads ``height / width / encoding / step / data``. ``shm_msgs/Image*`` uses a
    fixed-size ``data`` buffer, so only the first ``height * step`` bytes are valid.

    * ``rgb8`` / ``bgr8`` → ``uint8 HxWx3`` (RGB; bgr is flipped).
    * ``mono16`` / ``16uc1`` → ``float32 HxW`` in metres (raw * ``depth_scale``).
    * ``yuv422`` (UYVY) → ``uint8 HxWx3`` (RGB; BT.601 full-range conversion).

    Raises ``ValueError`` on an unsupported encoding. Assumes little-endian.
    """
    h = int(msg.height)
    w = int(msg.width)
    encoding = _encoding_to_str(msg.encoding)
    step = int(getattr(msg, "step", 0) or 0) or _default_step(w, encoding)
    raw = bytes(bytearray(msg.data))[: h * step]

    if encoding in _COLOR_ENCODINGS:
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(h, step)
        arr = rows[:, : w * 3].reshape(h, w, 3)
        if encoding == "bgr8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)

    if encoding in _DEPTH_ENCODINGS:
        if step % 2 != 0:
            raise ValueError(f"odd step {step} for 16-bit depth")
        rows = np.frombuffer(raw, dtype=np.uint16).reshape(h, step // 2)
        depth = rows[:, :w].astype(np.float32) * float(depth_scale)
        return np.ascontiguousarray(depth)

    if encoding in _YUV422_ENCODINGS:
        if w % 2 != 0:
            raise ValueError(f"odd width {w} for yuv422")
        # UYVY: every 4 bytes = 2 pixels [U, Y0, V, Y1]; U/V shared across the pair.
        groups = np.frombuffer(raw, dtype=np.uint8).reshape(h, step)[:, : w * 2]
        groups = groups.reshape(h, w // 2, 4).astype(np.float32)
        u, y0, v, y1 = groups[:, :, 0], groups[:, :, 1], groups[:, :, 2], groups[:, :, 3]
        y = np.empty((h, w), dtype=np.float32)
        y[:, 0::2] = y0
        y[:, 1::2] = y1
        cu = np.repeat(u - 128.0, 2, axis=1)
        cv = np.repeat(v - 128.0, 2, axis=1)
        # BT.601 full-range YUV -> RGB.
        r = y + 1.402 * cv
        g = y - 0.344136 * cu - 0.714136 * cv
        b = y + 1.772 * cu
        rgb = np.stack([r, g, b], axis=-1)
        return np.ascontiguousarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8))

    raise ValueError(f"unsupported encoding: {encoding}")
