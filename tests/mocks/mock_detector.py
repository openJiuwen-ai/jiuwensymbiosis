# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Mock detection segment function for testing vision pipelines."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def make_mock_seg_fn(
    *,
    score: float = 0.9,
    label: str = "object",
    box: list[float] | None = None,
    mask_shape: tuple[int, int] = (480, 640),
    returns_empty: bool = False,
) -> Callable[..., list[dict[str, Any]]]:
    """Return a mock segment function that satisfies the detector contract.

    Each call returns a list with a single detection dict (or empty if
    ``returns_empty=True``). The mask covers a 20x20 pixel region at the center.
    """
    _box = box or [300.0, 220.0, 340.0, 260.0]

    def segment_fn(image: Any, text_prompt: str = "") -> list[dict[str, Any]]:
        if returns_empty:
            return []
        mask = np.zeros(mask_shape, dtype=bool)
        cy, cx = mask_shape[0] // 2, mask_shape[1] // 2
        mask[cy - 10: cy + 10, cx - 10: cx + 10] = True
        return [
            {
                "mask": mask,
                "box": list(_box),
                "score": score,
                "label": label,
            }
        ]

    return segment_fn
