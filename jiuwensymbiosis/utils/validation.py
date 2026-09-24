# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared numeric validation for configuration and response fields.

Each caller passes its own full error message, so existing exception text and
type stay the contract of the calling site. The helpers only validate; any
``float`` coercion stays where it already happened.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["require_positive_number", "require_unit_interval"]


def require_positive_number(value: Any, *, message: str) -> None:
    """Require a finite positive real number (``bool`` is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(message)
    if value <= 0:
        raise ValueError(message)


def require_unit_interval(value: Any, *, message: str) -> None:
    """Require a finite real number within ``0 <= value <= 1``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(message)
    if not 0 <= value <= 1:
        raise ValueError(message)
