# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Derived from the instrument integration; licensed under Apache-2.0.

"""JSON and strict configuration primitives for instrument declarations."""

from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "1"
Identifier = Annotated[str, Field(min_length=1, max_length=200, pattern=r"\S")]
PositiveNumber = Annotated[float, Field(gt=0, allow_inf_nan=False)]
ConditionId = Identifier


def ensure_json_value(value: Any, path: str = "$") -> None:
    """Reject coercions, cycles, excessive nesting and nonfinite JSON values."""
    active: set[int] = set()

    def visit(item: Any, location: str, depth: int) -> None:
        if depth > 100:
            raise ValueError(f"{location}: JSON nesting exceeds 100 levels")
        # Only native JSON types are admitted; isinstance alone accepts coercible subclasses.
        if type(item) not in (type(None), str, bool, int, float, list, dict):
            raise ValueError(f"{location}: unsupported JSON value {type(item).__name__}")
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{location}: nonfinite numbers are not valid JSON")
            return
        if id(item) in active:
            raise ValueError(f"{location}: cyclic JSON is unsupported")
        active.add(id(item))
        try:
            if isinstance(item, list):
                for index, child in enumerate(item):
                    visit(child, f"{location}[{index}]", depth + 1)
            else:
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise ValueError(f"{location}: JSON object keys must be strings")
                    visit(key, location, depth)
                    visit(child, f"{location}.{key}", depth + 1)
        finally:
            active.remove(id(item))

    visit(value, path, 0)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
