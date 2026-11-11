# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""One coercion for the ``tool_args`` shape every rail receives.

openjiuwen types ``ToolCall.arguments`` as ``str`` and hands that JSON string to
``before_tool_call`` / ``on_tool_exception``; only the tool's ``invoke`` ever
sees a dict. A rail that checks ``isinstance(tool_args, dict)`` therefore misses
every ``robot_control`` dispatch (the action lives in the parsed payload), which
is exactly the class of bug this module prevents from being re-invented per rail.

``coerce_tool_args`` never raises: unparseable / non-dict / absent input becomes
``{}``, so enforcement degrades to "no params → no rejection" — the same
contract ``SafetyRail`` documented before this helper existed.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["coerce_tool_args"]


def coerce_tool_args(value: Any) -> dict[str, Any]:
    """Return ``value`` as a params dict, parsing a JSON string when needed.

    Args:
        value: Whatever the callback delivered — dict, JSON string, ``None``.

    Returns:
        The dict payload, or ``{}`` when the value is absent or not object-like.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
