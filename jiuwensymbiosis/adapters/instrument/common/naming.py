"""Stable physical-profile tool naming shared by both access paths."""

import hashlib
import re


def tool_name(device_id: str, action: str) -> str:
    """Stable device-qualified name shared by direct, Agent and MCP tools."""
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", device_id)[:32]
    suffix = hashlib.sha256(device_id.encode()).hexdigest()[:8]
    return f"{slug}_{suffix}__{action}"
