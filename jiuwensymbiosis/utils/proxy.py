# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Proxy environment hygiene.

openjiuwen's HTTP layer (httpx) requires the ``socksio`` package when
``ALL_PROXY=socks5://...`` is set, AND it routes localhost through the
proxy unless ``NO_PROXY`` is set. Both behaviors break local vLLM /
ollama / detection calls. Call ``clear_proxy_env()`` BEFORE importing
``openjiuwen.*`` — see sorting_agent.py for the original incident.
"""

from __future__ import annotations

import os

_PROXY_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def clear_proxy_env() -> dict[str, str]:
    """Pop proxy env vars, set NO_PROXY=*. Return the popped values for
    diagnostics. Idempotent.
    """
    popped: dict[str, str] = {}
    for k in _PROXY_VARS:
        v = os.environ.pop(k, None)
        if v is not None:
            popped[k] = v
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")
    # Stash a popped proxy so EXTERNAL calls that genuinely need it (e.g. the
    # cloud LLM planner reaching api.siliconflow.cn) can opt back into it, while
    # local services (detector / local vLLM on 127.0.0.1) stay OFF the proxy via
    # NO_PROXY=*. Consumers read ``JIUWEN_LLM_PROXY`` and pass it explicitly.
    proxy = (
        popped.get("https_proxy")
        or popped.get("HTTPS_PROXY")
        or popped.get("http_proxy")
        or popped.get("HTTP_PROXY")
        or popped.get("all_proxy")
        or popped.get("ALL_PROXY")
    )
    if proxy:
        os.environ.setdefault("JIUWEN_LLM_PROXY", proxy)
    return popped
