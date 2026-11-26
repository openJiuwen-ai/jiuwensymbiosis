# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""jiuwensymbiosis — robotics agent framework on top of openjiuwen."""

# Proxy hygiene: clear proxy env BEFORE any lazy attribute below imports the
# agent/openjiuwen httpx layer (which would route local vLLM/detection through a
# SOCKS proxy and require socksio). Must run at the very top of package init so
# even `from jiuwensymbiosis.utils.proxy import clear_proxy_env` (which triggers
# this __init__) finds the env already clean. Logic mirrors utils/proxy.py
# (single source of truth); kept inline here to avoid recursing through
# utils.__init__ (which imports agent.config → openjiuwen). See AGENTS.md.
import os as _os
from importlib import import_module as _import_module

_popped_proxy_env: dict[str, str] = {}
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    _v = _os.environ.pop(_k, None)
    if _v is not None:
        _popped_proxy_env[_k] = _v
_os.environ.setdefault("NO_PROXY", "*")
_os.environ.setdefault("no_proxy", "*")
_proxy = (
    _popped_proxy_env.get("https_proxy")
    or _popped_proxy_env.get("HTTPS_PROXY")
    or _popped_proxy_env.get("http_proxy")
    or _popped_proxy_env.get("HTTP_PROXY")
    or _popped_proxy_env.get("all_proxy")
    or _popped_proxy_env.get("ALL_PROXY")
)
if _proxy:
    _os.environ.setdefault("JIUWEN_LLM_PROXY", _proxy)
del _os, _popped_proxy_env, _proxy

# The public names are imported on first access. Importing the package alone must
# stay side-effect free: the introspection CLIs are console scripts inside this
# package, and openjiuwen logs at INFO during its import. An eager import here
# would write those lines to stdout before argparse or `introspect.emit` runs,
# corrupting `--json` output (#18). Lazy attributes keep `import jiuwensymbiosis`
# cheap for launchers and tests while preserving the public surface.
_LAZY_ATTRIBUTES = {
    "AgentRail": ("jiuwensymbiosis.agent", "AgentRail"),
    "LocalFunction": ("jiuwensymbiosis.agent", "LocalFunction"),
    "ModelSpec": ("jiuwensymbiosis.agent", "ModelSpec"),
    "Tool": ("jiuwensymbiosis.agent", "Tool"),
    "ToolCard": ("jiuwensymbiosis.agent", "ToolCard"),
    "ToolOutput": ("jiuwensymbiosis.agent", "ToolOutput"),
    "build_model": ("jiuwensymbiosis.agent", "build_model"),
    "build_robot_agent_config": ("jiuwensymbiosis.agent", "build_robot_agent_config"),
    "clear_proxy_env": ("jiuwensymbiosis.agent", "clear_proxy_env"),
    "run_fast_task": ("jiuwensymbiosis.agent", "run_fast_task"),
    "run_robot_task": ("jiuwensymbiosis.agent", "run_robot_task"),
    "build_robot_agent": ("jiuwensymbiosis.agent.builder", "build_robot_agent"),
    "RobotSession": ("jiuwensymbiosis.agent.session", "RobotSession"),
    "defaults": ("jiuwensymbiosis.api", "defaults"),
    "ACTIONS": ("jiuwensymbiosis.api.actions", "ACTIONS"),
    "ActionSpec": ("jiuwensymbiosis.api.actions", "ActionSpec"),
    "implements": ("jiuwensymbiosis.api.actions", "implements"),
    "BaseRobotApi": ("jiuwensymbiosis.api.base", "BaseRobotApi"),
    "BaseRobotEnv": ("jiuwensymbiosis.env.base", "BaseRobotEnv"),
    "RobotObservation": ("jiuwensymbiosis.env.base", "RobotObservation"),
    "build_robot_tools": ("jiuwensymbiosis.tools.builder", "build_robot_tools"),
    "InProcessCodeTool": ("jiuwensymbiosis.tools.inproc_code", "InProcessCodeTool"),
}

__all__ = list(_LAZY_ATTRIBUTES)

# Submodules that eager imports used to expose as package attributes.
_SUBMODULES = frozenset({"agent", "api", "env", "tools", "utils"})


def __getattr__(name: str):
    """Import a public name or submodule on first access, then cache it."""
    if name in _SUBMODULES:
        value = _import_module(f"{__name__}.{name}")
        globals()[name] = value
        return value
    try:
        module_name, attribute = _LAZY_ATTRIBUTES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(_import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
