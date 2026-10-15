# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Resolve config backend names without importing hardware or simulator clients."""

from __future__ import annotations

import importlib.util

__all__ = ["backend_module_name"]


def backend_module_name(name: str) -> str:
    """Resolve a simulator first, otherwise retain the hardware adapter convention.

    The YAML ``adapter:`` selector and ``build_<name>_session`` factory convention
    are shared by both namespaces. Probing the lightweight simulators package
    leaves backend dependencies deferred until the caller imports the result.
    """
    simulator = f"jiuwensymbiosis.simulators.{name}"
    if importlib.util.find_spec(simulator) is not None:
        return simulator
    return f"jiuwensymbiosis.adapters.{name}"
