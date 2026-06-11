# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Sidecar HTTP services that adapters spawn as subprocesses.

Each module in this package is independently runnable as
``python -m jiuwensymbiosis.serving.<name>``. They are imported lazily,
so the framework stays light-weight unless a sidecar is actually used.
"""
