# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Interactive adapter onboarding wizard for jiuwensymbiosis.

Two phases:
  A. Generate — ask in vendor terms (DOF, end-effector, camera...) and emit a
     consistent 6-file adapter + YAML that passes ``validate_adapter`` out of the
     box (driver bodies are runnable in-memory mocks marked with a sentinel).
  B. Complete — step the engineer through replacing each mock driver method with
     real SDK calls, re-checking after each, until the adapter actually drives.

Run with ``python scripts/new_adapter/main.py``.
"""
