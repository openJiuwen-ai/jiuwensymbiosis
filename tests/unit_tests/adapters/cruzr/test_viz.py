# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""DetectionViz debug window is a zero-side-effect no-op when disabled.

The visualizer is a hardware-debug aid; on headless CI / unit runs it must never import cv2,
open a window, or throw — so the default (disabled) path has to be fully inert."""

import sys

from jiuwensymbiosis.adapters.cruzr._viz import DetectionViz


def test_disabled_viz_is_inert():
    viz = DetectionViz(enabled=False)
    # No cv2/numpy captured, no panes rendered.
    assert viz._enabled is False
    assert viz._cv2 is None
    assert viz._panes == {}
    # Every call is a safe no-op (no exception, no state change).
    viz.update("waist", "box", object(), mask=object(), box=[0, 0, 1, 1], score=0.9, ok=True)
    viz.update("head", "table", None)
    viz.close()
    assert viz._panes == {}


def test_disabled_viz_never_imports_cv2():
    # Constructing a disabled viz must not pull cv2 into the interpreter (headless safety).
    had_cv2 = "cv2" in sys.modules
    DetectionViz(enabled=False)
    if not had_cv2:
        assert "cv2" not in sys.modules
