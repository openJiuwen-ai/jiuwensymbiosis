# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Cruzr-only detection visualizer: a single cv2 window with the waist + head camera panes,
each showing its prompt, the live frame, and the mask overlay. Debug aid; a no-op when cv2/GUI
is unavailable so headless runs and tests are unaffected. Displays already-grabbed frames only —
it opens NO camera subscription (respects the one-shot subprocess camera model)."""
from __future__ import annotations

import atexit
from typing import Any, Optional


class DetectionViz:
    """A single debug window showing the waist + head detection panes side by side.

    Self-disables permanently on any failure (missing cv2, no DISPLAY, a failing cv2 call), so
    every method is a safe no-op in headless runs and unit tests. Frames are DISPLAYED only — no
    camera subscription is opened here (grabbing stays with the one-shot subprocess workers)."""

    WINDOW = "cruzr detections (waist | head)"

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = bool(enabled)
        self._cv2 = None
        self._np = None
        self._panes: dict[str, Any] = {}   # camera -> rendered BGR pane
        if self._enabled:
            try:
                import cv2
                import numpy as np

                self._cv2, self._np = cv2, np
                atexit.register(self.close)
            except Exception:
                self._enabled = False

    def update(self, camera: str, prompt: str, rgb: Any, *,
               mask: Any = None, box: Any = None, score: Optional[float] = None,
               ok: bool = False) -> None:
        """Render one camera pane (rgb->bgr, mask alpha-overlay, box, score, prompt, ok/miss)
        and refresh the combined window. Any failure disables the viz permanently (no-op after)."""
        if not self._enabled:
            return
        try:
            cv2, np = self._cv2, self._np
            img = np.ascontiguousarray(rgb)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if mask is not None:
                m = np.asarray(mask).astype(bool)
                color = np.zeros_like(bgr)
                color[m] = (0, 0, 255)
                bgr = cv2.addWeighted(bgr, 1.0, color, 0.4, 0.0)
            if box is not None:
                x1, y1, x2, y2 = (int(v) for v in box[:4])
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{camera}: {prompt}" + (f"  s={score:.2f}" if score is not None else "")
            status = "OK" if ok else "MISS"
            scolor = (0, 200, 0) if ok else (0, 0, 255)
            cv2.putText(bgr, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(bgr, status, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, scolor, 2)
            self._panes[camera] = bgr
            self._show()
        except Exception:
            self._enabled = False

    def _show(self) -> None:
        cv2, np = self._cv2, self._np
        order = [self._panes[k] for k in ("waist", "head") if k in self._panes]
        if not order:
            return
        h = max(p.shape[0] for p in order)
        padded = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
                  for p in order]
        combined = padded[0] if len(padded) == 1 else np.hstack(padded)
        cv2.imshow(self.WINDOW, combined)
        cv2.waitKey(1)

    def close(self) -> None:
        if self._enabled and self._cv2 is not None:
            try:
                self._cv2.destroyAllWindows()
            except Exception:
                pass
