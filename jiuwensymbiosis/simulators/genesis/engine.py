"""Genesis SDK ownership for one isolated worker."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager


@contextmanager
def engine_session(startup):
    macos = sys.platform == "darwin"
    gs = nps = None
    initialized = nyx_initialized = False
    try:
        with startup.phase("imports"):
            import genesis as gs
            import quadrants

            if not macos:
                import gs_nyx.nyx_py_sdk as nps
        with startup.phase("genesis_init"):
            backend = os.environ.get("JIUWEN_GENESIS_MACOS_BACKEND", "cpu") if macos else "cuda"
            if macos and backend not in {"cpu", "metal"}:
                raise ValueError("JIUWEN_GENESIS_MACOS_BACKEND must be cpu or metal")
            gs.init(backend=getattr(gs, backend), logging_level="warning", performance_mode=True)
            initialized = True
        startup.collect_runtime(quadrants)
        if nps is not None:
            with startup.phase("nyx_startup"):
                nps.startup()
                nyx_initialized = True
        yield
    finally:
        try:
            if initialized:
                gs.destroy()
        finally:
            if nyx_initialized:
                nps.shutdown()
