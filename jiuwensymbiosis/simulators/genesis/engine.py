"""Genesis physics and renderer SDK ownership for one isolated worker."""

import os
import sys
from contextlib import contextmanager


def physics_backend() -> str:
    if sys.platform == "darwin":
        backend = os.environ.get("JIUWEN_GENESIS_MACOS_BACKEND", "cpu")
        if backend not in {"cpu", "metal"}:
            raise ValueError("JIUWEN_GENESIS_MACOS_BACKEND must be cpu or metal")
        return backend
    return "cpu"


@contextmanager
def engine_session(startup, *, render_dispatch):
    backend = physics_backend()
    gs = nps = None
    initialized = nyx_initialized = False
    try:
        with startup.phase("imports"):
            import genesis as gs
            import quadrants

            if render_dispatch is not None:
                import gs_nyx.nyx_py_sdk as nps
        with startup.phase("genesis_init"):
            gs.init(backend=getattr(gs, backend), logging_level="warning", performance_mode=True)
            initialized = True
        startup.collect_runtime(quadrants)
        if nps is not None:
            with startup.phase("nyx_startup"):
                render_dispatch.call(nps.startup)
                nyx_initialized = True
        yield backend
    finally:
        try:
            if initialized:
                gs.destroy()
        finally:
            if nyx_initialized:
                render_dispatch.call(nps.shutdown)
