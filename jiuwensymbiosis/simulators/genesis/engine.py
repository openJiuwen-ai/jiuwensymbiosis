"""Genesis CPU physics and GPU Nyx ownership for one isolated worker."""

from contextlib import contextmanager


def physics_backend() -> str:
    return "cpu"


@contextmanager
def engine_session(startup, *, render_dispatch):
    gs = nps = None
    initialized = nyx_initialized = False
    try:
        with startup.phase("imports"):
            import genesis as gs
            import gs_nyx.nyx_py_sdk as nps
            import quadrants
        with startup.phase("genesis_init"):
            gs.init(backend=gs.cpu, logging_level="warning", performance_mode=True)
            initialized = True
        startup.collect_runtime(quadrants)
        with startup.phase("nyx_startup"):
            render_dispatch.call(nps.startup)
            nyx_initialized = True
        yield
    finally:
        try:
            if initialized:
                gs.destroy()
        finally:
            if nyx_initialized:
                render_dispatch.call(nps.shutdown)
