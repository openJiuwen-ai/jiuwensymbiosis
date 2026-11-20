"""One physics owner and a bounded dispatcher for main-thread render APIs."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from typing import Any


class MainThreadCalls:
    def __init__(self):
        self._main = threading.get_ident()
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self.physics_thread_id: int | None = None
        self.physics_steps = 0
        self.frame_sequence = 0
        self._tick = None
        self._next_tick = 0.0
        self._error: BaseException | None = None

    def set_tick(self, tick) -> None:
        """Install a main-thread loop, independent of requests from physics."""
        if threading.get_ident() != self._main:
            raise RuntimeError("Render-loop registration requires the main thread")
        self._tick = tick
        self._next_tick = time.monotonic()

    def check_error(self) -> None:
        if self._error is not None:
            raise self._error

    def submit(self, function, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            self._queue.put_nowait((future, function, args, kwargs))
        except queue.Full as exc:
            raise RuntimeError("Main-thread render queue is full") from exc
        return future

    def call(self, function, *args, **kwargs) -> Any:
        if threading.get_ident() == self._main:
            return function(*args, **kwargs)
        return self.submit(function, *args, **kwargs).result()

    def snapshot(self) -> dict:
        return {
            "mode": "physics_thread_independent_main_render",
            "physics_thread_id": self.physics_thread_id,
            "render_thread_id": self._main,
            "physics_steps": self.physics_steps,
            "queued_calls": self._queue.qsize(),
            "independent_loop": self._tick is not None,
        }

    def serve(self, owner: threading.Thread) -> None:
        """Release each completed callback before accepting a replacement."""
        while owner.is_alive() or not self._queue.empty():
            try:
                timeout = min(0.05, max(0.0, self._next_tick - time.monotonic())) if self._tick else 0.05
                future, function, args, kwargs = self._queue.get(timeout=timeout)
            except queue.Empty:
                if self._tick is not None and owner.is_alive():
                    try:
                        delay = self._tick()
                        self._next_tick = time.monotonic() + max(0.001, delay)
                    except BaseException as exc:
                        # Keep servicing teardown RPCs; the owner reports the
                        # error and closes CUDA/Nyx on their required threads.
                        self._error = exc
                        self._tick = None
                continue
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
                del result
            finally:
                # The next callback may destroy this RGB view's allocation.
                del future, function, args, kwargs


def run_split(physical_owner) -> None:
    """Run SDK callbacks on the calling main thread until its owner cleans up."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("The render dispatcher must run on the process main thread")
    dispatch = MainThreadCalls()
    errors = []

    def physical():
        dispatch.physics_thread_id = threading.get_ident()
        try:
            physical_owner(dispatch)
        except BaseException as exc:
            errors.append(exc)

    owner = threading.Thread(target=physical, name="genesis-physics", daemon=False)
    owner.start()
    dispatch.serve(owner)
    owner.join()
    if errors:
        raise errors[0]
