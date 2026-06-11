# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""RobotSession — the lifecycle bag the rails and tools share.

A ``RobotSession`` owns:
- the env (hardware driver instance)
- the api (capability-mixin object that calls into env)
- optional sidecar processes (e.g. detection server)
- a ``globals_provider`` for ``InProcessCodeTool``: returns the dict
  injected as code-exec globals.

Lifecycle: ``with session: ...`` connects/disconnects the env and starts/
stops sidecars. Idempotent.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.env.base import BaseRobotEnv

logger = logging.getLogger(__name__)


@dataclass
class RobotSession:
    """Container for one robot+api+sidecars unit, with shared globals.

    Attributes:
        env: ``BaseRobotEnv`` instance (already constructed; not yet connected).
        api: ``BaseRobotApi`` instance bound to ``env``.
        name: Used in logging, prompts, and tool prefixes.
        sidecar_starters: Callables returning a context manager / closer.
            Each is entered on ``connect`` and exited on ``disconnect``.
            Use this for the detection subprocess, video recorder, etc.
        extra_globals: Extra names exposed to ``InProcessCodeTool``-executed
            code. The default exposes ``env`` and ``api``; add ``np``,
            ``time``, your own helpers here.
    """

    env: BaseRobotEnv
    api: BaseRobotApi
    name: str = "robot"
    sidecar_starters: list[Callable[[], Any]] = field(default_factory=list)
    extra_globals: dict[str, Any] = field(default_factory=dict)

    _stack: Optional[ExitStack] = field(default=None, init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)

    # ----------------------------------------------------------- context manager
    def __enter__(self) -> "RobotSession":
        """Enter context: connect and return self."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit context: disconnect."""
        self.disconnect()

    # ----------------------------------------------------------------- lifecycle
    def connect(self) -> None:
        """Connect the env and start all sidecars. Idempotent."""
        if self._connected:
            return
        self._stack = ExitStack()
        for starter in self.sidecar_starters:
            cm = starter()
            if hasattr(cm, "__enter__"):
                self._stack.enter_context(cm)
        try:
            self.env.connect()
        except Exception:
            self._stack.close()
            self._stack = None
            raise
        self._connected = True
        logger.info("RobotSession[%s] connected", self.name)

    def disconnect(self) -> None:
        """Disconnect the env and stop all sidecars. Idempotent."""
        if not self._connected:
            return
        try:
            self.env.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("RobotSession[%s] env.disconnect failed: %s", self.name, exc)
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        self._connected = False
        logger.info("RobotSession[%s] disconnected", self.name)

    # ------------------------------------------------------------------- globals
    def globals_provider(self) -> dict[str, Any]:
        """Return the dict that ``InProcessCodeTool`` injects on every run.

        Re-evaluated per call so updates to ``extra_globals`` (rare) propagate.
        """
        import numpy as np

        return {
            "env": self.env,
            "api": self.api,
            "np": np,
            **self.extra_globals,
        }

    # --------------------------------------------------------------- description
    def describe(self) -> dict[str, Any]:
        """Return a JSON-able summary of this session."""
        return {
            "name": self.name,
            "env": getattr(self.env, "name", type(self.env).__name__),
            "env_capabilities": sorted(self.env.capabilities),
            "api_capabilities": sorted(self.api.capabilities),
        }
