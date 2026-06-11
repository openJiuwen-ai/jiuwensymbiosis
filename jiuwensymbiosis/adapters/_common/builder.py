# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Generic factory for per-vendor session builders.

Each adapter exposes one ``build_xxx_session`` callable that:

  * accepts a config object directly (``build_xxx_session(cfg)``), OR
  * loads a YAML (``build_xxx_session.from_yaml(path)``), OR
  * loads a dict (``build_xxx_session.from_dict(data)``).

Wiring this up by hand in every adapter is pure boilerplate. ``make_builder``
takes (cfg_cls, env_cls, api_cls) plus optional callbacks for adapter-specific
session decoration (e.g. detector sidecar starter, extra_globals) and returns a
``_Builder`` instance with the three call shapes above. The adapter just does:

    build_xxx_session = make_builder(
        XxxConfig, XxxEnv, XxxApi,
        sidecar_builders=[_detector_sidecar_from_cfg],
        decorate=_set_extra_globals,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from jiuwensymbiosis.agent.session import RobotSession


SidecarBuilder = Callable[[Any], Any]
SessionDecorator = Callable[[RobotSession, Any], None]


def make_builder(
    cfg_cls: type,
    env_cls: type,
    api_cls: type,
    *,
    api_kwargs_from_cfg: Optional[Callable[[Any], dict]] = None,
    sidecar_builders: Optional[list[SidecarBuilder]] = None,
    decorate: Optional[SessionDecorator] = None,
):
    """Build a polymorphic session-factory callable.

    Args:
      cfg_cls: Config dataclass with ``from_yaml`` and ``from_dict`` classmethods.
      env_cls: ``BaseRobotEnv`` subclass; constructed as ``env_cls(cfg)``.
      api_cls: ``BaseRobotApi`` subclass; constructed as
        ``api_cls(env, **api_kwargs_from_cfg(cfg))`` if a kwargs callback is given,
        else ``api_cls(env)``.
      sidecar_builders: Each callable, given the cfg, returns either a context
        manager (e.g. ``detector_subprocess(...)``) or None. Only non-None returns
        are appended to the session's sidecar_starters. The order is preserved.
      decorate: Optional final-pass callback for storing things on the session.

    Returns the builder instance — callable directly with a cfg, with
    ``.from_yaml`` and ``.from_dict`` classmethods.
    """
    def _session_from_cfg(cfg: Any) -> RobotSession:
        env = env_cls(cfg)
        api_kwargs = api_kwargs_from_cfg(cfg) if api_kwargs_from_cfg else {}
        api = api_cls(env, **api_kwargs)

        sidecar_starters: list[Callable[[], Any]] = []
        if sidecar_builders:
            for build in sidecar_builders:
                cm_or_lambda = build(cfg)
                if cm_or_lambda is None:
                    continue
                # Accept either a context manager directly OR a zero-arg
                # callable that produces one (the latter is how RobotSession
                # calls it).
                if callable(cm_or_lambda):
                    sidecar_starters.append(cm_or_lambda)
                else:
                    sidecar_starters.append(lambda cm=cm_or_lambda: cm)

        session = RobotSession(
            env=env,
            api=api,
            name=getattr(cfg, "name", "robot"),
            sidecar_starters=sidecar_starters,
        )
        if decorate is not None:
            decorate(session, cfg)
        return session

    class _Builder:
        """Polymorphic session factory — call directly, or use ``.from_yaml`` / ``.from_dict``."""

        @staticmethod
        def __call__(cfg: Any) -> RobotSession:
            """Build a session directly from an in-memory config object."""
            return _session_from_cfg(cfg)

        @staticmethod
        def from_yaml(path: str | Path) -> RobotSession:
            """Build a session from a YAML config file at ``path``."""
            return _session_from_cfg(cfg_cls.from_yaml(path))

        @staticmethod
        def from_dict(data: dict[str, Any]) -> RobotSession:
            """Build a session from an in-memory config ``dict``."""
            return _session_from_cfg(cfg_cls.from_dict(data))

    return _Builder()
