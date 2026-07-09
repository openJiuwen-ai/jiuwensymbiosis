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
callable (a plain function carrying ``.from_yaml`` / ``.from_dict`` attributes)
with the three call shapes above. The adapter just does:

    build_xxx_session = make_builder(
        XxxConfig, XxxEnv, XxxApi,
        sidecar_builders=[_detector_sidecar_from_cfg],
        decorate=_set_extra_globals,
    )
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from jiuwensymbiosis.agent.session import RobotSession

SidecarBuilder = Callable[[Any], Any]
SessionDecorator = Callable[[RobotSession, Any], None]

# A field-mapping spec for declarative api_kwargs_from_cfg:
#   "cfg_field"          → pass cfg.cfg_field as api kwarg of the same name
#   "cfg_field:api_kwarg" → pass cfg.cfg_field as api kwarg "api_kwarg"
# A plain callable is also accepted (backward compat).
ApiKwargsSpec = Callable[[Any], dict] | list[str]


def _resolve_api_kwargs(spec: ApiKwargsSpec | None, cfg: Any) -> dict:
    """Turn a kwargs spec into the kwargs dict passed to ``api_cls(env, **kw)``.

    * ``None`` → no kwargs.
    * ``callable`` → ``spec(cfg)`` (legacy per-adapter extractor).
    * ``list[str]`` → declarative field mapping. Each item is either a bare
      cfg attribute name (passed through under the same name) or
      ``"cfg_attr:api_kwarg"`` to rename on the way through. ``cfg_attr`` may
      use dotted paths (e.g. ``detector.url``) to reach nested sub-configs.
    """
    if spec is None:
        return {}
    if callable(spec):
        return spec(cfg)
    kwargs: dict[str, Any] = {}
    for item in spec:
        if ":" in item:
            cfg_path, api_kwarg = item.split(":", 1)
        else:
            cfg_path = api_kwarg = item
        value = cfg
        for part in cfg_path.split("."):
            value = getattr(value, part)
        kwargs[api_kwarg] = value
    return kwargs


def make_detector_sidecar(cfg_attr: str = "detector"):
    """Return a ``SidecarBuilder`` that spawns the GroundingDINO+SAM2 server.

    Reads the detector sub-config from ``cfg.<cfg_attr>`` (default
    ``cfg.detector``) and, when its ``spawn`` flag is set, returns a zero-arg
    factory that produces a ``detector_subprocess(...)`` context manager.
    Returns ``None`` when spawning is disabled or no detector config is present,
    so an adapter without vision can still wire this in unconditionally.

    Mirrors the field shape of ``DetectorServerConfig``
    (``host/port/device/startup_timeout_s/gdino_model_id/...``).
    """

    def _build(cfg: Any) -> Callable | None:
        det = getattr(cfg, cfg_attr, None)
        if det is None or not getattr(det, "spawn", False):
            return None
        from jiuwensymbiosis.perception.detector_sidecar import detector_subprocess

        kwargs = {
            "host": det.host,
            "port": det.port,
            "device": det.device,
            "startup_timeout_s": det.startup_timeout_s,
            "gdino_model_id": det.gdino_model_id,
            "sam2_model_id": det.sam2_model_id,
            "box_threshold": det.box_threshold,
            "text_threshold": det.text_threshold,
            "use_sam2": det.use_sam2,
        }
        return lambda: detector_subprocess(**kwargs)

    return _build


def make_builder(
    cfg_cls: type,
    env_cls: type,
    api_cls: type,
    *,
    api_kwargs_from_cfg: ApiKwargsSpec | None = None,
    sidecar_builders: list[SidecarBuilder] | None = None,
    decorate: SessionDecorator | None = None,
):
    """Build a polymorphic session-factory callable.

    Args:
      cfg_cls: Config dataclass with ``from_yaml`` and ``from_dict`` classmethods.
      env_cls: ``BaseRobotEnv`` subclass; constructed as ``env_cls(cfg)``.
      api_cls: ``BaseRobotApi`` subclass; constructed as
        ``api_cls(env, **api_kwargs_from_cfg(cfg))`` if a kwargs spec is given,
        else ``api_cls(env)``.
      api_kwargs_from_cfg: Either a callable ``cfg -> dict`` (legacy), or a
        list of cfg-attribute mappings (declarative). Bare names pass through
        unchanged; ``"cfg_attr:api_kwarg"`` renames on the way to the Api.
        This removes the per-adapter field-shuffling function when cfg and Api
        already use (near-)matching names.
      sidecar_builders: Each callable, given the cfg, returns either a context
        manager (e.g. ``detector_subprocess(...)``) or None. Only non-None returns
        are appended to the session's sidecar_starters. The order is preserved.
      decorate: Optional final-pass callback for storing things on the session.

    Returns a callable ``build(cfg)`` that also exposes ``.from_yaml(path)``
    and ``.from_dict(dict)`` as attributes.
    """

    def _session_from_cfg(cfg: Any) -> RobotSession:
        env = env_cls(cfg)
        api_kwargs = _resolve_api_kwargs(api_kwargs_from_cfg, cfg)
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
                    # bridge cm→zero-arg factory; list typed list[Callable[[],Any]]
                    sidecar_starters.append(lambda cm=cm_or_lambda: cm)  # type: ignore[misc]

        session = RobotSession(
            env=env,
            api=api,
            name=getattr(cfg, "name", "robot"),
            sidecar_starters=sidecar_starters,
        )
        if decorate is not None:
            decorate(session, cfg)
        return session

    def build(cfg: Any) -> RobotSession:
        """Build a session directly from an in-memory config object."""
        return _session_from_cfg(cfg)

    def from_yaml(path: str | Path) -> RobotSession:
        """Build a session from a YAML config file at ``path``."""
        # cfg_cls is a config dataclass w/ from_yaml classmethod (factory contract)
        return _session_from_cfg(cfg_cls.from_yaml(path))  # type: ignore[attr-defined]

    def from_dict(data: dict[str, Any]) -> RobotSession:
        """Build a session from an in-memory config ``dict``."""
        # cfg_cls is a config dataclass w/ from_dict classmethod (factory contract)
        return _session_from_cfg(cfg_cls.from_dict(data))  # type: ignore[attr-defined]

    # function-attribute attachment pattern; mypy can't model fn.__dict__
    build.from_yaml = from_yaml  # type: ignore[attr-defined]
    build.from_dict = from_dict  # type: ignore[attr-defined]
    return build
