# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""One preview reconfiguration: validate, replace resources, commit or roll back.

The worker loop owns scheduling, loop-state reset, motion cancellation and scene
reset. This module owns one configuration switch's candidate resources,
teardown/creation order and the one place where ownership transfers; it also owns
the encoder replacement used when the video stream reconnects, so the encoder
type selector and close/create order have a single authority. A reconnect's
encoder failure remains process-fatal. CPU tests drive it with recording
factories and fault injection, without a Genesis SDK, CUDA or the event loop.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .preview_backends import preview_backend
from .runtime import create_runtime
from .scene_catalog import resolve_scene, validate_material_bundle

logger = logging.getLogger("jiuwensymbiosis.simulators.common.reconfigurator")


@dataclass(frozen=True)
class SwitchResources:
    """The resources a preview currently owns; exactly one owner per field."""

    runtime: Any
    encoder: Any
    settings: Any


def _world_changed(old: Any, requested: Any) -> bool:
    # planning_profile selects the runtime's arm planner, so it is part of the
    # world identity the runtime constructor consumes.
    return (requested.scene_id, requested.robot_mode, requested.robot_id, requested.planning_profile) != (
        old.scene_id,
        old.robot_mode,
        old.robot_id,
        old.planning_profile,
    )


def _renderer_changed(old: Any, requested: Any) -> bool:
    return (requested.width, requested.height, requested.spp, requested.render_mode) != (
        old.width,
        old.height,
        old.spp,
        old.render_mode,
    )


def _close_encoder(state: _SwitchState) -> None:
    """Close the old encoder; a failed close still makes the handle unusable."""
    try:
        state.encoder.close()
    finally:
        state.encoder = None


@dataclass
class _SwitchState:
    """Mutable resources during one switch, so a failure still owns every handle."""

    runtime: Any
    encoder: Any
    settings: Any
    revision: int
    started: float
    candidate: Any = None
    new_encoder: Any = None
    renderer_released: bool = False


@dataclass(frozen=True)
class SwitchResult:
    """One switch's effective resources and already-processed revision."""

    resources: SwitchResources
    revision: int
    error: str | None
    switch_ms: float
    replaced_world: bool


class ReconfigurationCleanupError(RuntimeError):
    """The switch failed and cleanup/restore could not confirm resource ownership."""

    def __init__(self, message: str, failures: tuple[BaseException, ...]) -> None:
        self.failures = failures
        details = "; ".join(f"{type(exc).__name__}: {exc}" for exc in failures)
        suffix = f"；清理或恢复未确认：{details}" if details else ""
        super().__init__(f"{message}{suffix}")


def precheck_scene(scene_id: str, directory: Path) -> None:
    """Reject a missing or unusable asset before releasing the current renderer."""
    asset = resolve_scene(scene_id)
    if asset.asset.preview_mode == "urdf":
        from ..genesis.urdf_compat import validate_urdf

        validate_urdf(asset.preview_path)
        validate_material_bundle(asset)
    elif asset.asset.preview_mode == "static_usd":
        with tempfile.TemporaryDirectory(prefix="check-", dir=directory) as check:
            from ..genesis.usd_scene import prepare_static_stage

            prepare_static_stage(asset.preview_path, Path(check) / "background.usda")


class Reconfigurator:
    """Apply one requested switch and return the resources that remain valid.

    The order is external behavior: precheck the candidate asset before tearing
    down the current renderer, close the old NVENC first, then replace Nyx
    resources, and only create the new encoder after the new frame size is known.
    A recoverable failure keeps the old settings and returns the old resources
    with an error; a second failure while cleaning up or restoring propagates so
    the worker never publishes a false ready.
    """

    def __init__(
        self,
        *,
        encoder_factory: Callable[..., Any],
        precheck: Callable[[str, Path], None] = precheck_scene,
        runtime_factory: Callable[..., Any] = create_runtime,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._precheck = precheck
        self._runtime_factory = runtime_factory
        self._encoder_factory = encoder_factory
        self._clock = clock

    def replace_encoder(self, encoder: Any | None, settings: Any, *, resolution: tuple[int, int]) -> Any:
        """Create the encoder and, on a reconnect, close the current handle first.

        A failed close still leaves the old handle unusable, so a replacement is
        created anyway; only creation failure remains fatal to the process.
        """
        if encoder is not None:
            try:
                encoder.close()
            except Exception:
                logger.exception("Video reconnect encoder close failed; rebuilding")
        return self._encoder_factory(settings, resolution=resolution)

    def apply(
        self,
        current: SwitchResources,
        requested: Any,
        revision: int,
        *,
        directory: Path,
        loading: Any,
    ) -> SwitchResult:
        """Apply one switch, returning the resources that remain valid."""
        if requested.backend != current.settings.backend:
            # The engine SDK (Genesis or MuJoCo) is initialized once per worker
            # process; the parent restarts the preview for a backend change.
            raise ValueError("backend changes require a fresh preview process")
        started = self._clock()
        old_settings = current.settings
        state = _SwitchState(
            runtime=current.runtime, encoder=current.encoder, settings=old_settings, revision=revision, started=started
        )
        try:
            replaced_world = self._replace(state, requested, directory=directory, loading=loading)
            return SwitchResult(
                resources=SwitchResources(runtime=state.runtime, encoder=state.encoder, settings=requested),
                revision=revision,
                error=None,
                switch_ms=round((self._clock() - started) * 1000, 1),
                replaced_world=replaced_world,
            )
        except ReconfigurationCleanupError:
            raise
        except Exception as exc:
            return self._recover(state, loading, exc)

    def _replace(self, state: _SwitchState, requested: Any, *, directory: Path, loading: Any) -> bool:
        """Release, build and commit the requested resources in their fixed order."""
        world_changed = _world_changed(state.settings, requested)
        renderer_changed = world_changed or _renderer_changed(state.settings, requested)
        if requested.scene_id != state.settings.scene_id and requested.scene_id != "demo":
            self._precheck(requested.scene_id, directory)
        _close_encoder(state)
        if world_changed:
            state.renderer_released = True
            state.runtime.close_renderer()
            state.candidate = self._runtime_factory(
                requested, directory, state.runtime.pose["revision"] + 1, startup=loading
            )
        elif renderer_changed:
            state.renderer_released = True
            state.runtime.close_renderer()
            with loading.phase("nyx_renderer"):
                state.runtime.build_renderer(requested)
        # Open NVENC only after Nyx finishes replacing CUDA resources:
        # opening it before teardown fails on repeated 720p/1080p switches.
        with loading.phase("encoder_init" if preview_backend(requested.backend).software_encoding else "nvenc"):
            state.new_encoder = self._encoder_factory(
                requested, resolution=(state.candidate or state.runtime).frame_size(requested.camera_name)
            )
        self._commit(state, loading)
        state.settings = requested
        state.runtime.settings = requested
        return world_changed

    def _commit(self, state: _SwitchState, loading: Any) -> None:
        """Transfer ownership only after the candidate world and encoder are ready."""
        if state.candidate is not None:
            try:
                state.runtime.close()
            except Exception as exc:
                # The old world may be partially destroyed; do not restore it.
                # Close the ready resources and let the worker fail instead of
                # publishing a ready that owns an ambiguous runtime.
                self._abort_commit(state, loading, exc)
            state.runtime = state.candidate
            state.candidate = None
        state.encoder = state.new_encoder
        state.new_encoder = None

    @staticmethod
    def _abort_commit(state: _SwitchState, loading: Any, exc: Exception) -> None:
        """Do not publish ready when the old runtime release is unconfirmed."""
        loading.finish(error=True)
        logger.exception("Simulation old runtime release failed")
        cleanup: list[BaseException] = []
        for resource in (state.candidate, state.new_encoder):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as close_error:
                cleanup.append(close_error)
        error = f"切换失败，旧场景释放未确认：{str(exc)[:1000]}"
        raise ReconfigurationCleanupError(error, tuple(cleanup)) from exc

    def _recover(self, state: _SwitchState, loading: Any, exc: Exception) -> SwitchResult:
        """Return the old resources after a recoverable switch failure."""
        old_settings = state.settings
        loading.finish(error=True)
        logger.exception("Simulation reconfiguration failed; restoring previous scene")
        error = f"切换失败，保留原场景和配置：{str(exc)[:1000]}"
        failures: list[BaseException] = []
        self._restore(state, old_settings, failures)
        if failures:
            raise ReconfigurationCleanupError(error, tuple(failures)) from exc
        return SwitchResult(
            resources=SwitchResources(runtime=state.runtime, encoder=state.encoder, settings=old_settings),
            revision=state.revision,
            error=error,
            switch_ms=round((self._clock() - state.started) * 1000, 1),
            replaced_world=False,
        )

    def _restore(self, state: _SwitchState, old_settings: Any, failures: list[BaseException]) -> None:
        """Close candidates and restore the old renderer/encoder, attempting all steps.

        Every step is attempted even after an earlier one fails: a candidate's
        failing close must not skip closing the new encoder or restoring the old
        renderer. Rebuild the encoder whenever the current handle is gone, which
        includes a failed ``close()``.
        """
        for resource, label in (
            (state.candidate, "candidate scene"),
            (state.new_encoder, "replacement encoder"),
        ):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as exc:
                logger.exception("Simulation %s cleanup failed", label)
                failures.append(exc)
        if state.renderer_released:
            try:
                state.runtime.close_renderer()
            except Exception as exc:
                logger.exception("Simulation renderer cleanup failed")
                failures.append(exc)
            try:
                state.runtime.build_renderer(old_settings)
            except Exception as exc:
                logger.exception("Simulation renderer restore failed")
                failures.append(exc)
        if state.encoder is None:
            try:
                state.encoder = self._encoder_factory(
                    old_settings, resolution=state.runtime.frame_size(old_settings.camera_name)
                )
            except Exception as exc:
                logger.exception("Simulation encoder restore failed")
                failures.append(exc)


__all__ = [
    "ReconfigurationCleanupError",
    "Reconfigurator",
    "SwitchResources",
    "SwitchResult",
    "precheck_scene",
]
