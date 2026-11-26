# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""CPU fault matrix for one preview reconfiguration, without SDK or GPU."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from jiuwensymbiosis.simulators.common import reconfigurator
from jiuwensymbiosis.simulators.common.camera import PreviewSettings
from jiuwensymbiosis.simulators.common.reconfigurator import (
    ReconfigurationCleanupError,
    Reconfigurator,
    SwitchResources,
)


class _Loading:
    def __init__(self) -> None:
        self.phases: list[str] = []
        self.finished_error: bool | None = None

    def phase(self, name: str):
        self.phases.append(name)
        return nullcontext()

    def finish(self, *, error: bool = False) -> dict:
        self.finished_error = error
        return {}


class _Encoder:
    def __init__(self, label: str, events: list[str], resolution: tuple[int, int]) -> None:
        self.label = label
        self.events = events
        self.resolution = resolution
        self.close_error: BaseException | None = None
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        self.events.append(f"close_encoder:{self.label}")
        if self.close_error is not None:
            raise self.close_error


class _Runtime:
    def __init__(self, name: str, settings: PreviewSettings, events: list[str]) -> None:
        self.name = name
        self.settings = settings
        self.events = events
        self.pose = {"revision": 3}
        self.info = {"instance_id": name}
        self.frame = (1280, 720) if name == "world" else (960, 540)
        self.build_failures: set[int] = set()
        self.close_error: BaseException | None = None
        self.renderer_close_error: BaseException | None = None
        self.renderer_close_error_after = 0
        self.renderer_close_calls = 0
        self.close_count = 0

    def frame_size(self, name: str) -> tuple[int, int]:
        self.events.append(f"frame_size:{self.name}:{name}")
        return self.frame

    def close_renderer(self) -> None:
        self.events.append(f"close_renderer:{self.name}")
        self.renderer_close_calls += 1
        if self.renderer_close_error is not None and self.renderer_close_calls > self.renderer_close_error_after:
            raise self.renderer_close_error

    def build_renderer(self, settings: PreviewSettings) -> None:
        self.events.append(f"build_renderer:{self.name}:{settings.width}")
        if settings.width in self.build_failures:
            raise RuntimeError(f"renderer {settings.width} failed")

    def close(self) -> None:
        self.close_count += 1
        self.events.append(f"close:{self.name}")
        if self.close_error is not None:
            raise self.close_error


class _Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.directory = tmp_path
        self.events: list[str] = []
        self.settings = PreviewSettings()
        self.runtime = _Runtime("world", self.settings, self.events)
        self.encoder = _Encoder("old", self.events, self.runtime.frame)
        self.encoder_calls: list[tuple[PreviewSettings, tuple[int, int]]] = []
        self.encoders: list[_Encoder] = []
        self.created: list[_Runtime] = []
        self.precheck_error: BaseException | None = None
        self.runtime_create_error: BaseException | None = None
        self.encoder_fail_fps: set[int] = set()
        self.candidate_close_error: BaseException | None = None
        self.loading = _Loading()
        self.reconfigurator = Reconfigurator(
            precheck=self._precheck,
            runtime_factory=self._runtime_factory,
            encoder_factory=self._encoder_factory,
            clock=lambda: 0.0,
        )

    def _precheck(self, scene_id: str, directory: Path) -> None:
        self.events.append(f"precheck:{scene_id}")
        if self.precheck_error is not None:
            raise self.precheck_error

    def _runtime_factory(self, settings, directory, revision=0, *, startup=None):
        self.events.append(f"create_runtime:{settings.scene_id}")
        if self.runtime_create_error is not None:
            raise self.runtime_create_error
        runtime = _Runtime("candidate", settings, self.events)
        runtime.close_error = self.candidate_close_error
        self.created.append(runtime)
        return runtime

    def _encoder_factory(self, settings, *, resolution):
        self.events.append(f"encoder:{settings.fps}:{resolution}")
        if settings.fps in self.encoder_fail_fps:
            raise RuntimeError(f"encoder {settings.fps} failed")
        self.encoder_calls.append((settings, resolution))
        encoder = _Encoder(f"encoder-{len(self.encoder_calls)}", self.events, resolution)
        self.encoders.append(encoder)
        return encoder

    def apply(self, requested: PreviewSettings):
        return self.reconfigurator.apply(
            SwitchResources(runtime=self.runtime, encoder=self.encoder, settings=self.settings),
            requested,
            7,
            directory=self.directory,
            loading=self.loading,
        )


def test_fps_and_camera_change_keeps_world_and_renderer(tmp_path):
    harness = _Harness(tmp_path)
    requested = replace(harness.settings, fps=30, camera_name="wrist")

    result = harness.apply(requested)

    assert result.error is None and result.replaced_world is False and result.revision == 7
    assert result.resources.runtime is harness.runtime
    assert result.resources.settings is requested
    assert harness.runtime.settings is requested
    assert harness.encoder.close_count == 1
    assert harness.encoder_calls == [(requested, (1280, 720))]
    assert not any(event.startswith(("close_renderer", "build_renderer")) for event in harness.events)
    assert harness.events.index("close_encoder:old") < harness.events.index("encoder:30:(1280, 720)")


def test_resolution_change_rebuilds_renderer_before_the_new_encoder(tmp_path):
    harness = _Harness(tmp_path)
    requested = replace(harness.settings, width=960, height=540)

    result = harness.apply(requested)

    assert result.error is None and result.replaced_world is False
    assert result.resources.runtime is harness.runtime
    assert harness.events == [
        "close_encoder:old",
        "close_renderer:world",
        "build_renderer:world:960",
        "frame_size:world:observer",
        "encoder:60:(1280, 720)",
    ]
    assert harness.loading.phases == ["nyx_renderer", "nvenc"]


def test_world_change_prechecks_then_commits_with_one_owner(tmp_path):
    harness = _Harness(tmp_path)
    requested = replace(harness.settings, scene_id="scene-a")

    result = harness.apply(requested)

    candidate = harness.created[0]
    assert result.error is None and result.replaced_world is True
    assert result.resources.runtime is candidate
    assert candidate.close_count == 0
    assert harness.runtime.close_count == 1
    assert harness.events == [
        "precheck:scene-a",
        "close_encoder:old",
        "close_renderer:world",
        "create_runtime:scene-a",
        "frame_size:candidate:observer",
        "encoder:60:(960, 540)",
        "close:world",
    ]


def test_precheck_failure_keeps_every_current_resource(tmp_path):
    harness = _Harness(tmp_path)
    harness.precheck_error = ValueError("missing asset")
    requested = replace(harness.settings, scene_id="scene-a")

    result = harness.apply(requested)

    assert "切换失败" in result.error and "missing asset" in result.error
    assert result.resources.runtime is harness.runtime and result.resources.encoder is harness.encoder
    assert result.resources.settings is harness.settings
    assert harness.encoder.close_count == 0 and harness.runtime.close_count == 0
    assert harness.loading.finished_error is True
    assert result.replaced_world is False and result.revision == 7


def test_candidate_scene_failure_restores_the_old_renderer_and_encoder(tmp_path):
    harness = _Harness(tmp_path)
    harness.runtime_create_error = RuntimeError("candidate failed")
    requested = replace(harness.settings, scene_id="scene-a")

    result = harness.apply(requested)

    assert "candidate failed" in result.error
    assert result.resources.runtime is harness.runtime
    assert result.resources.encoder is not harness.encoder
    assert harness.encoder.close_count == 1
    assert harness.events == [
        "precheck:scene-a",
        "close_encoder:old",
        "close_renderer:world",
        "create_runtime:scene-a",
        "close_renderer:world",
        "build_renderer:world:1280",
        "frame_size:world:observer",
        "encoder:60:(1280, 720)",
    ]


def test_new_encoder_failure_closes_the_candidate_and_restores_the_old_world(tmp_path):
    harness = _Harness(tmp_path)
    harness.encoder_fail_fps = {30}
    requested = replace(harness.settings, scene_id="scene-a", fps=30)

    result = harness.apply(requested)

    assert "encoder 30 failed" in result.error
    assert result.resources.runtime is harness.runtime
    assert result.resources.encoder is not harness.encoder
    assert harness.events == [
        "precheck:scene-a",
        "close_encoder:old",
        "close_renderer:world",
        "create_runtime:scene-a",
        "frame_size:candidate:observer",
        "encoder:30:(960, 540)",
        "close:candidate",
        "close_renderer:world",
        "build_renderer:world:1280",
        "frame_size:world:observer",
        "encoder:60:(1280, 720)",
    ]


def test_failed_old_runtime_release_is_fatal_and_still_closes_ready_resources(tmp_path):
    harness = _Harness(tmp_path)
    harness.runtime.close_error = RuntimeError("old runtime close failed")
    harness.candidate_close_error = RuntimeError("candidate close failed")
    requested = replace(harness.settings, scene_id="scene-a")

    with pytest.raises(ReconfigurationCleanupError) as excinfo:
        harness.apply(requested)

    failures = [str(failure) for failure in excinfo.value.failures]
    assert "candidate close failed" in failures
    assert "旧场景释放未确认" in str(excinfo.value)
    assert harness.encoder.close_count == 1
    assert len(harness.encoders) == 1  # the ready replacement, closed during abort
    assert harness.encoders[0].close_count == 1
    # The old world may be partially destroyed, so do not rebuild its renderer.
    assert not any(event.startswith("build_renderer") for event in harness.events)


def test_backend_change_is_rejected_before_touching_resources(tmp_path):
    harness = _Harness(tmp_path)
    requested = replace(harness.settings, backend="mujoco")

    with pytest.raises(ValueError, match="fresh preview process"):
        harness.apply(requested)

    assert harness.events == []
    assert harness.runtime.close_count == 0 and harness.encoder.close_count == 0


def test_encoder_close_failure_rebuilds_instead_of_leaving_it_dead(tmp_path):
    harness = _Harness(tmp_path)
    harness.encoder.close_error = RuntimeError("close failed")
    requested = replace(harness.settings, fps=30)

    result = harness.apply(requested)

    assert "close failed" in result.error
    assert result.resources.encoder is harness.encoders[0]
    assert harness.encoder_calls == [(harness.settings, (1280, 720))]
    assert harness.events == [
        "close_encoder:old",
        "frame_size:world:observer",
        "encoder:60:(1280, 720)",
    ]

    # When the replacement cannot be built either, the cleanup failure surfaces.
    harness = _Harness(tmp_path)
    harness.encoder.close_error = RuntimeError("close failed")
    harness.encoder_fail_fps = {60}
    requested = replace(harness.settings, fps=30)

    with pytest.raises(ReconfigurationCleanupError):
        harness.apply(requested)

    assert harness.encoder.close_count == 1
    assert harness.loading.finished_error is True


def test_replace_encoder_closes_and_recovers_a_failed_close(tmp_path):
    harness = _Harness(tmp_path)
    rebuilt = harness.reconfigurator.replace_encoder(harness.encoder, harness.settings, resolution=(320, 240))

    assert harness.encoder.close_count == 1
    assert rebuilt is harness.encoders[0]
    rebuilt.close_error = RuntimeError("close failed")
    rebuilt_again = harness.reconfigurator.replace_encoder(rebuilt, harness.settings, resolution=(320, 240))
    assert rebuilt_again is harness.encoders[1]


def test_rollback_failures_are_reported_together(tmp_path):
    harness = _Harness(tmp_path)
    harness.runtime.build_failures = {960, 1280}
    harness.runtime.renderer_close_error = RuntimeError("renderer close failed")
    harness.runtime.renderer_close_error_after = 1
    requested = replace(harness.settings, width=960, height=540)

    with pytest.raises(ReconfigurationCleanupError) as excinfo:
        harness.apply(requested)

    failures = [str(failure) for failure in excinfo.value.failures]
    assert any("renderer close failed" in failure for failure in failures)
    assert any("renderer 1280 failed" in failure for failure in failures)
    assert harness.encoder.close_count == 1
    assert len(harness.encoders) == 1
    assert harness.runtime.renderer_close_calls == 2
    assert harness.loading.finished_error is True


def test_precheck_scene_validates_urdf_and_materials(monkeypatch, tmp_path):
    asset = SimpleNamespace(
        asset=SimpleNamespace(preview_mode="urdf"),
        preview_path=tmp_path / "robot.urdf",
    )
    calls = []
    monkeypatch.setattr(reconfigurator, "resolve_scene", lambda scene_id: asset)
    monkeypatch.setattr(reconfigurator, "validate_material_bundle", lambda item: calls.append(("bundle", item)))
    from jiuwensymbiosis.simulators.genesis import urdf_compat

    monkeypatch.setattr(urdf_compat, "validate_urdf", lambda path: calls.append(("urdf", path)))

    reconfigurator.precheck_scene("scene-a", tmp_path)

    assert calls == [("urdf", asset.preview_path), ("bundle", asset)]
