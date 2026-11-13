"""Backend selection and advertised capabilities stay independent of display labels."""

from typing import get_args

import pytest

from jiuwensymbiosis.simulators.common.camera import PreviewSettings
from jiuwensymbiosis.simulators.common.preview_backends import (
    MODES,
    POLICY_MODES,
    RobotMode,
    installation_extra,
    preview_backend,
    required_modules,
)


def test_public_modes_have_exactly_one_factory():
    assert set(get_args(RobotMode)) == set(MODES)
    assert all(spec.module and spec.runtime for spec in MODES.values())


@pytest.mark.parametrize("platform", ["darwin", "linux"])
@pytest.mark.parametrize("mode", POLICY_MODES)
def test_native_install_hint_contains_all_required_extras(platform, mode):
    profile = preview_backend("mujoco", platform)
    assert profile.id == "mujoco" and profile.software_encoding
    assert not any(profile.features.values())
    assert installation_extra("mujoco", mode, platform) == f"mujoco,{mode}"
    assert {"mujoco", "onnxruntime"} <= set(required_modules("mujoco", mode, platform))
    assert ("bam" in required_modules("mujoco", mode, platform)) is (mode == "microduck")
    assert not {"genesis", "torch", "gs_nyx_plugin"} & set(required_modules("mujoco", mode, platform))


@pytest.mark.parametrize(
    "backend, mode",
    [("genesis", "jumper"), ("genesis", "microduck"), ("mujoco", "moz1_wheels"), ("unknown", "inspection")],
)
def test_invalid_engine_controller_combinations_fail_before_start(backend, mode):
    with pytest.raises(ValueError):
        PreviewSettings(backend=backend, robot_mode=mode)


def test_both_engines_support_inspection_without_a_policy_dependency():
    assert PreviewSettings().backend == "genesis"
    assert PreviewSettings(backend="mujoco").robot_mode == "inspection"
    assert "onnxruntime" not in required_modules("mujoco", "inspection")


def test_platform_changes_genesis_renderer_without_changing_robot_identity():
    assert preview_backend("genesis", "darwin").id == "genesis_raster"
    linux = preview_backend("genesis", "linux")
    assert linux.id == "genesis_nyx" and linux.depth and linux.truth and linux.pick
    with pytest.raises(ValueError):
        preview_backend("not-a-mode")
