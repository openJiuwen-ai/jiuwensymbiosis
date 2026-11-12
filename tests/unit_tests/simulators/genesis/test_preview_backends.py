"""Backend selection and advertised capabilities stay independent of display labels."""

from typing import get_args

import pytest

from jiuwensymbiosis.simulators.genesis.preview_backends import (
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
    profile = preview_backend(mode, platform)
    assert profile.id == "mujoco" and profile.software_encoding
    assert not any(profile.features.values())
    base = "genesis-macos" if platform == "darwin" else "genesis"
    assert installation_extra(mode, platform) == f"{base},{mode}"
    assert {"mujoco", "onnxruntime"} <= set(required_modules(mode, platform))
    assert ("bam" in required_modules(mode, platform)) is (mode == "microduck")


def test_platform_changes_genesis_renderer_without_changing_robot_identity():
    assert preview_backend("inspection", "darwin").id == "genesis_raster"
    linux = preview_backend("inspection", "linux")
    assert linux.id == "genesis_nyx" and linux.depth and linux.truth and linux.pick
    with pytest.raises(ValueError):
        preview_backend("not-a-mode")
