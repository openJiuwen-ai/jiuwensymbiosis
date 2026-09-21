"""Tests for immutable, source-aware adapter bindings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from jiuwensymbiosis.adapters._common.resources import common_resource_keys, complete_resource_keys
from jiuwensymbiosis.runtime.bindings import prepare_binding


def _piper_source(path: Path) -> Path:
    path.write_text("adapter: piper\ncan_port: can_test\n", encoding="utf-8")
    return path


def test_binding_rejects_string_resource_declaration_before_building(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from jiuwensymbiosis.adapters import piper
    from jiuwensymbiosis.runtime import Runtime

    source = _piper_source(tmp_path / "piper.yaml")
    construct = Mock(side_effect=AssertionError("must reject before session construction"))
    construct.config_factory = piper.build_piper_session.config_factory
    construct.resource_keys = lambda cfg: "can:can0"
    monkeypatch.setattr(piper, "build_piper_session", construct)
    runtime = Runtime(tmp_path / "ws", resource_directory=tmp_path / "locks")
    try:
        with pytest.raises(TypeError, match="resource_keys"):
            runtime.prepare_binding(source)
        construct.assert_not_called()
        assert not runtime.get_runtime_state()["operations"]
    finally:
        runtime.close()
        runtime.store.close()


def test_source_aliases_resolve_to_one_binding(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMERA_SERIAL", raising=False)
    source = _piper_source(tmp_path / "piper.yaml")
    alias = tmp_path / "alias.yaml"
    alias.symlink_to(source)
    workspace = tmp_path / "workspace"

    absolute = prepare_binding(source, workspace=workspace)
    aliased = prepare_binding(alias, workspace=workspace)
    monkeypatch.chdir(tmp_path)
    relative = prepare_binding(Path("piper.yaml"), workspace=workspace)

    assert absolute.source == source.resolve()
    assert aliased.source == absolute.source
    assert relative.source == absolute.source
    assert aliased.binding_id == absolute.binding_id == relative.binding_id


def test_snapshot_requires_a_real_source_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="configuration source"):
        prepare_binding(
            tmp_path / "missing.yaml",
            config_snapshot={"adapter": "piper", "can_port": "can_test"},
        )


def test_configuration_factory_is_called_once_before_repeated_session_builds(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMERA_SERIAL", raising=False)
    source = _piper_source(tmp_path / "piper.yaml")
    from jiuwensymbiosis.adapters.piper.session import build_piper_session

    factory = build_piper_session.config_factory
    original_from_dict = factory.from_dict
    calls = []

    def counted_from_dict(data):
        calls.append(data)
        return original_from_dict(data)

    monkeypatch.setattr(factory, "from_dict", counted_from_dict)
    binding = prepare_binding(source, workspace=tmp_path / "ws")
    assert len(calls) == 1

    binding.build_session()
    binding.build_session()
    assert len(calls) == 1


def test_config_snapshot_and_returned_data_are_deeply_isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMERA_SERIAL", raising=False)
    source = _piper_source(tmp_path / "piper.yaml")
    supplied = {
        "adapter": "piper",
        "can_port": "can_snapshot",
        "calib_path": "calibration/not-created.json",
        "agent": {"exec_mode": "fastagent"},
    }

    binding = prepare_binding(source, config_snapshot=supplied, workspace=tmp_path / "ws")
    supplied["can_port"] = "changed-after-prepare"
    supplied["agent"]["exec_mode"] = "changed"

    returned = binding.config_data()
    assert returned["can_port"] == "can_snapshot"
    assert returned["agent"]["exec_mode"] == "fastagent"
    returned["agent"]["exec_mode"] = "changed-through-copy"
    assert binding.config_data()["agent"]["exec_mode"] == "fastagent"

    session = binding.build_session()
    assert session.env.cfg.can_port == "can_snapshot"
    assert session._connected is False
    assert session.env.low_level is None


def test_known_relative_config_paths_resolve_even_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMERA_SERIAL", raising=False)
    source_dir = tmp_path / "nested" / "config"
    source_dir.mkdir(parents=True)
    source = source_dir / "cruzr.yaml"
    source.write_text("adapter: cruzr\n", encoding="utf-8")
    snapshot = {
        "adapter": "cruzr",
        "ros_workspace": "ros/ws",
        "camera_calib_path": "calib/camera.json",
        "urdf_path": "models/robot.urdf",
        "urdf_package_dir": "models/packages",
        "camera_fastdds_profiles_file": "dds/profile.xml",
    }

    binding = prepare_binding(source, config_snapshot=snapshot, workspace=tmp_path / "ws")
    resolved = binding.config_data()

    assert resolved["ros_workspace"] == str((source_dir / "ros/ws").resolve())
    assert resolved["camera_calib_path"] == str((source_dir / "calib/camera.json").resolve())
    assert resolved["urdf_path"] == str((source_dir / "models/robot.urdf").resolve())
    assert resolved["urdf_package_dir"] == str((source_dir / "models/packages").resolve())
    assert resolved["camera_fastdds_profiles_file"] == str((source_dir / "dds/profile.xml").resolve())


def test_adapter_owns_new_path_fields_without_runtime_changes(tmp_path, monkeypatch):
    from jiuwensymbiosis.runtime import bindings

    @dataclass
    class Config:
        path_fields: ClassVar[tuple[str, ...]] = ("vendor_model",)
        vendor_model: str

        @classmethod
        def from_dict(cls, data):
            return cls(data["vendor_model"])

    calls = []

    def builder(cfg, **_kwargs):
        calls.append(cfg)
        return SimpleNamespace(env=SimpleNamespace(cfg=cfg))

    monkeypatch.setattr(bindings, "_discover_adapter", lambda _: (Config, builder, lambda cfg: ("device:test",)))
    source = tmp_path / "robot.yaml"
    source.write_text("adapter: custom\nvendor_model: models/vendor.bin\ncalib_path: not-owned\n")
    binding = prepare_binding(source)
    assert binding.config_data()["vendor_model"] == str(tmp_path / "models/vendor.bin")
    assert binding.config_data()["calib_path"] == "not-owned"
    assert binding.build_session().env.cfg.vendor_model == str(tmp_path / "models/vendor.bin")
    assert len(calls) == 1


def test_camera_environment_default_is_captured_for_resources_and_session(tmp_path, monkeypatch):
    source = tmp_path / "piper.yaml"
    source.write_text("adapter: piper\ncan_port: can_test\ncamera_serial: file-serial\n", encoding="utf-8")
    monkeypatch.setenv("CAMERA_SERIAL", "camera-from-env")

    binding = prepare_binding(source, workspace=tmp_path / "ws")
    assert "camera:camera-from-env" in binding.resources
    assert "camera:file-serial" not in binding.resources

    monkeypatch.setenv("CAMERA_SERIAL", "camera-after-binding")
    session = binding.build_session()
    assert session.env.cfg.camera_serial == "camera-from-env"
    assert session._connected is False
    assert session.env.low_level is None


def test_adapter_device_resources_use_canonical_local_identities(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMERA_SERIAL", raising=False)
    source = _piper_source(tmp_path / "piper.yaml")
    binding = prepare_binding(source, workspace=tmp_path / "ws")
    assert "can:can_test" in binding.resources

    serial_link = tmp_path / "serial-alias"
    serial_target = tmp_path / "serial-device"
    serial_target.touch()
    serial_link.symlink_to(serial_target)
    from jiuwensymbiosis.adapters.so101.session import build_so101_session

    cfg = build_so101_session.config_factory.from_dict(
        {
            "port": str(serial_link),
            "home_use_init_pose": True,
            "safety_validated": True,
            "joint_limits": {
                "shoulder_pan": [-90, 90],
                "shoulder_lift": [-90, 90],
                "elbow_flex": [-90, 90],
                "wrist_flex": [-90, 90],
                "wrist_roll": [-90, 90],
            },
        }
    )
    assert tuple(build_so101_session.resource_keys(cfg)) == (f"serial:{os.path.realpath(serial_target)}",)

    from jiuwensymbiosis.adapters.cruzr.session import build_cruzr_session

    monkeypatch.setenv("ROS_DOMAIN_ID", "27")
    cruzr_cfg = build_cruzr_session.config_factory.from_dict({"command_topic": "/robot/command"})
    assert tuple(build_cruzr_session.resource_keys(cruzr_cfg)) == ("ros:27:/robot/command",)


def test_common_resource_keys_use_local_port_lock_and_skip_remote_services():
    def detector(host: str, *, spawn: bool = True):
        return SimpleNamespace(
            camera_serial="camera-1",
            detector=SimpleNamespace(host=host, port=8114, spawn=spawn),
        )

    expected = ("camera:camera-1", "detector:local:8114")
    for host in ("localhost", "127.0.0.1", "0.0.0.0", "192.168.1.42", "robot.local"):
        assert common_resource_keys(detector(host)) == expected
    assert common_resource_keys(detector("detector.example", spawn=False)) == ("camera:camera-1",)
    with pytest.raises(ValueError, match="host"):
        common_resource_keys(detector("", spawn=True))


def test_physical_device_id_supplements_derived_keys():
    cfg = SimpleNamespace(camera_serial="cam", detector=SimpleNamespace(spawn=False))
    keys = complete_resource_keys(
        ("can:can_left",),
        cfg,
        physical_device_id="piper-serial-42",
    )
    assert keys == ("camera:cam", "can:can_left", "physical:piper-serial-42")


def test_adapter_without_device_key_requires_physical_device_id():
    cfg = SimpleNamespace(camera_serial="camera", detector=SimpleNamespace(spawn=False))
    with pytest.raises(ValueError, match="physical_device_id"):
        complete_resource_keys((), cfg)
    assert complete_resource_keys((), cfg, physical_device_id="mobile-robot") == (
        "camera:camera",
        "physical:mobile-robot",
    )


def test_without_sidecars_preserves_captured_resources_and_build_behavior(tmp_path, monkeypatch):
    source = _piper_source(tmp_path / "piper.yaml")
    monkeypatch.setenv("CAMERA_SERIAL", "camera-before")
    with_sidecar = prepare_binding(source, workspace=tmp_path / "ws")
    assert "detector:local:8114" in with_sidecar.resources

    monkeypatch.setenv("CAMERA_SERIAL", "camera-after")
    without_sidecar = with_sidecar.without_sidecars()
    assert without_sidecar is not with_sidecar
    assert without_sidecar.binding_id != with_sidecar.binding_id
    assert "detector:local:8114" not in without_sidecar.resources
    assert "camera:camera-before" in without_sidecar.resources
    assert "camera:camera-after" not in without_sidecar.resources
    session = without_sidecar.build_session()
    assert session.env.cfg.camera_serial == "camera-before"
    assert session.sidecar_starters == []
    assert session._connected is False


def test_cruzr_ros_domain_is_frozen_with_the_binding(tmp_path, monkeypatch):
    source = tmp_path / "cruzr.yaml"
    source.write_text("adapter: cruzr\n", encoding="utf-8")
    monkeypatch.setenv("ROS_DOMAIN_ID", "27")
    binding = prepare_binding(
        source,
        config_snapshot={"adapter": "cruzr", "command_topic": "/robot/command"},
        workspace=tmp_path / "ws",
    )
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")

    without_sidecars = binding.without_sidecars()
    assert "ros:27:/robot/command" in without_sidecars.resources
    assert "ros:42:/robot/command" not in without_sidecars.resources


def test_binding_repr_does_not_expose_config_values(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMERA_SERIAL", raising=False)
    source = tmp_path / "piper.yaml"
    source.write_text(
        "adapter: piper\ncan_port: can_test\nmodel:\n  api_key: never-print-this-token\n",
        encoding="utf-8",
    )
    binding = prepare_binding(source, workspace=tmp_path / "ws")
    assert "never-print-this-token" not in repr(binding)
    assert binding.binding_id not in {"", "never-print-this-token"}
