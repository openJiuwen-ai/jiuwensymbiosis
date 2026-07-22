# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.piper.config."""

from __future__ import annotations

import pytest
import yaml

from jiuwensymbiosis.adapters.piper.config import (
    PiperConfig,
    _extract_detector_from_api_servers,
)


class TestPiperConfigDefaults:
    def test_from_dict_flat(self):
        cfg = PiperConfig.from_dict({"can_port": "can_right", "move_speed": 80})
        assert cfg.can_port == "can_right"
        assert cfg.move_speed == 80
        assert cfg.tool_offset_mm == 135.8

    def test_from_dict_nested(self):
        data = {
            "env": {
                "cfg": {
                    "low_level": {
                        "can_port": "can_left",
                        "move_speed": 30,
                        "tool_offset_mm": 100.0,
                    },
                    "prompt": "pick the box",
                },
            },
        }
        cfg = PiperConfig.from_dict(data)
        assert cfg.can_port == "can_left"
        assert cfg.move_speed == 30
        assert cfg.tool_offset_mm == 100.0
        assert cfg.task_prompt == "pick the box"

    def test_from_dict_defaults(self):
        cfg = PiperConfig.from_dict({})
        assert cfg.can_port == "can_left"
        assert cfg.z_min_safe_mm == 50.0
        assert cfg.joint_limits is None

    def test_from_dict_joint_limits_normalises_tuples(self):
        """YAML loads inner bounds as lists; from_dict must coerce to tuples."""
        cfg = PiperConfig.from_dict({"joint_limits": {"J1": [-360.0, 360.0], "J2": [-135.0, 135.0]}})
        assert cfg.joint_limits == {"J1": (-360.0, 360.0), "J2": (-135.0, 135.0)}
        # inner bounds are tuples, not lists (matches the dataclass annotation)
        assert all(isinstance(v, tuple) for v in cfg.joint_limits.values())

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # wrong top-level type (not a dict) → None, no crash
            ([-360.0, 360.0], None),
            ("not-a-dict", None),
            (5, None),
            # inner value wrong arity/type → that joint dropped
            ({"J1": 5}, None),  # scalar, not iterable
            ({"J1": [1, 2, 3]}, None),  # 3 elements, not 2
            ({"J1": ["a", "b"]}, None),  # not float-coercible
            # mixed: one good + one bad → keep the good, drop the bad
            ({"J1": [-360.0, 360.0], "J2": "bad"}, {"J1": (-360.0, 360.0)}),
        ],
        ids=[
            "list-instead-of-dict",
            "string-instead-of-dict",
            "scalar",
            "inner-scalar",
            "inner-3-tuple",
            "inner-non-float",
            "mixed-good-and-bad",
        ],
    )
    def test_from_dict_joint_limits_malformed_drops_safely(self, raw, expected):
        cfg = PiperConfig.from_dict({"joint_limits": raw})
        assert cfg.joint_limits == expected


class TestExtractDetectorFromApiServers:
    def test_with_grounding_dino(self):
        servers = [
            {
                "_target_": "jiuwensymbiosis.serving.grounding_dino_sam2_server",
                "host": "192.168.1.10",
                "port": 9000,
            }
        ]
        cfg = _extract_detector_from_api_servers(servers)
        assert cfg.url == "http://192.168.1.10:9000"
        assert cfg.spawn is True

    def test_empty_servers(self):
        cfg = _extract_detector_from_api_servers([])
        assert cfg.url == "http://127.0.0.1:8114"

    def test_no_matching_server(self):
        servers = [{"_target_": "other_server", "host": "1.2.3.4", "port": 9999}]
        cfg = _extract_detector_from_api_servers(servers)
        assert cfg.url == "http://127.0.0.1:8114"

    @pytest.mark.parametrize(
        ("field_name", "expected"),
        [
            ("port", 8114),
            ("startup_timeout_s", 300.0),
            ("box_threshold", 0.35),
            ("text_threshold", 0.25),
            ("use_sam2", True),
        ],
    )
    def test_null_uses_field_default(self, field_name, expected):
        servers = [{"_target_": "x.grounding_dino_sam2_server", field_name: None}]
        cfg = _extract_detector_from_api_servers(servers)
        assert getattr(cfg, field_name) == expected

    @pytest.mark.parametrize("field_name", ["port", "startup_timeout_s", "box_threshold", "text_threshold"])
    def test_invalid_number_names_field(self, field_name):
        servers = [{"_target_": "x.grounding_dino_sam2_server", field_name: "bad"}]
        with pytest.raises(ValueError, match=rf"api_servers detector\.{field_name}"):
            _extract_detector_from_api_servers(servers)

    @pytest.mark.parametrize("field_name", ["port", "startup_timeout_s", "box_threshold", "text_threshold"])
    @pytest.mark.parametrize("value", [True, False])
    def test_boolean_is_rejected_for_number(self, field_name, value):
        servers = [{"_target_": "x.grounding_dino_sam2_server", field_name: value}]
        with pytest.raises(ValueError, match=rf"api_servers detector\.{field_name}"):
            _extract_detector_from_api_servers(servers)

    def test_invalid_boolean_names_field(self):
        servers = [{"_target_": "x.grounding_dino_sam2_server", "use_sam2": "false"}]
        with pytest.raises(ValueError, match=r"api_servers detector\.use_sam2"):
            _extract_detector_from_api_servers(servers)


class TestPiperConfigFromYaml:
    def test_from_yaml(self, tmp_path):
        data = {
            "can_port": "can_left",
            "move_speed": 40,
            "tool_offset_mm": 135.8,
        }
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump(data), encoding="utf-8")
        cfg = PiperConfig.from_yaml(p)
        assert cfg.can_port == "can_left"
        assert cfg.move_speed == 40

    def test_from_yaml_joint_limits(self, tmp_path):
        data = {"joint_limits": {"J1": [-360.0, 360.0], "J2": [-135.0, 135.0]}}
        p = tmp_path / "limits.yaml"
        p.write_text(yaml.dump(data), encoding="utf-8")
        cfg = PiperConfig.from_yaml(p)
        assert cfg.joint_limits == {"J1": (-360.0, 360.0), "J2": (-135.0, 135.0)}
        assert isinstance(cfg.joint_limits["J1"], tuple)


class TestEnvVarOverrides:
    """Environment variable overrides for detector model IDs and camera serial."""

    def test_gdino_model_id_env_override(self, monkeypatch):
        monkeypatch.setenv("GDINO_MODEL_ID", "my-org/custom-gdino")
        servers = [
            {
                "_target_": "jiuwensymbiosis.serving.grounding_dino_sam2_server",
                "gdino_model_id": "IDEA-Research/grounding-dino-base",
            }
        ]
        cfg = _extract_detector_from_api_servers(servers)
        assert cfg.gdino_model_id == "my-org/custom-gdino"

    def test_sam2_model_id_env_override(self, monkeypatch):
        monkeypatch.setenv("SAM2_MODEL_ID", "my-org/custom-sam2")
        servers = [
            {
                "_target_": "jiuwensymbiosis.serving.grounding_dino_sam2_server",
                "sam2_model_id": "facebook/sam2.1-hiera-large",
            }
        ]
        cfg = _extract_detector_from_api_servers(servers)
        assert cfg.sam2_model_id == "my-org/custom-sam2"

    def test_env_override_no_yaml_value(self, monkeypatch):
        """Env var should apply even when YAML doesn't set the field (defaults are used)."""
        monkeypatch.setenv("GDINO_MODEL_ID", "env-only-dino")
        monkeypatch.setenv("SAM2_MODEL_ID", "env-only-sam2")
        cfg = _extract_detector_from_api_servers([])
        assert cfg.gdino_model_id == "env-only-dino"
        assert cfg.sam2_model_id == "env-only-sam2"

    def test_camera_serial_env_override(self, monkeypatch):
        monkeypatch.setenv("CAMERA_SERIAL", "999999999999")
        cfg = PiperConfig.from_dict({"camera_serial": "123456789012"})
        assert cfg.camera_serial == "999999999999"

    def test_camera_serial_env_override_no_yaml(self, monkeypatch):
        monkeypatch.setenv("CAMERA_SERIAL", "999999999999")
        cfg = PiperConfig.from_dict({})
        assert cfg.camera_serial == "999999999999"
