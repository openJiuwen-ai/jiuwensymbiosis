# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.so101.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from jiuwensymbiosis.adapters.so101.config import So101Config

_ARM_LIMITS = {
    "shoulder_pan": (-90.0, 90.0),
    "shoulder_lift": (-90.0, 90.0),
    "elbow_flex": (-90.0, 90.0),
    "wrist_flex": (-90.0, 90.0),
    "wrist_roll": (-180.0, 180.0),
}


def _base_kwargs(**overrides) -> dict:
    base: dict = {
        "port": "/dev/ttyUSB0",
        "home_joints_deg": [0.0, 0.0, 0.0, 0.0, 0.0],
        "joint_limits": _ARM_LIMITS,
    }
    base.update(overrides)
    return base


class TestMaxRelativeTarget:
    """The high-priority sync from §A3: float-only, dict rejected, int normalised."""

    def test_float_passes_through(self):
        cfg = So101Config(**_base_kwargs(max_relative_target=5.0))
        assert cfg.max_relative_target == 5.0
        assert isinstance(cfg.max_relative_target, float)

    def test_int_normalised_to_float_via_loader(self):
        # Int normalisation happens in the loader (from_dict), per plan §A3.
        cfg = So101Config.from_dict(_base_kwargs(max_relative_target=5))
        assert isinstance(cfg.max_relative_target, float)
        assert cfg.max_relative_target == 5.0

    def test_dict_form_rejected_in_from_dict(self):
        with pytest.raises(ValueError, match="must be a float, not a dict"):
            So101Config.from_dict(_base_kwargs(max_relative_target={"shoulder_pan.pos": 1.0}))

    def test_none_allowed(self):
        # ``None`` remains supported when the settle over-compensation path is
        # disabled; combining them would remove the only firmware slew guard.
        cfg = So101Config(**_base_kwargs(max_relative_target=None, settle_overcompensate=False))
        assert cfg.max_relative_target is None

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            So101Config.from_dict(_base_kwargs(max_relative_target=float("nan")))

    def test_none_rejected_when_overcompensation_enabled(self):
        with pytest.raises(ValueError, match="settle_overcompensate=True"):
            So101Config(**_base_kwargs(max_relative_target=None))


class TestDetectorConfig:
    def test_nested_detector_is_preserved(self):
        cfg = So101Config.from_dict(
            {
                **_base_kwargs(camera_serial="camera"),
                "detector": {"url": "http://127.0.0.1:9000", "spawn": True, "port": 9000},
            }
        )
        assert cfg.detector.url == "http://127.0.0.1:9000"
        assert cfg.detector.spawn is True
        assert cfg.detector.port == 9000

    def test_spawn_address_is_derived_from_url(self):
        cfg = So101Config.from_dict(
            {
                **_base_kwargs(camera_serial="camera"),
                "detector": {"url": "http://localhost:9123", "spawn": True},
            }
        )
        assert cfg.detector.host == "localhost"
        assert cfg.detector.port == 9123

    def test_unknown_detector_field_is_rejected(self):
        with pytest.raises(ValueError, match="unknown detector fields.*spwan"):
            So101Config.from_dict(
                {
                    **_base_kwargs(camera_serial="camera"),
                    "detector": {"spwan": True},
                }
            )

    def test_spawn_rejects_non_http_url(self):
        with pytest.raises(ValueError, match="absolute http URL"):
            So101Config.from_dict(
                {
                    **_base_kwargs(camera_serial="camera"),
                    "detector": {"url": "https://localhost:9123", "spawn": True},
                }
            )


class TestMaxRelativeTargetDirectConstruction:
    """Direct So101Config(...) bypasses from_dict; post_init must enforce too."""

    def test_dict_rejected_on_direct_construction(self):
        with pytest.raises(ValueError, match="must be a float, not a dict"):
            So101Config(**_base_kwargs(max_relative_target={"shoulder_pan.pos": 1.0}))

    def test_negative_rejected_on_direct_construction(self):
        with pytest.raises(ValueError, match="must be > 0"):
            So101Config(**_base_kwargs(max_relative_target=-1.0))

    def test_nan_rejected_on_direct_construction(self):
        with pytest.raises(ValueError, match="finite"):
            So101Config(**_base_kwargs(max_relative_target=float("nan")))

    def test_bool_rejected_on_direct_construction(self):
        # bool is an int subclass but is not a valid motor-step limit.
        with pytest.raises(ValueError, match="must be a number|finite"):
            So101Config(**_base_kwargs(max_relative_target=True))

    def test_zero_rejected(self):
        with pytest.raises(ValueError, match="must be > 0"):
            So101Config(**_base_kwargs(max_relative_target=0.0))

    def test_int_normalised_to_float_on_direct_construction(self):
        # LeRobot's ensure_safe_goal_position does isinstance(mrt, float); an int
        # is NOT a float subclass and would raise TypeError on the first motion.
        # __post_init__ must normalise int -> float so direct construction is safe.
        cfg = So101Config(**_base_kwargs(max_relative_target=3))
        assert cfg.max_relative_target == 3.0
        assert isinstance(cfg.max_relative_target, float), (
            "max_relative_target must be a float, not int — LeRobot's "
            "ensure_safe_goal_position rejects non-float values."
        )


class TestJointLimits:
    def test_exact_keys_required(self):
        bad = dict(_ARM_LIMITS)
        bad.pop("wrist_roll")
        with pytest.raises(ValueError, match="missing"):
            So101Config(**_base_kwargs(joint_limits=bad))

    def test_extra_key_rejected(self):
        bad = dict(_ARM_LIMITS)
        bad["extra_joint"] = (-10.0, 10.0)
        with pytest.raises(ValueError, match="unexpected"):
            So101Config(**_base_kwargs(joint_limits=bad))

    def test_unordered_pair_rejected(self):
        bad = dict(_ARM_LIMITS)
        bad["shoulder_pan"] = (90.0, -90.0)  # lo > hi
        with pytest.raises(ValueError, match="ordered"):
            So101Config(**_base_kwargs(joint_limits=bad))


class TestHomeJoints:
    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="5 arm joints"):
            So101Config(**_base_kwargs(home_joints_deg=[0.0, 0.0, 0.0]))

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            So101Config(**_base_kwargs(home_joints_deg=[0, 0, 0, 0, float("inf")]))


class TestSafetyValidated:
    def test_defaults_false(self):
        cfg = So101Config(**_base_kwargs())
        assert cfg.safety_validated is False

    @pytest.mark.parametrize("value", [1, "true", None])
    def test_non_bool_rejected(self, value):
        with pytest.raises(ValueError, match="safety_validated must be bool"):
            So101Config(**_base_kwargs(safety_validated=value))

    def test_shipped_config_is_fail_closed_and_within_urdf_wrist_limit(self):
        repo_root = Path(__file__).resolve().parents[4]
        cfg = So101Config.from_yaml(repo_root / "configs" / "so101" / "so101.yaml")
        assert cfg.safety_validated is False
        wrist_lo, wrist_hi = cfg.joint_limits["wrist_roll"]
        assert -157.2 < wrist_lo < wrist_hi < 162.8


class TestGripperSettleS:
    def test_non_negative(self):
        cfg = So101Config(**_base_kwargs(gripper_settle_s=0.0))
        assert cfg.gripper_settle_s == 0.0

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="gripper_settle_s"):
            So101Config(**_base_kwargs(gripper_settle_s=-0.1))

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            So101Config(**_base_kwargs(gripper_settle_s=float("nan")))


class TestOrientationTolerance:
    def test_none_allowed(self):
        cfg = So101Config(**_base_kwargs(ik_orientation_tolerance_deg=None))
        assert cfg.ik_orientation_tolerance_deg is None

    def test_non_negative(self):
        cfg = So101Config(**_base_kwargs(ik_orientation_tolerance_deg=5.0))
        assert cfg.ik_orientation_tolerance_deg == 5.0

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="ik_orientation_tolerance_deg"):
            So101Config(**_base_kwargs(ik_orientation_tolerance_deg=-1.0))


class TestFromDictNested:
    def test_legacy_nested_layout(self):
        data = {
            "env": {
                "cfg": {
                    "low_level": {
                        "port": "/dev/ttyUSB0",
                        "home_joints_deg": [0, 0, 0, 0, 0],
                        "joint_limits": _ARM_LIMITS,
                        "max_relative_target": 3,
                    },
                    "prompt": "pick the cup",
                }
            }
        }
        cfg = So101Config.from_dict(data)
        assert cfg.port == "/dev/ttyUSB0"
        assert cfg.max_relative_target == 3.0
        assert isinstance(cfg.max_relative_target, float)
        assert cfg.task_prompt == "pick the cup"


class TestFromYamlPathResolution:
    """from_yaml must resolve relative urdf_path/calibration_dir against the
    YAML directory unconditionally (not only when the target exists), and must
    expand ``~``."""

    def _write_yaml(self, tmp_path, body: str) -> Path:
        p = tmp_path / "so101.yaml"
        p.write_text(body, encoding="utf-8")
        return Path(p)

    def test_relative_calibration_dir_resolved_even_when_absent(self, tmp_path):
        """A relative calibration_dir that does NOT yet exist must still be
        resolved against the YAML dir — LeRobot creates it during calibrate,
        so only-when-exists resolution would leave a cwd-relative path."""
        p = self._write_yaml(
            tmp_path,
            "env:\n  cfg:\n    low_level:\n"
            "      port: /dev/ttyUSB0\n"
            "      home_joints_deg: [0, 0, 0, 0, 0]\n"
            "      joint_limits:\n"
            "        shoulder_pan: [-90, 90]\n        shoulder_lift: [-90, 90]\n"
            "        elbow_flex: [-90, 90]\n        wrist_flex: [-90, 90]\n"
            "        wrist_roll: [-180, 180]\n"
            "      calibration_dir: calib/\n",
        )
        cfg = So101Config.from_yaml(p)
        assert cfg.calibration_dir == str((tmp_path / "calib").resolve())
        # The dir does not exist — resolution is unconditional.
        assert not (tmp_path / "calib").exists()

    def test_relative_urdf_path_resolved_against_yaml_dir(self, tmp_path):
        (tmp_path / "robot.urdf").write_text("<robot/>", encoding="utf-8")
        p = self._write_yaml(
            tmp_path,
            "env:\n  cfg:\n    low_level:\n"
            "      port: /dev/ttyUSB0\n"
            "      home_joints_deg: [0, 0, 0, 0, 0]\n"
            "      joint_limits:\n"
            "        shoulder_pan: [-90, 90]\n        shoulder_lift: [-90, 90]\n"
            "        elbow_flex: [-90, 90]\n        wrist_flex: [-90, 90]\n"
            "        wrist_roll: [-180, 180]\n"
            "      urdf_path: robot.urdf\n",
        )
        cfg = So101Config.from_yaml(p)
        assert cfg.urdf_path == str((tmp_path / "robot.urdf").resolve())

    def test_absolute_path_passed_through(self, tmp_path):
        abs_urdf = tmp_path / "abs.urdf"
        abs_urdf.write_text("<robot/>", encoding="utf-8")
        p = self._write_yaml(
            tmp_path,
            f"env:\n  cfg:\n    low_level:\n"
            f"      port: /dev/ttyUSB0\n"
            f"      home_joints_deg: [0, 0, 0, 0, 0]\n"
            f"      joint_limits:\n"
            f"        shoulder_pan: [-90, 90]\n        shoulder_lift: [-90, 90]\n"
            f"        elbow_flex: [-90, 90]\n        wrist_flex: [-90, 90]\n"
            f"        wrist_roll: [-180, 180]\n"
            f"      urdf_path: {abs_urdf}\n",
        )
        cfg = So101Config.from_yaml(p)
        assert cfg.urdf_path == str(abs_urdf.resolve())
