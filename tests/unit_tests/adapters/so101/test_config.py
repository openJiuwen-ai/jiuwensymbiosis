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
        cfg = So101Config(**_base_kwargs(max_relative_target=None))
        assert cfg.max_relative_target is None

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            So101Config.from_dict(_base_kwargs(max_relative_target=float("nan")))

    def test_none_is_the_default_with_overcompensation(self):
        cfg = So101Config(**_base_kwargs())
        assert cfg.max_relative_target is None
        assert cfg.settle_overcompensate is True


class TestCartesianOrientationPolicy:
    def test_defaults_to_preserve(self):
        cfg = So101Config(**_base_kwargs())
        assert cfg.cartesian_orientation_policy == "preserve"
        assert cfg.grasp_orientation is None

    def test_policy_is_normalised(self):
        cfg = So101Config(**_base_kwargs(cartesian_orientation_policy=" TOP_DOWN "))
        assert cfg.cartesian_orientation_policy == "top_down"

    @pytest.mark.parametrize("value", ["free", "", 1, None])
    def test_invalid_policy_rejected(self, value):
        with pytest.raises(ValueError, match="cartesian_orientation_policy"):
            So101Config(**_base_kwargs(cartesian_orientation_policy=value))

    def test_grasp_orientation_is_normalised(self):
        cfg = So101Config(**_base_kwargs(grasp_orientation={"rx": 145, "ry": -10.0, "rz": 90}))
        assert cfg.grasp_orientation == {"rx": 145.0, "ry": -10.0, "rz": 90.0}

    @pytest.mark.parametrize(
        "value",
        [
            [145.0, -10.0, 90.0],
            {"rx": 145.0, "ry": -10.0},
            {"rx": 145.0, "ry": -10.0, "rz": float("nan")},
        ],
    )
    def test_invalid_grasp_orientation_rejected(self, value):
        with pytest.raises(ValueError, match="grasp_orientation"):
            So101Config(**_base_kwargs(grasp_orientation=value))


class TestPlaceZOffsetMigration:
    def test_new_name_defaults_to_75_mm(self):
        cfg = So101Config(**_base_kwargs())
        assert cfg.place_z_offset_mm == 75.0

    def test_new_name_is_canonical(self):
        cfg = So101Config(**_base_kwargs(place_z_offset_mm=20.0))
        assert cfg.place_z_offset_mm == 20.0

    def test_deprecated_alias_is_accepted_and_logged(self, caplog):
        caplog.set_level("WARNING", logger="jiuwensymbiosis.adapters.so101.config")

        cfg = So101Config.from_dict(_base_kwargs(chip_thickness_mm=20.0))

        assert cfg.place_z_offset_mm == 20.0
        assert "chip_thickness_mm is deprecated" in caplog.text
        assert "place_z_offset_mm" in caplog.text

    def test_new_and_deprecated_names_conflict_even_when_values_match(self):
        with pytest.raises(ValueError, match="cannot both be configured"):
            So101Config.from_dict(
                _base_kwargs(
                    place_z_offset_mm=20.0,
                    chip_thickness_mm=20.0,
                )
            )


class TestMotionConfig:
    def test_direct_defaults_separate_normal_and_fast_rates(self):
        cfg = So101Config(**_base_kwargs())
        assert cfg.trajectory_hz == 20.0
        assert cfg.max_joint_vel_dps == 35.0
        assert cfg.max_cartesian_vel_mm_s == 60.0
        assert cfg.fast_control_hz == 5.0
        assert cfg.servo_max_joint_vel_dps == 35.0
        assert cfg.servo_max_cartesian_vel_mm_s == 60.0
        assert cfg.tracking_error_deg == 4.0
        assert cfg.max_joint_step_deg == pytest.approx(1.75)
        assert cfg.servo_max_joint_step_deg == pytest.approx(3.5)
        assert cfg.tracking_error_deg > cfg.servo_max_joint_step_deg
        assert cfg.servo_min_send_period_s == pytest.approx(0.2)
        assert cfg.cartesian_interp_step_mm == pytest.approx(3.0)
        assert cfg.servo_max_cartesian_step_mm == pytest.approx(6.0)
        assert cfg.fast_move_timeout_s == pytest.approx(20.0)
        assert cfg.fast_absolute_timeout_s == pytest.approx(60.0)

    def test_explicit_normal_rate_recomputes_inherited_fast_steps(self):
        cfg = So101Config(**_base_kwargs(trajectory_hz=10.0))
        assert cfg.max_joint_step_deg == pytest.approx(3.5)
        assert cfg.cartesian_interp_step_mm == pytest.approx(6.0)
        assert cfg.servo_max_joint_step_deg == pytest.approx(7.0)
        assert cfg.servo_max_cartesian_step_mm == pytest.approx(12.0)
        assert cfg.servo_min_send_period_s == pytest.approx(0.2)

    def test_explicit_fast_rate_only_recomputes_send_period(self):
        cfg = So101Config(**_base_kwargs(fast_control_hz=10.0))
        assert cfg.max_joint_step_deg == pytest.approx(1.75)
        assert cfg.cartesian_interp_step_mm == pytest.approx(3.0)
        assert cfg.servo_max_joint_step_deg == pytest.approx(3.5)
        assert cfg.servo_max_cartesian_step_mm == pytest.approx(6.0)
        assert cfg.servo_min_send_period_s == pytest.approx(0.1)

    def test_explicit_fast_steps_override_normal_inheritance(self):
        cfg = So101Config(
            **_base_kwargs(
                servo_max_joint_step_deg=0.8,
                servo_max_cartesian_step_mm=1.5,
            )
        )
        assert cfg.servo_max_joint_step_deg == pytest.approx(0.8)
        assert cfg.servo_max_cartesian_step_mm == pytest.approx(1.5)

    def test_grouped_motion_block_is_normalised(self):
        cfg = So101Config.from_dict(
            {
                **_base_kwargs(),
                "motion": {
                    "trajectory_hz": 10.0,
                    "max_joint_vel_dps": 20.0,
                    "max_cartesian_vel_mm_s": 40.0,
                    "fast_control_hz": 4.0,
                    "servo_max_joint_vel_dps": 20.0,
                    "servo_max_cartesian_vel_mm_s": 30.0,
                    "tracking_error_deg": 2.5,
                },
            }
        )
        assert cfg.trajectory_hz == 10.0
        assert cfg.max_joint_step_deg == pytest.approx(2.0)
        assert cfg.cartesian_interp_step_mm == pytest.approx(4.0)
        assert cfg.fast_control_hz == pytest.approx(4.0)
        assert cfg.servo_max_joint_step_deg == pytest.approx(4.0)
        assert cfg.servo_max_cartesian_step_mm == pytest.approx(8.0)
        assert cfg.servo_min_send_period_s == pytest.approx(0.25)
        assert cfg.servo_max_cartesian_vel_mm_s == pytest.approx(30.0)
        assert cfg.tracking_error_deg == pytest.approx(2.5)

    def test_grouped_motion_conflict_is_rejected(self):
        with pytest.raises(ValueError, match=r"conflicting motion\.trajectory_hz"):
            So101Config.from_dict({**_base_kwargs(trajectory_hz=20.0), "motion": {"trajectory_hz": 30.0}})

    def test_removed_flat_profile_is_rejected(self):
        with pytest.raises(ValueError, match="motion_profile was removed"):
            So101Config.from_dict(_base_kwargs(motion_profile="balanced"))

    def test_removed_grouped_profile_is_rejected(self):
        with pytest.raises(ValueError, match="unknown motion field 'profile'"):
            So101Config.from_dict({**_base_kwargs(), "motion": {"profile": "balanced"}})

    def test_runtime_steps_are_derived_from_velocity_and_rate(self):
        cfg = So101Config(**_base_kwargs())
        assert cfg.max_joint_step_deg == pytest.approx(cfg.max_joint_vel_dps / cfg.trajectory_hz)
        assert cfg.cartesian_interp_step_mm == pytest.approx(cfg.max_cartesian_vel_mm_s / cfg.trajectory_hz)
        assert cfg.servo_max_joint_step_deg == pytest.approx(2.0 * cfg.max_joint_step_deg)
        assert cfg.servo_max_cartesian_step_mm == pytest.approx(2.0 * cfg.cartesian_interp_step_mm)
        assert cfg.servo_min_send_period_s == pytest.approx(1.0 / cfg.fast_control_hz)
        assert cfg.joint_tolerance_deg == pytest.approx(1.5)
        assert cfg.settle_soft_tolerance_deg == pytest.approx(3.0)
        assert cfg.settle_soft_samples == 5
        assert cfg.settle_max_z_undershoot_mm == pytest.approx(10.0)
        assert cfg.settle_z_only_lift_enabled is True
        assert cfg.settle_z_only_lift_step_mm == pytest.approx(2.0)
        assert cfg.settle_z_only_lift_max_joint_offset_deg == pytest.approx(4.0)
        assert "normal=20Hz/35deg/s/60mm/s" in cfg.motion_summary()
        assert "fast=5Hz/3.5deg/6mm-step" in cfg.motion_summary()
        assert "source=direct" in cfg.motion_summary()
        assert "settle=1.5/3deg" in cfg.motion_summary()

    def test_soft_settle_must_not_be_tighter_than_strict_settle(self):
        with pytest.raises(ValueError, match="settle_soft_tolerance_deg"):
            So101Config(
                **_base_kwargs(
                    joint_tolerance_deg=2.0,
                    settle_soft_tolerance_deg=1.5,
                )
            )

    def test_z_undershoot_tolerance_must_be_non_negative(self):
        with pytest.raises(ValueError, match="settle_max_z_undershoot_mm"):
            So101Config(
                **_base_kwargs(
                    settle_max_z_undershoot_mm=-0.1,
                )
            )

    def test_grouped_soft_settle_values_are_loaded(self):
        cfg = So101Config.from_dict(
            {
                **_base_kwargs(),
                "motion": {
                    "settle_soft_tolerance_deg": 2.5,
                    "settle_soft_samples": 4,
                    "settle_max_z_undershoot_mm": 2.0,
                },
            }
        )
        assert cfg.settle_soft_tolerance_deg == pytest.approx(2.5)
        assert cfg.settle_soft_samples == 4
        assert cfg.settle_max_z_undershoot_mm == pytest.approx(2.0)

    def test_grouped_safety_and_grasp_values_are_normalised(self):
        cfg = So101Config.from_dict(
            {
                **_base_kwargs(),
                "safety": {
                    "table_z_mm": -15.0,
                    "gripper_lowest_offset_mm": 20.0,
                    "minimum_floor_margin_mm": 9.0,
                },
                "grasp": {"payload_protrusion_mm": 30.0},
            }
        )
        assert cfg.table_z_mm == -15.0
        assert cfg.gripper_lowest_offset_mm == 20.0
        assert cfg.payload_protrusion_mm == 30.0
        assert cfg.minimum_floor_margin_mm == 9.0

    def test_grouped_safety_conflict_is_rejected(self):
        with pytest.raises(ValueError, match=r"conflicting safety\.table_z_mm"):
            So101Config.from_dict(
                {
                    **_base_kwargs(table_z_mm=-15.0),
                    "safety": {"table_z_mm": -20.0},
                }
            )


class TestDetectorConfig:
    def test_api_servers_detector_extracted(self):
        cfg = So101Config.from_dict(
            {
                **_base_kwargs(camera_serial="camera"),
                "api_servers": [
                    {
                        "_target_": "jiuwensymbiosis.serving.grounding_dino_sam2_server.main",
                        "host": "127.0.0.1",
                        "port": 9000,
                    },
                ],
            }
        )
        assert cfg.detector.url == "http://127.0.0.1:9000"
        assert cfg.detector.spawn is True
        assert cfg.detector.port == 9000

    def test_spawn_address_is_derived_from_url(self):
        # The extractor builds url="http://localhost:9123"; DetectorServerConfig
        # __post_init__ then derives host/port back from that url (spawn=True).
        cfg = So101Config.from_dict(
            {
                **_base_kwargs(camera_serial="camera"),
                "api_servers": [
                    {
                        "_target_": "jiuwensymbiosis.serving.grounding_dino_sam2_server.main",
                        "host": "localhost",
                        "port": 9123,
                    },
                ],
            }
        )
        assert cfg.detector.host == "localhost"
        assert cfg.detector.port == 9123

    def test_no_api_servers_yields_fail_closed_default(self):
        # No api_servers entry -> DetectorServerConfig() default (spawn=False).
        cfg = So101Config.from_dict({**_base_kwargs(camera_serial="camera")})
        assert cfg.detector.url == "http://127.0.0.1:8114"
        assert cfg.detector.spawn is False

    def test_unknown_detector_field_is_rejected(self):
        # DetectorServerConfig.from_dict still rejects unknown keys — the
        # extractor does not go through from_dict, but the class contract holds.
        from jiuwensymbiosis.adapters.so101.config import DetectorServerConfig

        with pytest.raises(ValueError, match="unknown detector fields.*spwan"):
            DetectorServerConfig.from_dict({"spwan": True})

    def test_spawn_rejects_non_http_url(self):
        # __post_init__ rejects https (non-http) url when spawn=True.
        from jiuwensymbiosis.adapters.so101.config import DetectorServerConfig

        with pytest.raises(ValueError, match="absolute http URL"):
            DetectorServerConfig(url="https://localhost:9123", spawn=True)

    def test_api_servers_env_override(self, monkeypatch):
        monkeypatch.setenv("GDINO_MODEL_ID", "env-gdino")
        cfg = So101Config.from_dict(
            {
                **_base_kwargs(camera_serial="camera"),
                "api_servers": [
                    {
                        "_target_": "jiuwensymbiosis.serving.grounding_dino_sam2_server.main",
                        "gdino_model_id": "IDEA-Research/grounding-dino-base",
                    },
                ],
            }
        )
        assert cfg.detector.gdino_model_id == "env-gdino"

    def test_env_override_without_api_servers(self, monkeypatch):
        monkeypatch.setenv("GDINO_MODEL_ID", "env-only-gdino")
        monkeypatch.setenv("SAM2_MODEL_ID", "env-only-sam2")

        cfg = So101Config.from_dict(_base_kwargs())

        assert cfg.detector.spawn is False
        assert cfg.detector.gdino_model_id == "env-only-gdino"
        assert cfg.detector.sam2_model_id == "env-only-sam2"

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
    def test_api_server_null_uses_field_default(self, field_name, expected):
        cfg = So101Config.from_dict(
            {
                **_base_kwargs(),
                "api_servers": [{"_target_": "x.grounding_dino_sam2_server", field_name: None}],
            }
        )
        assert getattr(cfg.detector, field_name) == expected

    @pytest.mark.parametrize("field_name", ["port", "startup_timeout_s", "box_threshold", "text_threshold"])
    def test_invalid_api_server_number_names_field(self, field_name):
        with pytest.raises(ValueError, match=rf"api_servers detector\.{field_name}"):
            So101Config.from_dict(
                {
                    **_base_kwargs(),
                    "api_servers": [{"_target_": "x.grounding_dino_sam2_server", field_name: "bad"}],
                }
            )

    @pytest.mark.parametrize("field_name", ["port", "startup_timeout_s", "box_threshold", "text_threshold"])
    @pytest.mark.parametrize("value", [True, False])
    def test_api_server_boolean_is_rejected_for_number(self, field_name, value):
        with pytest.raises(ValueError, match=rf"api_servers detector\.{field_name}"):
            So101Config.from_dict(
                {
                    **_base_kwargs(),
                    "api_servers": [{"_target_": "x.grounding_dino_sam2_server", field_name: value}],
                }
            )

    def test_invalid_api_server_boolean_names_field(self):
        with pytest.raises(ValueError, match=r"api_servers detector\.use_sam2"):
            So101Config.from_dict(
                {
                    **_base_kwargs(),
                    "api_servers": [{"_target_": "x.grounding_dino_sam2_server", "use_sam2": "false"}],
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


class TestHomeUseInitPose:
    def test_defaults_false(self):
        cfg = So101Config(**_base_kwargs())
        assert cfg.home_use_init_pose is False

    def test_init_pose_allows_missing_home_joints(self):
        # home_use_init_pose=True: home_joints_deg may be omitted entirely.
        kw = _base_kwargs()
        kw.pop("home_joints_deg")
        cfg = So101Config(home_use_init_pose=True, **kw)
        assert cfg.home_use_init_pose is True
        assert cfg.home_joints_deg is None  # filled at connect() time

    def test_init_pose_still_validates_given_home(self):
        # A caller may still pass home_joints_deg alongside the flag; it must be
        # length/finite-checked (the flag doesn't relax validation of an explicit list).
        with pytest.raises(ValueError, match="5 arm joints"):
            So101Config(home_use_init_pose=True, **_base_kwargs(home_joints_deg=[0.0, 0.0, 0.0]))

    def test_missing_home_without_flag_rejected(self):
        kw = _base_kwargs()
        kw.pop("home_joints_deg")
        with pytest.raises(ValueError, match="home_use_init_pose"):
            So101Config(**kw)

    @pytest.mark.parametrize("value", [1, "true", None])
    def test_non_bool_rejected(self, value):
        with pytest.raises(ValueError, match="home_use_init_pose must be bool"):
            So101Config(**_base_kwargs(home_use_init_pose=value))


class TestSafetyValidated:
    def test_defaults_false(self):
        cfg = So101Config(**_base_kwargs())
        assert cfg.safety_validated is False

    @pytest.mark.parametrize("value", [1, "true", None])
    def test_non_bool_rejected(self, value):
        with pytest.raises(ValueError, match="safety_validated must be bool"):
            So101Config(**_base_kwargs(safety_validated=value))


class TestZMaxSafe:
    """Upper Z bound on the control frame: optional ceiling symmetric to z_min."""

    def test_default_none(self):
        cfg = So101Config(**_base_kwargs())
        assert cfg.z_max_safe_mm is None

    def test_none_explicit(self):
        cfg = So101Config(**_base_kwargs(z_max_safe_mm=None))
        assert cfg.z_max_safe_mm is None

    def test_float_normalised_and_kept(self):
        cfg = So101Config(**_base_kwargs(z_max_safe_mm=121))
        assert cfg.z_max_safe_mm == 121.0
        assert isinstance(cfg.z_max_safe_mm, float)

    def test_below_floor_rejected(self):
        # Ceiling must be strictly above the floor; equal also rejected.
        with pytest.raises(ValueError, match="z_max_safe_mm .* must be > z_min_safe_mm"):
            So101Config(**_base_kwargs(z_min_safe_mm=30.0, z_max_safe_mm=30.0))

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="z_max_safe_mm must be finite"):
            So101Config(**_base_kwargs(z_max_safe_mm=float("inf")))


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


class TestGripperContactDetection:
    def test_defaults(self):
        cfg = So101Config(**_base_kwargs())
        assert cfg.gripper_contact_min_travel == 5.0
        assert cfg.gripper_contact_stall_tolerance == 0.5
        assert cfg.gripper_contact_stall_samples == 5
        assert cfg.gripper_contact_hold_offset == 1.0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("gripper_contact_min_travel", 0.0),
            ("gripper_contact_min_travel", float("nan")),
            ("gripper_contact_stall_tolerance", 0.0),
            ("gripper_contact_stall_tolerance", float("inf")),
            ("gripper_contact_hold_offset", -0.1),
            ("gripper_contact_hold_offset", float("nan")),
            ("gripper_contact_stall_samples", 0),
            ("gripper_contact_stall_samples", 1.5),
            ("gripper_contact_stall_samples", True),
        ],
    )
    def test_invalid_values_rejected(self, field, value):
        with pytest.raises(ValueError, match=field):
            So101Config(**_base_kwargs(**{field: value}))


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


class TestServoGoalTolerance:
    def test_default_is_one_mm(self):
        cfg = So101Config(**_base_kwargs())
        assert cfg.servo_goal_tolerance_mm == 1.0

    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
    def test_non_positive_or_non_finite_rejected(self, value):
        with pytest.raises(ValueError, match="servo_goal_tolerance_mm"):
            So101Config(**_base_kwargs(servo_goal_tolerance_mm=value))


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
