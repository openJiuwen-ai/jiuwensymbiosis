# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""calibration_view:写回运行配置、产物落盘约定与失败项建议映射。"""

from __future__ import annotations

import numpy as np
import pytest

from jiuwensymbiosis.calibration.domain.models import (
    AxxbResidualReport,
    CalibrationQualityReport,
    ReprojectionReport,
    TargetConsistencyReport,
    VerifyStat,
)
from jiuwensymbiosis.calibration.domain.quality import (
    EyeInHandAcceptancePolicy,
    EyeToHandAcceptancePolicy,
    ObservabilityReport,
)
from jiuwensymbiosis_gui.workbench.pages.calibration_view import _output_relpath, _remedy_for, _write_calib_path

_CONFIG = """\
env:
  cfg:
    low_level:
      port: "/dev/ttyACM0"
      # 本机验收软限位（deg）；更换机械臂后必须重新确认。
      joint_limits:
        shoulder_pan: [-110.0, 110.0]
      camera_mount: "eye_to_hand"
      calib_path: "so101_calib.json"
      camera_fps: 30
"""


@pytest.fixture(autouse=True)
def isolate_runtime_dir(tmp_path, monkeypatch):
    """Keep local resource locks and runtime state inside pytest's temporary directory."""
    monkeypatch.setenv("JIUWENSYMBIOSIS_RUNTIME_DIR", str(tmp_path / "runtime"))


class TestWriteCalibPath:
    def test_updates_only_that_key_and_keeps_comments(self, tmp_path):
        """配置里的注释记录了字段物理含义与验收状态,重写 YAML 会把它们全丢掉。"""
        config = tmp_path / "so101.yaml"
        config.write_text(_CONFIG, encoding="utf-8")
        artifact = tmp_path / "calibration" / "so101_eye_to_hand.json"
        artifact.parent.mkdir()
        artifact.write_text("{}", encoding="utf-8")

        written = _write_calib_path(config, artifact)

        text = config.read_text(encoding="utf-8")
        assert written == "calibration/so101_eye_to_hand.json"
        assert 'calib_path: "calibration/so101_eye_to_hand.json"' in text
        assert "# 本机验收软限位（deg）；更换机械臂后必须重新确认。" in text
        assert 'port: "/dev/ttyACM0"' in text
        assert "camera_fps: 30" in text

    def test_keeps_the_original_indentation(self, tmp_path):
        config = tmp_path / "so101.yaml"
        config.write_text(_CONFIG, encoding="utf-8")
        _write_calib_path(config, tmp_path / "c.json")
        line = next(x for x in config.read_text(encoding="utf-8").splitlines() if "calib_path" in x)
        assert line.startswith("      calib_path:")

    def test_backs_up_the_user_maintained_file(self, tmp_path):
        config = tmp_path / "so101.yaml"
        config.write_text(_CONFIG, encoding="utf-8")
        _write_calib_path(config, tmp_path / "c.json")
        assert (tmp_path / "so101.yaml.bak").read_text(encoding="utf-8") == _CONFIG

    def test_absolute_path_when_artifact_is_outside_the_config_dir(self, tmp_path):
        config = tmp_path / "cfg" / "so101.yaml"
        config.parent.mkdir()
        config.write_text(_CONFIG, encoding="utf-8")
        artifact = tmp_path / "elsewhere" / "c.json"
        assert _write_calib_path(config, artifact) == str(artifact)

    def test_refuses_when_the_key_is_absent_instead_of_appending_blindly(self, tmp_path):
        config = tmp_path / "so101.yaml"
        config.write_text("env:\n  cfg:\n    low_level:\n      port: /dev/ttyACM0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="没有 calib_path"):
            _write_calib_path(config, tmp_path / "c.json")


class TestRemedy:
    @pytest.mark.parametrize(
        ("check", "fragment"),
        [
            ("observability_flange_axes", "两条不同的轴"),
            ("observability_duplicates", "重复"),
            ("target_consistency", "刚性"),
        ],
    )
    def test_known_checks_map_to_an_actionable_step(self, check, fragment):
        remedy = _remedy_for(check)
        assert remedy is not None and fragment in remedy

    def test_suffixed_check_names_still_match(self):
        assert _remedy_for("observability_flange_axes_x") == _remedy_for("observability_flange_axes")

    def test_unknown_check_returns_none_rather_than_inventing_advice(self):
        assert _remedy_for("some_future_gate") is None

    def test_reproj_invalid_is_not_confused_with_reproj(self):
        """``reproj`` 是 ``reproj_invalid`` 的前缀:必须取最长匹配,否则两者共用一条建议。"""
        assert _remedy_for("reproj_invalid") != _remedy_for("reproj")

    def test_every_gate_either_policy_can_raise_has_a_remedy(self):
        """两套验收策略的失败项都要有中文建议。

        本体只是二选一地挑一套策略,不会引入新键 —— 会引入新键的是 policy 新增门禁,
        而那时 ``_REMEDY`` 不会报错,只会静默少一条建议。这条断言就是那道拦截。
        """
        raised: set[str] = set()
        for policy in (EyeInHandAcceptancePolicy(), EyeToHandAcceptancePolicy()):
            for finite in (True, False):
                raised |= set(policy.decide(_failing_quality(finite=finite)).failed_checks)

        # 先确认样本真的把两套门禁都踩响了,否则下面的覆盖断言会空转。
        assert {"reproj", "axxb_rot", "axxb_trans", "board_origin_spread"} <= raised
        assert {"observability_camera_sep", "observability_flange_rot", "target_consistency_trans"} <= raised
        assert {"reproj_invalid", "quality_nonfinite"} <= raised

        assert sorted(check for check in raised if _remedy_for(check) is None) == []


class TestOutputRelpath:
    def test_mount_is_part_of_the_filename(self):
        """同一本体日后同时支持两种安装方式时,两份产物不能互相覆盖。"""
        assert _output_relpath("piper", "eye_in_hand", {}) == "piper/calibration/piper_eye_in_hand.json"
        assert _output_relpath("piper", "eye_to_hand", {}) == "piper/calibration/piper_eye_to_hand.json"

    def test_new_body_needs_no_profile_line(self):
        assert _output_relpath("newarm", "eye_to_hand", {}) == "newarm/calibration/newarm_eye_to_hand.json"

    def test_profile_override_wins(self):
        assert _output_relpath("so101", "eye_to_hand", {"output_relpath": "x/y.json"}) == "x/y.json"

    def test_unknown_mount_still_yields_a_path(self):
        assert _output_relpath("newarm", "", {}) == "newarm/calibration/newarm.json"


def _failing_quality(*, finite: bool) -> CalibrationQualityReport:
    """一份把两套策略的门禁尽量全部踩掉的质量报告。

    ``finite=False`` 换成 NaN,用来触发 ``*_nonfinite`` 那一档 —— 它与"数值有限但全部
    超标"是互斥的两条分支,所以两个变体的失败项取并集才是完整覆盖。
    """
    bad = float("nan") if not finite else 1.0e6
    stat = VerifyStat(bad, bad, bad)
    weak = bad if not finite else 0.0
    return CalibrationQualityReport(
        reprojection=ReprojectionReport(per_view_rms_px=(), summary=VerifyStat(bad, bad, bad)),
        observability=ObservabilityReport(
            n_valid_pairs=1,
            n_relative_axes=0,
            max_relative_rotation_deg=weak,
            max_relative_translation_mm=weak,
            max_axis_separation_deg=weak,
            n_relative_axes_camera=0,
            max_relative_rotation_camera_deg=weak,
            max_relative_translation_camera_mm=weak,
            max_axis_separation_camera_deg=weak,
            n_duplicates=1,
            svd_condition_min=float("inf"),
        ),
        axxb_residual=AxxbResidualReport(rotation_deg=stat, translation_mm=stat),
        target_consistency=TargetConsistencyReport(
            invariant_frame="flange_target",
            mean_transform=np.eye(4),
            translation_residual_mm=stat,
            rotation_residual_deg=stat,
        ),
        cross_check=None,
    )
