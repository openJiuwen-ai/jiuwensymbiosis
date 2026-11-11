# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""errors:失败码契约 + 携带它们的异常类型。"""

from __future__ import annotations

from jiuwensymbiosis import errors
from jiuwensymbiosis.errors import (
    DETECTION_REASONS,
    ERROR_CODES,
    DetectionError,
    DetectorStartError,
    GraspNotConfirmedError,
    JiuwenSymbiosisError,
    SafetyViolationError,
    error_code,
)


def test_detection_reasons_are_part_of_the_code_set():
    assert DETECTION_REASONS <= ERROR_CODES


def test_reuses_the_same_object_as_the_result_contract():
    # 结果契约的 reason 集合与 code 表必须是同一份,否则两边会各自漂移
    from jiuwensymbiosis import contracts

    assert contracts.DETECTION_REASONS is DETECTION_REASONS


def test_typed_errors_carry_their_code():
    assert error_code(SafetyViolationError("refused")) == errors.SAFETY_REJECTED
    assert error_code(GraspNotConfirmedError("no contact")) == errors.GRASP_NOT_CONFIRMED
    assert error_code(DetectorStartError("no port")) == errors.DETECTOR_START_TIMEOUT
    assert error_code(DetectionError("miss", code="no_camera")) == "no_camera"


def test_plain_exception_has_no_code():
    assert error_code(RuntimeError("boom")) == ""


def test_adapter_defined_codes_survive():
    # 适配器早有自己的 .code 约定(CartesianServoError 等),不能被框架的集合挡掉
    class DriverError(ValueError):
        code = "cartesian_bounds_rejected"

    assert error_code(DriverError("rejected")) == "cartesian_bounds_rejected"


def test_safety_violation_stays_a_value_error():
    # LLM 自纠契约 + 运动代码里的 except ValueError 都依赖这个类型
    assert issubclass(SafetyViolationError, ValueError)
    assert issubclass(SafetyViolationError, JiuwenSymbiosisError)


def test_safety_violation_declines_recovery():
    # 预检拒绝发生在下发前,RecoveryRail 不应为它 home/release
    assert SafetyViolationError("refused").skip_recovery is True
