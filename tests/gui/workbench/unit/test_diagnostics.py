# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""diagnostics:失败原因分类(纯逻辑,证据强度从具体到通用)。"""

from __future__ import annotations

import pytest

from jiuwensymbiosis import errors
from jiuwensymbiosis_gui.workbench import diagnostics
from jiuwensymbiosis_gui.workbench.diagnostics import FIX_USE_HF_MIRROR, FIX_USE_LOCAL_MODEL, diagnose


def test_detector_startup_failure_offers_model_fixes():
    # 检测器起不来时端口始终不开,真实错误串只有这一种形态(detector_sidecar.py);
    # needle 必须跟它逐字对齐,否则规则形同虚设。
    d = diagnose("RuntimeError: detector server did not start on 127.0.0.1:8114 within 300s")
    assert "超时" in d.title
    assert FIX_USE_LOCAL_MODEL in d.fixes and FIX_USE_HF_MIRROR in d.fixes


def test_detector_startup_cause_covers_missing_model_not_only_network():
    d = diagnose("RuntimeError: detector server did not start on 127.0.0.1:8114 within 300s")
    assert "模型" in d.cause and "huggingface" in d.cause  # 不把成因写死成"网络不通"


def test_port_in_use():
    d = diagnose("RuntimeError: address already in use")
    assert "端口" in d.title
    assert d.fixes == ()


def test_port_word_in_a_message_is_not_port_in_use():
    # "port" + "in use" 两个泛词凑巧同现不足以判端口占用
    d = diagnose("RuntimeError: serial port /dev/ttyACM0 is in use by another process")
    assert "端口被占用" not in d.title


def test_llm_auth_failure():
    d = diagnose("openai.AuthenticationError: Invalid API key provided")
    assert "鉴权" in d.title


def test_missing_api_key_is_auth_not_fallback():
    # openjiuwen 的报错用词是 "api_key is required",而非 "invalid key/401"
    d = diagnose(
        "[181002] model service config error, reason: model client config api_key is required for OpenAI client."
    )
    assert "鉴权" in d.title
    assert "API Key" in d.title or "API Key" in d.cause


def test_llm_endpoint_unreachable_not_confused_with_detector():
    d = diagnose("httpx.ConnectError: [Errno -2] Name or service not known")
    assert "连不上" in d.title


def test_detection_miss_is_no_detection_card():
    # fast 路径检测未命中中止:错误诊断给中文卡,而非把英文糊在相机下方
    d = diagnose(
        "RuntimeError: detection for 'black box' produced no usable result (reason=no_valid_depth); "
        "later steps read 'black_box.<field>' — aborting instead of crashing downstream"
    )
    assert d.title == "没识别到目标物体"
    assert d.fixes == ()


def test_out_of_reach_is_reach_card():
    d = diagnose("RuntimeError: [Piper] EndPose target OUT OF REACH — Arm Status=TARGET_POS_EXCEEDS_LIMIT")
    assert "可达" in d.title


# 真机原始报错(2026-09-22)。它同时踩了两个坑:驱动的措辞是 "UNREACHABLE / out of
# envelope" 而非 "out of reach",所以可达卡漏判;落到鉴权卡时,末端位姿的 Z 坐标
# " 403.873" 命中了子串 " 403",于是一次运动学失败被报成"API Key 没填或不对"。
_UNREACHABLE_WITH_403_COORDINATE = (
    "[runner] step 3 failed: goto_xyzr({'x': 'pick.x', 'y': 'pick.y', 'z': 'pick.grasp_z + 40'}): "
    "RuntimeError: [Piper] EndPose not executing — arm did not move from start after 2.0s; "
    "target likely UNREACHABLE / out of envelope "
    "(target=(285.3, 64.4, 252.6, 180.0, 30.0, 178.0), "
    "last=(142.524, -7.331, 403.873, -174.945, 39.333, 178.038)). Aborted; arm held in place."
)


def test_unreachable_target_carrying_a_403_coordinate_is_reach_card_not_auth():
    d = diagnose(_UNREACHABLE_WITH_403_COORDINATE)
    assert "可达" in d.title
    assert "鉴权" not in d.title


def test_a_pose_coordinate_is_never_read_as_an_http_auth_code():
    # 数值与 HTTP 状态码同形但前后接着数字/小数点:不能算鉴权信号。
    for coordinate in ("403.873", "142.403", "1401", "401.0", "-403.5", "401", "403", "-403"):
        d = diagnose(f"RuntimeError: something we have no rule for, last=(1.0, {coordinate}, 2.0)")
        assert d.title == "运行失败", coordinate


def test_real_http_auth_codes_still_match():
    for text in (
        "openai.APIStatusError: 401 Unauthorized",
        "httpx.HTTPStatusError: server returned HTTP 403",
        "model service error (status 401)",
        "httpx.HTTPStatusError: 403 Forbidden",
        "model service error (status_code=403)",
        "model service error: Error code: 401",
        "requests.HTTPError: 403 Client Error: Forbidden for url: https://model.example/v1",
    ):
        assert "鉴权" in diagnose(text).title, text


def test_detector_service_unreachable_is_not_blamed_on_arm_reach():
    # detector_client 的连不上也含 "unreachable",但那是检测服务,不是机械臂到不了。
    d = diagnose("RuntimeError: detector service at http://127.0.0.1:8114 unreachable: 3 attempts failed")
    assert "可达" not in d.title


@pytest.mark.parametrize(
    "error",
    (
        "httpx.ConnectError: [Errno 101] Network is unreachable",
        "OSError: [Errno 113] Host is unreachable",
        "RuntimeError: model service unreachable",
    ),
)
def test_network_unreachable_is_not_blamed_on_arm_reach(error):
    assert diagnose(error).title == "运行失败"


@pytest.mark.parametrize("error", ("RuntimeError: hardware fault E401", "RuntimeError: device 403B failed"))
def test_hardware_identifiers_are_not_http_auth_status(error):
    assert diagnose(error).title == "运行失败"


@pytest.mark.parametrize(
    "error",
    (
        "ValueError: target unreachable, rejected",
        "ValueError: near singularity or unreachable orientation; target rejected",
        "ValueError: goto_pose waypoint: IK raised ValueError; target likely unreachable or on a singularity.",
        "RuntimeError: ik_unreachable",
        "RuntimeError: lift_unreachable",
        "RuntimeError: unreachable_any_lifter",
        "RuntimeError: place_unreachable_any_lifter",
    ),
)
def test_adapter_reachability_failures_keep_the_reach_card(error):
    assert "可达" in diagnose(error).title


def test_no_camera_is_camera_card_not_no_detection():
    # 相机没连上时 reason=no_camera 也被包进 "produced no usable result",不能误诊成"没识别到物体"
    d = diagnose("RuntimeError: detection for 'black box' produced no usable result (reason=no_camera); aborting")
    assert d.title == "没读到相机画面"


def test_detector_unavailable_routes_to_model_not_ready():
    d = diagnose("RuntimeError: detection for 'box' produced no usable result (reason=detector_unavailable); aborting")
    assert "模型" in d.title
    assert FIX_USE_LOCAL_MODEL in d.fixes


def test_fallback_is_conservative():
    d = diagnose("some totally unexpected traceback with no known signature")
    assert d.title == "运行失败"
    assert d.fixes == ()
    assert d.steps == ()  # 兜底不写"去翻日志"这类话术,交给页面的高级用户提示


def test_module_exports_fix_keys():
    assert diagnostics.FIX_USE_LOCAL_MODEL == "use_local_model"
    assert diagnostics.FIX_USE_HF_MIRROR == "use_hf_mirror"


def test_track_grasp_timeout_reads_as_camera_not_object():
    # track_grasp collapses a dead camera into "not detected"; a frame-timeout in
    # the log tail must route it to the camera card, not "没识别到目标物体".
    d = diagnose(
        "RuntimeError: target 'banana' not detected",
        log_tail="[SO-101 vision] grab_frames error: Frame didn't arrive within 2000ms",
    )
    assert d.title == "没读到相机画面"


def test_not_detected_without_camera_evidence_stays_no_detection():
    d = diagnose("RuntimeError: target 'banana' not detected", log_tail="[runner] track_grasp 'banana' approach")
    assert d.title == "没识别到目标物体"


def test_code_wins_over_the_text_rules():
    # 源头已经写下 code,就不该再由文本猜:这里文本明明是"检测没结果"
    d = diagnose("RuntimeError: detection produced no usable result", code="no_camera")
    assert d.title == "没读到相机画面"


def test_code_lookup_survives_a_text_that_looks_like_something_else():
    d = diagnose("RuntimeError: serial port /dev/ttyACM0 error", code="grasp_not_confirmed")
    assert d.title == "夹爪合拢后没夹到东西"


def test_safety_rejection_code_gets_its_own_card():
    d = diagnose("SafetyViolationError: SafetyRail: refusing goto_xyzr: z=-9 below z_floor=0", code="safety_rejected")
    assert d.title == "动作被安全护栏拦下"
    assert d.steps  # 给出可操作的下一步,而不是只报错


def test_unknown_code_falls_back_to_the_text_rules():
    # 适配器自带的 code 没有对应卡时,不能吃掉文本判断
    d = diagnose("RuntimeError: target 'banana' not detected", code="hardware_send_mismatch")
    assert d.title == "没识别到目标物体"


def test_every_framework_code_has_a_card():
    # errors.ERROR_CODES 是框架自己写得出来的码,每一个都必须有卡,否则源头明明报了准
    # 信、界面却只能显示兜底。反向不成立:适配器自带的码(cartesian_bounds_rejected)
    # 也可以映射进来,它们不在 ERROR_CODES 里。
    assert errors.ERROR_CODES <= set(diagnostics._CODE_TABLE)


def test_servo_cartesian_bounds_rejection_shares_the_safety_card():
    d = diagnose(
        "So101CartesianServoError: cartesian_bounds_rejected: servo_to_pose target: z=-9 below driver z_min_safe=0",
        code="cartesian_bounds_rejected",
    )
    assert d.title == "动作被安全护栏拦下"


def test_the_two_cards_word_the_same_check_the_same_way():
    # 抓取扑空与目标越界都要查感知/标定;两处措辞若各写各的,用户会当成两件事。
    grasp = diagnose("x", code="grasp_not_confirmed")
    safety = diagnose("x", code="safety_rejected")
    shared = set(grasp.steps) & set(safety.steps)
    assert len(shared) == 1


class TestArmBusCards:
    """CAN 与串口分开认:so101 没有 CAN,不能被指去激活 CAN。"""

    def test_can_error_reads_as_can(self):
        assert diagnose("OSError: [Errno 19] can0: No such device").title == "机械臂连接失败"
        assert "CAN" in diagnose("OSError: can0 no such device").cause

    def test_serial_error_does_not_mention_can(self):
        d = diagnose("SerialException: could not open port /dev/ttyACM0")
        assert d.title == "机械臂连接失败"
        assert "CAN" not in d.cause
        assert "串口" in d.cause

    def test_a_bare_device_error_is_not_blamed_on_the_arm(self):
        # "no such device" 相机也会报;泛词不该被机械臂卡吃掉。
        assert diagnose("RuntimeError: no such device").title == "运行失败"


def test_every_rule_needs_a_main_error_signal():
    # 结构不变量:没有任何一条规则可以只凭日志尾命中。强信号可以是子串或正则。
    assert all(rule.err_needles or rule.err_patterns for rule in diagnostics._RULES)


def test_noisy_log_tail_alone_does_not_hijack_an_unknown_error():
    noisy = "\n".join(
        [
            "INFO jiuwensymbiosis: opening serial port /dev/ttyACM0",
            "INFO jiuwensymbiosis: probe returned 401",
            "WARNING jiuwensymbiosis: address already in use",
            "WARNING jiuwensymbiosis: CUDA out of memory",
        ]
    )
    d = diagnose("RuntimeError: something we have no rule for", log_tail=noisy)
    assert d.title == "运行失败"


def test_structured_reason_beats_generic_hardware_wording():
    # 相机报错里带 "serial"(相机序列号),不能被"机械臂连接失败"那条泛词规则抢走
    d = diagnose("RuntimeError: camera serial 34AB no frames; detection produced no usable result (reason=no_camera)")
    assert d.title == "没读到相机画面"


def test_non_detection_failure_not_hijacked_by_stray_frame_timeout():
    # A grasp-not-confirmed failure that merely happens to carry a frame-timeout
    # line in its log tail must NOT be mis-attributed to the camera.
    d = diagnose(
        "RuntimeError: grasp_not_confirmed: gripper closed without object contact",
        log_tail="[SO-101 vision] grab_frames error: Frame didn't arrive within 2000ms",
    )
    assert d.title != "没读到相机画面"
