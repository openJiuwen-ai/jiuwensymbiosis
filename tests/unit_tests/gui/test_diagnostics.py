# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""diagnostics:失败原因分类(纯逻辑,证据强度从具体到通用)。"""

from __future__ import annotations

from jiuwensymbiosis.gui import diagnostics
from jiuwensymbiosis.gui.diagnostics import FIX_USE_HF_MIRROR, FIX_USE_LOCAL_MODEL, diagnose


def test_gdino_missing_weights_is_model_not_ready():
    d = diagnose(
        "detector server exited (code 1) before becoming ready",
        log_tail="[detector] OSError: ... does not appear to have a file named model.safetensors",
    )
    assert "模型" in d.title  # 面向用户,标题讲"模型未就绪"而非"权重"
    assert FIX_USE_LOCAL_MODEL in d.fixes and FIX_USE_HF_MIRROR in d.fixes


def test_sam2_processor_missing_is_model_not_ready():
    # SAM2 缺 processor_config.json,应归入"模型未就绪",且文案不能只提 GroundingDINO
    d = diagnose(
        "detector server exited (code 1) before becoming ready",
        log_tail="[detector] OSError: Can't load processor for 'facebook/sam2.1-hiera-large' ... processor_config.json",
    )
    assert "模型" in d.title
    assert "SAM2" in d.cause
    assert FIX_USE_LOCAL_MODEL in d.fixes and FIX_USE_HF_MIRROR in d.fixes


def test_detector_timeout_network_mentions_network():
    d = diagnose(
        "detector server not ready on 127.0.0.1:8114 within 300s",
        log_tail="[detector] httpx ... Connection to huggingface.co timed out",
    )
    assert "超时" in d.title
    assert "网络" in d.cause or "huggingface" in d.cause
    assert FIX_USE_LOCAL_MODEL in d.fixes and FIX_USE_HF_MIRROR in d.fixes


def test_detector_crash_without_weight_signature_is_generic_detector():
    d = diagnose("detector server exited (code 1) before becoming ready", log_tail="[detector] some other error")
    assert d.title == "视觉检测器启动失败"
    assert FIX_USE_LOCAL_MODEL in d.fixes


def test_port_in_use():
    d = diagnose("RuntimeError: address already in use")
    assert "端口" in d.title
    assert d.fixes == ()


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


def test_fallback_is_conservative():
    d = diagnose("some totally unexpected traceback with no known signature")
    assert d.title == "运行失败"
    assert d.fixes == ()
    assert d.steps == ()  # 兜底不写"去翻日志"这类话术,交给页面的高级用户提示


def test_module_exports_fix_keys():
    assert diagnostics.FIX_USE_LOCAL_MODEL == "use_local_model"
    assert diagnostics.FIX_USE_HF_MIRROR == "use_hf_mirror"
