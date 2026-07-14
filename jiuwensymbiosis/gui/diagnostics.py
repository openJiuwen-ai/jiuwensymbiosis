# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""把一次运行失败翻译成"人话"诊断:标题 + 原因 + 怎么办 + 可点的修复。

纯逻辑、不依赖 Qt,便于单测。输入是异常文本 + 最近日志尾(``log_tail``,含检测器
子进程转发来的 ``[detector]`` 行)。判定用一张**规则表**:每条规则 = 一个匹配谓词 +
一条预置诊断,按"证据强度"从具体到通用排列,命中第一条即返回;都不中就退回通用卡——
**绝不臆断**具体原因(免得像"写死 huggingface.co"那样误导)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "Diagnosis",
    "diagnose",
    "FIX_USE_LOCAL_MODEL",
    "FIX_USE_HF_MIRROR",
]

# 修复动作 key —— 界面据此决定显示哪些修复按钮(标签/行为在 run_page 里映射)。
FIX_USE_LOCAL_MODEL = "use_local_model"
FIX_USE_HF_MIRROR = "use_hf_mirror"


@dataclass(frozen=True)
class Diagnosis:
    """一条面向用户的诊断结论。

    Attributes:
        title: 简短中文标题(一句话说清"是什么错")。
        cause: 一句话原因。
        steps: "怎么办"要点。
        fixes: 适用的一键修复 key(见本模块常量);界面据此显示按钮。
    """

    title: str
    cause: str
    steps: tuple[str, ...] = ()
    fixes: tuple[str, ...] = ()


def _has(text: str, *needles: str) -> bool:
    """``text`` 是否包含任一子串。"""
    return any(n in text for n in needles)


# ------------------------------------------------------------------ 预置诊断
# 文案先给判断("错误可能是源于……");涉及模型来源的问题把"怎么办"交给界面的两栏「解决方法」
# 承接(fixes),不在文案里让用户去翻日志——查看日志只作为熟悉系统者的兜底提示。
_MODEL_NOT_READY = Diagnosis(
    title="视觉检测模型未就绪",
    cause="错误可能是源于视觉检测模型(GroundingDINO 或 SAM2)没能就绪——本机缺少对应的模型文件,或之前没下载完整。",
    fixes=(FIX_USE_LOCAL_MODEL, FIX_USE_HF_MIRROR),
)
_DETECTOR_TIMEOUT = Diagnosis(
    title="视觉检测模型下载/加载超时",
    cause="错误可能是源于下载视觉检测模型时网络不通(连不上国外的 huggingface.co),或模型加载太慢。",
    fixes=(FIX_USE_LOCAL_MODEL, FIX_USE_HF_MIRROR),
)
_DETECTOR_CRASH = Diagnosis(
    title="视觉检测器启动失败",
    cause="错误可能是源于视觉检测器启动异常,较常见的是它的模型没就绪或来源不对。",
    fixes=(FIX_USE_LOCAL_MODEL, FIX_USE_HF_MIRROR),
)
_PORT_IN_USE = Diagnosis(
    title="检测器端口被占用",
    cause="错误可能是源于检测器要用的端口被别的程序占用了(常见于上一次没退出干净)。",
    steps=("结束占用该端口的程序后重试,或在配置里把检测器端口换一个。",),
)
_LLM_AUTH = Diagnosis(
    title="大模型鉴权失败:API Key 没填或不对",
    cause="错误可能是源于大模型服务的 API Key 没填(留空)或填得不对。",
    steps=("到「配置 → 模型」里把 API Key 填上(SiliconFlow 等需要鉴权的服务必填),并确认服务地址正确。",),
)
_LLM_ENDPOINT = Diagnosis(
    title="连不上大模型服务",
    cause="错误可能是源于填写的大模型服务地址不通(网络不通,或地址写错了)。",
    steps=("到「配置 → 模型」里检查服务地址,确认它在本机能打开。",),
)
_GPU_OOM = Diagnosis(
    title="显存不足",
    cause="错误可能是源于显卡显存不够,模型没能加载或运行起来。",
    steps=("关掉占显存的其它程序后重试,或换用更小的模型。",),
)
_ARM_CAN = Diagnosis(
    title="机械臂连接失败",
    cause="错误可能是源于连不上机械臂(CAN 接口没激活,或线没接好)。",
    steps=("确认 CAN 已激活、线缆已接;或先用「模拟模式」不接硬件跑一遍验证流程。",),
)
_NO_CAMERA = Diagnosis(
    title="没读到相机画面",
    cause="错误可能是源于相机没连上/没被识别到(没插好、被别的程序占用,或配置里相机序列号不对),视觉拿不到画面。",
    steps=("确认相机已插好、未被其它程序占用;并在「配置」里核对相机序列号/分辨率后重试。",),
)
_NO_DETECTION = Diagnosis(
    title="没识别到目标物体",
    cause="错误可能是源于视觉没能识别/定位到目标物体(物体不在画面里、被遮挡,或深度/光照不佳),动作序列因此中止。",
    steps=("确认目标物体在相机视野内、没被挡住、光照充足;必要时调整物体摆放或相机角度后重试。",),
)
_OUT_OF_REACH = Diagnosis(
    title="目标超出机械臂可达范围",
    cause="错误可能是源于目标位置超出了机械臂的可达空间或关节限位,动作被中止(机械臂已停在原地)。",
    steps=("把目标移到机械臂工作范围内(更靠近基座)后重试;并确认标定/工作区设置正确。",),
)
_FALLBACK = Diagnosis(
    title="运行失败",
    cause="暂时无法自动判断具体原因。",
)

# ------------------------------------------------------------------ 规则表
# 每条 = (谓词(err, both) -> bool, 诊断)。``err`` 是异常文本,``both`` 是异常+日志尾,
# 均小写。按证据强度从具体到通用排列,命中第一条即返回。
_Matcher = Callable[[str, str], bool]
_RULES: tuple[tuple[_Matcher, Diagnosis], ...] = (
    (
        lambda err, both: (
            "detector server exited" in err
            and _has(
                both, "does not appear to have a file", "no file named", "can't load processor", "processor_config"
            )
        ),
        _MODEL_NOT_READY,
    ),
    (lambda err, both: "detector server" in err and _has(err, "not ready", "within"), _DETECTOR_TIMEOUT),
    (lambda err, both: "detector server exited" in err, _DETECTOR_CRASH),
    (lambda err, both: _has(both, "address already in use") or ("port" in both and "in use" in both), _PORT_IN_USE),
    (
        lambda err, both: _has(
            both,
            "unauthorized",
            "invalid api key",
            "invalid_api_key",
            "api_key is required",
            "api key is required",
            "authentication",
            " 401",
            " 403",
        ),
        _LLM_AUTH,
    ),
    (
        lambda err, both: (
            "detector" not in both
            and _has(both, "getaddrinfo", "name or service not known", "failed to establish", "connection refused")
        ),
        _LLM_ENDPOINT,
    ),
    (lambda err, both: _has(both, "out of memory", "cuda oom", "cublas_status_alloc_failed"), _GPU_OOM),
    (lambda err, both: _has(both, "can_left", "socketcan", "no such device", "serial", "can0"), _ARM_CAN),
    (lambda err, both: _has(err, "out of reach", "exceeds_limit", "out_of_reach"), _OUT_OF_REACH),
    # no_camera / detector_unavailable 必须排在「没识别到目标物体」之前:它们也含
    # "produced no usable result" 子串,否则相机/检测器问题会被误诊成"物体没识别到"。
    (lambda err, both: _has(both, "no_camera", "no camera"), _NO_CAMERA),
    (lambda err, both: _has(both, "detector_unavailable"), _MODEL_NOT_READY),
    (lambda err, both: _has(err, "produced no usable result", "not detected", "no_valid_depth"), _NO_DETECTION),
)


def diagnose(error_text: str, log_tail: str = "") -> Diagnosis:
    """按规则表返回第一条命中的诊断;都不中则返回通用卡。"""
    err = (error_text or "").lower()
    both = f"{err}\n{(log_tail or '').lower()}"
    for matches, diagnosis in _RULES:
        if matches(err, both):
            return diagnosis
    return _FALLBACK
