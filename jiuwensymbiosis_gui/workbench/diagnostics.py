# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""把一次运行失败翻译成"人话"诊断:标题 + 原因 + 怎么办 + 可点的修复。

纯逻辑、不依赖 Qt,便于单测。输入是**主错误串** + 最近日志尾(``log_tail``)。判定用一张
**规则表**,命中第一条即返回;都不中就退回通用卡——**绝不臆断**具体原因(免得像"写死
huggingface.co"那样误导)。

两条硬约束,破了就会像"日志里撞到一个 serial 就说机械臂没连上"那样误诊:

1. **强信号只看主错误串**。日志尾是整段运行的噪声(默认 400 行),泛词(``serial`` /
   ``401`` / ``port``)在里面撞上纯属巧合;日志尾只能作为已命中规则的**佐证**。
2. **自家的结构化 reason 优先于外部文本推断**。``no_camera`` / ``detector_unavailable``
   等是我们自己写出来的,排在靠前;鉴权/网络/显存这些源头在三方库、只能靠文本猜的规则排在最后。
"""

from __future__ import annotations

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
# 文案先给判断(命中是启发式的,故留"可能"二字);涉及模型来源的问题把"怎么办"交给界面的两栏「解决方法」
# 承接(fixes),不在文案里让用户去翻日志——查看日志只作为熟悉系统者的兜底提示。
_MODEL_NOT_READY = Diagnosis(
    title="视觉检测模型未就绪",
    cause="视觉检测模型(GroundingDINO 或 SAM2)可能没能就绪:本机缺少对应的模型文件,或之前没下载完整。",
    fixes=(FIX_USE_LOCAL_MODEL, FIX_USE_HF_MIRROR),
)
_DETECTOR_TIMEOUT = Diagnosis(
    title="视觉检测模型下载/加载超时",
    cause="视觉检测器可能没能在超时前就绪:模型没下载完整、连不上国外的 huggingface.co,或加载太慢。",
    fixes=(FIX_USE_LOCAL_MODEL, FIX_USE_HF_MIRROR),
)
_PORT_IN_USE = Diagnosis(
    title="检测器端口被占用",
    cause="检测器要用的端口可能被别的程序占用了(常见于上一次没退出干净)。",
    steps=("结束占用该端口的程序后重试,或在配置里把检测器端口换一个。",),
)
_LLM_AUTH = Diagnosis(
    title="大模型鉴权失败:API Key 没填或不对",
    cause="大模型服务的 API Key 可能没填(留空)或填得不对。",
    steps=("到「配置 → 模型」里把 API Key 填上(SiliconFlow 等需要鉴权的服务必填),并确认服务地址正确。",),
)
_LLM_ENDPOINT = Diagnosis(
    title="连不上大模型服务",
    cause="填写的大模型服务地址可能不通(网络不通,或地址写错了)。",
    steps=("到「配置 → 模型」里检查服务地址,确认它在本机能打开。",),
)
_GPU_OOM = Diagnosis(
    title="显存不足",
    cause="显卡显存可能不够,模型没能加载或运行起来。",
    steps=("关掉占显存的其它程序后重试,或换用更小的模型。",),
)
_ARM_CAN = Diagnosis(
    title="机械臂连接失败",
    cause="可能连不上机械臂(CAN 接口没激活,或线没接好)。",
    steps=("确认 CAN 已激活、线缆已接后重试。",),
)
_ARM_SERIAL = Diagnosis(
    title="机械臂连接失败",
    cause="可能连不上机械臂(串口设备不存在、被占用,或没有访问权限)。",
    steps=("确认机械臂已上电、串口线已接;到「配置」核对端口(如 /dev/ttyACM0)。",),
)
_NO_CAMERA = Diagnosis(
    title="没读到相机画面",
    cause="相机可能没连上/没被识别到(没插好、被别的程序占用,或配置里相机序列号不对),视觉拿不到画面。",
    steps=("确认相机已插好、未被其它程序占用;并在「配置」里核对相机序列号/分辨率后重试。",),
)
_NO_DETECTION = Diagnosis(
    title="没识别到目标物体",
    cause="视觉可能没能识别/定位到目标物体(物体不在画面里、被遮挡,或深度/光照不佳),动作序列因此中止。",
    steps=("确认目标物体在相机视野内、没被挡住、光照充足;必要时调整物体摆放或相机角度后重试。",),
)
_OUT_OF_REACH = Diagnosis(
    title="目标超出机械臂可达范围",
    cause="目标位置可能超出了机械臂的可达空间或关节限位,动作被中止(机械臂已停在原地)。",
    steps=("把目标移到机械臂工作范围内(更靠近基座)后重试;并确认标定/工作区设置正确。",),
)
# 抓取点与运动目标都由感知加标定算出,两张卡指的是同一个检查——共用一份文案,免得
# 各写各的,让用户以为是两件事。
_CHECK_PERCEPTION = "检查感知与标定是否有偏差;必要时到「工具 → 手眼标定」重做标定。"

# 下面两张卡只由 code 命中(失败点自己写下的机器码),所以原因是确定的、不用"可能"。
_SAFETY_REJECTED = Diagnosis(
    title="动作被安全护栏拦下",
    cause="目标位置超出了设定的安全范围(低于安全高度、越过工作区边界,或关节超出软限位),动作在执行前被拦下,机械臂停在原地。",
    steps=(
        "照错误信息里越界的那一项,到「配置」核对安全高度 / 工作区边界 / 关节软限位。",
        _CHECK_PERCEPTION,
    ),
)
_GRASP_NOT_CONFIRMED = Diagnosis(
    title="夹爪合拢后没夹到东西",
    cause="夹爪已合拢到全闭位置,中间没有物体,动作序列因此中止。",
    steps=(_CHECK_PERCEPTION, "确认物体尺寸在夹爪行程内。"),
)
_FALLBACK = Diagnosis(
    title="运行失败",
    cause="暂时无法自动判断具体原因。",
)

# ------------------------------------------------------------------ 规则表
_DETECTION_SHAPED = ("produced no usable result", "not detected", "no_valid_depth")
_FRAME_TIMEOUT = ("grab_frames error", "frame didn't arrive", "frame did not arrive")


@dataclass(frozen=True)
class _Rule:
    """一条规则:主错误串命中 ``err_needles`` 才算数,日志尾只做佐证。

    Attributes:
        diagnosis: 命中后返回的诊断卡。
        err_needles: 主错误串里任一命中即满足(强信号,必须有)。
        log_needles: 非空时,还要求日志尾里任一命中(佐证,用来在同一种错误形态里做区分)。
        err_excludes: 主错误串里任一命中则本条作废(排除更具体的子系统)。
    """

    diagnosis: Diagnosis
    err_needles: tuple[str, ...]
    log_needles: tuple[str, ...] = ()
    err_excludes: tuple[str, ...] = ()

    def matches(self, err: str, log: str) -> bool:
        if not _has(err, *self.err_needles):
            return False
        if self.err_excludes and _has(err, *self.err_excludes):
            return False
        return not self.log_needles or _has(log, *self.log_needles)


# 顺序即优先级,命中第一条即返回。前半段是自家写出来的结构化 reason,后半段是只能靠
# 文本猜的外部错误(鉴权/网络/显存/接口),needle 越泛排越后。
_RULES: tuple[_Rule, ...] = (
    # 检测器 sidecar 起不来时端口始终不开,唯一的错误形态就是这句超时(见
    # perception/detector_sidecar.py);子进程 stderr 没接进 logging,拿不到更细的证据。
    _Rule(_DETECTOR_TIMEOUT, err_needles=("detector server did not start",)),
    # no_camera / detector_unavailable 必须排在「没识别到目标物体」之前:它们也含
    # "produced no usable result" 子串,否则相机/检测器问题会被误诊成"物体没识别到"。
    _Rule(_NO_CAMERA, err_needles=("no_camera", "no camera")),
    _Rule(_MODEL_NOT_READY, err_needles=("detector_unavailable",)),
    _Rule(_OUT_OF_REACH, err_needles=("out of reach", "exceeds_limit", "out_of_reach")),
    # 相机一停出帧,track_grasp/track_detect 会把"没帧"塌缩成"没检出"(报 not detected)。
    # 失败是检测形态、且日志尾显示取帧超时时,归因到相机而非"物体没摆好"。
    _Rule(_NO_CAMERA, err_needles=_DETECTION_SHAPED, log_needles=_FRAME_TIMEOUT),
    _Rule(_NO_DETECTION, err_needles=_DETECTION_SHAPED),
    _Rule(_PORT_IN_USE, err_needles=("address already in use",)),
    _Rule(
        _LLM_AUTH,
        err_needles=(
            "unauthorized",
            "invalid api key",
            "invalid_api_key",
            "api_key is required",
            "api key is required",
            "authentication",
            " 401",
            " 403",
        ),
    ),
    _Rule(
        _LLM_ENDPOINT,
        err_needles=("getaddrinfo", "name or service not known", "failed to establish", "connection refused"),
        err_excludes=("detector",),
    ),
    _Rule(_GPU_OOM, err_needles=("out of memory", "cuda oom", "cublas_status_alloc_failed")),
    # 总线分开认:so101 走串口、没有 CAN,一张 CAN 卡会把它引去查根本不存在的东西。
    # 泛词 "no such device" 两边都不收——相机的设备错误也长这样,宁可落到兜底卡。
    _Rule(_ARM_CAN, err_needles=("can_left", "socketcan", "can0")),
    _Rule(_ARM_SERIAL, err_needles=("serial", "ttyacm", "ttyusb")),
)


# ------------------------------------------------------------------ code 查表
# 失败点自己写下的机器码(jiuwensymbiosis.errors)→ 诊断卡。命中即返回,不再做文本推断:
# 源头已经确定的事,轮不到这里猜。没有 code 的失败(三方库抛的鉴权/网络/显存)才走规则表。
_CODE_TABLE: dict[str, Diagnosis] = {
    "no_camera": _NO_CAMERA,
    "no_detection": _NO_DETECTION,
    "empty_mask": _NO_DETECTION,
    "no_valid_depth": _NO_DETECTION,
    "detector_unavailable": _MODEL_NOT_READY,
    "detector_start_timeout": _DETECTOR_TIMEOUT,
    "safety_rejected": _SAFETY_REJECTED,
    "grasp_not_confirmed": _GRASP_NOT_CONFIRMED,
    # 适配器自带的码(见 adapters/_common/kinematic_driver.py):servo 下发前的笛卡尔
    # 边界拒绝,与 safety_rejected 是同一件事,同卡。
    "cartesian_bounds_rejected": _SAFETY_REJECTED,
}


def diagnose(error_text: str, log_tail: str = "", *, code: str = "") -> Diagnosis:
    """先按失败点给出的 ``code`` 精确查表;没有 code 才按规则表猜,都不中返回通用卡。"""
    by_code = _CODE_TABLE.get(code)
    if by_code is not None:
        return by_code
    err = (error_text or "").lower()
    log = (log_tail or "").lower()
    for rule in _RULES:
        if rule.matches(err, log):
            return rule.diagnosis
    return _FALLBACK
