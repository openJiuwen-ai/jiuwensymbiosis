# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Generic task runner (jiuwensymbiosis) — one entry for every robot + task.

Pick the robot with ``--config`` (the YAML's ``adapter:`` field selects the
adapter from the registry; ``--robot`` overrides). Give the task at run time
with ``--query "..."`` or ``--voice`` — the **task is not in the config**.
Nothing here is task- or robot-specific, so a new task is a new ``--query`` and
a new robot is a new ``--config`` — no new Python file.

Default execution is ``fastagent`` (declared in the shipped configs): one LLM
call compiles the task (+ the capability-generic SKILL.md files) into an action
sequence, then it runs with no per-step LLM. Pass ``--stepagent`` for the
per-step LLM path (single-step debugging / verification); ``--mock`` implies it.

Usage::

    jiuwensymbiosis-run --config configs/cruzr/cruzr.yaml --query "把箱子搬到桌上"
    jiuwensymbiosis-run --config configs/piper/piper.yaml   --query "把瓶子放到左边"

For a dry run without hardware or a real LLM (piper only), pass ``--mock``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, cast

import yaml

from jiuwensymbiosis.utils.proxy import clear_proxy_env  # noqa: E402 - call it before the package imports below

clear_proxy_env()

from jiuwensymbiosis import RobotSession, run_robot_task  # noqa: E402 - after clear_proxy_env() (proxy hygiene)
from jiuwensymbiosis.agent import ModelSpec, RobotAgentConfig  # noqa: E402 - after clear_proxy_env() (proxy hygiene)

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    """加载 YAML 配置文件并返回字典。"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _robot_session_builders() -> dict[str, Any]:
    """适配器注册表：机器人名 → 会话构建器（暴露 ``.from_yaml(path)``）。

    支持多机器人的关键：新增一款机器人只需在此注册一条。机器人由 config 的 ``adapter:``
    字段选中（``--robot`` 覆盖），其余代码无需改动。
    """
    from jiuwensymbiosis.adapters.cruzr import build_cruzr_session
    from jiuwensymbiosis.adapters.piper import build_piper_session
    from jiuwensymbiosis.adapters.so101 import build_so101_session

    return {"piper": build_piper_session, "cruzr": build_cruzr_session, "so101": build_so101_session}


def _resolve_robot(args: argparse.Namespace, raw: dict[str, Any]) -> str:
    """机器人身份：``--robot`` > config 顶层 ``adapter:`` > piper（向后兼容）。"""
    return args.robot or raw.get("adapter") or "piper"


def _build_session(args: argparse.Namespace, raw: dict[str, Any]) -> RobotSession:
    """构建 RobotSession：piper 的 --mock 用 MockArmEnv 干跑，否则按机器人从注册表加载。"""
    robot = _resolve_robot(args, raw)
    if args.mock and robot == "piper":
        from jiuwensymbiosis.api.actions import (
            CLOSE_GRIPPER,
            GET_GRASP_INFO_SIMPLE,
            GET_HOME_POSE,
            GET_POSE,
            GOTO_XYZR,
            OPEN_GRIPPER,
            PIXEL_TO_BASE_XYZ,
            implements,
        )
        from jiuwensymbiosis.api.base import BaseRobotApi
        from jiuwensymbiosis.env.mock import MockArmEnv

        class _MockPiperApi(BaseRobotApi):
            """无硬件环境下的 Piper 机械臂模拟 API。

            每个方法都绑定共享动作词表里的一条契约，和真实适配器同形——mock 换掉的
            只是「怎么做」，不是「做什么」。``home`` 由 BaseRobotApi 提供。
            """

            # Narrow the base ``env`` (BaseRobotEnv) to the mock subtype so the
            # selftest tools can see ``move`` / ``set_suction`` / ``home_pose``.
            env: MockArmEnv

            @implements(GET_POSE)
            def get_pose(self) -> dict:
                """获取当前位姿。"""
                return self.env.get_observation().pose or {}

            @implements(GET_HOME_POSE)
            def get_home_pose(self) -> dict:
                """获取初始位姿。"""
                return self.env.home_pose

            @implements(GOTO_XYZR)
            def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
                """移动到指定坐标 (x, y, z, r)。"""
                self.env.move(x, y, z, r)

            @implements(CLOSE_GRIPPER)
            def close_gripper(self, force_n: float | None = None) -> dict:
                """关闭夹爪（模拟吸合）。"""
                self.env.set_suction(True)
                return {"ok": True, "state": "closed"}

            @implements(OPEN_GRIPPER)
            def open_gripper(self, width_mm: float = 70.0) -> dict:
                """打开夹爪（模拟释放）。"""
                self.env.set_suction(False)
                return {"ok": True, "state": "open"}

            @implements(GET_GRASP_INFO_SIMPLE)
            def get_grasp_info_simple(self, object_name: str) -> dict:
                """获取抓取目标的位姿信息（返回模拟值）。"""
                hp = self.env.home_pose
                return {
                    "ok": True,
                    "position": [hp["x"] + 30, hp["y"], hp["z"] - 200],
                    "score": 0.9,
                    "pixel_uv": [320, 240],
                    "depth_m": 0.20,
                }

            @implements(PIXEL_TO_BASE_XYZ)
            def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
                """将像素坐标 + 深度转换为基坐标系下的三维坐标（返回模拟值）。"""
                hp = self.env.home_pose
                return {"x": hp["x"] + 30, "y": hp["y"], "z": hp["z"] - 200}

        env = MockArmEnv()
        api = _MockPiperApi(env)
        return RobotSession(env=env, api=api, name="piper_mock")

    builders = _robot_session_builders()
    build = builders.get(robot)
    if build is None:
        raise ValueError(f"unknown robot {robot!r}; registered: {sorted(builders)}")
    return cast(RobotSession, build.from_yaml(args.config))


def _build_model_spec(raw: dict[str, Any], args: argparse.Namespace) -> ModelSpec:
    """从 YAML 配置和 CLI 参数构建 ModelSpec，CLI 参数优先级高于配置文件。"""
    spec_data = raw.get("model") or {}
    spec = ModelSpec(**spec_data) if spec_data else ModelSpec()
    if args.server_url:
        spec.api_base = args.server_url.rstrip("/").removesuffix("/chat/completions")
    if args.model:
        spec.model_name = args.model
    # Key priority: --api-key > $OPENJIUWEN_API_KEY > YAML model.api_key.
    if args.api_key:
        spec.api_key = args.api_key
    elif os.environ.get("OPENJIUWEN_API_KEY"):
        spec.api_key = os.environ["OPENJIUWEN_API_KEY"]
    return spec


def _resolve_query(raw: dict[str, Any], args: argparse.Namespace) -> str:
    """解析用户任务：优先 --query，其次 YAML 里可选的 prompt（一般不存在）；都没有返回空串。

    config 不再内置任务——任务由 --query 或 --voice 在运行时给出。返回空串时由调用方
    （main）要求用户提供，不再默默跑一个默认任务。
    """
    if args.query:
        return cast(str, args.query)
    body = (raw.get("env", {}).get("cfg", {}) or {}).get("prompt")
    if body:
        return cast(str, body)
    return ""


def _voice_enabled(args: argparse.Namespace) -> bool:
    """是否进入语音模式（显式 --voice，或给了一次性文本/音频输入）。"""
    return bool(args.voice or args.voice_text or args.voice_audio_file)


def _run_voice(
    session: RobotSession,
    agent_cfg: RobotAgentConfig,
    conv_id: str,
    raw: dict[str, Any],
    args: argparse.Namespace,
) -> dict:
    """语音模式：把 VoiceLoop 的 on_command 回调接到 run_robot_task。

    语音层是机器人无关的；这里是它与框架的唯一接缝（文本进、反馈出）。换 N2 时只换
    session，本函数不变。详见 design/voice-control-integration.md。
    """
    import numpy as np

    from jiuwensymbiosis.voice import (
        FileAudioSource,
        FixedASRBackend,
        VoiceConfig,
        VoiceLoop,
        result_to_speech,
    )

    voice_cfg = VoiceConfig.from_dict(raw.get("voice"))
    if args.tts:
        voice_cfg.tts_backend = args.tts
    if args.asr_device:
        voice_cfg.asr_device = args.asr_device
    if args.no_wake:
        voice_cfg.wake_enabled = False
    logger.info(
        "  voice : wake=%s(%s) asr=%s@%s tts=%s",
        voice_cfg.wake_word,
        "on" if voice_cfg.wake_enabled else "off",
        voice_cfg.asr_backend,
        voice_cfg.asr_device,
        voice_cfg.tts_backend,
    )

    def on_command(text: str) -> str:
        logger.info("[voice] 指令 → agent: %s", text)
        reply = result_to_speech(run_robot_task(session, text, agent_cfg, conversation_id=conv_id))
        logger.info("[voice] agent → 反馈: %s", reply)
        return reply

    # 一次性文本/音频用 mock/file 注入；否则用配置里的实时麦克风后端。
    asr = audio = None
    one_shot = False
    if args.voice_text is not None:
        asr = FixedASRBackend([args.voice_text])
        audio = FileAudioSource([np.ones(480, dtype=np.int16)])  # 占位音频；FixedASR 忽略内容
        one_shot = True
    elif args.voice_audio_file is not None:
        audio = FileAudioSource([args.voice_audio_file])  # 真实 ASR 走配置后端
        one_shot = True

    loop = VoiceLoop(voice_cfg, on_command, asr=asr, audio=audio)
    if one_shot or args.voice_once:
        cmd = loop.run_once()
        if cmd:
            loop.handle_command(cmd)
        else:
            logger.info("[voice] 未得到有效指令（--voice-text 需含唤醒词，或加 --no-wake）")
        loop.wait()
        return {"ok": True, "mode": "voice", "one_shot": True}
    loop.run_forever()
    return {"ok": True, "mode": "voice"}


def main() -> int:
    """通用任务入口：按 config 的 adapter 建会话，用 --query 给任务，执行并输出结果。"""
    p = argparse.ArgumentParser(description="Generic task runner (jiuwensymbiosis).")
    p.add_argument("--config", required=True, help="Path to a robot config YAML (its adapter: field picks the robot).")
    p.add_argument("--query", help="User task, e.g. --query \"把箱子搬到桌上\". The task is not in the config.")
    p.add_argument(
        "--server-url",
        default=None,
        help="Override the LLM endpoint base URL from the YAML (without /chat/completions).",
    )
    p.add_argument("--model", default=None, help="Override the model name from the YAML.")
    p.add_argument(
        "--api-key",
        default=None,
        help=("Override the LLM API key (overrides YAML model.api_key)."),
    )
    p.add_argument("--mock", action="store_true",
                   help="Piper-only dry run: MockArmEnv + offline model. Implies --stepagent (no real LLM).")
    p.add_argument(
        "--robot",
        default=None,
        help="Override the config's adapter: field (registry key: piper / so101 / cruzr).",
    )
    p.add_argument(
        "--stepagent",
        action="store_true",
        help="Force exec_mode=stepagent (per-step LLM) for single-step debugging / verification. "
        "Default: the config's exec_mode (fastagent — compile once, no per-step LLM).",
    )
    p.add_argument(
        "--no-skill",
        action="store_true",
        help="Override config: disable the SkillUseRail + robot_control dispatcher.",
    )
    p.add_argument(
        "--mode",
        choices=["tool", "code", "hybrid"],
        default="hybrid",
        help="Agent mode: tool-calling, code-as-action, or both.",
    )
    p.add_argument(
        "--no-visual-feedback", action="store_true", help="Override config: disable VisualFeedbackRail."
    )
    # --- fastagent tuning (real-time servo tracking at track_detect steps) ---
    p.add_argument(
        "--control-hz",
        type=float,
        default=10.0,
        help="fastagent: servo control-loop rate (Hz). Start low on the Piper (firmware EndPoseCtrl).",
    )
    p.add_argument(
        "--servo-step-mm",
        type=float,
        default=5.0,
        help="fastagent: max linear move per servo tick (mm, slew limit).",
    )
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument(
        "--workspace",
        default=None,
        help=(
            "Agent workspace directory (default: ~/.openjiuwen/{session_name}_workspace/). "
            "Matches openjiuwen CLI's --workspace; resolution priority is "
            "--workspace > $OPENJIUWEN_WORKSPACE > default."
        ),
    )
    p.add_argument("--debug", action="store_true")
    # --- voice mode (语音前端；详见 design/voice-control-integration.md) ---
    p.add_argument(
        "--voice",
        action="store_true",
        help="语音模式：麦克风→唤醒词「九问九问」→ASR→agent→TTS，持续监听(Ctrl-C 退出)。"
        '读 --config 里可选的 voice: 块，缺省用默认值。需 pip install -e ".[voice]"。',
    )
    p.add_argument(
        "--voice-text",
        default=None,
        help="语音模式一次性：直接注入这段转写文本(不需麦克风/funasr)，走完整唤醒→派发流程。",
    )
    p.add_argument(
        "--voice-audio-file",
        default=None,
        help="语音模式一次性：对该 WAV 走真实 ASR(需 .[voice] 依赖)，跑一次。",
    )
    p.add_argument("--voice-once", action="store_true", help="语音模式只监听一次麦克风指令后退出。")
    p.add_argument("--no-wake", action="store_true", help="语音模式关闭唤醒词，整句当指令。")
    p.add_argument("--tts", choices=["null", "chattts"], default=None, help="语音模式覆盖 TTS 后端。")
    p.add_argument("--asr-device", default=None, help="语音模式覆盖 ASR 设备(cuda:0/cpu)。")
    args = p.parse_args()

    # Configure logging up front so the voice-listening phase (before the first
    # agent build, which is what otherwise sets logging up) is visible. Without
    # this, only WARNING+ shows and ``--voice`` looks "stuck" while it is in fact
    # listening / loading the ASR model — all of which log at INFO.
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg_path = Path(args.config).resolve()
    raw = _load_yaml(cfg_path)

    # 任务不在 config 里：非语音模式必须由 --query 给出；语音模式由说话给出。
    query = _resolve_query(raw, args)
    if not _voice_enabled(args) and not query.strip():
        logger.error(
            '未提供任务：config 不内置任务。请用 --query "..." 给出要执行的任务，'
            "或用 --voice / --voice-text / --voice-audio-file 由语音给出。"
        )
        return 2

    robot = _resolve_robot(args, raw)
    try:
        session = _build_session(args, raw)
    except ImportError as exc:
        logger.error("failed to import the %r adapter (%s).", robot, exc)
        return 2
    spec = _build_model_spec(raw, args)

    # exec_mode: the config declares it (fastagent by default); --stepagent or
    # --mock (offline, no real LLM to compile) force the per-step stepagent path.
    exec_mode = "stepagent" if (args.mock or args.stepagent) else (raw.get("agent") or {}).get("exec_mode", "fastagent")

    logger.info("=== jiuwensymbiosis task runner ===")
    logger.info("  robot : %s", robot)
    logger.info("  config: %s", cfg_path)
    logger.info(
        "  exec  : %s",
        "fastagent (compile-once, no per-step LLM)" if exec_mode == "fastagent" else "stepagent (per-step LLM)",
    )
    logger.info("  model : %s @ %s", spec.model_name, spec.api_base)
    logger.info("  query : %s", query[:120] + ("..." if len(query) > 120 else ""))
    logger.info("")

    exec_config = None
    if exec_mode == "fastagent":
        from jiuwensymbiosis.agent.fast import SkillExecConfig
        from jiuwensymbiosis.agent.fast.realtime import ServoConfig

        exec_config = SkillExecConfig(
            servo=ServoConfig(control_hz=args.control_hz, max_lin_step_mm=args.servo_step_mm),
        )
    with session:
        # YAML ``agent:`` block is the declarative base (exec_mode / rails / trace /
        # logging live there); CLI flags override only when explicitly given.
        agent_cfg = RobotAgentConfig.from_dict(raw.get("agent"))
        agent_cfg.model_spec = spec
        # --mock: offline model so the YAML placeholder api_key/api_base is never
        # validated against a real client (mirrors MockArmEnv for the LLM side).
        if args.mock:
            from jiuwensymbiosis.agent.mock_model import build_mock_model

            agent_cfg.model = build_mock_model()
        agent_cfg.mode = args.mode
        agent_cfg.max_iterations = args.max_iter
        agent_cfg.exec_mode = exec_mode
        agent_cfg.exec_config = exec_config
        if args.no_visual_feedback:
            agent_cfg.enable_visual_feedback = False
        if args.no_skill:
            agent_cfg.enable_skill = False
        if args.debug:
            agent_cfg.log_level = "DEBUG"
        if args.workspace:
            agent_cfg.workspace = args.workspace
        conv_id = f"task-{uuid.uuid4().hex[:8]}"
        if _voice_enabled(args):
            result = _run_voice(session, agent_cfg, conv_id, raw, args)
        else:
            result = run_robot_task(session, query, agent_cfg, conversation_id=conv_id)

    logger.info("=== Agent result ===")
    if isinstance(result, dict):
        logger.info(json.dumps(result, ensure_ascii=False, indent=2, default=repr))
    else:
        logger.info(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
