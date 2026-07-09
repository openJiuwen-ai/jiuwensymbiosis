# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Piper vision-driven robot demo (jiuwensymbiosis).

Builds a Piper session (6-DoF AgileX arm over CAN + parallel gripper + wrist
RealSense + open-vocab detection). The *workflow* (which skills, step ordering,
failure handling) comes from the capability-generic SKILL.md files (loaded via
``enable_skill=True`` → SkillUseRail + the ``robot_control`` dispatcher). The
**task is not in the config** — give it at run time via ``--query "..."`` or
``--voice``; the compiler turns it (+ the SKILL.md) into an action sequence.
Pass ``--no-skill`` to revert to the fully prompt-driven hybrid.

Usage::

    PYTHONPATH=/xxx/jiuwensymbiosis/  /xxx/python examples/piper_pick_demo.py
        --config configs/piper/piper.yaml
        --no-visual-feedback --query "<你的任务>"
        --max-iter 30 2>&1 | tail -40

For a dry run without hardware (mock arm + gripper), pass ``--mock``.
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

    这是 demo 支持多机器人的关键：新增一款机器人只需在此注册一条，``--robot <name>``
    即可选中它，其余代码无需改动。当前内置 Piper；其它适配器实现好后加进来即可。
    """
    from jiuwensymbiosis.adapters.piper import build_piper_session

    return {"piper": build_piper_session}


def _build_session(args: argparse.Namespace, raw: dict[str, Any]) -> RobotSession:
    """构建 RobotSession：piper 的 --mock 用 MockArmEnv 干跑，否则按 --robot 从注册表加载。"""
    robot = getattr(args, "robot", "piper")
    if args.mock and robot == "piper":
        from jiuwensymbiosis.api.base import BaseRobotApi
        from jiuwensymbiosis.api.decorators import robot_tool
        from jiuwensymbiosis.api.mixins import (
            MotionMixin,
            ParallelGripperMixin,
            VisionMixin,
        )
        from jiuwensymbiosis.env.mock import MockArmEnv

        class _MockPiperApi(MotionMixin, ParallelGripperMixin, VisionMixin, BaseRobotApi):
            """无硬件环境下的 Piper 机械臂模拟 API。"""

            # Narrow the base ``env`` (BaseRobotEnv) to the mock subtype so the
            # selftest tools can see ``move`` / ``set_suction`` / ``home_pose``.
            env: MockArmEnv

            @robot_tool(desc="home", tags=["motion"])
            def home(self) -> None:
                """回归机械臂初始位姿。"""
                self.env.home()

            @robot_tool
            def get_pose(self) -> dict:
                """获取当前位姿。"""
                return self.env.get_observation().pose or {}

            @robot_tool
            def get_home_pose(self) -> dict:
                """获取初始位姿。"""
                return self.env.home_pose

            @robot_tool(tags=["motion"])
            def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
                """移动到指定坐标 (x, y, z, r)。"""
                self.env.move(x, y, z, r)

            @robot_tool(tags=["grasp"])
            def close_gripper(self, force_n: float | None = None) -> dict:
                """关闭夹爪（模拟吸合）。"""
                self.env.set_suction(True)
                return {"ok": True, "state": "closed"}

            @robot_tool(tags=["grasp"])
            def open_gripper(self, width_mm: float = 70.0) -> dict:
                """打开夹爪（模拟释放）。"""
                self.env.set_suction(False)
                return {"ok": True, "state": "open"}

            @robot_tool
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

            @robot_tool
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
    session，本函数不变。详见 docs/voice-control-integration-design.md。
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
    """Piper 视觉抓取演示入口：解析参数、构建会话和 agent，执行抓取任务并输出结果。"""
    p = argparse.ArgumentParser(description="Piper vision pick-box demo (jiuwensymbiosis).")
    p.add_argument("--config", required=True, help="Path to a configs/piper/*.yaml.")
    p.add_argument("--query", help="User query for the agent. Defaults to the YAML's task prompt.")
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
    p.add_argument("--mock", action="store_true", help="Use MockArmEnv — no hardware required.")
    p.add_argument(
        "--robot",
        default="piper",
        help="选择机器人（见适配器注册表 _robot_session_builders）；默认 piper。加新机器人只需注册一条。",
    )
    p.add_argument(
        "--no-skill",
        action="store_true",
        help="Disable the SkillUseRail + robot_control dispatcher (revert to the "
        "old fully prompt-driven hybrid). Default: skills ON (visual_pick / "
        "visual_place SKILL.md drive the pick-place workflow).",
    )
    p.add_argument(
        "--mode",
        choices=["tool", "code", "hybrid"],
        default="hybrid",
        help="Agent mode: tool-calling, code-as-action, or both.",
    )
    p.add_argument(
        "--no-visual-feedback", action="store_true", help="Disable VisualFeedbackRail (use a non-VLM model)."
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast path (exec_mode=fast): plan once with the LLM (selecting "
        "skills), then run an in-process real-time Perceive+Act servo loop with "
        "NO LLM per step. Default: the per-step LLM agent (slow).",
    )
    # --- fast-path tuning (real-time servo tracking) ---
    p.add_argument(
        "--control-hz",
        type=float,
        default=10.0,
        help="Fast path: servo control-loop rate (Hz). Start low on the Piper (firmware EndPoseCtrl).",
    )
    p.add_argument(
        "--servo-step-mm",
        type=float,
        default=5.0,
        help="Fast path: max linear move per servo tick (mm, slew limit).",
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
    # --- voice mode (语音前端；详见 docs/voice-control-integration-design.md) ---
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

    try:
        session = _build_session(args, raw)
    except ImportError as exc:
        logger.error("failed to import piper adapter (%s).", exc)
        return 2
    spec = _build_model_spec(raw, args)

    logger.info("=== jiuwensymbiosis piper pick-box demo ===")
    logger.info("  config: %s", cfg_path)
    logger.info("  mode  : %s", args.mode)
    logger.info("  model : %s @ %s", spec.model_name, spec.api_base)
    logger.info(
        "  skill : %s", "OFF (prompt-driven hybrid)" if args.no_skill else "ON (visual_pick / visual_place SKILL.md)"
    )
    logger.info("  query : %s", query[:120] + ("..." if len(query) > 120 else ""))
    logger.info("")

    logger.info("  exec  : %s", "FAST (plan-once + real-time servo loop)" if args.fast else "AGENT (per-step LLM)")
    exec_config = None
    if args.fast:
        from jiuwensymbiosis.agent.fast import SkillExecConfig
        from jiuwensymbiosis.agent.fast.realtime import ServoConfig

        exec_config = SkillExecConfig(
            servo=ServoConfig(control_hz=args.control_hz, max_lin_step_mm=args.servo_step_mm),
        )
        logger.info(
            "  loop  : SKILL.md workflow + tracking (no approach/lift offsets; control_hz=%.1f, step=%.1fmm)",
            args.control_hz,
            args.servo_step_mm,
        )
    with session:
        # YAML ``agent:`` block is the declarative base (trace/logging/mode
        # knobs live there); CLI flags override on top. ``model_spec`` is owned
        # by the ``model:`` block (via ``_build_model_spec``) and assigned here.
        agent_cfg = RobotAgentConfig.from_dict(raw.get("agent"))
        agent_cfg.model_spec = spec
        # --mock: drive the agent with an offline model so the YAML placeholder
        # api_key/api_base are never validated against a real client (mirrors
        # MockArmEnv for the LLM side). build_robot_agent then short-circuits
        # on config.model and skips build_model entirely.
        if args.mock:
            from jiuwensymbiosis.agent.mock_model import build_mock_model

            agent_cfg.model = build_mock_model()
        agent_cfg.mode = args.mode
        agent_cfg.enable_visual_feedback = not args.no_visual_feedback
        agent_cfg.enable_skill = not args.no_skill
        agent_cfg.max_iterations = args.max_iter
        if args.debug:
            agent_cfg.log_level = "DEBUG"
        if args.workspace:
            agent_cfg.workspace = args.workspace
        agent_cfg.exec_mode = "fast" if args.fast else "agent"
        agent_cfg.exec_config = exec_config
        conv_id = f"piper-demo-{uuid.uuid4().hex[:8]}"
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
