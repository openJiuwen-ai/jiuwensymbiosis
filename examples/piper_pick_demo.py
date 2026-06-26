# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Piper vision-driven pick-box demo (jiuwensymbiosis).

Builds a Piper session (6-DoF AgileX arm over CAN + parallel gripper + wrist
RealSense + open-vocab detection). By default the pick-place *workflow* (step ordering, failure
handling) comes from the capability-generic ``visual_pick`` / ``visual_place``
SKILL.md (loaded via ``enable_skill=True`` → SkillUseRail + the ``robot_control``
dispatcher); the YAML ``prompt`` only states the high-level task + task-specific
constants. Pass ``--no-skill`` to revert to the old fully prompt-driven hybrid
(the agent orchestrates home / get_grasp_info_simple / goto_xyzr / open_gripper /
close_gripper directly from a step-by-step prompt).

Usage::

    PYTHONPATH=/xxx/jiuwensymbiosis/  /xxx/python examples/piper_pick_demo.py
        --config configs/piper/pick_box.yaml
        --no-visual-feedback --query "把高盒子放到矮盒子上"
        --max-iter 30 2>&1 | tail -40

For a dry run without hardware (mock arm + gripper), pass ``--mock``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml

from jiuwensymbiosis.utils.proxy import clear_proxy_env  # noqa: E402

clear_proxy_env()

from jiuwensymbiosis import RobotSession, build_robot_agent
from jiuwensymbiosis.agent import ModelSpec, RobotAgentConfig

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    """加载 YAML 配置文件并返回字典。"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_session(args: argparse.Namespace, raw: dict[str, Any]) -> RobotSession:
    """构建 RobotSession：mock 模式下创建模拟会话，否则从 YAML 配置加载 Piper 适配器。"""
    if args.mock:
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
                return self.env.home_pose()

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
                hp = self.env.home_pose()
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
                hp = self.env.home_pose()
                return {"x": hp["x"] + 30, "y": hp["y"], "z": hp["z"] - 200}

        env = MockArmEnv()
        api = _MockPiperApi(env)
        return RobotSession(env=env, api=api, name="piper_mock")

    from jiuwensymbiosis.adapters.piper import build_piper_session
    return build_piper_session.from_yaml(args.config)


def _build_model_spec(raw: dict[str, Any], args: argparse.Namespace) -> ModelSpec:
    """从 YAML 配置和 CLI 参数构建 ModelSpec，CLI 参数优先级高于配置文件。"""
    spec_data = raw.get("model") or {}
    spec = ModelSpec(**spec_data) if spec_data else ModelSpec()
    if args.server_url:
        spec.api_base = args.server_url.rstrip("/").removesuffix("/chat/completions")
    if args.model:
        spec.model_name = args.model
    if args.api_key:
        spec.api_key = args.api_key
    return spec


def _resolve_query(raw: dict[str, Any], args: argparse.Namespace) -> str:
    """解析用户查询：优先取 --query 参数，其次 YAML 中的 prompt 字段，最后返回默认值。"""
    if args.query:
        return args.query
    body = (raw.get("env", {}).get("cfg", {}) or {}).get("prompt")
    if body:
        return body
    return "Run the configured pick-box task to completion."


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
    args = p.parse_args()


    cfg_path = Path(args.config).resolve()
    raw = _load_yaml(cfg_path)

    try:
        session = _build_session(args, raw)
    except ImportError as exc:
        logger.error("failed to import piper adapter (%s).", exc)
        return 2
    spec = _build_model_spec(raw, args)
    query = _resolve_query(raw, args)

    logger.info("=== jiuwensymbiosis piper pick-box demo ===")
    logger.info("  config: %s", cfg_path)
    logger.info("  mode  : %s", args.mode)
    logger.info("  model : %s @ %s", spec.model_name, spec.api_base)
    logger.info(
        "  skill : %s", "OFF (prompt-driven hybrid)" if args.no_skill else "ON (visual_pick / visual_place SKILL.md)"
    )
    logger.info("  query : %s", query[:120] + ("..." if len(query) > 120 else ""))
    logger.info("")

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
        agent = build_robot_agent(session, config=agent_cfg)
        conv_id = f"piper-demo-{uuid.uuid4().hex[:8]}"
        result = asyncio.run(agent.invoke({"query": query, "conversation_id": conv_id}))

    logger.info("=== Agent result ===")
    if isinstance(result, dict):
        logger.info(json.dumps(result, ensure_ascii=False, indent=2, default=repr))
    else:
        logger.info(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
