# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SO-101 eye-to-hand visual-pick demo.

The fast mode compiles ``visual_pick`` once and, when the connected SO-101
session advertises ``vision.eye_to_hand`` + ``motion.servo``, selects the
runner-owned ``track_grasp`` operation (absolute approach + descend).  The
agent mode retains the normal per-step LLM workflow.

Usage::

    python examples/so101_pick_demo.py \
        --config configs/so101/so101_left.yaml \
        --query "把香蕉抓起来" --fast --api-key <KEY>
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

from jiuwensymbiosis.utils.proxy import clear_proxy_env  # noqa: E402 - before package imports

clear_proxy_env()

from jiuwensymbiosis import run_robot_task  # noqa: E402 - after proxy hygiene
from jiuwensymbiosis.adapters.so101 import build_so101_session  # noqa: E402
from jiuwensymbiosis.agent import ModelSpec, RobotAgentConfig  # noqa: E402

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _build_model_spec(raw: dict[str, Any], args: argparse.Namespace) -> ModelSpec:
    spec_data = raw.get("model") or {}
    spec = ModelSpec(**spec_data) if spec_data else ModelSpec()
    if args.server_url:
        spec.api_base = args.server_url.rstrip("/").removesuffix("/chat/completions")
    if args.model:
        spec.model_name = args.model
    if args.api_key:
        spec.api_key = args.api_key
    elif os.environ.get("OPENJIUWEN_API_KEY"):
        spec.api_key = os.environ["OPENJIUWEN_API_KEY"]
    return spec


def _fast_preflight(session: Any) -> list[str]:
    """Return missing prerequisites for the SO-101 absolute visual loop."""
    caps = set(getattr(session.env, "capabilities", frozenset()))
    required = {"motion.servo", "vision.eye_to_hand", "vision.detection", "grasp.parallel"}
    missing = sorted(required - caps)
    required_bindings = {
        "api.get_pose": getattr(session.api, "get_pose", None),
        "api.get_grasp_info_simple": getattr(session.api, "get_grasp_info_simple", None),
    }
    missing.extend(name for name, fn in required_bindings.items() if not callable(fn))
    servo_to_tip = getattr(session.api, "servo_to_tip", None)
    servo_to_flange = getattr(session.env, "servo_to_flange", None)
    if not callable(servo_to_tip) and not callable(servo_to_flange):
        missing.append("api.servo_to_tip or env.servo_to_flange")
    driver = getattr(session.env, "low_level", None)
    if driver is not None and not bool(getattr(driver, "has_calibration", False)):
        missing.append("loaded eye-to-hand calibration (T_base_cam)")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-101 eye-to-hand visual pick demo.")
    parser.add_argument("--config", required=True, help="Path to configs/so101/*.yaml")
    parser.add_argument("--query", required=True, help="Natural-language task, e.g. 把香蕉抓起来")
    parser.add_argument("--server-url", default=None, help="Override LLM base URL")
    parser.add_argument("--model", default=None, help="Override LLM model name")
    parser.add_argument("--api-key", default=None, help="Override LLM API key")
    parser.add_argument("--fast", action="store_true", help="Compile once and run track_grasp fast path")
    parser.add_argument("--no-skill", action="store_true", help="Disable skill mode (incompatible with --fast)")
    parser.add_argument("--mode", choices=["tool", "code", "hybrid"], default="hybrid")
    parser.add_argument("--no-visual-feedback", action="store_true")
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--servo-step-mm", type=float, default=5.0)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.fast and args.no_skill:
        parser.error("--no-skill is incompatible with --fast: fast compilation uses SKILL.md")
    if not (1.0 <= args.control_hz <= 100.0):
        parser.error(f"--control-hz must be in [1, 100], got {args.control_hz}")
    if not (0.5 <= args.servo_step_mm <= 50.0):
        parser.error(f"--servo-step-mm must be in [0.5, 50], got {args.servo_step_mm}")
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config_path = Path(args.config).expanduser().resolve()
    raw = _load_yaml(config_path)
    spec = _build_model_spec(raw, args)
    session = cast(Any, build_so101_session.from_yaml(config_path))

    exec_config = None
    if args.fast:
        from jiuwensymbiosis.agent.fast import SkillExecConfig
        from jiuwensymbiosis.agent.fast.realtime import ServoConfig

        exec_config = SkillExecConfig(
            servo=ServoConfig(control_hz=args.control_hz, max_lin_step_mm=args.servo_step_mm),
        )

    logger.info("=== SO-101 eye-to-hand visual pick ===")
    logger.info("config=%s model=%s @ %s", config_path, spec.model_name, spec.api_base)
    logger.info("mode=%s exec=%s query=%s", args.mode, "fast" if args.fast else "agent", args.query)

    with session:
        if args.fast:
            missing = _fast_preflight(session)
            if missing:
                logger.error("fast visual loop prerequisites missing: %s", ", ".join(missing))
                return 2
        agent_cfg = RobotAgentConfig.from_dict(raw.get("agent"))
        agent_cfg.model_spec = spec
        agent_cfg.mode = args.mode
        agent_cfg.enable_skill = not args.no_skill
        agent_cfg.enable_visual_feedback = not args.no_visual_feedback
        agent_cfg.max_iterations = args.max_iter
        agent_cfg.exec_mode = "fast" if args.fast else "agent"
        agent_cfg.exec_config = exec_config
        if args.debug:
            agent_cfg.log_level = "DEBUG"
        if args.workspace:
            agent_cfg.workspace = args.workspace
        result = run_robot_task(
            session,
            args.query,
            agent_cfg,
            conversation_id=f"so101-demo-{uuid.uuid4().hex[:8]}",
        )

    if isinstance(result, dict):
        logger.info("=== Result ===\n%s", json.dumps(result, ensure_ascii=False, indent=2, default=repr))
    else:
        logger.info("=== Result ===\n%s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
