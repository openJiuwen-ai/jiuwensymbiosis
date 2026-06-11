# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Piper continuous watch + pick-and-place (jiuwensymbiosis).

A long-running control loop (NOT an LLM agent): the arm returns to the **single
initial / home observe pose** and watches. Each round it observes once and judges
**by vision** whether the task is already done:

  * the pick object is already ON the place target  → task complete → keep waiting;
  * the pick object is somewhere else               → pick it and place it on the target;
  * nothing detected                                → keep waiting.

So after a pick-and-place the object sits on the target and the arm judges the task
complete and waits; when you move the object off the target, the next round sees it
is no longer on the target and runs another pick-and-place. At startup with nothing
in view it just waits at the initial pose until a target appears.

What to pick / where to place comes from the YAML's ``slot_pick`` block
(``chip_object_name`` / ``slot_object_name``) — a deployment task, set once. The
observe pose is the robot's home/initial pose (``home_use_init_pose``), already set
by the operator; no separate observe pose is configured.

Usage::

    # real robot (Ctrl-C to stop):
    */python examples/piper_watch_pick_place.py \\
        --config configs/piper/watch_pick_box.yaml

    # no hardware — mock simulates: place → "complete, waiting" → object removed → pick again:
    #   ... examples/piper_watch_pick_place.py --config configs/piper/watch_pick_box.yaml \\
    #         --mock --max-rounds 8 --poll 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from jiuwensymbiosis.utils.proxy import clear_proxy_env  # noqa: E402

clear_proxy_env()

from jiuwensymbiosis.adapters.piper.slot_pick import load_piper_slot_pick_config
from jiuwensymbiosis.tools.slot_pick import (
    GripperStrategy,
    geometric_completion_judge,
    make_vlm_completion_judge,
    run_watch_pick_place,
)

logger = logging.getLogger("piper_watch")


def _build_vlm_judge(config_path: str, args) -> Optional[Any]:
    """Build a VLM completion judge from the YAML's ``model`` block (+ CLI overrides),
    falling back to the geometric judge when the VLM is unreachable / unclear."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    spec = raw.get("model") or {}
    api_base = args.server_url or spec.get("api_base")
    model_name = args.vlm_model or spec.get("model_name")
    api_key = args.api_key or spec.get("api_key")
    if not model_name:
        logger.warning("--vlm: no model_name (YAML model.model_name or --vlm-model); using geometric judge.")
        return None
    logger.info("VLM completion judge: %s @ %s (geometric fallback)", model_name, api_base)
    return make_vlm_completion_judge(
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        fallback=geometric_completion_judge,
    )


# --------------------------------------------------------------------- mock
class _SceneFakeApi:
    """Fake api that models the scene so the vision-completion behaviour is visible
    without hardware:

      * the place target (white box) is always present;
      * the pick object (black box) starts at the pick location (away from the
        target); after it is placed (gripper released while holding) it sits ON the
        target → detected near the target → judged "already done";
      * after ``remove_after`` "already done" looks, it is moved back to the pick
        location (simulating the operator clearing the place spot) → the next round
        picks-and-places it again.
    """

    def __init__(self, cfg, *, remove_after: int = 2) -> None:
        """初始化场景模拟器。"""
        self._home = {"x": 200.0, "y": 0.0, "z": 350.0, "r": 0.0}
        self._pose = dict(self._home)
        self._slot_name = cfg.slot_object_name
        self._chip_name = cfg.chip_object_name
        self._white = [330.0, 110.0, 45.0]
        self._pick_xyz = [290.0, -110.0, 40.0]
        self._black = "pick"  # "pick" (away) | "placed" (on the target)
        self._holding = False
        # Round-driven state so detection is idempotent within a round (it can be
        # called more than once per round, e.g. by a VLM judge's geometric fallback).
        self._round = 0
        self._placed_round = -1
        self._remove_after = max(1, remove_after)

    def get_home_pose(self) -> dict:
        """返回初始位姿。"""
        return dict(self._home)

    def get_pose(self) -> dict:
        """返回当前位姿。"""
        return dict(self._pose)

    def home(self) -> None:
        """回归初始位姿，并递增轮次计数器。"""
        self._round += 1
        self._pose = dict(self._home)

    def goto_xyzr(self, x: float, y: float, z: float, r=None) -> None:
        """移动到指定坐标 (x, y, z, r)。"""
        self._pose = {"x": x, "y": y, "z": z, "r": r if r is not None else self._pose["r"]}

    def open_gripper(self, width_mm: float = 70.0) -> dict:
        """打开夹爪；若正在夹持则视为放置到目标上。"""
        if self._holding:
            self._holding = False
            self._black = "placed"
            self._placed_round = self._round
        return {"ok": True, "state": "open"}

    def close_gripper(self, force_n=None) -> dict:
        """关闭夹爪，标记为夹持状态。"""
        self._holding = True
        return {"ok": True, "state": "closed"}

    def get_grasp_info_simple(self, object_name: str) -> dict:
        """获取抓取目标的位姿信息（模拟场景中的物体位置变化）。"""
        if object_name == self._slot_name:
            return {"ok": True, "position": list(self._white), "score": 0.9}
        # pick object: it sits on the target after placement, until the operator
        # "removes" it remove_after rounds later (idempotent: depends on the round).
        if self._black == "placed" and (self._round - self._placed_round) > self._remove_after:
            self._black = "pick"
        if self._black == "placed":
            return {"ok": True, "position": [self._white[0], self._white[1], self._white[2] + 80], "score": 0.9}
        return {"ok": True, "position": list(self._pick_xyz), "score": 0.9}


def _make_on_status(cfg):
    """创建状态回调函数，用于在日志中输出每轮观测/抓放结果。"""
    reasons = {
        "no_object": f"未检测到「{cfg.chip_object_name}」",
        "no_place_target": f"未检测到放置目标「{cfg.slot_object_name}」",
        "already_done": f"任务已完成（「{cfg.chip_object_name}」已在「{cfg.slot_object_name}」上）",
    }

    def _cb(ev: dict) -> None:
        """输出单轮观测或抓放的状态日志。"""
        if ev["action"] == "act":
            ok = (ev.get("result") or {}).get("ok")
            logger.info(
                "round %d: 抓放%s（累计 %d 次）；回初始位继续观测",
                ev["round"],
                "完成 ✓" if ok else "失败 ✗",
                ev["placements"],
            )
        else:
            logger.info("round %d: %s → 停在初始位观测…", ev["round"], reasons.get(ev["reason"], ev["reason"]))

    return _cb


def main() -> int:
    """Piper 持续观测抓放演示入口：解析参数，加载配置，启动抓放循环。"""
    p = argparse.ArgumentParser(description="Piper continuous watch pick-and-place.")
    p.add_argument("--config", required=True, help="Path to a configs/piper/*.yaml with a slot_pick: block.")
    p.add_argument("--mock", action="store_true", help="No hardware: simulate place → complete → removed → repeat.")
    p.add_argument("--max-rounds", type=int, default=None, help="Stop after N rounds (mock/testing). Default: forever.")
    p.add_argument("--poll", type=float, default=1.0, help="Seconds to wait between observations when idle.")
    p.add_argument(
        "--mock-remove-after",
        type=int,
        default=2,
        help="(mock) rounds the object stays on the target before it is 'removed'.",
    )
    p.add_argument(
        "--vlm",
        action="store_true",
        help="Judge task completion with a VLM (look at the camera image and answer "
        "'is the object on the target?') instead of the geometric xy check. "
        "Needs a VLM model (YAML model.model_name / --vlm-model); falls back to "
        "geometric when the VLM is unreachable.",
    )
    p.add_argument("--vlm-model", default=None, help="VLM model name (overrides YAML model.model_name).")
    p.add_argument("--server-url", default=None, help="VLM endpoint base URL (overrides YAML model.api_base).")
    p.add_argument("--api-key", default=None, help="VLM API key (overrides YAML model.api_key).")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_piper_slot_pick_config(args.config)
    judge = _build_vlm_judge(args.config, args) if args.vlm else None
    logger.info(
        "watch: 抓「%s」放到「%s」上；回初始位持续观测（完成判定：%s）…",
        cfg.chip_object_name,
        cfg.slot_object_name,
        "VLM 视觉理解" if judge is not None else "几何 xy",
    )

    if args.mock:
        api = _SceneFakeApi(cfg, remove_after=args.mock_remove_after)
        strategy = GripperStrategy(
            api,
            max_reach_radius_mm=cfg.max_reach_radius_mm,
            safe_travel_z_min_mm=cfg.safe_travel_z_min_mm,
        )
        summary = run_watch_pick_place(
            api,
            cfg,
            strategy,
            poll_interval_s=args.poll,
            max_rounds=args.max_rounds,
            on_status=_make_on_status(cfg),
            is_task_complete=judge,
        )
        logger.info("=== watch summary (mock) === %s", summary)
        return 0

    from jiuwensymbiosis.adapters.piper import build_piper_session

    session = build_piper_session.from_yaml(args.config)
    with session:
        strategy = GripperStrategy(
            session.api,
            max_reach_radius_mm=cfg.max_reach_radius_mm,
            safe_travel_z_min_mm=cfg.safe_travel_z_min_mm,
        )
        logger.info("Ctrl-C 停止。")
        summary = run_watch_pick_place(
            session.api,
            cfg,
            strategy,
            poll_interval_s=args.poll,
            max_rounds=args.max_rounds,
            on_status=_make_on_status(cfg),
            is_task_complete=judge,
        )
    logger.info("=== watch summary === %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
