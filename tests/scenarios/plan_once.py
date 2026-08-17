# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Plan one task for one body and print the sequence — the production path, no hardware.

Same chain ``agent/run.py:run_fast_task`` runs (``parse_task`` → ``plan_task`` →
``parse_sequence``), with the pre-plan perception supplied as JSON instead of read
off a live detector. Use it to check what a body would do about a task without
owning that body.

Usage::

    python tests/scenarios/plan_once.py --config configs/so101/so101.yaml \\
        --query "抓起桌面上的香蕉" --api-key sk-...
    python tests/scenarios/plan_once.py --config configs/cruzr/cruzr.yaml \\
        --query "把所有箱子搬到紫色桌子上" --scene scene.json --repeat 3 --api-key sk-...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jiuwensymbiosis.utils.proxy import clear_proxy_env  # noqa: E402 - before the package imports below

clear_proxy_env()

from jiuwensymbiosis import introspect  # noqa: E402 - after clear_proxy_env() (proxy hygiene)
from jiuwensymbiosis.agent.fast.planner import parse_task, plan_task  # noqa: E402
from jiuwensymbiosis.agent.fast.registry import SkillRegistry  # noqa: E402
from jiuwensymbiosis.agent.fast.sequence import parse_sequence  # noqa: E402
from jiuwensymbiosis.agent.run import _action_param_sig  # noqa: E402
from jiuwensymbiosis.api.world_state import WorldState  # noqa: E402
from jiuwensymbiosis.skills import SKILLS_DIR  # noqa: E402
from jiuwensymbiosis.tools.robot_control_tool import _build_action_index  # noqa: E402


def _load_scene(spec: str | None) -> Any:
    """Scene from a JSON file path or an inline JSON string; ``None`` = plan scene-blind."""
    if not spec:
        return None
    p = Path(spec)
    return json.loads(p.read_text(encoding="utf-8") if p.is_file() else spec)


def _extra_ops(module_path: str | None) -> dict[str, Any]:
    """Load extra planner-visible actions from ``path.py:ClassName`` (test-local bodies)."""
    if not module_path:
        return {}
    path, _, cls_name = module_path.partition(":")
    import importlib.util

    spec = importlib.util.spec_from_file_location("_extra_ops", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_extra_ops"] = mod
    spec.loader.exec_module(mod)
    return _build_action_index(getattr(mod, cls_name)(), planner_only=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Plan one task for one body (real LLM, no hardware).")
    p.add_argument("--config", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--scene", default=None, help="JSON file or inline JSON: the pre-plan perception.")
    p.add_argument("--skills-dir", default=None, help="Extra SKILL.md directory to register.")
    p.add_argument("--extra-ops", default=None, help="path/to/mod.py:ClassName supplying extra actions.")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--model", default=None)
    p.add_argument("--api-base", default=None, help="Override the YAML endpoint (some configs ship a placeholder).")
    args = p.parse_args()

    raw = introspect.load_config(args.config)
    session = introspect.build_session(args.config)
    model = raw.get("model") or {}
    api_base = args.api_base or model.get("api_base", "")
    model_name = args.model or model.get("model_name", "")
    temperature = float(model.get("temperature", 0.0))

    index = _build_action_index(session.api, planner_only=True)
    index.update(_extra_ops(args.extra_ops))
    reg = SkillRegistry()
    reg.register_dir(SKILLS_DIR)
    if args.skills_dir:
        reg.register_dir(Path(args.skills_dir))
    world = WorldState.snapshot(session)
    scene = _load_scene(args.scene)

    print(f"=== {raw.get('name', args.config)} · {args.query}")
    print(f"  能力    : {', '.join(sorted(session.api.capabilities))}")
    print(f"  词表    : {', '.join(sorted(index))}")
    print(f"  技能库  : {', '.join(s['name'] for s in reg.skills_markdown())}")
    print(f"  起始状态: {sorted(world.tokens)}")

    ok_all = True
    for run_no in range(1, args.repeat + 1):
        intent = parse_task(args.query, api_base=api_base, api_key=args.api_key,
                            model_name=model_name, temperature=temperature)
        print(f"\n  --- 第 {run_no} 次 ---")
        print(f"  解析意图: targets={intent['targets']} references={intent['references']} "
              f"grounding={intent['grounding']} destination={intent['destination']!r} "
              f"mode={intent['mode']} count={intent['count']} intent={intent['intent']}")
        try:
            planned = plan_task(
                args.query,
                skills_md=reg.skills_markdown(),
                action_index=index,
                allowed_ops=index,
                api_base=api_base,
                api_key=args.api_key,
                model_name=model_name,
                temperature=temperature,
                api_capabilities=sorted(session.api.capabilities),
                action_sigs={n: _action_param_sig(f) for n, f in index.items()},
                scene=scene,
                grounding=intent.get("grounding") or {},
                world_block=world.as_prompt_block(),
                world_tokens=sorted(world.tokens) or None,
            )
        except RuntimeError as exc:
            print(f"  ❌ 规划失败: {exc}")
            ok_all = False
            continue
        print(f"  规划层级: tier={planned.tier} skills={list(planned.skills or ())}"
              + (f" reason={planned.reason}" if planned.reason else ""))
        for i, step in enumerate(planned.sequence):
            bind = f"  bind={step['bind']}" if step.get("bind") else ""
            body = json.dumps(step.get("params") or step.get("body") or {}, ensure_ascii=False)
            print(f"    {i:>2}. {step.get('op')}{bind}  {body}")
        try:
            parse_sequence(planned.sequence, allowed_ops=index, initial_state=sorted(world.tokens) or None,
                           grounding=intent.get("grounding") or {})
            print("  ✅ 契约校验通过")
        except Exception as exc:  # noqa: BLE001 - a rejected plan is a result, not a crash
            print(f"  ❌ 契约校验拒绝: {exc}")
            ok_all = False
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
