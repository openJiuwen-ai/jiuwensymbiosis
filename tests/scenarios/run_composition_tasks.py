# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Composition probes — tasks no single SKILL.md covers end to end.

Each built-in skill describes one closed job (pick, place, move). These tasks
need the planner to *combine* them: iterate over an unknown number of targets,
chain two pick-places with different destinations, obey an order the task states
rather than the one the template suggests, or derive something with no skill at
all. What is under test is the composition, not any one skill.

Runs the production chain unchanged (``parse_task`` → ``plan_task`` →
``parse_sequence``) against the real cruzr action index, with the pre-plan
perception supplied as a fixture. ``pull`` / ``push`` come from
``run_cruzr_scenarios._DoorOps`` — test-local, never shipped on a body.

Usage::

    python tests/scenarios/run_composition_tasks.py --api-key sk-... [--repeat 2] [--only C1 C3]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jiuwensymbiosis.utils.proxy import clear_proxy_env  # noqa: E402 - before the package imports below

clear_proxy_env()

from jiuwensymbiosis import introspect  # noqa: E402 - after clear_proxy_env() (proxy hygiene)
from jiuwensymbiosis.agent.fast.planner import parse_task, plan_task  # noqa: E402
from jiuwensymbiosis.agent.fast.registry import SkillRegistry  # noqa: E402
from jiuwensymbiosis.agent.fast.sequence import parse_sequence  # noqa: E402
from jiuwensymbiosis.agent.run import _action_param_sig, _blocked_access  # noqa: E402
from jiuwensymbiosis.api.world_state import WorldState  # noqa: E402
from jiuwensymbiosis.skills import SKILLS_DIR  # noqa: E402
from jiuwensymbiosis.tools.robot_control_tool import _build_action_index  # noqa: E402

_HERE = Path(__file__).resolve().parent


def _door_ops_module() -> Any:
    spec = importlib.util.spec_from_file_location("_cruzr_scen", _HERE / "run_cruzr_scenarios.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cruzr_scen"] = mod
    spec.loader.exec_module(mod)
    return mod


def _obj(name: str, forward_mm: float, *, y_mm: float = 60.0, z_mm: float = 700.0,
         width_mm: float = 300.0, height_mm: float = 250.0, reachable: bool = False) -> dict:
    return {
        "object": name,
        "center_mm": [forward_mm, y_mm, z_mm],
        "distance_mm": (forward_mm**2 + y_mm**2) ** 0.5,
        "forward_mm": forward_mm,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "front_x_mm": forward_mm - width_mm / 2.0,
        "back_x_mm": forward_mm + width_mm / 2.0,
        "top_z_mm": z_mm + height_mm / 2.0,
        "score": 0.9,
        "reachable": reachable,
    }


def _kind(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("table", "desk", "shelf", "counter")):
        return "surface"
    if any(k in n for k in ("cabinet", "cupboard", "drawer", "locker")):
        return "cabinet"
    return "small"


def _reference_obj(name: str, i: int) -> dict:
    kind = _kind(name)
    if kind == "surface":
        return _obj(name, 2400.0 + 400.0 * i, y_mm=-800.0 + 1600.0 * i, z_mm=380.0, width_mm=900.0, height_mm=750.0)
    if kind == "cabinet":
        return _obj(name, 1600.0, y_mm=-150.0, z_mm=600.0, width_mm=800.0, height_mm=1200.0)
    return _obj(name, 700.0, y_mm=-260.0, width_mm=180.0, height_mm=120.0)


def scene_n_targets(n: int, *, seen_refs: tuple[str, ...] = ("all",)) -> Callable[[dict], dict]:
    """``n`` instances of the target in view; references listed in ``seen_refs`` also in view."""

    def build(intent: dict) -> dict:
        target = (intent["targets"] or ["carton"])[0]
        objs = [_obj(target, 900.0 + 350.0 * i, y_mm=-200.0 + 200.0 * i) for i in range(n)]
        refs, missing = [], []
        for i, r in enumerate(intent["references"]):
            visible = "all" in seen_refs or any(s in r.lower() for s in seen_refs)
            (refs if visible else missing).append(_reference_obj(r, i) if visible else r)
        out: dict[str, Any] = {"count": len(objs), "objects": objs}
        if refs:
            out["references"] = refs
        if missing:
            out["missing"] = missing
        return out

    return build


def scene_box_in_cabinet(intent: dict) -> dict:
    """The cabinet is in view, the box is not — it is shut inside. Surfaces not in view."""
    target = (intent["targets"] or ["carton"])[0]
    refs, missing = [], [target]
    for i, r in enumerate(intent["references"]):
        if _kind(r) == "cabinet":
            refs.append(_reference_obj(r, i))
        else:
            missing.append(r)
    return {"count": 0, "objects": [], "references": refs, "missing": missing}


def scene_two_sources(intent: dict) -> dict:
    """One box beside the banana (in view) + one shut in the cabinet (not in view)."""
    target = (intent["targets"] or ["carton"])[0]
    refs, missing = [], []
    for i, r in enumerate(intent["references"]):
        if _kind(r) in ("cabinet", "small"):
            refs.append(_reference_obj(r, i))
        else:
            missing.append(r)
    return {"count": 1, "objects": [_obj(target, 850.0, y_mm=-180.0)], "references": refs, "missing": missing}


def scene_swap(intent: dict) -> dict:
    """One box on the purple table (in view), one in the cabinet (not in view)."""
    target = (intent["targets"] or ["carton"])[0]
    refs = [_reference_obj(r, i) for i, r in enumerate(intent["references"])]
    return {"count": 1, "objects": [_obj(target, 2400.0, y_mm=-700.0, z_mm=800.0)], "references": refs}


# --------------------------------------------------------------------------- #
@dataclass
class Task:
    id: str
    title: str
    query: str
    world: str
    scene: Callable[[dict], dict]
    tests: str
    checks: Callable[[list[str], list[dict]], list[tuple[str, bool, str]]] = lambda ops, seq: []
    door_ops: bool = True
    notes: list[str] = field(default_factory=list)


def _idx(ops: list[str], name: str) -> int:
    return ops.index(name) if name in ops else -1


def _has_loop(seq: list[dict]) -> bool:
    return any("loop" in s for s in seq)


def _check_loop(ops: list[str], seq: list[dict]) -> list[tuple[str, bool, str]]:
    copies = ops.count("dual_arm_grasp")
    return [
        ("用循环而不是复制 N 遍", _has_loop(seq), f"loop={_has_loop(seq)} grasp 出现 {copies} 次"),
        ("没有把同一段动作抄多遍", copies <= 1, f"dual_arm_grasp×{copies}"),
    ]


def _check_two_chains(ops: list[str], seq: list[dict]) -> list[tuple[str, bool, str]]:
    return [
        ("两次抓取（两个来源各一次）", ops.count("dual_arm_grasp") == 2, f"grasp×{ops.count('dual_arm_grasp')}"),
        ("两次放置（两个目的地各一次）", ops.count("dual_arm_place") == 2, f"place×{ops.count('dual_arm_place')}"),
        ("柜子那一路先开门", _idx(ops, "pull") >= 0 and _idx(ops, "pull") < len(ops) - 1, f"pull@{_idx(ops, 'pull')}"),
    ]


def _check_close_after(ops: list[str], seq: list[dict]) -> list[tuple[str, bool, str]]:
    pull, push = _idx(ops, "pull"), _idx(ops, "push")
    place = _idx(ops, "dual_arm_place")
    return [
        ("开门在抓取前", pull >= 0 and pull < _idx(ops, "dual_arm_grasp"), f"pull@{pull}"),
        ("关门动作存在", push >= 0, f"push@{push}"),
        ("关门排在放下之后（手必须空）", push > place >= 0, f"push@{push} place@{place}"),
    ]


def _check_no_grasp(ops: list[str], seq: list[dict]) -> list[tuple[str, bool, str]]:
    return [
        ("没有抓取动作（任务说了不用拿）", "dual_arm_grasp" not in ops, f"ops={ops}"),
        ("有观察类动作", any(o in ops for o in ("analyze_scene", "search_target", "locate_for_grasp")), f"ops={ops}"),
    ]


def _check_order_reversed(ops: list[str], seq: list[dict]) -> list[tuple[str, bool, str]]:
    push, grasp = _idx(ops, "push"), _idx(ops, "dual_arm_grasp")
    return [
        ("关门排在搬运之前（照任务给的顺序）", push >= 0 and grasp >= 0 and push < grasp, f"push@{push} grasp@{grasp}"),
    ]


def _place_into_shut_cabinet(seq: list[dict]) -> tuple[bool, str]:
    """A place whose surface is the cabinet, with no ``pull`` before it — putting a box
    through a closed door. The place-side target is whatever the last approach/locate named."""
    target, pulled = "", False
    for i, s in enumerate(seq):
        op = s.get("op")
        params = s.get("params") or {}
        if op == "pull":
            pulled = True
        elif op in ("approach_for_place", "locate_for_place"):
            target = str(params.get("object_name") or "")
        elif op == "dual_arm_place" and _kind(target) == "cabinet" and not pulled:
            return True, f"step {i}: 往 {target!r} 里放，但此前没有 pull"
    return False, ""


def _check_swap(ops: list[str], seq: list[dict]) -> list[tuple[str, bool, str]]:
    g, p = ops.count("dual_arm_grasp"), ops.count("dual_arm_place")
    blind_place, why = _place_into_shut_cabinet(seq)
    return [
        ("两个箱子各搬一次", g >= 2 and p >= 2, f"grasp×{g} place×{p}"),
        ("没有同时持两件（抓-放交替）", _alternating(ops), f"顺序={[o for o in ops if o in ('dual_arm_grasp', 'dual_arm_place')]}"),
        ("往柜子里放之前柜门已经拉开", not blind_place, why),
    ]


def _alternating(ops: list[str]) -> bool:
    held = False
    for o in ops:
        if o == "dual_arm_grasp":
            if held:
                return False
            held = True
        elif o == "dual_arm_place":
            if not held:
                return False
            held = False
    return True


TASKS: list[Task] = [
    Task(
        id="C1",
        title="未知数量的同类目标 → 循环",
        query="把桌上所有的箱子都搬到紫色桌子上",
        world="视野里 3 个箱子；紫色桌子可见。",
        scene=scene_n_targets(3),
        tests="能否识别「所有」是数量未知的迭代，输出 loop 构造而不是把 pick-place 抄 3 遍。",
        checks=_check_loop,
    ),
    Task(
        id="C2",
        title="两个来源、两个目的地 → 两条链",
        query="把香蕉旁的箱子放到紫色桌子上，把柜子里的箱子放到白色桌子上",
        world="香蕉旁的箱子可见；柜子可见但里面的箱子不可见（门关着）；两张桌子都不可见。",
        scene=scene_two_sources,
        tests="能否把一句话拆成两条独立的 pick-place 链，且只在柜子那一路插入开门。",
        checks=_check_two_chains,
    ),
    Task(
        id="C3",
        title="任务要求的收尾动作 → 受契约约束的排序",
        query="把柜子里的箱子拿出来放到紫色桌子上，然后把柜门关上",
        world="柜子可见、箱子不可见；紫色桌子不可见。",
        scene=scene_box_in_cabinet,
        tests="push 需要空手（requires payload.clear），所以关门只能排在放下之后——是否推出了这个约束。",
        checks=_check_close_after,
    ),
    Task(
        id="C4",
        title="技能库不覆盖的任务 → 落到契约层",
        query="转过身去看看柜子里都有什么，不用拿东西",
        world="柜子在身后，视野内什么都没有。",
        scene=lambda intent: {"count": 0, "objects": [], "missing": [*intent["targets"], *intent["references"]]},
        tests="没有一个 SKILL.md 描述「只看不拿」；能否不硬套 visual_pick，改用观察类动作组合。",
        checks=_check_no_grasp,
    ),
    Task(
        id="C5",
        title="任务给的顺序与模板顺序相反",
        query="先把柜门关上，再把香蕉旁的箱子搬到紫色桌子上",
        world="柜门开着、柜子可见；香蕉旁的箱子可见；紫色桌子不可见。",
        scene=scene_two_sources,
        tests="模板顺序是「先搬运后收尾」，任务要求反过来；能否照任务而不是照模板。",
        checks=_check_order_reversed,
    ),
    Task(
        id="C6",
        title="交换位置 → 需要中转的序列化",
        query="把紫色桌子上的箱子和柜子里的箱子交换位置",
        world="紫色桌子上的箱子可见；柜子可见、里面的箱子不可见。",
        scene=scene_swap,
        tests="只有一副手，交换必须序列化（且严格说需要一个中转位）。这是最难的一条，看它怎么退化。",
        checks=_check_swap,
    ),
]


def run_one(task: Task, ctx: dict[str, Any], run_no: int) -> dict[str, Any]:
    spec = ctx["spec"]
    intent = parse_task(task.query, api_base=spec["api_base"], api_key=spec["api_key"],
                        model_name=spec["model_name"], temperature=spec["temperature"])
    scene = task.scene(intent)
    blocked = _blocked_access(scene, intent.get("grounding") or {})
    index = dict(ctx["action_index"])
    if task.door_ops:
        index.update(ctx["door_index"])
    rec: dict[str, Any] = {"task": task.id, "run": run_no, "query": task.query,
                           "intent": intent, "scene": scene}
    try:
        planned = plan_task(
            task.query,
            skills_md=ctx["skills_md"] if task.door_ops else ctx["skills_md_base"],
            action_index=index,
            allowed_ops=index,
            api_base=spec["api_base"],
            api_key=spec["api_key"],
            model_name=spec["model_name"],
            temperature=spec["temperature"],
            api_capabilities=ctx["capabilities"],
            action_sigs={n: _action_param_sig(f) for n, f in index.items()},
            scene=scene,
            grounding=intent.get("grounding") or {},
            blocked_access=blocked,
            world_block=ctx["world_block"],
            world_tokens=ctx["world_tokens"],
        )
    except RuntimeError as exc:
        rec.update({"ok": False, "error": str(exc), "checks": []})
        return rec

    seq = planned.sequence
    ops = [s.get("op") for s in seq if s.get("op")]
    for s in seq:  # a loop body's ops count as ops of the plan
        if "loop" in s:
            ops.extend(b.get("op") for b in (s["loop"].get("body") or []) if b.get("op"))
    rec.update({"tier": planned.tier, "skills": list(planned.skills or ()), "reason": planned.reason,
                "sequence": seq, "ops": ops})

    checks: list[tuple[str, bool, str]] = []
    try:
        parse_sequence(seq, allowed_ops=index, initial_state=ctx["world_tokens"],
                       grounding=intent.get("grounding") or {}, blocked_access=blocked)
        checks.append(("契约校验通过", True, ""))
    except Exception as exc:  # noqa: BLE001 - a rejected plan is a result, not a crash
        checks.append(("契约校验通过", False, str(exc)))
    checks.extend(task.checks(ops, seq))
    rec["checks"] = [{"name": n, "pass": bool(p), "detail": d} for n, p, d in checks]
    rec["ok"] = all(c["pass"] for c in rec["checks"])
    return rec


def _print(rec: dict[str, Any]) -> None:
    print(f"\n  --- 第 {rec['run']} 次 ---")
    i = rec.get("intent") or {}
    print(f"  解析意图: targets={i.get('targets')} refs={i.get('references')} "
          f"dest={i.get('destination')!r} mode={i.get('mode')} count={i.get('count')}")
    if rec.get("error"):
        print(f"  ❌ 规划失败: {rec['error'][:300]}")
        return
    print(f"  规划层级: tier={rec['tier']} skills={rec.get('skills')}"
          + (f"\n  降级原因: {rec['reason'][:200]}" if rec.get("reason") else ""))
    for n, step in enumerate(rec["sequence"]):
        if "loop" in step:
            lp = step["loop"]
            print(f"    {n:>2}. LOOP detect={lp.get('detect', {}).get('op')} "
                  f"{json.dumps(lp.get('detect', {}).get('params', {}), ensure_ascii=False)} "
                  f"bind={lp.get('bind')} max_iters={lp.get('max_iters')}")
            for m, b in enumerate(lp.get("body") or []):
                print(f"        {m:>2}. {b.get('op')}  {json.dumps(b.get('params') or {}, ensure_ascii=False)}")
            continue
        bind = f"  bind={step['bind']}" if step.get("bind") else ""
        print(f"    {n:>2}. {step.get('op')}{bind}  {json.dumps(step.get('params') or {}, ensure_ascii=False)}")
    for c in rec["checks"]:
        mark = "✅" if c["pass"] else "❌"
        print(f"  {mark} {c['name']}" + (f"   [{c['detail'][:200]}]" if c["detail"] and not c["pass"] else ""))


def main() -> int:
    p = argparse.ArgumentParser(description="LLM composition probes on cruzr (real LLM, no hardware).")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs/cruzr/cruzr.yaml"))
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    raw = introspect.load_config(args.config)
    session = introspect.build_session(args.config)
    model = raw.get("model") or {}
    door = _door_ops_module()
    base_reg = SkillRegistry()
    base_reg.register_dir(SKILLS_DIR)
    full_reg = SkillRegistry()
    full_reg.register_dir(SKILLS_DIR)
    full_reg.register_dir(_HERE / "skills")
    world = WorldState.snapshot(session)
    ctx = {
        "spec": {"api_base": model.get("api_base", ""), "api_key": args.api_key,
                 "model_name": args.model or model.get("model_name", ""),
                 "temperature": float(model.get("temperature", 0.0))},
        "action_index": _build_action_index(session.api, planner_only=True),
        "door_index": _build_action_index(door._DoorOps(), planner_only=True),  # noqa: SLF001 - test-local body
        "capabilities": sorted(session.api.capabilities),
        "skills_md": full_reg.skills_markdown(),
        "skills_md_base": base_reg.skills_markdown(),
        "world_block": world.as_prompt_block(),
        "world_tokens": sorted(world.tokens) or None,
    }

    print("=== LLM 组合能力测试（cruzr，真实 LLM，无硬件）===")
    print(f"  模型  : {ctx['spec']['model_name']} @ {ctx['spec']['api_base']}")
    print(f"  技能库: {', '.join(s['name'] for s in ctx['skills_md'])}")

    wanted = set(args.only) if args.only else None
    records: list[dict[str, Any]] = []
    for t in TASKS:
        if wanted and t.id not in wanted:
            continue
        print(f"\n{'=' * 78}\n{t.id} · {t.title}\n  任务  : {t.query}\n  世界  : {t.world}\n  考点  : {t.tests}")
        for run_no in range(1, args.repeat + 1):
            rec = run_one(t, ctx, run_no)
            records.append(rec)
            _print(rec)

    print(f"\n{'=' * 78}\n=== 汇总 ===")
    for t in TASKS:
        runs = [r for r in records if r["task"] == t.id]
        if not runs:
            continue
        ok = sum(1 for r in runs if r.get("ok"))
        fails: dict[str, int] = {}
        for r in runs:
            for c in r.get("checks", []):
                if not c["pass"]:
                    fails[c["name"]] = fails.get(c["name"], 0) + 1
        note = "" if not fails else "  未过：" + "; ".join(f"{k}×{v}" for k, v in fails.items())
        print(f"  {t.id}: {ok}/{len(runs)}{note}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(records, ensure_ascii=False, indent=2, default=repr),
                                       encoding="utf-8")
        print(f"\n  明细已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
