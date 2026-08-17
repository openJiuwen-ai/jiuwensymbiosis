# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cruzr task-planning scenarios — the real two-call planner, no hardware.

Each scenario fixes what perception WOULD have reported (the ``scene`` block
``agent/run.py:_perceive_scene`` builds from the standing detector) and then runs
the production planning path unchanged: ``parse_task`` (LLM①) → ``plan_task``
(LLM②, tier 1 skills → tier 2 contracts) → ``parse_sequence`` (the contract
validator). The action index and capabilities come from the real cruzr session,
which builds offline — only the detector's answer is supplied by hand, because
that is exactly what the scenarios vary.

``pull`` / ``push`` exist only here (``tests/scenarios/skills/`` + the two
``@robot_tool`` stubs below): scenario 4 assumes such skills, and assuming them
must not mean shipping them.

Usage::

    python tests/scenarios/run_cruzr_scenarios.py --api-key sk-...
    python tests/scenarios/run_cruzr_scenarios.py --api-key sk-... --only 4a --repeat 1
"""

from __future__ import annotations

import argparse
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
from jiuwensymbiosis.api.decorators import robot_tool  # noqa: E402
from jiuwensymbiosis.api.world_state import WorldState  # noqa: E402
from jiuwensymbiosis.skills import SKILLS_DIR  # noqa: E402
from jiuwensymbiosis.tools.robot_control_tool import _build_action_index  # noqa: E402

SCENARIO_SKILLS_DIR = Path(__file__).resolve().parent / "skills"


# --------------------------------------------------------------------------- #
# The two actions scenario 4 assumes. Test-local: never registered on a body.
# --------------------------------------------------------------------------- #
class _DoorOps:
    """``pull`` / ``push`` as the planner would see them if a body offered them.

    Two declarations carry the whole contract, and neither says anything about doors:
    ``requires=[payload.clear]`` (pulling needs free hands) and ``opens_access`` /
    ``closes_access`` (this action clears, or restores, whatever blocks the thing it
    acts on — see ``api/state.py`` §3). Everything else the validator needs it derives
    from the sequence and from what the pre-plan look measured.
    """

    capabilities: frozenset[str] = frozenset()

    @robot_tool(
        desc=(
            "Pull an openable thing (door / cabinet door / drawer) towards the robot to open it, "
            "exposing what is inside. object_name is the door or drawer itself, never the item inside. "
            "Needs free hands. Failure → {ok: False, reason}."
        ),
        requires=["payload.clear"],
        invalidates=["body.home"],
        opens_access=True,
        tags=["motion"],
        planner_visible=True,
    )
    def pull(self, object_name: str, distance_m: float = 0.4) -> dict:
        return {"ok": True, "object": object_name, "distance_m": distance_m}

    @robot_tool(
        desc=(
            "Push an openable thing shut (or push an obstacle aside) away from the robot. "
            "object_name is the door / drawer / obstacle itself. Needs free hands. "
            "Failure → {ok: False, reason}."
        ),
        requires=["payload.clear"],
        invalidates=["body.home"],
        closes_access=True,
        tags=["motion"],
        planner_visible=True,
    )
    def push(self, object_name: str, distance_m: float = 0.4) -> dict:
        return {"ok": True, "object": object_name, "distance_m": distance_m}


# --------------------------------------------------------------------------- #
# Scene fixtures — the shape agent/run.py:_perceive_scene returns
# --------------------------------------------------------------------------- #
def _obj(name: str, forward_mm: float, *, y_mm: float = 60.0, z_mm: float = 700.0,
         width_mm: float = 300.0, height_mm: float = 250.0, reachable: bool | None = None) -> dict:
    """One detected instance, with the fields detect_all_object_geometry emits."""
    o: dict[str, Any] = {
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
    }
    if reachable is not None:
        o["reachable"] = reachable
    return o


def _nothing_seen(intent: dict) -> dict:
    """Looked for target + references, saw neither — the body simply isn't facing them."""
    names = [*intent["targets"], *intent["references"]]
    return {"count": 0, "objects": [], "missing": names}


def _target_and_place_seen(*, reachable: bool) -> Callable[[dict], dict]:
    """Target visible (at a distance matching ``reachable``), reference + destination visible."""

    def build(intent: dict) -> dict:
        target = intent["targets"][0] if intent["targets"] else "carton"
        refs = intent["references"]
        forward = 550.0 if reachable else 1900.0
        objs = [_obj(target, forward, reachable=reachable)]
        ref_objs = [
            _obj(r, forward + 120.0, y_mm=-260.0, width_mm=180.0, height_mm=120.0, reachable=reachable)
            if _is_side_reference(r)
            else _obj(r, 2600.0, y_mm=-900.0, width_mm=900.0, height_mm=750.0, z_mm=380.0, reachable=False)
            for r in refs
        ]
        return {"count": len(objs), "objects": objs, "references": ref_objs}

    return build


def _is_side_reference(name: str) -> bool:
    """A reference standing next to the target (the banana), vs the far-off destination."""
    n = name.lower()
    return not any(k in n for k in ("table", "desk", "shelf", "counter"))


def _cabinet_seen_box_not(intent: dict) -> dict:
    """The cabinet is in view; the box the task names is not — it is shut inside it.

    This is the whole of what the framework can say about a closed door: there is no
    door-state token in ``KNOWN_STATE_TOKENS`` and no door field in the scene block.
    """
    target = intent["targets"][0] if intent["targets"] else "carton"
    refs = [r for r in intent["references"] if "cabinet" in r.lower() or "cupboard" in r.lower()]
    others = [r for r in intent["references"] if r not in refs]
    return {
        "count": 0,
        "objects": [],
        "references": [_obj(r, 1600.0, y_mm=-150.0, z_mm=600.0, width_mm=800.0,
                            height_mm=1200.0, reachable=False) for r in (refs or ["cabinet"])],
        "missing": [target, *others],
    }


# --------------------------------------------------------------------------- #
# Checks — what a correct plan must look like, stated as contract facts
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    id: str
    title: str
    query: str
    world: str
    scene: Callable[[dict], dict]
    door_ops: bool = False
    extra_checks: list[str] = field(default_factory=list)


def _index_of(ops: list[str], name: str) -> int:
    return ops.index(name) if name in ops else -1


def _params_for(seq: list[dict], name: str) -> dict:
    for s in seq:
        if s.get("op") == name:
            return s.get("params") or {}
    return {}


def _all_params_text(seq: list[dict]) -> str:
    return json.dumps(seq, ensure_ascii=False).lower()


def _check_common(ops: list[str], seq: list[dict], intent: dict) -> list[tuple[str, bool, str]]:
    """The pick→carry→place spine every one of scenarios 1–3 must produce."""
    grasp = _index_of(ops, "dual_arm_grasp")
    place = _index_of(ops, "dual_arm_place")
    appr_g = _index_of(ops, "approach_for_grasp")
    appr_p = _index_of(ops, "approach_for_place")
    loc_p = _index_of(ops, "locate_for_place")
    lift = _index_of(ops, "lift_to_clearance")
    text = _all_params_text(seq)
    ref = (intent["references"] or [""])[0].lower()
    out = [
        ("抓取前先搜索并驱近 (approach_for_grasp → dual_arm_grasp)",
         appr_g >= 0 and grasp >= 0 and appr_g < grasp,
         f"approach_for_grasp@{appr_g} grasp@{grasp}"),
        ("抓取后抬到搬运净空 (lift_to_clearance)", lift > grasp >= 0, f"lift@{lift}"),
        ("放置前先驱近放置面 (approach_for_place)", appr_p >= 0 and appr_p < place,
         f"approach_for_place@{appr_p} place@{place}"),
        ("底盘移动后重新感知放置面 (locate_for_place 在 approach_for_place 之后)",
         loc_p > appr_p >= 0, f"locate_for_place@{loc_p}"),
        ("先抓后放 (dual_arm_place 在 dual_arm_grasp 之后)", place > grasp >= 0, f"place@{place}"),
        ("参照物限定并入抓取侧 (reference/relation)",
         bool(ref) and ref in text and "relation" in text, f"reference={ref!r} 出现={ref in text}"),
        ("未盲转 / 盲走 (无 rotate_base / navigate_relative)",
         "rotate_base" not in ops and "navigate_relative" not in ops,
         f"ops={[o for o in ops if o in ('rotate_base', 'navigate_relative')]}"),
    ]
    return out


def _check_door(ops: list[str], seq: list[dict], intent: dict) -> list[tuple[str, bool, str]]:
    """Scenario 4: the door must be opened, with free hands, before the box is grasped."""
    pull = _index_of(ops, "pull")
    grasp = _index_of(ops, "dual_arm_grasp")
    target = (intent["targets"] or ["carton"])[0].lower()
    pull_obj = str(_params_for(seq, "pull").get("object_name", "")).lower()
    return [
        ("编出了开门动作 (pull)", pull >= 0, f"pull@{pull}"),
        ("先开门再抓箱 (pull → dual_arm_grasp)", pull >= 0 and grasp >= 0 and pull < grasp,
         f"pull@{pull} grasp@{grasp}"),
        ("pull 的对象是柜/门而非目标箱", bool(pull_obj) and target not in pull_obj,
         f"pull.object_name={pull_obj!r} target={target!r}"),
        ("没有任务未要求的关门动作 (push)", "push" not in ops, f"push@{_index_of(ops, 'push')}"),
    ]


SCENARIOS: list[Scenario] = [
    Scenario(
        id="1",
        title="看不到目标和放置位；抓取目标超出工作范围",
        query="帮我把香蕉旁的箱子放到紫色桌子上",
        world="规划前对目标、参照物、放置面都检测过，一个都不在视野里；目标实际也在双臂工作范围之外，必须先驶近。",
        scene=_nothing_seen,
    ),
    Scenario(
        id="2",
        title="看不到目标和放置位；抓取目标未超出工作范围",
        query="帮我把香蕉旁的箱子放到紫色桌子上",
        world="同场景 1 的感知输入（都没看见）；区别在于目标其实就在身侧、转过去即可够到，无需长距离驶近。",
        scene=_nothing_seen,
    ),
    Scenario(
        id="3a",
        title="看得到目标和放置位；目标超出工作范围",
        query="帮我把香蕉旁的箱子放到紫色桌子上",
        world="目标 1.9m 处可见、可达=否；香蕉可见；紫色桌子 2.6m 处可见、可达=否。",
        scene=_target_and_place_seen(reachable=False),
    ),
    Scenario(
        id="3b",
        title="看得到目标和放置位；目标在工作范围内",
        query="帮我把香蕉旁的箱子放到紫色桌子上",
        world="目标 0.55m 处可见、可达=是；香蕉可见；紫色桌子可见但不可达（放置仍需驶近）。",
        scene=_target_and_place_seen(reachable=True),
    ),
    Scenario(
        id="4a",
        title="柜门关着（只由感知隐含：柜子可见、箱子未见）",
        query="帮我把柜子里的箱子放到紫色桌子上",
        world="柜子 1.6m 处可见、可达=否；箱子在【未见】里（关在柜门后）；紫色桌子未见。门的开合状态框架无法表达，只能由此推断。",
        scene=_cabinet_seen_box_not,
        door_ops=True,
    ),
    Scenario(
        id="4b",
        title="柜门关着（在任务文本里明说）",
        query="帮我把柜子里的箱子放到紫色桌子上。柜子门关着",
        world="感知输入同 4a；区别是任务文本直接告诉规划器门是关的。",
        scene=_cabinet_seen_box_not,
        door_ops=True,
    ),
]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _skills_md(*, door_ops: bool) -> list[dict[str, Any]]:
    reg = SkillRegistry()
    reg.register_dir(SKILLS_DIR)
    if door_ops:
        reg.register_dir(SCENARIO_SKILLS_DIR)
    return reg.skills_markdown()


def run_one(sc: Scenario, ctx: dict[str, Any], run_no: int) -> dict[str, Any]:
    """One scenario, one repetition: parse → perceive (fixture) → plan → validate → judge."""
    spec = ctx["spec"]
    intent = parse_task(
        sc.query, api_base=spec["api_base"], api_key=spec["api_key"],
        model_name=spec["model_name"], temperature=spec["temperature"],
    )
    scene = sc.scene(intent)
    blocked = _blocked_access(scene, intent.get("grounding") or {})
    index = dict(ctx["action_index"])
    if sc.door_ops:
        index.update(ctx["door_index"])
    sigs = {name: _action_param_sig(fn) for name, fn in index.items()}

    record: dict[str, Any] = {
        "scenario": sc.id, "run": run_no, "query": sc.query,
        "intent": intent, "scene": scene,
    }
    try:
        planned = plan_task(
            sc.query,
            skills_md=_skills_md(door_ops=sc.door_ops),
            action_index=index,
            allowed_ops=index,
            api_base=spec["api_base"],
            api_key=spec["api_key"],
            model_name=spec["model_name"],
            temperature=spec["temperature"],
            api_capabilities=ctx["capabilities"],
            action_sigs=sigs,
            scene=scene,
            grounding=intent.get("grounding") or {},
            blocked_access=blocked,
            world_block=ctx["world_block"],
            world_tokens=ctx["world_tokens"],
        )
    except RuntimeError as exc:
        record.update({"ok": False, "error": f"planning failed: {exc}", "checks": []})
        return record

    seq = planned.sequence
    ops = [s.get("op") for s in seq]
    record.update({"tier": planned.tier, "skills": list(planned.skills or ()),
                   "reason": planned.reason, "sequence": seq})

    checks: list[tuple[str, bool, str]] = []
    try:
        parse_sequence(seq, allowed_ops=index, initial_state=ctx["world_tokens"],
                       grounding=intent.get("grounding") or {}, blocked_access=blocked)
        checks.append(("契约校验通过 (parse_sequence)", True, ""))
    except Exception as exc:  # noqa: BLE001 - a rejected plan is a result, not a crash
        checks.append(("契约校验通过 (parse_sequence)", False, str(exc)))

    checks.extend(_check_common(ops, seq, intent))
    if sc.door_ops:
        checks.extend(_check_door(ops, seq, intent))
    record["checks"] = [{"name": n, "pass": bool(p), "detail": d} for n, p, d in checks]
    record["ok"] = all(c["pass"] for c in record["checks"])
    return record


def _print_run(rec: dict[str, Any]) -> None:
    print(f"\n--- 场景 {rec['scenario']} · 第 {rec['run']} 次 ---")
    intent = rec.get("intent") or {}
    print(f"  解析意图 : targets={intent.get('targets')} references={intent.get('references')} "
          f"grounding={intent.get('grounding')} destination={intent.get('destination')!r}")
    if rec.get("error"):
        print(f"  ❌ {rec['error']}")
        return
    print(f"  规划层级 : tier={rec['tier']} skills={rec.get('skills')}")
    print("  动作序列 :")
    for i, step in enumerate(rec["sequence"]):
        params = json.dumps(step.get("params") or {}, ensure_ascii=False)
        bind = f"  bind={step['bind']}" if step.get("bind") else ""
        print(f"    {i:>2}. {step.get('op')}{bind}  {params}")
    for c in rec["checks"]:
        mark = "✅" if c["pass"] else "❌"
        detail = f"   [{c['detail']}]" if c["detail"] and not c["pass"] else ""
        print(f"  {mark} {c['name']}{detail}")


def main() -> int:
    p = argparse.ArgumentParser(description="Cruzr task-planning scenarios (real LLM, no hardware).")
    p.add_argument("--config", default="configs/cruzr/cruzr.yaml")
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", default=None, help="Override the YAML model name.")
    p.add_argument("--repeat", type=int, default=3, help="Runs per scenario (the planner is stochastic).")
    p.add_argument("--only", nargs="*", default=None, help="Scenario ids, e.g. --only 1 4a")
    p.add_argument("--json-out", default=None, help="Write every record to this JSON file.")
    args = p.parse_args()

    raw = introspect.load_config(args.config)
    session = introspect.build_session(args.config)
    model = raw.get("model") or {}
    index = _build_action_index(session.api, planner_only=True)
    world = WorldState.snapshot(session)
    ctx = {
        "spec": {
            "api_base": model.get("api_base", ""),
            "api_key": args.api_key,
            "model_name": args.model or model.get("model_name", ""),
            "temperature": float(model.get("temperature", 0.0)),
        },
        "action_index": index,
        "door_index": _build_action_index(_DoorOps(), planner_only=True),
        "capabilities": sorted(session.api.capabilities),
        "world_block": world.as_prompt_block(),
        "world_tokens": sorted(world.tokens) or None,
    }

    print("=== cruzr 任务场景测试（真实 LLM 编译，无硬件）===")
    print(f"  模型     : {ctx['spec']['model_name']} @ {ctx['spec']['api_base']}")
    print(f"  本体能力 : {', '.join(ctx['capabilities'])}")
    print(f"  规划词表 : {', '.join(sorted(index))}")
    print(f"  门动作   : {', '.join(sorted(ctx['door_index']))}（仅场景 4 注入）")
    print(f"  起始状态 : {ctx['world_tokens']}")

    wanted = set(args.only) if args.only else None
    records: list[dict[str, Any]] = []
    for sc in SCENARIOS:
        if wanted and sc.id not in wanted:
            continue
        print(f"\n{'=' * 78}\n场景 {sc.id}：{sc.title}\n  任务：{sc.query}\n  世界：{sc.world}")
        for run_no in range(1, args.repeat + 1):
            rec = run_one(sc, ctx, run_no)
            records.append(rec)
            _print_run(rec)

    print(f"\n{'=' * 78}\n=== 汇总 ===")
    for sc in SCENARIOS:
        runs = [r for r in records if r["scenario"] == sc.id]
        if not runs:
            continue
        passed = sum(1 for r in runs if r.get("ok"))
        fails: dict[str, int] = {}
        for r in runs:
            for c in r.get("checks", []):
                if not c["pass"]:
                    fails[c["name"]] = fails.get(c["name"], 0) + 1
        note = "" if not fails else "  失败项：" + "; ".join(f"{k}×{v}" for k, v in fails.items())
        print(f"  场景 {sc.id}: {passed}/{len(runs)} 全项通过{note}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(records, ensure_ascii=False, indent=2, default=repr),
                                       encoding="utf-8")
        print(f"\n  明细已写入 {args.json_out}")
    return 0 if all(r.get("ok") for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
