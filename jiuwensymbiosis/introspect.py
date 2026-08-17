# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Machine-readable views of a body: its actions, its skills, its current state.

These are the entry points an LLM (or a coding agent authoring a new skill) reads
instead of the adapter source or the SKILL.md prose. Everything a planner needs to
derive an order is here: each action's params, result fields, and pre-conditions /
effects; each skill's contract; and what currently holds.

The action and skill views need no hardware — they are built from a config, and
the body is constructed but never connected. ``state`` does connect, since
"what is true right now" cannot be answered offline.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from jiuwensymbiosis.api.decorators import ToolMeta

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Read a robot config YAML."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_session(config_path: str) -> Any:
    """Build (but do not connect) the session a config describes.

    The adapter is named by the config's ``adapter:`` field and resolved by the
    same convention the GUI registry uses — ``jiuwensymbiosis.adapters.<name>``
    exports ``build_<name>_session`` with a ``.from_dict``. So a new body needs no
    change here.
    """
    raw = load_config(config_path)
    adapter = str(raw.get("adapter") or "piper")
    module = importlib.import_module(f"jiuwensymbiosis.adapters.{adapter}")
    factory = getattr(module, f"build_{adapter}_session")
    return factory.from_dict(raw)


def _contract(meta: ToolMeta) -> dict[str, Any]:
    """One action's full planning contract, JSON-able."""
    return {
        "name": meta.name,
        "description": meta.description,
        "planner_visible": meta.planner_visible,
        "capability": meta.capability,
        "tags": list(meta.tags),
        "params": meta.input_params,
        "returns": meta.returns,
        "requires": list(meta.requires),
        "provides": list(meta.provides),
        "invalidates": list(meta.invalidates),
        "produces_location": meta.produces_location,
        "consumes_location": meta.consumes_location,
        "invalidates_locations": meta.invalidates_locations,
    }


def action_contracts(session: Any) -> list[dict[str, Any]]:
    """Every action this body exposes, with its contract, sorted by name."""
    from jiuwensymbiosis.tools.robot_control_tool import _build_action_index

    index = _build_action_index(session.api, env=session.env)
    out = []
    for name in sorted(index):
        meta = getattr(index[name], "__robot_tool__", None)
        if isinstance(meta, ToolMeta):
            out.append(_contract(meta))
    return out


def skill_contracts() -> list[dict[str, Any]]:
    """The skill library's planning view (no markdown)."""
    from jiuwensymbiosis.agent.fast import DEFAULT_REGISTRY

    return DEFAULT_REGISTRY.catalogue()


def action_vocabulary() -> list[dict[str, Any]]:
    """Every action the FRAMEWORK defines, independent of any robot.

    ``action_contracts`` answers "what can this robot do"; this answers "what is
    there to do at all, and what capability does each need". Writing a skill that
    survives being moved to another body needs the second question: it tells you
    which actions are universal (no gate), which are conditional, and therefore
    which branches a SKILL.md has to carry.
    """
    from jiuwensymbiosis.api.actions import ACTIONS
    from jiuwensymbiosis.api.decorators import schema_from_typeddict

    return [
        {
            "name": spec.name,
            "description": spec.description,
            "capability": spec.capability,
            "params": list(spec.params),
            "required_params": list(spec.required_params),
            "returns": schema_from_typeddict(spec.result),
            "requires": list(spec.requires),
            "provides": list(spec.provides),
            "invalidates": list(spec.invalidates),
            "produces_location": spec.produces_location,
            "consumes_location": spec.consumes_location,
            "invalidates_locations": spec.invalidates_locations,
            "tags": list(spec.tags),
            "planner_visible": spec.planner_visible,
        }
        for spec in sorted(ACTIONS.values(), key=lambda s: (s.capability or "", s.name))
    ]


def _param_sig(schema: dict[str, Any]) -> str:
    """Render a JSON-Schema params object as ``(a, b?)`` — ``?`` = has a default."""
    props = schema.get("properties") or {}
    required = set(schema.get("required") or ())
    return "(" + ", ".join(f"{n}{'' if n in required else '?'}" for n in props) + ")"


def _effects(entry: dict[str, Any]) -> str:
    """Render the pre-conditions / effects of an action or skill as one line."""
    bits = []
    for key, label in (("requires", "需要"), ("provides", "产生"), ("invalidates", "作废")):
        if entry.get(key):
            bits.append(f"{label}{entry[key]}")
    if entry.get("produces_location"):
        bits.append("产生位置")
    if entry.get("consumes_location"):
        bits.append("消费位置")
    if entry.get("invalidates_locations"):
        bits.append("位置全失效")
    return "  ".join(bits)


def render_actions(contracts: list[dict[str, Any]]) -> str:
    """Human-readable action table (the ``--json`` form is the machine one).

    Marks which entries the planner actually sees, and separates each body's own
    note from the shared contract — the note is the only part that stops being true
    when the same plan runs on another robot.
    """
    lines = [f"# 可用动作 ({len(contracts)})", ""]
    for c in contracts:
        fields = sorted(c["returns"].get("properties") or {})
        flag = "" if c.get("planner_visible", True) else "   [调试/标定，不进规划器词表]"
        gate = f"   [能力门 {c['capability']}]" if c.get("capability") else ""
        lines.append(f"{c['name']}{_param_sig(c['params'])}{gate}{flag}")
        lines.append(f"    {c['description']}")
        if fields:
            lines.append(f"    返回: {{{', '.join(fields)}}}")
        eff = _effects(c)
        if eff:
            lines.append(f"    {eff}")
        lines.append("")
    return "\n".join(lines)


def render_vocabulary(specs: list[dict[str, Any]]) -> str:
    """Human-readable view of the framework-wide action vocabulary, grouped by capability."""
    lines = [f"# 动作词表 ({len(specs)}) —— 全框架共享，与本体无关", ""]
    current = object()
    for spec in specs:
        cap = spec["capability"]
        if cap != current:
            current = cap
            lines.append(f"## {cap or '（无能力门 —— 所有本体都有）'}")
            lines.append("")
        flag = "" if spec["planner_visible"] else "   [调试/标定，不进规划器词表]"
        lines.append(f"{spec['name']}{_param_sig({'properties': dict.fromkeys(spec['params'], {}), 'required': spec['required_params']})}{flag}")
        lines.append(f"    {spec['description']}")
        fields = sorted(spec["returns"].get("properties") or {})
        if fields:
            lines.append(f"    返回: {{{', '.join(fields)}}}")
        eff = _effects(spec)
        if eff:
            lines.append(f"    {eff}")
        lines.append("")
    return "\n".join(lines)


def render_skills(skills: list[dict[str, Any]]) -> str:
    """Human-readable skill table."""
    lines = [f"# Skill Library ({len(skills)})", ""]
    for s in skills:
        lines.append(f"{s['name']}")
        lines.append(f"    {s['description']}")
        if s.get("capabilities"):
            lines.append(f"    能力门: {s['capabilities']}")
        eff = _effects(s)
        if eff:
            lines.append(f"    {eff}")
        lines.append("")
    return "\n".join(lines)


def render_state(state: dict[str, Any]) -> str:
    """Human-readable world-state snapshot."""
    lines = ["# 当前世界状态", ""]
    lines.append(f"状态 token: {state['tokens'] or '（未知）'}")
    lines.append(f"能力: {state['capabilities']}")
    if state.get("pose"):
        lines.append(f"位姿: {state['pose']}")
    if state.get("joints"):
        lines.append(f"关节: {state['joints']}")
    if state.get("locations"):
        lines.append("已感知:")
        for loc in state["locations"]:
            lines.append(f"    {loc['referent']}: {loc['position_mm']} ({loc['sensed_by']}, {loc['age_s']}s 前)")
    if state.get("extra"):
        lines.append(f"其他: {state['extra']}")
    return "\n".join(lines)


def emit(payload: Any, text: str, *, as_json: bool) -> None:
    """Print the machine or the human rendering of one view."""
    print(json.dumps(payload, ensure_ascii=False, indent=2) if as_json else text)
