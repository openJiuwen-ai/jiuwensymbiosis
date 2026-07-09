# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SkillPatchProposal generation (P3 of the Trace Feedback Loop).

``docs/trace-feedback-loop-design.md`` §4.4: turn ``FailureCluster``s into
human-review SKILL.md patch proposals. First pass is deterministic — no LLM,
no SKILL.md parsing. ``target_skill`` is always ``"<unresolved>"``; skill
matching is deferred to a later phase.

Two non-obvious invariants:

- **Recovery advice is a global post-process, not a pattern branch.** The
  ``FailureSignature`` records the first ``success=False`` rail event as the
  failure cause, but ``RecoveryRail/recover`` is normally ``success=True`` (it
  *fixed* the failure), so it never lands in the signature. A cluster that hit
  the SafetyRail pattern can still have recovery events in its examples — the
  recovery hint must be appended after the main template for *every* cluster,
  by scanning ``examples[*].rail_events``, not by matching the signature.
- **No SKILL.md reads.** Templates cite generic section names
  (``## 参数取值约定`` / ``## 标准 Workflow`` / ``## 失败处理``) as placement
  hints. The proposal is prose + hint; the human picks the actual section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jiuwensymbiosis.trace_feedback.analysis import FailureCluster, FailureEvidence, FailureSignature

_VISION_TOOLS = frozenset({"analyze_scene", "get_grasp_info_simple", "pixel_to_base_xyz"})
_RISKS = (
    "建议基于聚类证据，未在真实硬件验证。",
    "target_skill 未自动确定，需人审确认目标 SKILL.md。",
)
_VALIDATION = ("人审修改后，用 `pytest -m integration` 或 `--mock` demo 复现原失败场景，确认不再触发。",)


@dataclass
class SkillPatchProposal:
    target_skill: str
    summary: str
    evidence_signatures: list[FailureSignature]
    examples: list[FailureEvidence]
    proposed_diff: str
    confidence: Literal["low", "medium", "high"]
    risks: list[str]
    validation_suggestions: list[str]


def propose_skill_patches(clusters: list[FailureCluster]) -> list[SkillPatchProposal]:
    """Generate one deterministic proposal per cluster. No source files modified."""
    return [_proposal_for(c) for c in clusters]


def _proposal_for(c: FailureCluster) -> SkillPatchProposal:
    main = _main_template(c)
    main = _append_recovery_advice(main, c.examples)
    pattern_desc = _pattern_description(c.signature)
    return SkillPatchProposal(
        target_skill="<unresolved>",
        summary=f"{c.count} 次 {pattern_desc}，建议人审定 skill 后改 SKILL.md。",
        evidence_signatures=[c.signature],
        examples=list(c.examples[:2]),
        proposed_diff=main,
        confidence=_confidence(c.count),
        risks=list(_RISKS),
        validation_suggestions=list(_VALIDATION),
    )


def _main_template(c: FailureCluster) -> str:
    sig = c.signature
    if sig.rail_name == "SafetyRail" and sig.kind == "reject":
        reason = sig.reason_norm
        if "z" in reason or "floor" in reason or "below" in reason:
            return (
                f"在相关 SKILL.md 的『## 参数取值约定』或『## 标准 Workflow』章节，补充："
                f"调用 `{sig.tool_name}` 时 `z` 必须 ≥ `env.z_min_safe`，否则被 SafetyRail 拒绝。"
                "建议加 pre-check 或失败后上抬 z 重试。"
            )
        if "out of bounds" in reason or "x" in reason or "y" in reason:
            return (
                f"在相关 SKILL.md 的『## 参数取值约定』章节，补充 workspace XY 边界约束："
                f"调用 `{sig.tool_name}` 前确认 `(x,y)` 在 `env.workspace_bounds` 内。"
            )
        if "joint" in reason or "limit" in reason or "q" in reason:
            return (
                "在相关 SKILL.md 的『## 参数取值约定』章节，补充 `move_joint` 的 `q` "
                "长度/范围校验，对齐 `env.joint_limits`。"
            )
    if sig.rail_name is None and sig.tool_name in _VISION_TOOLS:
        return (
            f"在相关 SKILL.md 的『## 标准 Workflow』章节，补充视觉确认步骤或 prompt/target 消歧，"
            f"避免 `{sig.tool_name}` 失败导致后续动作空跑。"
        )
    return f"复核 `{sig.reason_norm or sig.tool_name}` 失败模式，在相关章节加 guard / retry / 参数约束。"


def _append_recovery_advice(diff: str, examples: list[FailureEvidence]) -> str:
    has_recovery = any(
        isinstance(ev, dict) and ev.get("rail_name") == "RecoveryRail" and ev.get("kind") == "recover"
        for ex in examples
        for ev in ex.rail_events
    )
    if not has_recovery:
        return diff
    return (
        diff + "\n\n另：本类失败的 examples 显示 RecoveryRail 已 home+release，"
        "建议在『## 失败处理』补『动作失败后重新 `get_observation` 确认末端空载与位姿再继续』。"
    )


def _pattern_description(sig: FailureSignature) -> str:
    if sig.rail_name == "SafetyRail":
        return f"SafetyRail/{sig.kind or 'reject'} 失败（{sig.reason_norm or '未知原因'}）"
    if sig.tool_name in _VISION_TOOLS:
        return f"视觉工具 {sig.tool_name} 失败"
    return f"{sig.tool_name} 失败（{sig.reason_norm or '未知原因'}）"


def _confidence(count: int) -> Literal["low", "medium", "high"]:
    if count >= 5:
        return "high"
    if count >= 3:
        return "medium"
    return "low"
