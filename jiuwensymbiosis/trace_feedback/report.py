# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Render failure clusters and patch proposals as JSON / markdown.

P2/P3 report output of ``docs/trace-feedback-loop-design.md`` §4.3. All renderers
are pure functions over already-built data — no re-analysis, no file reads — so
they can be snapshotted or piped to any writer.
"""

from __future__ import annotations

import json
from typing import Any

from jiuwensymbiosis.trace_feedback.analysis import (
    FailureCluster,
    FailureEvidence,
    FailureSignature,
    TraceCorpus,
)


def render_clusters_json(clusters: list[FailureCluster]) -> str:
    """JSON serialization of clusters (round-trippable via ``json.loads``)."""
    return json.dumps({"clusters": [_cluster_to_dict(c) for c in clusters]}, ensure_ascii=False, indent=2)


def _cluster_to_dict(c: FailureCluster) -> dict:
    return {
        "signature": _signature_to_dict(c.signature),
        "count": c.count,
        "examples": [_evidence_to_dict(e) for e in c.examples],
        "affected_conversations": sorted(c.affected_conversations),
    }


def _signature_to_dict(s: FailureSignature) -> dict:
    return {
        "skill_hint": s.skill_hint,
        "tool_name": s.tool_name,
        "rail_name": s.rail_name,
        "kind": s.kind,
        "reason_norm": s.reason_norm,
        "param_bucket": [list(pair) for pair in s.param_bucket],
    }


def _evidence_to_dict(e: FailureEvidence) -> dict:
    return {
        "trace_path": str(e.trace_path),
        "conversation_id": e.conversation_id,
        "step": e.step,
        "tool_name": e.tool_name,
        "error": e.error,
        "input_params": e.input_params,
        "rail_events": e.rail_events,
    }


def render_failure_report(
    clusters: list[FailureCluster],
    *,
    corpus: TraceCorpus | None = None,
) -> str:
    """Markdown report: overview + one section per cluster.

    Empty input still renders an overview line (honest "0 failures" rather than
    a blank page).
    """
    n_traces = len(corpus.traces) if corpus is not None else 0
    n_failures = sum(c.count for c in clusters)
    lines = [
        "# Trace Failure Report",
        "",
        f"- traces analyzed: {n_traces}",
        f"- failed steps clustered: {n_failures}",
        f"- clusters: {len(clusters)}",
        "",
    ]
    if not clusters:
        lines.append("_(no recurring failure clusters above min size)_")
        return "\n".join(lines)
    for i, c in enumerate(clusters, 1):
        lines.extend(_render_cluster_section(i, c))
    return "\n".join(lines)


def _render_cluster_section(idx: int, c: FailureCluster) -> list[str]:
    sig = c.signature
    rail = sig.rail_name or "tool failure"
    kind = sig.kind or ""
    header = f"## Cluster {idx} — {sig.tool_name} / {rail}"
    if kind:
        header += f" / {kind}"
    lines = [
        header,
        "",
        f"- count: **{c.count}**",
        f"- affected conversations: {sorted(c.affected_conversations) or '—'}",
        f"- reason (normalised): `{sig.reason_norm or '—'}`",
        f"- param bucket: `{_fmt_bucket(sig.param_bucket)}`",
        "",
    ]
    for ex in c.examples:
        lines.append(f"- **{ex.trace_path.name}:step {ex.step}** — {ex.tool_name}")
        if ex.error:
            lines.append(f"  - error: `{ex.error}`")
        for ev in ex.rail_events:
            if isinstance(ev, dict):
                lines.append(f"  - rail: {ev.get('rail_name')}/{ev.get('kind')} {ev.get('detail', {})}")
    lines.append("")
    return lines


def _fmt_bucket(bucket: tuple[tuple[str, str], ...]) -> str:
    if not bucket:
        return "—"
    return ", ".join(f"{k}={v}" for k, v in bucket)


def render_patch_proposals(proposals: list[Any]) -> str:
    """Markdown rendering of SkillPatchProposal list (P3).

    Accepts the ``SkillPatchProposal`` dataclass from ``patches`` via duck typing
    to avoid an import cycle (patches imports report for nothing, report stays
    independent).
    """
    if not proposals:
        return "# Skill Patch Proposals\n\n_(no proposals — no clusters above min size)_\n"
    lines = ["# Skill Patch Proposals", ""]
    for i, p in enumerate(proposals, 1):
        lines.extend(_render_proposal_section(i, p))
    return "\n".join(lines)


def _render_proposal_section(idx: int, p: Any) -> list[str]:
    sigs = p.evidence_signatures
    sig_lines = ", ".join(f"`{s.tool_name}/{s.rail_name or '—'}/{s.kind or '—'}`" for s in sigs) or "—"
    lines = [
        f"## Proposal {idx} — target: `{p.target_skill}`",
        "",
        f"- confidence: **{p.confidence}**",
        f"- summary: {p.summary}",
        "",
        "### Proposed diff (human review required)",
        "",
        "```",
        p.proposed_diff,
        "```",
        "",
        f"- evidence signatures: {sig_lines}",
    ]
    for ex in p.examples:
        lines.append(f"- example: `{ex.trace_path.name}:step {ex.step}` — {ex.tool_name}: {ex.error or '—'}")
    if p.risks:
        lines.append("")
        lines.append("### Risks")
        for r in p.risks:
            lines.append(f"- {r}")
    if p.validation_suggestions:
        lines.append("")
        lines.append("### Validation suggestions")
        for v in p.validation_suggestions:
            lines.append(f"- {v}")
    lines.append("")
    return lines
