# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Offline failure extraction and clustering for the Trace Feedback Loop.

P2 of ``docs/trace-feedback-loop-design.md`` §4.4: load trace JSON written by
``TraceRail``, pull each failed step plus its neighbour context into a
``FailureEvidence``, normalise it into a hashable ``FailureSignature``, and
group repeated failures into ``FailureCluster``.

Two non-obvious invariants:

- **Failure detection uses a defensive default**: ``entry.get("success", True)
  is False or bool(entry.get("error"))``. Missing ``success`` defaults to True
  so a stale/hand-written trace without the field is not misread as all-failed.
  ``success`` is normally a trustworthy backfilled field (``trace.py`` after
  ``on_tool_exception``), but the offline tool eats historical and hand-written
  fixtures, so it cannot assume the field is present.
- **``FailureSignature`` is the only frozen dataclass**: it is a cluster key and
  must hash. Other records stay mutable and keep their ``list``/``set`` fields
  as-is (matching the design doc) — no tuple/frozenset contortions elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)

_NUM_RE = re.compile(r"[-+]?\d+\.?\d*")


@dataclass
class TraceRecord:
    """One loaded trace file: its source path and parsed JSON dict."""

    path: Path
    data: dict


@dataclass
class TraceCorpus:
    """A batch of loaded traces.

    ``traces`` holds ``TraceRecord``s (path bound to data, no parallel array to
    drift out of sync). ``root`` is metadata (parent of the first path); frames
    are referenced by absolute ``frame_path`` inside each entry, kept as-is.
    """

    root: Path
    traces: list[TraceRecord]
    frames_root: Path | None = None


@dataclass(frozen=True)
class FailureSignature:
    """Hashable identity of one failed step — the cluster key.

    ``skill_hint`` is reserved for future skill matching; first pass leaves it
    ``None`` (trace JSON records no active-skill name). ``param_bucket`` is a
    tuple of ``(field, bucket)`` pairs so the whole signature is hashable.
    """

    skill_hint: str | None
    tool_name: str
    rail_name: str | None
    kind: str | None
    reason_norm: str
    param_bucket: tuple[tuple[str, str], ...]


@dataclass
class FailureEvidence:
    """A failed step plus its neighbour context, ready for reporting."""

    trace_path: Path
    conversation_id: str
    step: int
    tool_name: str
    input_params: dict
    error: str | None
    output_summary: str
    rail_events: list[dict]
    log_events: list[dict]
    frame_path: Path | None
    before_context: list[dict] = field(default_factory=list)
    after_context: list[dict] = field(default_factory=list)


@dataclass
class FailureCluster:
    """All evidence sharing one ``FailureSignature``."""

    signature: FailureSignature
    count: int
    examples: list[FailureEvidence]
    affected_conversations: set[str]


# --------------------------------------------------------------------------- load
def load_trace_corpus(paths: list[Path], *, frames_root: Path | None = None) -> TraceCorpus:
    """Load trace JSON files, skipping unreadable ones with a warning.

    A bad file never fails the whole batch — the offline tool processes what it
    can and reports what it dropped.
    """
    records: list[TraceRecord] = []
    root = Path.cwd()
    for p in paths:
        p = Path(p)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("trace_feedback: skipping unreadable trace %s: %s", p, exc)
            continue
        if not isinstance(data, dict):
            logger.warning("trace_feedback: skipping non-dict trace %s", p)
            continue
        if not records:
            root = p.parent
        records.append(TraceRecord(path=p, data=data))
    return TraceCorpus(root=root, traces=records, frames_root=frames_root)


# ------------------------------------------------------------------------ extract
def extract_failure_evidence(
    corpus: TraceCorpus,
    *,
    context_steps: int = 2,
) -> list[FailureEvidence]:
    """Pull each failed step plus up to ``context_steps`` neighbours on each side."""
    out: list[FailureEvidence] = []
    for rec in corpus.traces:
        entries = rec.data.get("entries", [])
        if not isinstance(entries, list):
            logger.warning("trace_feedback: %s has non-list entries, skipping", rec.path)
            continue
        cid = str(rec.data.get("conversation_id", "")) or ""
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            if not _is_failed(entry):
                continue
            before = [e for e in entries[max(0, i - context_steps) : i] if isinstance(e, dict)]
            after = [e for e in entries[i + 1 : i + 1 + context_steps] if isinstance(e, dict)]
            out.append(_to_evidence(rec, entry, cid, before, after))
    return out


def _is_failed(entry: dict) -> bool:
    # Missing success defaults to True — do not misread a stale/hand-written trace.
    return entry.get("success", True) is False or bool(entry.get("error"))


def _to_evidence(
    rec: TraceRecord,
    entry: dict,
    conversation_id: str,
    before: list[dict],
    after: list[dict],
) -> FailureEvidence:
    fp = entry.get("frame_path")
    # step may be None or non-numeric in a hand-written/historical fixture;
    # default 0 rather than crash the batch (TraceRail writes 1-based ints).
    try:
        step = int(entry.get("step") or 0)
    except (TypeError, ValueError):
        step = 0
    return FailureEvidence(
        trace_path=rec.path,
        conversation_id=conversation_id,
        step=step,
        tool_name=str(entry.get("tool_name", "")),
        input_params=entry.get("input_params") or {},
        error=entry.get("error"),
        output_summary=str(entry.get("output_summary", "")),
        rail_events=list(entry.get("rail_events") or []),
        log_events=list(entry.get("log_events") or []),
        frame_path=Path(fp) if fp else None,
        before_context=before,
        after_context=after,
    )


# ---------------------------------------------------------------------- signature
def build_failure_signature(evidence: FailureEvidence) -> FailureSignature:
    """Normalise one failed step into a cluster key.

    Numbers in the reason are masked to ``<num>`` so ``z=-50 below z_floor=10``
    and ``z=-99 below z_floor=10`` collide. ``param_bucket`` buckets scalar
    params by sign + magnitude (no env/config reads — trace has no workspace or
    joint-limit info).
    """
    rail_name, kind, reason = _first_failing_rail(evidence.rail_events)
    if reason is None:
        reason = evidence.error or ""
    reason_norm = _normalize_reason(reason)
    return FailureSignature(
        skill_hint=None,
        tool_name=evidence.tool_name,
        rail_name=rail_name,
        kind=kind,
        reason_norm=reason_norm,
        param_bucket=_param_bucket(evidence.input_params),
    )


def _first_failing_rail(rail_events: list[dict]) -> tuple[str | None, str | None, str | None]:
    """The root-cause rail event, or (None, None, None) if none.

    Only ``SafetyRail/reject`` (a pre-check that blocked the call) is a failure
    *cause*. ``RecoveryRail/recover`` is a post-failure *remedy* — even when
    ``home_ok=False`` it is a failed recovery attempt, not the original cause,
    so it must not be reported as the cluster's rail_name (that would mislabel a
    tool-exception cluster as a recovery problem). ``VisualFeedback/inject_frame``
    is likewise never a cause. With no Safety reject, the cause lives in
    ``entry.error`` and the caller falls back to that.
    """
    for ev in rail_events:
        if not isinstance(ev, dict):
            continue
        if ev.get("rail_name") == "SafetyRail" and ev.get("success") is False:
            detail = ev.get("detail") or {}
            reason = detail.get("reason") if isinstance(detail, dict) else None
            return ev.get("rail_name"), ev.get("kind"), reason
    return None, None, None


def _normalize_reason(reason: str) -> str:
    return _NUM_RE.sub("<num>", reason.strip()).lower()


def _param_bucket(params: dict) -> tuple[tuple[str, str], ...]:
    if not isinstance(params, dict):
        return ()
    out: list[tuple[str, str]] = []
    motion = ("x", "y", "z", "r")
    for k in motion:
        if k in params:
            out.append((k, _bucket_scalar(params[k])))
    if "q" in params:
        out.append(("q", _bucket_q(params["q"])))
    for k in ("object_name", "target", "prompt"):
        if k in params:
            out.append((k, _bucket_str(params[k])))
    return tuple(out)


def _bucket_scalar(v: Any) -> str:
    if v is None:
        return "<none>"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "<non-numeric>"
    if f != f:  # NaN
        return "<nan>"
    if f in (float("inf"), float("-inf")):
        return "<inf>"
    if f == 0:
        sign = "zero"
    elif f < 0:
        sign = "neg"
    else:
        sign = "pos"
    a = abs(f)
    if a < 1:
        mag = "abs<1"
    elif a < 10:
        mag = "1-10"
    elif a < 100:
        mag = "10-100"
    else:
        mag = ">=100"
    return f"{sign}/{mag}"


def _bucket_q(v: Any) -> str:
    if v is None:
        return "<none>"
    if not isinstance(v, (list, tuple)):
        return "<non-list>"
    n = len(v)
    non_finite = any(not _is_finite(x) for x in v)
    if non_finite:
        return f"len={n}/non-finite"
    return f"len={n}"


def _is_finite(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def _bucket_str(v: Any) -> str:
    if v is None:
        return "<none>"
    s = str(v).strip().lower()
    if not s:
        return "<empty>"
    if len(s) > 40:
        # Deterministic hash (sha256, not builtin hash() — that is
        # per-process-randomized and would make cluster keys unstable across
        # runs). 8 hex chars is enough to disambiguate long prompts.
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]
        return f"long/sha={h}"
    return s


# ------------------------------------------------------------------------ cluster
def cluster_failures(
    evidence: list[FailureEvidence],
    *,
    min_size: int = 2,
) -> list[FailureCluster]:
    """Group evidence by ``FailureSignature``, dropping clusters below ``min_size``."""
    groups: dict[FailureSignature, list[FailureEvidence]] = {}
    for ev in evidence:
        sig = build_failure_signature(ev)
        groups.setdefault(sig, []).append(ev)
    clusters: list[FailureCluster] = []
    for sig, evs in groups.items():
        if len(evs) < min_size:
            continue
        clusters.append(
            FailureCluster(
                signature=sig,
                count=len(evs),
                examples=evs[:3],
                affected_conversations={e.conversation_id for e in evs if e.conversation_id},
            )
        )
    clusters.sort(key=lambda c: c.count, reverse=True)
    return clusters
