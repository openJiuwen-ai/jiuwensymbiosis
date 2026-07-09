# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Trace Feedback Loop offline analysis (P2/P3).

See ``docs/trace-feedback-loop-design.md`` §3.3 / §4.4. Loads trace JSON written
by ``TraceRail``, clusters failed steps by normalised signature, and renders
failure reports + skill patch proposals for human review.
"""

from jiuwensymbiosis.trace_feedback.analysis import (
    FailureCluster,
    FailureEvidence,
    FailureSignature,
    TraceCorpus,
    TraceRecord,
    build_failure_signature,
    cluster_failures,
    extract_failure_evidence,
    load_trace_corpus,
)
from jiuwensymbiosis.trace_feedback.patches import SkillPatchProposal, propose_skill_patches

__all__ = [
    "TraceRecord",
    "TraceCorpus",
    "FailureSignature",
    "FailureEvidence",
    "FailureCluster",
    "SkillPatchProposal",
    "load_trace_corpus",
    "extract_failure_evidence",
    "build_failure_signature",
    "cluster_failures",
    "propose_skill_patches",
]
