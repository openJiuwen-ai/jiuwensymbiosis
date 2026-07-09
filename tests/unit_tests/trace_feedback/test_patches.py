# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for trace_feedback.patches — see design §7.2."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from jiuwensymbiosis.trace_feedback.analysis import (
    FailureCluster,
    FailureEvidence,
    FailureSignature,
)
from jiuwensymbiosis.trace_feedback.patches import propose_skill_patches
from tests.unit_tests.trace_feedback.conftest import recovery_event, safety_reject


def _ev(
    *,
    tool_name: str = "goto_xyzr",
    error: str = "boom",
    rail: list[dict] | None = None,
    cid: str = "conv-1",
) -> FailureEvidence:
    return FailureEvidence(
        trace_path=Path("/tmp/t.json"),
        conversation_id=cid,
        step=1,
        tool_name=tool_name,
        input_params={},
        error=error,
        output_summary="",
        rail_events=rail or [],
        log_events=[],
        frame_path=None,
        before_context=[],
        after_context=[],
    )


def _sig(**over) -> FailureSignature:
    base = FailureSignature(
        skill_hint=None,
        tool_name="goto_xyzr",
        rail_name="SafetyRail",
        kind="reject",
        reason_norm="z=<num> below z_floor=<num>",
        param_bucket=(("z", "neg/10-100"),),
    )
    return dataclasses.replace(base, **over) if over else base


def _cluster(
    count: int = 3, *, sig: FailureSignature | None = None, examples: list[FailureEvidence] | None = None
) -> FailureCluster:
    return FailureCluster(
        signature=sig or _sig(),
        count=count,
        examples=examples or [_ev()],
        affected_conversations={"c1"},
    )


class TestProposePatterns:
    def test_safety_z_floor_mentions_z_min(self):
        c = _cluster(sig=_sig(reason_norm="z=<num> below z_floor=<num>"))
        p = propose_skill_patches([c])[0]
        assert "z_min_safe" in p.proposed_diff
        assert "SafetyRail" in p.proposed_diff

    def test_safety_xy_bounds_mentions_workspace(self):
        c = _cluster(sig=_sig(reason_norm="x=<num> out of bounds [<num>, <num>]"))
        p = propose_skill_patches([c])[0]
        assert "workspace_bounds" in p.proposed_diff

    def test_safety_joint_mentions_joint_limits(self):
        c = _cluster(sig=_sig(reason_norm="q[1]=<num> out of limits [<num>, <num>]"))
        p = propose_skill_patches([c])[0]
        assert "joint_limits" in p.proposed_diff

    def test_vision_tool_mentions_disambiguation(self):
        sig = _sig(rail_name=None, kind=None, tool_name="analyze_scene", reason_norm="")
        c = _cluster(sig=sig)
        p = propose_skill_patches([c])[0]
        assert "视觉确认" in p.proposed_diff or "消歧" in p.proposed_diff

    def test_current_vision_tool_names_use_vision_template(self):
        for tool_name in ("get_grasp_info_simple", "pixel_to_base_xyz"):
            sig = _sig(rail_name=None, kind=None, tool_name=tool_name, reason_norm="")
            c = _cluster(sig=sig)
            p = propose_skill_patches([c])[0]
            assert "视觉确认" in p.proposed_diff or "消歧" in p.proposed_diff

    def test_fallback_pattern(self):
        sig = _sig(rail_name=None, kind=None, tool_name="close_gripper", reason_norm="gripper timeout")
        c = _cluster(sig=sig)
        p = propose_skill_patches([c])[0]
        assert "guard" in p.proposed_diff or "retry" in p.proposed_diff or "参数约束" in p.proposed_diff


class TestRecoveryPostProcess:
    def test_safety_cluster_with_recovery_gets_both(self):
        # Main template (SafetyRail z) + recovery post-process append.
        ev = _ev(rail=[safety_reject("z=-50 below z_floor=10"), recovery_event()])
        c = _cluster(examples=[ev])
        p = propose_skill_patches([c])[0]
        assert "z_min_safe" in p.proposed_diff
        assert "get_observation" in p.proposed_diff
        assert "失败处理" in p.proposed_diff

    def test_fallback_with_recovery_gets_recovery_hint(self):
        ev = _ev(tool_name="close_gripper", error="gripper timeout", rail=[recovery_event()])
        sig = _sig(rail_name=None, kind=None, tool_name="close_gripper", reason_norm="gripper timeout")
        c = _cluster(sig=sig, examples=[ev])
        p = propose_skill_patches([c])[0]
        assert "get_observation" in p.proposed_diff

    def test_no_recovery_no_append(self):
        ev = _ev(rail=[safety_reject("z=-50 below z_floor=10")])
        c = _cluster(examples=[ev])
        p = propose_skill_patches([c])[0]
        assert "get_observation" not in p.proposed_diff


class TestProposalFields:
    def test_target_skill_always_unresolved(self):
        c = _cluster()
        p = propose_skill_patches([c])[0]
        assert p.target_skill == "<unresolved>"

    def test_confidence_by_count(self):
        assert propose_skill_patches([_cluster(count=5)])[0].confidence == "high"
        assert propose_skill_patches([_cluster(count=3)])[0].confidence == "medium"
        assert propose_skill_patches([_cluster(count=2)])[0].confidence == "low"

    def test_risks_non_empty_and_mention_human_review(self):
        p = propose_skill_patches([_cluster()])[0]
        assert p.risks
        assert any("人审" in r for r in p.risks)

    def test_examples_capped_at_two(self):
        evs = [_ev() for _ in range(5)]
        c = _cluster(examples=evs)
        p = propose_skill_patches([c])[0]
        assert len(p.examples) == 2

    def test_evidence_signatures_has_cluster_sig(self):
        sig = _sig(reason_norm="z=<num> below z_floor=<num>")
        c = _cluster(sig=sig)
        p = propose_skill_patches([c])[0]
        assert p.evidence_signatures == [sig]

    def test_empty_clusters_empty_proposals(self):
        assert propose_skill_patches([]) == []
