#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""jiuwensymbiosis 离线 trace 失败分析与 skill 补丁建议工具.

把 ``TraceRail`` 落盘的 trace JSON 批量加载、抽取失败 step、按归一化签名
聚类，产出失败报告和（人审用）SKILL.md 补丁建议。所有输出写入 ``--out``
目录，不修改任何源文件。

用法::

    # 分析整个 trace 目录
    python scripts/analyze_traces.py --trace-dir ~/.jiuwensymbiosis/piper_workspace/traces \
        --out reports/trace_feedback/latest --min-cluster-size 3

    # 分析单条 trace（调试用）
    python scripts/analyze_traces.py --trace path/to/one_trace.json

输出文件（``--out`` 目录下）:

  - ``failure_clusters.json`` — 机器可读的聚类结果
  - ``failure_report.md`` — 人读的失败报告
  - ``skill_patch_proposals.md`` — 人审用 SKILL.md 补丁建议

退出码:

  - 0: 正常完成。包括「有合法 trace 但没有失败 step」——输出空报告，不是错误。
  - 1: 输入错误——路径不存在、未给 trace 来源、或加载后无合法 trace
    （含目录里有 json 但全部损坏的情况，比输出空报告诚实）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Proxy hygiene is handled at package import time (jiuwensymbiosis/__init__.py
# clears proxy env before pulling openjiuwen). See AGENTS.md "Proxy Hygiene".
from jiuwensymbiosis.trace_feedback import (
    cluster_failures,
    extract_failure_evidence,
    load_trace_corpus,
    propose_skill_patches,
)
from jiuwensymbiosis.trace_feedback.report import (
    render_clusters_json,
    render_failure_report,
    render_patch_proposals,
)
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger("analyze_traces")

_DEFAULT_OUT = Path("reports/trace_feedback/latest")


def run(
    trace_paths: list[Path],
    *,
    out_dir: Path = _DEFAULT_OUT,
    min_cluster_size: int = 3,
    context_steps: int = 2,
) -> int:
    """Run the full P2+P3 analysis pipeline and write reports.

    Returns a process exit code (0 success, 1 no valid traces). Pure I/O at the
    edges (read JSON, write reports); analysis is delegated to ``trace_feedback``.
    """
    if not trace_paths:
        logger.error("未提供任何 trace 路径")
        return 1
    corpus = load_trace_corpus(trace_paths)
    if not corpus.traces:
        logger.error("加载后无合法 trace（路径不存在或全部损坏）")
        return 1
    evidence = extract_failure_evidence(corpus, context_steps=context_steps)
    clusters = cluster_failures(evidence, min_size=min_cluster_size)
    proposals = propose_skill_patches(clusters)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "failure_clusters.json").write_text(render_clusters_json(clusters), encoding="utf-8")
    (out_dir / "failure_report.md").write_text(render_failure_report(clusters, corpus=corpus), encoding="utf-8")
    (out_dir / "skill_patch_proposals.md").write_text(render_patch_proposals(proposals), encoding="utf-8")

    n_fail = sum(c.count for c in clusters)
    logger.info(
        "分析完成: %d traces, %d failures, %d clusters, %d proposals → %s",
        len(corpus.traces),
        n_fail,
        len(clusters),
        len(proposals),
        out_dir,
    )
    return 0


def _resolve_paths(args: argparse.Namespace) -> list[Path]:
    if args.trace:
        return [Path(args.trace)]
    if args.trace_dir:
        return sorted(Path(args.trace_dir).glob("*.json"))
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="analyze_traces",
        description="离线分析 jiuwensymbiosis trace，产出失败聚类报告和 skill 补丁建议。",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace-dir", help="trace JSON 目录（递归不展开，只取顶层 *.json）")
    src.add_argument("--trace", help="单条 trace JSON 文件（调试用）")
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help=f"输出目录（默认 {_DEFAULT_OUT}）")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="聚类最小样本数（默认 3）")
    parser.add_argument("--context-steps", type=int, default=2, help="失败 step 前后保留的上下文步数（默认 2）")
    args = parser.parse_args()

    paths = _resolve_paths(args)
    if not paths:
        logger.error("未找到任何 trace 文件（--trace-dir 下无 *.json 或 --trace 路径不存在）")
        return 1
    if args.trace and not Path(args.trace).is_file():
        logger.error("--trace 指定的文件不存在: %s", args.trace)
        return 1
    return run(
        paths,
        out_dir=Path(args.out),
        min_cluster_size=args.min_cluster_size,
        context_steps=args.context_steps,
    )


if __name__ == "__main__":
    sys.exit(main())
