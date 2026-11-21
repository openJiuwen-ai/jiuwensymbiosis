"""Compare optional appearance models on identical frozen RGB-D and YOLOE detections."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--clip-model", type=Path)
    parser.add_argument("--dinov3-model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--appearance-weight", type=float, default=0.15)
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        from jiuwensymbiosis.spatial_memory import MemoryPolicy
        from jiuwensymbiosis.spatial_memory.appearance import TransformersAppearance
        from jiuwensymbiosis.spatial_memory.artifacts import read_detections, read_observation
        from jiuwensymbiosis.spatial_memory.evaluation import evaluate_appearance, evaluate_review
        from jiuwensymbiosis.spatial_memory.geometry import project
        from jiuwensymbiosis.spatial_memory.replay import replay

    policy = MemoryPolicy(appearance_weight=args.appearance_weight)
    manifest = json.loads((args.sequence / "sequence.json").read_text())
    annotations = json.loads(args.annotations.read_text())
    for index in range(len(manifest["frames"])):
        batch = read_detections(args.detections / f"{index:03d}")
        if "appearance" in json.loads(batch.provenance_json) or any(d.embedding is not None for d in batch.instances):
            raise ValueError("Benchmark requires raw YOLOE detections, without previously attached appearance features")
    args.output.mkdir(parents=True, exist_ok=False)
    methods = [("rgb", None)]
    if args.clip_model:
        methods.append(("clip", args.clip_model))
    if args.dinov3_model:
        methods.extend((name, args.dinov3_model) for name in ["dinov3-cls", "dinov3-patch"])
    report = {
        "sequence": str(args.sequence.resolve()),
        "annotations": str(args.annotations.resolve()),
        "reviewer": annotations["reviewer"],
        "methods": {},
    }
    input_digest = None
    for kind, directory in methods:
        with contextlib.redirect_stdout(sys.stderr):
            started = time.perf_counter()
            encoder = TransformersAppearance(directory, kind=kind, device=args.device) if directory else None
            load_seconds = time.perf_counter() - started
            started = time.perf_counter()
            summary = replay(
                args.sequence, args.output / kind, frozen_detections=args.detections, appearance=encoder, policy=policy
            )
            elapsed = time.perf_counter() - started
            if input_digest is not None and summary["input_evidence_sha256"] != input_digest:
                raise RuntimeError("Benchmark methods did not use identical input evidence")
            input_digest = summary["input_evidence_sha256"]
            vectors = {}
            for index, frame in enumerate(manifest["frames"]):
                observation = read_observation(args.sequence / frame)
                batch = read_detections(args.output / kind / "detections" / f"{index:03d}")
                for detection in batch.instances:
                    vectors[index, detection.local_id] = (
                        detection.embedding
                        if detection.embedding is not None
                        else project(observation, detection, policy)[1]
                    )
            metrics = {
                "association": evaluate_review(summary, annotations),
                "appearance": evaluate_appearance(vectors, annotations),
                "input_evidence_sha256": input_digest,
                "model_load_s": load_seconds,
                "replay_with_reports_s": elapsed,
                "feature_timings": encoder.timings if encoder else [],
                "feature_provenance": encoder.provenance if encoder else {"kind": "depth-supported-rgb-histogram"},
                "entity_count": summary["status"]["entity_count"],
                "needs_view_history_count": summary["status"]["needs_view_history_count"],
            }
            report["methods"][kind] = metrics
            (args.output / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            del encoder
        # Stdout is the benchmark's JSON-lines output.
        result = {
            "method": kind,
            "association": metrics["association"]["association"],
            "appearance_top1": metrics["appearance"]["retrieval_top1"],
            "feature_s": sum(x["elapsed_s"] for x in metrics["feature_timings"]),
        }
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
