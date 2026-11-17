"""Local replay, evidence visualization and read-only CLI inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jiuwensymbiosis.spatial_memory import SpatialMemory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Record head-camera RGB-D from the resident Genesis preview")
    capture.add_argument("output", type=Path)
    capture.add_argument("--map-frame", required=True)
    capture.add_argument("--frames", type=int, default=10)
    capture.add_argument("--interval", type=float, default=0.5)
    capture.add_argument("--depth-stride", type=int, default=4)
    capture.add_argument("--allow-motion", action="store_true")
    capture.add_argument("--state-dir", type=Path)
    scan = commands.add_parser("scan", help="Bounded Rule-only scan using a resident Genesis RobotSession")
    scan.add_argument("output", type=Path)
    scan.add_argument("--map-frame", required=True)
    scan.add_argument("--weights", type=Path, required=True)
    scan.add_argument("--sha256", required=True)
    scan.add_argument("--device", default="0")
    scan.add_argument("--state-dir", type=Path)
    scan.add_argument("--max-frames", type=int, default=16)
    scan.add_argument("--max-seconds", type=float, default=600)
    scan.add_argument("--max-distance", type=float, default=2)
    scan.add_argument("--depth-stride", type=int, default=4)
    build = commands.add_parser("replay")
    build.add_argument("sequence", type=Path)
    build.add_argument("output", type=Path)
    mode = build.add_mutually_exclusive_group(required=True)
    mode.add_argument("--weights", type=Path)
    mode.add_argument("--detections", type=Path)
    build.add_argument("--sha256")
    build.add_argument("--device", default="0")
    build.add_argument("--appearance", choices=["clip", "dinov3-cls", "dinov3-patch"])
    build.add_argument("--appearance-model", type=Path)
    build.add_argument("--appearance-sha256")
    build.add_argument("--appearance-weight", type=float, default=0.15)
    evaluate = commands.add_parser("evaluate", help="Evaluate a separately annotated detection subset")
    evaluate.add_argument("summary", type=Path)
    evaluate.add_argument("annotations", type=Path)
    visualize = commands.add_parser("visualize", help="Export a self-contained 3-D evidence viewer")
    visualize.add_argument("database", type=Path)
    visualize.add_argument("output", type=Path)
    visualize.add_argument("--sequence", type=Path)
    visualize.add_argument("--telemetry", type=Path)
    visualize.add_argument("--scan-report", type=Path)
    visualize.add_argument("--navigation", type=Path)
    visualize.add_argument("--snapshot", action="store_true", help="Also export a static plot (requires matplotlib)")
    for name in ("query", "entity", "status"):
        command = commands.add_parser(name)
        command.add_argument("database", type=Path)
        if name == "query":
            command.add_argument("--label")
            command.add_argument("--near", type=float, nargs=3)
            command.add_argument("--radius", type=float, default=1.0)
        if name == "entity":
            command.add_argument("entity_id")
    args = parser.parse_args()
    result: object
    if args.command == "scan":
        from jiuwensymbiosis.simulators.genesis.memory_scan_cli import scan_resident

        result = scan_resident(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if result["stop_reason"] in {"failed", "stop_unconfirmed"}:
            raise SystemExit(1)
        if result["stop_reason"] in {"blocked", "cancelled"}:
            raise SystemExit(2)
        return
    if args.command == "capture":
        from jiuwensymbiosis.simulators.common.resident import ResidentPreview
        from jiuwensymbiosis.simulators.genesis.spatial_capture import capture_sequence

        preview = ResidentPreview(state_dir=args.state_dir)
        try:
            preview.connect()
            result = capture_sequence(
                preview,
                args.output,
                map_frame=args.map_frame,
                frames=args.frames,
                interval_s=args.interval,
                depth_stride=args.depth_stride,
                allow_motion=args.allow_motion,
            )
        finally:
            preview.close()
    elif args.command == "replay":
        from jiuwensymbiosis.spatial_memory.geometry import MemoryPolicy
        from jiuwensymbiosis.spatial_memory.replay import replay

        detector = None
        if args.weights:
            if not args.sha256:
                parser.error("--weights requires the verified --sha256")
            from jiuwensymbiosis.spatial_memory.yoloe import YoloEPromptFree

            detector = YoloEPromptFree(args.weights, args.sha256, device=args.device)
        appearance = None
        if args.appearance:
            if not args.appearance_model:
                parser.error("--appearance requires --appearance-model")
            from jiuwensymbiosis.spatial_memory.appearance import TransformersAppearance

            appearance = TransformersAppearance(
                args.appearance_model, kind=args.appearance, device=args.device, expected_sha256=args.appearance_sha256
            )
        summary = replay(
            args.sequence,
            args.output,
            detector=detector,
            frozen_detections=args.detections,
            appearance=appearance,
            policy=MemoryPolicy(appearance_weight=args.appearance_weight),
        )
        result = {"output": str(args.output.resolve()), "status": summary["status"]}
    elif args.command == "evaluate":
        from jiuwensymbiosis.spatial_memory.evaluation import evaluate_review

        result = evaluate_review(json.loads(args.summary.read_text()), json.loads(args.annotations.read_text()))
    elif args.command == "visualize":
        from jiuwensymbiosis.spatial_memory.visualization import write_viewer

        result = write_viewer(
            args.database,
            args.output,
            sequence=args.sequence,
            telemetry=args.telemetry,
            scan_report=args.scan_report,
            navigation=args.navigation,
            snapshot=args.snapshot,
        )
    else:
        with SpatialMemory(args.database, readonly=True) as memory:
            if args.command == "query":
                result = memory.query_objects(label=args.label, near=args.near, radius_m=args.radius)
            elif args.command == "entity":
                result = memory.get_entity(args.entity_id)
            else:
                result = memory.get_status()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
