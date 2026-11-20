"""One ingestion path for captured sequences, model runs and frozen-output replay."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from jiuwensymbiosis.spatial_memory.appearance import AppearanceEncoder
from jiuwensymbiosis.spatial_memory.artifacts import read_detections, read_observation, write_detections
from jiuwensymbiosis.spatial_memory.geometry import MemoryPolicy
from jiuwensymbiosis.spatial_memory.memory import SpatialMemory
from jiuwensymbiosis.spatial_memory.models import Detections, canonical, discovery_fingerprint


class Discovery(Protocol):
    def detect(self, rgb: np.ndarray) -> Detections: ...


def replay(
    sequence: Path,
    output: Path,
    *,
    detector: Discovery | None = None,
    frozen_detections: Path | None = None,
    appearance: AppearanceEncoder | None = None,
    policy: MemoryPolicy | None = None,
) -> dict:
    """Write a fresh result directory. Failed inference propagates, never looks empty."""
    if (detector is None) == (frozen_detections is None):
        raise ValueError("Choose exactly one live detector or frozen detections directory")
    manifest = json.loads((sequence / "sequence.json").read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("frames"):
        raise ValueError("Expected a nonempty version-1 observation sequence")
    frames = []
    for name in manifest["frames"]:
        path = (sequence / name).resolve()
        if not path.is_relative_to(sequence.resolve()):
            raise ValueError("Frame path escapes sequence")
        frames.append(read_observation(path))
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    input_fingerprints = []
    try:
        with SpatialMemory(output / "memory.sqlite", policy=policy) as memory:
            for index, observation in enumerate(frames):
                batch = (
                    detector.detect(observation.rgb)
                    if detector is not None
                    else read_detections(cast(Path, frozen_detections) / f"{index:03d}")
                )
                if appearance is not None:
                    batch = appearance.enrich(observation.rgb, batch)
                write_detections(output / "detections" / f"{index:03d}", batch, observation.depth_m.shape)
                receipt = memory.ingest(observation, batch)
                input_fingerprints.append(discovery_fingerprint(observation, batch))
                receipts.append(receipt)
                (output / "receipts.json").write_text(canonical(receipts) + "\n")
            summary = {
                "sequence": str(sequence.resolve()),
                "sequence_manifest_sha256": hashlib.sha256((sequence / "sequence.json").read_bytes()).hexdigest(),
                "evidence_sha256": hashlib.sha256(
                    canonical([receipt["evidence_fingerprint"] for receipt in receipts]).encode()
                ).hexdigest(),
                "input_evidence_sha256": hashlib.sha256(canonical(input_fingerprints).encode()).hexdigest(),
                "status": memory.get_status(),
                "objects": memory.query_objects(limit=1000),
                "receipts": receipts,
            }
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            _report(output, frames, receipts, summary)
        with SpatialMemory(output / "memory.sqlite", readonly=True) as reopened:
            if reopened.query_objects(limit=1000) != summary["objects"]:
                raise RuntimeError("Memory restart changed query results")
        return summary
    except Exception as exc:
        (output / "failure.json").write_text(
            canonical({"error_type": type(exc).__name__, "error": str(exc), "committed_frames": len(receipts)}) + "\n"
        )
        raise


def _report(output: Path, frames: list, receipts: list, summary: dict) -> None:
    from PIL import Image, ImageDraw, ImageFont

    names = {o["entity_id"]: f"O{i + 1:03d}" for i, o in enumerate(summary["objects"])}
    panels = []
    for index, (observation, receipt) in enumerate(zip(frames, receipts, strict=True)):
        batch = read_detections(output / "detections" / f"{index:03d}")
        image = Image.fromarray(observation.rgb).convert("RGBA")
        tint = np.zeros((*observation.depth_m.shape, 4), dtype=np.uint8)
        labels = []
        rows = []
        for detection, item in zip(batch.instances, receipt["assignments"], strict=True):
            identity = names.get(item["entity_id"], item["status"])
            color = tuple(hashlib.sha256(identity.encode()).digest()[:3])
            if item["status"] != "excluded":
                tint[detection.mask] = (*color, 50)
                ys, xs = np.nonzero(detection.mask)
                labels.append(
                    (
                        (int(np.median(xs)), int(np.median(ys))),
                        f"[{detection.local_id}] {identity if item['entity_id'] else '?'}",
                        color,
                    )
                )
            quality = item.get("geometry_quality") or {}
            depth_samples = quality.get("valid_depth_pixels", "—")
            coverage = quality.get("depth_coverage_fraction")
            coverage_text = f"{coverage:.1%}" if coverage is not None else "—"
            rows.append(
                f"<tr><td>{html.escape(detection.local_id)}</td><td>{html.escape(detection.label)}</td><td>{identity}</td><td>{html.escape(item.get('reason', item['status']))}</td><td>{depth_samples} / {coverage_text}</td><td>{item['point_count']}</td></tr>"
            )
        image = Image.alpha_composite(image, Image.fromarray(tint))
        draw = ImageDraw.Draw(image)
        used_y: list[int] = []
        font = ImageFont.load_default(size=12)
        for position, label, color in labels:
            x, y = position
            while any(abs(y - prior) < 13 for prior in used_y) and y - position[1] < 28:
                y += 14
            y = min(y, image.height - 16)
            used_y.append(y)
            if y != position[1]:
                draw.line((position, (min(x, image.width - 95), y)), fill="white", width=1)
            draw.text(
                (min(x, image.width - 95), y),
                label,
                fill="white",
                font=font,
                stroke_width=1,
                stroke_fill=(*color, 255),
            )
        image.convert("RGB").save(output / f"frame-{index:03d}.jpg", quality=95)
        Image.fromarray(observation.rgb).save(output / f"raw-{index:03d}.jpg", quality=95)
        panels.append(
            f'<section><h2>View {index + 1}</h2><p>{html.escape(observation.observation_id)}</p><div class="images"><img src="raw-{index:03d}.jpg"><img src="frame-{index:03d}.jpg"></div><table><tr><th>Instance</th><th>Raw label</th><th>Identity</th><th>Decision</th><th>Depth samples / mask coverage</th><th>Surface voxels</th></tr>{"".join(rows)}</table></section>'
        )
    objects = []
    for item in summary["objects"]:
        labels_text = ", ".join(item["labels"])
        objects.append(
            f"<tr><td>{names[item['entity_id']]}<br><small>{item['entity_id']}</small></td><td>{html.escape(labels_text)}</td><td>{item['state']}</td><td>{item['observation_count']}</td><td>{[round(x, 3) for x in item['observed_center_m']]}</td></tr>"
        )
    content = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>空间记忆第一阶段实测</title>
<style>body{{font:16px/1.6 system-ui;margin:32px auto;max-width:1400px;padding:0 24px;color:#203040;background:#f4f7fa}}section{{background:white;padding:24px;margin:24px 0;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}small{{font-size:11px}}.images{{display:flex;gap:12px}}img{{width:calc(50% - 6px);object-fit:contain}}pre{{white-space:pre-wrap}}@media(max-width:700px){{.images{{display:block}}img{{width:100%}}}}</style>
<h1>空间记忆：多视角证据与实体 ID</h1><p>已处理 {len(frames)} 帧，记录 {len(objects)} 个候选实体。确认仅表示不同视角支持；不表示类别正确、物体完整或房间扫描完成。</p>
<p>历史待补看记录：{summary["status"]["needs_view_history_count"]}；最后一帧待补看：{summary["status"]["latest_observation_needs_view_count"]}。它们是逐次检测记录，不是独立物体数或已调度任务数。所有类别均为尚未核验的模型标签；深度覆盖率不表示透明表面测量正确。</p>
<p>定位来源：{html.escape(frames[0].pose_source)}；关联：三维表面重叠 + {html.escape(summary["status"].get("appearance_kind", "rgb-histogram"))}。图中 O 编号对应实体表，? 表示需要补看，方括号为该帧检测编号。完整记录见 <a href="summary.json">summary.json</a>。</p>
<section><h2>持久实体</h2><table><tr><th>ID</th><th>Label evidence</th><th>State</th><th>Views</th><th>Observed centre (m)</th></tr>{"".join(objects)}</table></section>{"".join(panels)}</html>"""
    (output / "review.html").write_text(content)
