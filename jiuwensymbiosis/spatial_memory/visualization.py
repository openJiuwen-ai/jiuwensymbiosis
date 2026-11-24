"""Self-contained temporal 3-D evidence viewer, built from committed observations."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from jiuwensymbiosis.spatial_memory.geometry import project
from jiuwensymbiosis.spatial_memory.memory import SpatialMemory
from jiuwensymbiosis.utils.geometry import pixels_and_depths_to_camera_xyz


def _sample(points: np.ndarray, count: int) -> list:
    if points.size == 0:
        return []
    return [
        list(map(float, row))
        for row in np.round(points[np.linspace(0, len(points) - 1, min(count, len(points)), dtype=int)], 3)
    ]


def write_viewer(
    database: Path,
    output: Path,
    *,
    telemetry: Path | None = None,
    snapshot: bool = False,
) -> dict:
    """Read-only export; points are display samples, not invented completed meshes."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Viewer export requires the spatial-memory extra") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    with SpatialMemory(database, readonly=True) as memory:
        objects = memory.query_objects(limit=1000)
        indices = {item["entity_id"]: i for i, item in enumerate(objects)}
        for row in memory.connection.execute("SELECT id, receipt FROM observations ORDER BY rowid"):
            observation, detections = memory.get_observation(row[0])
            receipt = json.loads(row[1])
            jpeg = io.BytesIO()
            Image.fromarray(observation.rgb).save(jpeg, format="JPEG", quality=82)
            k, t = observation.intrinsics, observation.map_from_camera
            v, u = np.nonzero(np.isfinite(observation.depth_m))
            chosen = np.linspace(0, len(v) - 1, min(600, len(v)), dtype=int)
            v, u = v[chosen], u[chosen]
            z = observation.depth_m[v, u]
            context = pixels_and_depths_to_camera_xyz(u, v, z, k) / 1000.0
            context = context @ t[:3, :3].T + t[:3, 3]
            parts = []
            for detection, item in zip(detections.instances, receipt["assignments"], strict=True):
                if item["status"] == "excluded":
                    continue
                points, _, _ = project(observation, detection, memory.policy)
                ys, xs = np.nonzero(detection.mask)
                parts.append(
                    {
                        "object": indices.get(item["entity_id"], -1),
                        "local_id": detection.local_id,
                        "label": detection.label,
                        "score": detection.score,
                        "status": item["status"],
                        "reason": item.get("reason"),
                        "association_score": item.get("association_score"),
                        "points": _sample(points, 128),
                        "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                        "candidates": item.get("candidates", []),
                        "geometry_quality": item.get("geometry_quality"),
                    }
                )
            frames.append(
                {
                    "id": observation.observation_id,
                    "time": observation.timestamp_s,
                    "image": "data:image/jpeg;base64," + base64.b64encode(jpeg.getvalue()).decode(),
                    "camera": t.round(6).tolist(),
                    "pose_source": observation.pose_source,
                    "context": _sample(context, 600),
                    "parts": parts,
                }
            )
        data = {
            "frames": frames,
            "objects": objects,
            "memory_status": memory.get_status(),
            "trajectory": json.loads(telemetry.read_text()) if telemetry else [],
        }
    if not frames:
        raise ValueError("Cannot visualize an empty memory")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("viewer.html").read_text()
    output.write_text(template.replace("__MEMORY_DATA__", payload), encoding="utf-8")
    result = {
        "html": str(output.resolve()),
        "frames": len(frames),
        "objects": len(objects),
        "display_geometry": "sampled_observed_surfaces",
        "agent_tools_added": False,
        "source_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2))
    if snapshot:
        write_snapshot(data, output.with_suffix(".png"))
    return result


def write_snapshot(data: dict, output: Path) -> None:
    """Optional standard plotting export of the same evidence, not a browser screenshot."""
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError("PNG export requires the spatial-memory extra") from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    with plt.style.context("dark_background"):
        figure = plt.figure(figsize=(15, 7), facecolor="#0c111b")
        ax = figure.add_subplot(121, projection="3d", facecolor="#0c111b")
        plan = figure.add_subplot(122, facecolor="#0c111b")
        context = np.asarray([p for f in data["frames"] for p in f["context"]])
        if len(context):
            ax.scatter(*context.T, s=0.4, color="#728197", alpha=0.22)
        for identity in range(len(data["objects"])):
            observed_points = []
            for frame in data["frames"]:
                for part in frame["parts"]:
                    if part["object"] == identity:
                        observed_points.extend(part["points"])
            if not observed_points:
                continue
            points = np.asarray(observed_points)
            color = colors.hsv_to_rgb([((identity * 137.508 + 15) % 360) / 360, 0.55, 0.95])
            ax.scatter(*points.T, s=1.5, color=color, alpha=0.8)
            plan.scatter(points[:, 0], points[:, 1], s=1, color=color, alpha=0.35)
            center = np.median(points, axis=0)
            if len(points) >= 128:
                ax.text(*center, f"O{identity + 1:03d}", fontsize=6, color=color)
        trajectory = np.asarray([row["robot"]["position_m"] for row in data["trajectory"]])
        if len(trajectory):
            ax.plot(*trajectory.T, color="#6fe0d0", linewidth=2)
            plan.plot(trajectory[:, 0], trajectory[:, 1], color="#6fe0d0", linewidth=2, label="measured base path")
            plan.scatter(trajectory[0, 0], trajectory[0, 1], marker="o", color="white", s=50, label="start")
            plan.scatter(trajectory[-1, 0], trajectory[-1, 1], marker="x", color="#ff9a66", s=60, label="end")
            plan.legend(loc="upper right", fontsize=8)
        ax.set(xlabel="X / m", ylabel="Y / m", zlabel="Z / m", title="Observed surfaces and persistent IDs")
        ax.view_init(elev=58, azim=-65)
        ax.set_box_aspect([1, 1, 0.55])
        plan.set(
            xlabel="X / m", ylabel="Y / m", title="Top view / measured travel and observed objects", aspect="equal"
        )
        plan.grid(alpha=0.15)
        status = data["memory_status"]
        figure.suptitle(
            f"Spatial memory  |  {len(data['frames'])} RGB-D frames  |  "
            f"{len(data['objects'])} candidate entities  |  {status['appearance_kind']}",
            fontsize=14,
        )
        figure.text(
            0.02,
            0.02,
            "Display samples of observed surfaces. Model labels and identities may be wrong; "
            "hidden space remains unknown.",
            fontsize=9,
            color="#adbcd0",
        )
        figure.tight_layout(rect=(0, 0.04, 1, 0.94))
        figure.savefig(output, dpi=150, facecolor=figure.get_facecolor())
        plt.close(figure)
