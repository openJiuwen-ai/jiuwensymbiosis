"""Transactional persistent objects, with immutable frame evidence and read-only queries."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from jiuwensymbiosis.spatial_memory.geometry import (
    MemoryPolicy,
    compare_surfaces,
    different_view,
    downsample,
    project,
    surface_in_view,
)
from jiuwensymbiosis.spatial_memory.models import (
    Detection,
    Detections,
    Observation,
    canonical,
    evidence_fingerprint,
    observation_from_arrays,
    pack,
    unpack,
)
from jiuwensymbiosis.spatial_memory.view_tasks import update_tasks

SCHEMA_VERSION = 1
ASSOCIATION_VERSION = "depth-visible-redundant-mask-symmetric-surface-v7"


class SpatialMemory:
    """Single-threaded owner; SQLite serializes concurrent writers at ingestion.

    One database describes one static scene epoch / map / clock / calibration.
    Queries expose observed surfaces, never grasp poses or complete inventories.
    """

    def __init__(self, path: str | Path, *, policy: MemoryPolicy | None = None, readonly: bool = False):
        self.path = Path(path).resolve()
        self.readonly = readonly
        self.policy = policy or MemoryPolicy()
        if readonly:
            self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        try:
            if not readonly:
                self.connection.executescript("""
                    CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS observations (
                        id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, metadata TEXT NOT NULL,
                        detections TEXT NOT NULL, payload BLOB NOT NULL, receipt TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS entities (id TEXT PRIMARY KEY, record TEXT NOT NULL, points BLOB NOT NULL);
                    CREATE TABLE IF NOT EXISTS view_tasks (id TEXT PRIMARY KEY, record TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS assignments (
                        observation_id TEXT NOT NULL REFERENCES observations(id), local_id TEXT NOT NULL,
                        entity_id TEXT REFERENCES entities(id), record TEXT NOT NULL,
                        PRIMARY KEY(observation_id, local_id));
                """)
                with self.connection:
                    self.connection.execute(
                        "INSERT OR IGNORE INTO metadata VALUES ('schema', ?)", (str(SCHEMA_VERSION),)
                    )
                    self.connection.execute(
                        "INSERT OR IGNORE INTO metadata VALUES ('policy', ?)", (canonical(asdict(self.policy)),)
                    )
                    self.connection.execute(
                        "INSERT OR IGNORE INTO metadata VALUES ('association', ?)", (ASSOCIATION_VERSION,)
                    )
            association = self._meta("association")
            compatible = association == ASSOCIATION_VERSION or (
                readonly
                and association
                in {
                    "symmetric-surface-rgb-histogram-v2",
                    "frustum-symmetric-surface-rgb-histogram-v3",
                    "frustum-symmetric-surface-optional-appearance-v4",
                    "frustum-symmetric-surface-optional-appearance-v5",
                }
            )
            if self._meta("schema") != str(SCHEMA_VERSION) or not compatible:
                raise ValueError("Unsupported memory schema or association version")
            stored_policy_text = self._meta("policy")
            if stored_policy_text is None:
                raise ValueError("Missing memory policy")
            stored_policy = json.loads(stored_policy_text)
            if policy is None:
                self.policy = MemoryPolicy(**stored_policy)
            elif stored_policy != asdict(policy):
                raise ValueError("Policy changed; rebuild into a new database")
        except Exception:
            self.connection.close()
            raise

    def _meta(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SpatialMemory:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def ingest(self, observation: Observation, detections: Detections) -> dict:
        """Commit a successful inference atomically; identical retries are idempotent."""
        if self.readonly:
            raise ValueError("Cannot ingest into a read-only memory")
        if any(d.mask.shape != observation.depth_m.shape for d in detections.instances):
            raise ValueError("Detection mask must match original RGB-D dimensions")
        fingerprint = evidence_fingerprint(observation, detections)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            old = self.connection.execute(
                "SELECT fingerprint, receipt FROM observations WHERE id=?", (observation.observation_id,)
            ).fetchone()
            if old:
                if old["fingerprint"] != fingerprint:
                    raise ValueError("Observation ID conflicts with immutable evidence")
                return {**json.loads(old["receipt"]), "duplicate": True}
            scope = canonical(
                {
                    k: observation.metadata()[k]
                    for k in ("scene_epoch", "map_frame", "clock_id", "calibration_id", "pose_source")
                }
            )
            provenance = json.loads(detections.provenance_json)
            model = canonical(
                {
                    k: provenance.get(k)
                    for k in (
                        "model_id",
                        "weights_sha256",
                        "vocabulary_sha256",
                        "inference",
                        "kind_policy",
                        "ultralytics_version",
                        "appearance",
                    )
                }
            )
            for key, value in (("scope", scope), ("model", model)):
                existing = self._meta(key)
                if existing is not None and existing != value:
                    raise ValueError(f"Incompatible {key}; start a separate memory")
                self.connection.execute("INSERT OR IGNORE INTO metadata VALUES (?, ?)", (key, value))
            previous = self._meta("timestamp")
            if previous is not None and observation.timestamp_s < float(previous):
                raise ValueError("Out-of-order capture; replay observations in capture order")
            self.connection.execute("DELETE FROM metadata WHERE key='scan_report'")
            receipt = self._associate(observation, detections)
            receipt["evidence_fingerprint"] = fingerprint
            arrays = observation.arrays()
            arrays["masks"] = (
                np.stack([d.mask for d in detections.instances])
                if detections.instances
                else np.empty((0, *observation.depth_m.shape), dtype=bool)
            )
            detection_metadata = canonical(
                {"instances": [d.metadata() for d in detections.instances], "provenance": provenance}
            )
            self.connection.execute(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)",
                (
                    observation.observation_id,
                    fingerprint,
                    canonical(observation.metadata()),
                    detection_metadata,
                    pack(**arrays),
                    canonical(receipt),
                ),
            )
            for assignment in receipt["assignments"]:
                self.connection.execute(
                    "INSERT INTO assignments VALUES (?, ?, ?, ?)",
                    (
                        observation.observation_id,
                        assignment["local_id"],
                        assignment["entity_id"],
                        canonical(assignment),
                    ),
                )
            self.connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('timestamp', ?)", (str(observation.timestamp_s),)
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('last_observation', ?)", (observation.observation_id,)
            )
            return receipt

    def _associate(self, observation: Observation, detections: Detections) -> dict:
        entities = [
            {"record": json.loads(row["record"]), "points": unpack(row["points"])["points"]}
            for row in self.connection.execute("SELECT * FROM entities ORDER BY id")
        ]
        projected: list[tuple[np.ndarray, np.ndarray, dict | None]] = [
            project(observation, d, self.policy) if d.kind == "object" else (np.empty((0, 3)), np.zeros(24), None)
            for d in detections.instances
        ]
        projected = [
            (points, detection.embedding if detection.embedding is not None else histogram, quality)
            for detection, (points, histogram, quality) in zip(detections.instances, projected, strict=True)
        ]
        scores = np.zeros((len(projected), len(entities)))
        visible = [
            surface_in_view(entity["points"], observation, occlusion_margin_m=self.policy.surface_distance_m)
            for entity in entities
        ]
        comparisons: list[dict[int, dict]] = [{} for _ in projected]
        reasons: dict[int, str] = {}
        for i, (points, histogram, quality) in enumerate(projected):
            if detections.instances[i].kind != "object":
                reasons[i] = "non_object"
            elif len(points) < self.policy.min_points:
                reasons[i] = (
                    "insufficient_surface_support"
                    if quality is not None and quality["valid_depth_pixels"] >= self.policy.min_points
                    else "insufficient_depth"
                )
            else:
                for j, entity in enumerate(entities):
                    if not len(visible[j]):
                        continue
                    comparison = compare_surfaces(
                        points, histogram, visible[j], entity["record"]["appearance_gallery"], self.policy
                    )
                    comparison.update(
                        entity_point_count=len(entity["points"]), visible_entity_point_count=len(visible[j])
                    )
                    scores[i, j] = comparison["score"]
                    comparisons[i][j] = comparison
        # Near-identical masks share one association attempt, never two votes.
        # Use a direct representative comparison (not transitive union-find), so
        # a chain of overlaps cannot merge distinct endpoints. Labels are evidence,
        # not identity; raw detections remain intact in the immutable observation.
        aliases: dict[int, int] = {}
        representatives: list[int] = []
        for i in sorted(
            range(len(projected)), key=lambda i: (-detections.instances[i].score, detections.instances[i].local_id)
        ):
            item = detections.instances[i]
            if item.kind != "object" or i in reasons:
                continue
            for j in representatives:
                prior = detections.instances[j]
                union = np.count_nonzero(item.mask | prior.mask)
                if union and np.count_nonzero(item.mask & prior.mask) / union >= 0.90:
                    aliases[i] = j
                    reasons[i] = "redundant_mask"
                    break
            else:
                representatives.append(i)
        # Near-duplicate masks are not independent object evidence. Keep both raw outputs.
        for i, first in enumerate(detections.instances):
            if first.kind != "object" or i in aliases:
                continue
            for j in range(i):
                second = detections.instances[j]
                if second.kind == "object" and j not in aliases:
                    intersection = np.count_nonzero(first.mask & second.mask)
                    union = np.count_nonzero(first.mask | second.mask)
                    if intersection / union > 0.65:
                        reasons[i] = reasons[j] = "overlapping_masks"
        for i in range(len(projected)):
            ranked = np.sort(scores[i])[::-1]
            if len(ranked) > 1 and ranked[1] > 0 and ranked[0] - ranked[1] < self.policy.ambiguity_margin:
                reasons.setdefault(i, "ambiguous_identity")
            elif len(ranked) and 0 < ranked[0] < self.policy.min_score:
                reasons.setdefault(i, "weak_identity")
        # Competing detections for one entity can be over-segmentation. Don't pick a winner by row order.
        for j in range(len(entities)):
            contenders = [
                i for i in range(len(projected)) if i not in reasons and scores[i, j] >= self.policy.min_score
            ]
            if len(contenders) > 1:
                best = sorted(contenders, key=lambda i: scores[i, j], reverse=True)
                if scores[best[0], j] - scores[best[1], j] < self.policy.ambiguity_margin:
                    for i in contenders:
                        reasons[i] = "competing_instances"
        # Each detection has its own zero-score rejection slot, so matching is never forced.
        augmented = np.column_stack((scores, np.zeros((len(projected), len(projected)))))
        for i in reasons:
            augmented[i] = 0
        matches = dict(zip(*linear_sum_assignment(-augmented), strict=True)) if len(projected) else {}
        assignments = []
        for i, detection in enumerate(detections.instances):
            points, histogram, quality = projected[i]
            candidates = [
                {"entity_id": entities[j]["record"]["entity_id"], **comparisons[i][j]}
                for j in np.argsort(-scores[i])
                if scores[i, j] > 0
            ][:5]
            rejected = [
                {"entity_id": entities[j]["record"]["entity_id"], **comparison}
                for j, comparison in sorted(comparisons[i].items(), key=lambda pair: pair[1]["bbox_gap_m"])
                if comparison.get("rejection") == "insufficient_surface_overlap"
            ][:5]
            item = {
                **detection.metadata(),
                "entity_id": None,
                "candidates": candidates,
                "rejected_candidates": rejected,
                "point_count": len(points),
                "geometry_quality": quality,
            }
            # Full vectors remain in immutable detections; receipts are compact decision evidence.
            item.pop("embedding", None)
            if i in reasons:
                item.update(status="excluded" if reasons[i] == "non_object" else "needs_view", reason=reasons[i])
            else:
                j = matches.get(i, len(entities))
                if j < len(entities) and scores[i, j] >= self.policy.min_score:
                    record = entities[j]["record"]
                    merged = downsample(np.concatenate((entities[j]["points"], points)), self.policy)
                    if different_view(record["first_camera_pose"], observation.map_from_camera, self.policy):
                        record["state"] = "confirmed"
                    record["observation_count"] += 1
                    record["labels"][detection.label] = record["labels"].get(detection.label, 0) + 1
                    record["appearance_gallery"] = (record["appearance_gallery"] + [histogram.tolist()])[-8:]
                    item.update(status="matched", association_score=float(scores[i, j]))
                elif candidates:
                    item.update(status="needs_view", reason="assignment_conflict")
                    assignments.append(item)
                    continue
                else:
                    entity_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            canonical([observation.scene_epoch, observation.observation_id, detection.local_id]),
                        )
                    )
                    record = {
                        "entity_id": entity_id,
                        "state": "provisional",
                        "labels": {detection.label: 1},
                        "observation_count": 1,
                        "first_observation": observation.observation_id,
                        "first_camera_pose": observation.map_from_camera.tolist(),
                        "appearance_gallery": [histogram.tolist()],
                        "geometry_kind": "observed_surface",
                        "pose_source": observation.pose_source,
                        "map_frame": observation.map_frame,
                        "scene_epoch": observation.scene_epoch,
                        "clock_id": observation.clock_id,
                    }
                    merged = points
                    item.update(status="created")
                record.update(
                    last_observation=observation.observation_id,
                    last_seen_s=observation.timestamp_s,
                    bounds_m=[merged.min(0).tolist(), merged.max(0).tolist()],
                    observed_center_m=np.median(merged, axis=0).tolist(),
                    point_count=len(merged),
                )
                item["entity_id"] = record["entity_id"]
                self.connection.execute(
                    "INSERT INTO entities VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET record=excluded.record, points=excluded.points",
                    (record["entity_id"], canonical(record), pack(points=merged)),
                )
            assignments.append(item)
        by_id = {item["local_id"]: item for item in assignments}
        for i, j in aliases.items():
            item = by_id[detections.instances[i].local_id]
            representative = by_id[detections.instances[j].local_id]
            item.update(
                status="duplicate_detection",
                reason="redundant_mask",
                duplicate_of=representative["local_id"],
                entity_id=representative["entity_id"],
            )
        tasks = update_tasks(
            self.get_view_tasks(include_surfaces=True), observation, assignments, projected, self.policy
        )
        for task in tasks:
            self.connection.execute(
                "INSERT INTO view_tasks VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET record=excluded.record",
                (task["task_id"], canonical(task)),
            )
        return {
            "observation_id": observation.observation_id,
            "duplicate": False,
            "assignments": assignments,
            "needs_view": [a for a in assignments if a["status"] == "needs_view"],
            "inventory_complete": False,
        }

    def get_view_tasks(self, *, include_surfaces: bool = False) -> list[dict]:
        """Return current revisit state, separately from immutable per-frame failures."""
        if not self.connection.execute("SELECT 1 FROM sqlite_master WHERE name='view_tasks'").fetchone():
            return []
        tasks = [json.loads(row[0]) for row in self.connection.execute("SELECT record FROM view_tasks ORDER BY rowid")]
        if not include_surfaces:
            for task in tasks:
                task.pop("surface_m", None)
                task.pop("last_attempt_pose", None)
        return tasks

    def query_objects(
        self,
        *,
        label: str | None = None,
        near: tuple[float, float, float] | None = None,
        radius_m: float = 1.0,
        limit: int = 100,
    ) -> list[dict]:
        """Case-insensitive label substring and distance to the observed surface centre."""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be in [1, 1000]")
        if near is not None and (
            len(near) != 3 or not np.isfinite(near).all() or not np.isfinite(radius_m) or radius_m <= 0
        ):
            raise ValueError("near requires three finite metre coordinates and positive radius_m")
        results = []
        for row in self.connection.execute("SELECT record FROM entities ORDER BY id"):
            record: dict = json.loads(row[0])
            if label and not any(label.casefold() in name.casefold() for name in record["labels"]):
                continue
            if near is not None:
                distance = float(np.linalg.norm(np.asarray(record["observed_center_m"]) - near))
                if distance > radius_m:
                    continue
                record["distance_m"] = distance
            for key in ("appearance_gallery", "first_camera_pose"):
                record.pop(key)
            record["location_semantics"] = "observed surface estimate; not a grasp or navigation target"
            record["semantic_status"] = "unverified_model_labels"
            results.append(record)
        if near is not None:
            results.sort(key=lambda r: (r["distance_m"], r["entity_id"]))
        return results[:limit]

    def get_entity(self, entity_id: str) -> dict | None:
        row = self.connection.execute("SELECT record FROM entities WHERE id=?", (entity_id,)).fetchone()
        if row is None:
            return None
        record: dict = json.loads(row[0])
        record["semantic_status"] = "unverified_model_labels"
        record["location_semantics"] = "observed surface estimate; not a grasp or navigation target"
        for key in ("appearance_gallery", "first_camera_pose"):
            record.pop(key)
        record["evidence"] = [
            {"observation_id": row[0], **json.loads(row[1])}
            for row in self.connection.execute(
                "SELECT observation_id, record FROM assignments WHERE entity_id=? ORDER BY rowid", (entity_id,)
            )
        ]
        return record

    def get_observation(self, observation_id: str) -> tuple[Observation, Detections]:
        row = self.connection.execute("SELECT * FROM observations WHERE id=?", (observation_id,)).fetchone()
        if row is None:
            raise KeyError(observation_id)
        arrays = unpack(row["payload"])
        masks = arrays.pop("masks")
        detection_metadata = json.loads(row["detections"])
        return (
            observation_from_arrays(json.loads(row["metadata"]), arrays),
            Detections(
                tuple(
                    Detection(**meta, mask=mask)
                    for meta, mask in zip(detection_metadata["instances"], masks, strict=True)
                ),
                canonical(detection_metadata["provenance"]),
            ),
        )

    def record_scan_report(self, report: dict) -> None:
        """Store the scan assessment separately from immutable observation evidence."""
        if self.readonly:
            raise ValueError("Cannot record a scan in read-only memory")
        retained = {k: v for k, v in report.items() if k != "memory_status"}
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO metadata VALUES ('scan_report', ?)", (canonical(retained),))

    def get_status(self) -> dict:
        appearance = json.loads(self._meta("model") or "{}").get("appearance") or {}
        needs = [
            {"observation_id": row[0], **json.loads(row[1])}
            for row in self.connection.execute("SELECT observation_id, record FROM assignments ORDER BY rowid")
            if json.loads(row[1])["status"] == "needs_view"
        ]
        last_observation = self._meta("last_observation")
        latest_needs = [item for item in needs if item["observation_id"] == last_observation]
        tasks = self.get_view_tasks()
        return {
            "schema_version": SCHEMA_VERSION,
            "association_version": self._meta("association"),
            "appearance_kind": appearance.get("kind", "rgb-histogram"),
            "scope": json.loads(self._meta("scope") or "null"),
            "policy": asdict(self.policy),
            "observation_count": self.connection.execute("SELECT count(*) FROM observations").fetchone()[0],
            "entity_count": self.connection.execute("SELECT count(*) FROM entities").fetchone()[0],
            "last_observation": last_observation,
            "needs_view": needs,
            "needs_view_semantics": "historical_per_detection_records_not_unique_objects_or_open_tasks",
            "needs_view_history_count": len(needs),
            "view_tasks": tasks,
            "view_task_counts": {
                state: sum(task["state"] == state for task in tasks) for state in ("open", "resolved", "deferred")
            },
            "needs_view_history_by_reason": dict(Counter(item["reason"] for item in needs)),
            "latest_observation_needs_view": latest_needs,
            "latest_observation_needs_view_count": len(latest_needs),
            "confirmed_semantics": "multi_view_identity_support_not_semantic_accuracy_or_complete_geometry",
            "scan_assessment": json.loads(self._meta("scan_report") or "null"),
            "scan_complete": None,
            "scan_completion_reason": "stage1_does_not_evaluate_room_coverage",
        }
