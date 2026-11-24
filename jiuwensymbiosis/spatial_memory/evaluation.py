"""Offline review of annotated detections; annotations never enter association."""

from __future__ import annotations

from collections import Counter

import numpy as np


def evaluate_review(summary: dict, annotations: dict) -> dict:
    """Score selected same/different pairs and raw labels, not full-room recall.

    References address a receipt's zero-based frame index and original local ID.
    Unresolved pairs stay in the denominator; repeated pairs cannot inflate scores.
    """
    if annotations.get("schema_version") != 1 or not annotations.get("reviewer"):
        raise ValueError("Require version-1 annotations with a named reviewer")
    if annotations.get("sequence_manifest_sha256") != summary.get("sequence_manifest_sha256"):
        raise ValueError("Annotations belong to a different observation sequence")
    accepted_hashes = {
        value for value in (summary.get("evidence_sha256"), summary.get("input_evidence_sha256")) if value
    }
    if not summary.get("evidence_sha256") or annotations.get("evidence_sha256") not in accepted_hashes:
        raise ValueError("Annotations belong to different observation/detection evidence")
    lookup = {
        (index, item["local_id"]): item
        for index, receipt in enumerate(summary["receipts"])
        for item in receipt["assignments"]
    }

    def resolve(reference: dict) -> tuple[tuple[int, str], dict]:
        frame, local_id = reference.get("frame"), reference.get("local_id")
        if type(frame) is not int or not isinstance(local_id, str):
            raise ValueError("Review reference requires integer frame and string local_id")
        key = frame, local_id
        if key not in lookup or lookup[key]["kind"] != "object":
            raise ValueError(f"Review reference is not an object detection: {key}")
        return key, lookup[key]

    pairs, labels = [], []
    seen_pairs: set = set()
    seen_labels: set = set()
    for row in annotations.get("pairs", []):
        first_key, first = resolve(row["first"])
        second_key, second = resolve(row["second"])
        pair_key = tuple(sorted((first_key, second_key)))
        if first_key == second_key or pair_key in seen_pairs or row["relation"] not in {"same", "different"}:
            raise ValueError("Review pairs must be distinct, unique and same/different")
        seen_pairs.add(pair_key)
        a, b = first["entity_id"], second["entity_id"]
        if a is None or b is None:
            verdict = "unresolved"
        elif row["relation"] == "same":
            verdict = "correct" if a == b else "fragmented"
        else:
            verdict = "correct" if a != b else "false_merge"
        pairs.append({**row, "first_entity_id": a, "second_entity_id": b, "verdict": verdict})
    for row in annotations.get("labels", []):
        key, detection = resolve(row)
        accepted = row.get("acceptable_labels")
        if key in seen_labels or not isinstance(accepted, list) or not accepted:
            raise ValueError("Label reviews must be unique and provide acceptable_labels")
        if any(not isinstance(label, str) or not label.strip() for label in accepted):
            raise ValueError("Acceptable labels must be nonempty strings")
        seen_labels.add(key)
        correct = detection["label"].casefold() in {label.casefold() for label in accepted}
        labels.append({**row, "observed_label": detection["label"], "correct": correct})
    if not pairs and not labels:
        raise ValueError("Review contains no annotations")
    true_positive = sum(row["relation"] == "same" and row["verdict"] == "correct" for row in pairs)
    false_positive = sum(row["verdict"] == "false_merge" for row in pairs)
    false_negative = sum(row["relation"] == "same" and row["verdict"] != "correct" for row in pairs)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    f1 = (
        2 * true_positive / (2 * true_positive + false_positive + false_negative)
        if pairs and 2 * true_positive + false_positive + false_negative
        else None
    )
    return {
        "reviewer": annotations["reviewer"],
        "sequence_manifest_sha256": summary["sequence_manifest_sha256"],
        "evidence_sha256": summary["evidence_sha256"],
        "scope": "annotated_detection_subset_not_inventory_precision_recall_or_room_completion",
        "pair_count": len(pairs),
        "association": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "unresolved_count": sum(row["verdict"] == "unresolved" for row in pairs),
            "same_pairs": sum(row["relation"] == "same" for row in pairs),
            "different_pairs": sum(row["relation"] == "different" for row in pairs),
        },
        "pair_outcomes": dict(Counter(row["verdict"] for row in pairs)),
        "pair_correct_fraction": sum(row["verdict"] == "correct" for row in pairs) / len(pairs) if pairs else None,
        "label_count": len(labels),
        "label_correct_count": sum(row["correct"] for row in labels),
        "pairs": pairs,
        "labels": labels,
    }


def evaluate_appearance(vectors: dict[tuple[int, str], np.ndarray], annotations: dict) -> dict:
    """Threshold-free pair AUC and retrieval with explicitly annotated negative identities.

    Same-pair components supply identity groups; unconnected groups are NOT
    presumed different. Only annotated different-group edges supply negatives.
    Earliest observations form the gallery and later observations are queries.
    """
    parent: dict[tuple[int, str], tuple[int, str]] = {}

    def reference(row):
        key = (row["frame"], row["local_id"])
        if key not in vectors:
            raise ValueError(f"Missing feature reference: {key}")
        parent.setdefault(key, key)
        return key

    def root(key):
        while parent[key] != key:
            key = parent[key]
        return key

    rows = []
    for row in annotations.get("pairs", []):
        a, b = reference(row["first"]), reference(row["second"])
        rows.append((a, b, row["relation"]))
        if row["relation"] == "same":
            parent[root(b)] = root(a)
    groups: dict[tuple[int, str], list] = {}
    for key in parent:
        groups.setdefault(root(key), []).append(key)
    negatives = set()
    pair_scores = []
    for a, b, relation in rows:
        if relation == "different":
            if root(a) == root(b):
                raise ValueError("Contradictory identity annotations")
            negatives.add(frozenset([root(a), root(b)]))
        av, bv = vectors[a], vectors[b]
        valid = bool(np.linalg.norm(av) > 0 and np.linalg.norm(bv) > 0)
        pair_scores.append(
            {"first": a, "second": b, "relation": relation, "cosine": float(np.dot(av, bv)) if valid else None}
        )
    positive = [r["cosine"] for r in pair_scores if r["relation"] == "same" and r["cosine"] is not None]
    negative = [r["cosine"] for r in pair_scores if r["relation"] == "different" and r["cosine"] is not None]
    auc = None
    if positive and negative:
        difference = np.asarray(positive)[:, None] - np.asarray(negative)[None, :]
        auc = float(np.mean((difference > 0) + 0.5 * (difference == 0)))
    anchors = {identity: min(keys) for identity, keys in groups.items()}
    retrieval = []
    for identity, keys in groups.items():
        candidates = {
            other: anchor
            for other, anchor in anchors.items()
            if other == identity or frozenset([identity, other]) in negatives
        }
        if len(candidates) < 2:
            continue
        for key in sorted(keys):
            if key[0] <= anchors[identity][0]:
                continue
            scores = sorted(
                [(float(np.dot(vectors[key], vectors[anchor])), other, anchor) for other, anchor in candidates.items()],
                reverse=True,
            )
            valid = bool(
                np.linalg.norm(vectors[key]) > 0
                and all(np.linalg.norm(vectors[anchor]) > 0 for anchor in candidates.values())
            )
            tied = len(scores) > 1 and abs(scores[0][0] - scores[1][0]) < 1e-8
            retrieval.append(
                {
                    "query": key,
                    "target": anchors[identity],
                    "predicted": scores[0][2],
                    "candidate_count": len(scores),
                    "correct": valid and not tied and scores[0][1] == identity,
                    "tie": tied,
                    "feature_available": valid,
                }
            )
    return {
        "scope": "annotated_pairs_and_explicit_negative_gallery_only",
        "threshold_selection": "none; threshold-free pilot, not a held-out generalization claim",
        "pair_count": len(pair_scores),
        "unavailable_pairs": sum(row["cosine"] is None for row in pair_scores),
        "same_pair_count": len(positive),
        "different_pair_count": len(negative),
        "pair_roc_auc": auc,
        "same_cosine_mean": float(np.mean(positive)) if positive else None,
        "different_cosine_mean": float(np.mean(negative)) if negative else None,
        "retrieval_queries": len(retrieval),
        "retrieval_correct": sum(row["correct"] for row in retrieval),
        "retrieval_top1": sum(row["correct"] for row in retrieval) / len(retrieval) if retrieval else None,
        "pairs": pair_scores,
        "retrieval": retrieval,
    }
