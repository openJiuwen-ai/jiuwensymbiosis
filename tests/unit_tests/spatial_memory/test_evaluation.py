"""Review metrics preserve uncertainty and cannot silently score another run."""

import copy

import numpy as np
import pytest

from jiuwensymbiosis.spatial_memory.evaluation import evaluate_appearance, evaluate_review


def fixture():
    summary = {
        "sequence_manifest_sha256": "sequence-a",
        "evidence_sha256": "evidence-a",
        "receipts": [
            {"assignments": [{"local_id": "0", "kind": "object", "entity_id": "chair-a", "label": "stool"}]},
            {
                "assignments": [
                    {"local_id": "0", "kind": "object", "entity_id": "chair-b", "label": "chair"},
                    {"local_id": "1", "kind": "object", "entity_id": None, "label": "chair"},
                ]
            },
        ],
    }
    review = {
        "schema_version": 1,
        "reviewer": "test-reviewer",
        "sequence_manifest_sha256": "sequence-a",
        "evidence_sha256": "evidence-a",
        "pairs": [
            {"first": {"frame": 0, "local_id": "0"}, "second": {"frame": 1, "local_id": "0"}, "relation": "same"},
            {
                "first": {"frame": 0, "local_id": "0"},
                "second": {"frame": 1, "local_id": "1"},
                "relation": "different",
            },
        ],
        "labels": [{"frame": 0, "local_id": "0", "acceptable_labels": ["chair"]}],
    }
    return summary, review


def test_fragments_and_unresolved_pairs_are_not_counted_as_success():
    summary, review = fixture()
    before = copy.deepcopy(summary)
    report = evaluate_review(summary, review)
    assert report["pair_outcomes"] == {"fragmented": 1, "unresolved": 1}
    assert report["pair_correct_fraction"] == 0 and report["label_correct_count"] == 0
    assert summary == before


@pytest.mark.parametrize("invalid", ["sequence", "detections", "duplicate", "reference"])
def test_invalid_annotation_scope_is_rejected(invalid):
    summary, review = fixture()
    if invalid == "sequence":
        review["sequence_manifest_sha256"] = "sequence-b"
    elif invalid == "detections":
        review["evidence_sha256"] = "other-evidence"
    elif invalid == "duplicate":
        review["pairs"].append(review["pairs"][0])
    else:
        review["pairs"][0]["first"]["frame"] = 99
    with pytest.raises(ValueError):
        evaluate_review(summary, review)


def test_optional_features_do_not_change_annotation_scope_or_admit_missing_hash():
    summary, review = fixture()
    summary.update(input_evidence_sha256="evidence-a", evidence_sha256="with-clip")
    assert evaluate_review(summary, review)["pair_count"] == 2
    review.pop("evidence_sha256")
    summary["input_evidence_sha256"] = None
    with pytest.raises(ValueError, match="evidence"):
        evaluate_review(summary, review)


def test_pair_precision_recall_keeps_false_merges_and_unresolved_same_pairs():
    summary, review = fixture()
    summary["receipts"][1]["assignments"][0]["entity_id"] = None
    summary["receipts"][1]["assignments"][1]["entity_id"] = "chair-a"
    summary["receipts"].append(copy.deepcopy(summary["receipts"][0]))
    review["pairs"].append(
        {"first": {"frame": 0, "local_id": "0"}, "second": {"frame": 2, "local_id": "0"}, "relation": "same"}
    )
    report = evaluate_review(summary, review)
    assert report["pair_correct_fraction"] == pytest.approx(1 / 3)
    assert report["association"] == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "unresolved_count": 1,
        "same_pairs": 2,
        "different_pairs": 1,
    }


def test_appearance_gallery_uses_explicit_negatives_and_retains_ties_or_missing_features():
    def pair(a, b, relation):
        return {
            "first": {"frame": a[0], "local_id": a[1]},
            "second": {"frame": b[0], "local_id": b[1]},
            "relation": relation,
        }

    vectors = {
        (0, "a"): np.array([1.0, 0]),
        (1, "a"): np.array([1.0, 0]),
        (0, "b"): np.array([0.0, 1]),
        (1, "b"): np.array([0.0, 1]),
        (0, "unknown"): np.array([1.0, 0]),
        (1, "unknown"): np.array([1.0, 0]),
    }
    annotation = {"pairs": [pair((0, key), (1, key), "same") for key in ("a", "b", "unknown")]}
    annotation["pairs"].append(pair((0, "a"), (0, "b"), "different"))
    report = evaluate_appearance(vectors, annotation)
    assert report["retrieval_queries"] == 2 and report["retrieval_correct"] == 2
    vectors[(1, "a")] = np.array([1.0, 1.0]) / np.sqrt(2)
    vectors[(1, "b")] = np.zeros(2)
    report = evaluate_appearance(vectors, annotation)
    assert report["retrieval_queries"] == 2 and report["retrieval_correct"] == 0
    assert report["unavailable_pairs"] == 1
