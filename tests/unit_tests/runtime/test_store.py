# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for SQLite-backed runtime job persistence."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest

from jiuwensymbiosis.runtime.store import JobStore, RequestConflictError


def _initial(job_id: str, *, phase: str = "queued") -> dict[str, str]:
    return {"job_id": job_id, "binding_id": "binding-1", "phase": phase}


def _event_kinds(store: JobStore, job_id: str) -> list[str]:
    batch = store.read_events(job_id, limit=1000)
    return [event["kind"] for event in batch["events"]]


def test_create_persists_snapshot_and_accepted_event(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "runtime.sqlite3"
    store = JobStore(db_path)
    try:
        snapshot, created = store.create("request-1", "fingerprint-1", _initial("job-1"))

        assert created is True
        assert snapshot == _initial("job-1")
        assert store.get("job-1") == snapshot
        event = store.read_events("job-1")["events"][0]
        assert event["job_id"] == "job-1"
        assert event["seq"] == 1
        assert event["kind"] == "accepted"
        assert event["data"] == {"phase": "queued"}
        assert datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).tzinfo is not None
        assert db_path.parent.is_dir()
    finally:
        store.close()


def test_concurrent_duplicate_create_across_connections_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    stores = [JobStore(path), JobStore(path)]
    try:

        def create(index: int) -> tuple[dict[str, object], bool]:
            return stores[index % 2].create(
                "same-request",
                "same-fingerprint",
                _initial(f"candidate-{index}"),
            )

        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(create, range(48)))

        assert sum(created for _, created in results) == 1
        job_ids = {snapshot["job_id"] for snapshot, _ in results}
        assert len(job_ids) == 1
        assert all(kinds == ["accepted"] for kinds in (_event_kinds(store, next(iter(job_ids))) for store in stores))
    finally:
        for store in stores:
            store.close()


def test_request_id_with_different_fingerprint_conflicts(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    try:
        store.create("request-1", "fingerprint-1", _initial("job-1"))

        with pytest.raises(RequestConflictError, match="different fingerprint"):
            store.create("request-1", "fingerprint-2", _initial("job-2"))

        assert store.get("job-1") == _initial("job-1")
        with pytest.raises(KeyError):
            store.get("job-2")
    finally:
        store.close()


def test_events_use_independent_non_destructive_cursors(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    try:
        store.create("request-1", "fingerprint-1", _initial("job-1"))
        store.append_event("job-1", "started", {"phase": "running"})
        store.append_event("job-1", "progress", {"step": 1})

        first_client = store.read_events("job-1", after_seq=0, limit=2)
        second_client = store.read_events("job-1", after_seq=0, limit=1)
        first_client_next = store.read_events("job-1", after_seq=first_client["next_seq"])

        assert [event["seq"] for event in first_client["events"]] == [1, 2]
        assert [event["seq"] for event in second_client["events"]] == [1]
        assert [event["seq"] for event in first_client_next["events"]] == [3]
        assert first_client["gap"] is False
        assert first_client["oldest_available_seq"] == 1
        assert store.read_events("job-1", after_seq=3)["next_seq"] == 3
    finally:
        store.close()


def test_old_event_cursor_reports_gap_without_losing_request_identity(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    try:
        store.create("request", "fingerprint", _initial("job"))
        # Seed the last retained sequence directly, then cross the retention boundary.
        with store._connection:
            store._connection.execute("UPDATE events SET seq = 10000 WHERE job_id = 'job'")
        store.append_event("job", "progress", {})
        batch = store.read_events("job", 0)
        assert batch["gap"]
        assert batch["oldest_available_seq"] == 10000
        assert store.find_request("request", "fingerprint")["job_id"] == "job"
    finally:
        store.close()


def test_event_sequence_stays_monotonic_across_store_instances(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    stores = [JobStore(path), JobStore(path)]
    try:
        stores[0].create("request-1", "fingerprint-1", _initial("job-1"))

        def append(index: int) -> dict[str, object]:
            return stores[index % 2].append_event("job-1", "progress", {"n": index})

        with ThreadPoolExecutor(max_workers=12) as pool:
            events = list(pool.map(append, range(48)))

        assert sorted(event["seq"] for event in events) == list(range(2, 50))
        read = stores[0].read_events("job-1", limit=1000)
        assert [event["seq"] for event in read["events"]] == list(range(1, 50))
    finally:
        for store in stores:
            store.close()


def test_update_and_event_roll_back_together_on_sql_failure(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    store = JobStore(db_path)
    try:
        store.create("request-1", "fingerprint-1", _initial("job-1"))
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_boom_event BEFORE INSERT ON events
                WHEN NEW.kind = 'boom'
                BEGIN SELECT RAISE(ABORT, 'injected event failure'); END
                """
            )

        with pytest.raises(sqlite3.IntegrityError, match="injected event failure"):
            store.update(
                "job-1",
                {"phase": "failed"},
                event_kind="boom",
                event_data={"reason": "test"},
            )

        assert store.get("job-1")["phase"] == "queued"
        batch = store.read_events("job-1")
        assert [event["kind"] for event in batch["events"]] == ["accepted"]
    finally:
        store.close()


def test_create_rolls_back_record_when_accepted_event_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    store = JobStore(db_path)
    try:
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_accepted_event BEFORE INSERT ON events
                WHEN NEW.kind = 'accepted'
                BEGIN SELECT RAISE(ABORT, 'injected accept failure'); END
                """
            )

        with pytest.raises(sqlite3.IntegrityError, match="injected accept failure"):
            store.create("request-1", "fingerprint-1", _initial("job-1"))

        with pytest.raises(KeyError):
            store.get("job-1")
        assert store.unfinished() == []
    finally:
        store.close()


def test_reopen_restores_current_snapshot_events_and_unfinished_jobs(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    store = JobStore(path)
    terminal = ("succeeded", "failed", "cancelled", "incomplete", "blocked")
    try:
        store.create("active-request", "active-fingerprint", _initial("active-job", phase="running"))
        for phase in terminal:
            store.create(f"{phase}-request", f"{phase}-fingerprint", _initial(f"{phase}-job", phase=phase))
        store.update(
            "active-job",
            {"result": {"ok": True}},
            event_kind="completed",
            event_data={"phase": "running"},
        )
    finally:
        store.close()
        store.close()

    reopened = JobStore(path)
    try:
        assert reopened.get("active-job")["result"] == {"ok": True}
        assert [event["kind"] for event in reopened.read_events("active-job")["events"]] == [
            "accepted",
            "completed",
        ]
        assert [job["job_id"] for job in reopened.unfinished()] == ["active-job"]
    finally:
        reopened.close()


def test_update_validates_immutable_job_id_and_json_payload(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    try:
        store.create("request-1", "fingerprint-1", _initial("job-1"))

        with pytest.raises(ValueError, match="job_id is immutable"):
            store.update("job-1", {"job_id": "job-2"})
        with pytest.raises(TypeError):
            store.append_event("job-1", "unsupported", {"value": object()})
        with pytest.raises(ValueError, match="after_seq"):
            store.read_events("job-1", after_seq=-1)
        with pytest.raises(ValueError, match="limit"):
            store.read_events("job-1", limit=0)
        with pytest.raises(ValueError, match="limit"):
            store.read_events("job-1", limit=1001)
        assert store.get("job-1")["phase"] == "queued"
    finally:
        store.close()
