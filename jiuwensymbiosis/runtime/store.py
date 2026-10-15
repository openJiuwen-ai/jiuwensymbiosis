# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SQLite persistence for runtime job snapshots and ordered events.

The store owns JSON snapshots and append-only event rows. It has no dependency
on the runtime coordinator or GUI, so execution policy stays with its caller.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["JobStore", "RequestConflictError"]

_TERMINAL_PHASES = frozenset({"succeeded", "failed", "cancelled", "incomplete", "blocked"})
_MAX_EVENT_LIMIT = 1000


class RequestConflictError(ValueError):
    """A request ID was reused with a different request fingerprint."""


def _encode_object(value: dict[str, Any], *, label: str) -> str:
    """Encode a JSON object without lossy fallbacks or non-standard numbers."""
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a dict")
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _decode_object(encoded: str, *, label: str) -> dict[str, Any]:
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise ValueError(f"stored {label} must be a JSON object")
    return value


def _timestamp() -> str:
    """Return a UTC timestamp suitable for persisted event records."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class JobStore:
    """Thread-safe SQLite store for task snapshots and per-job event streams.

    Writes run in ``BEGIN IMMEDIATE`` transactions so job state and its event
    commit together and event sequence numbers stay monotonic across store
    instances. A process-local lock protects the shared SQLite connection.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            str(self.path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL UNIQUE,
                fingerprint TEXT NOT NULL,
                snapshot_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                kind TEXT NOT NULL,
                data_json TEXT NOT NULL,
                PRIMARY KEY (job_id, seq)
            );
            """
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("JobStore is closed")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Hold the connection lock and roll back every unsuccessful write."""
        self._lock.acquire()
        try:
            self._ensure_open()
            self._connection.execute("BEGIN IMMEDIATE")
        except BaseException:
            self._lock.release()
            raise
        try:
            yield self._connection
            self._connection.commit()
        except BaseException:
            if self._connection.in_transaction:
                # Roll back even on cancellation/interrupt so a half-written
                # snapshot can never survive an interrupted transaction.
                self._connection.rollback()
            raise
        finally:
            self._lock.release()

    def create(self, request_id: str, fingerprint: str, initial: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Create a job and its accepted event, or return an idempotent retry.

        The initial snapshot must contain a non-empty ``job_id``. The return
        value is ``(snapshot, created)``; an existing request with the same
        fingerprint returns its current snapshot and ``False``.
        """
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("fingerprint must be a non-empty string")

        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT fingerprint, snapshot_json FROM jobs WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                if existing["fingerprint"] != fingerprint:
                    raise RequestConflictError(f"request_id {request_id!r} has a different fingerprint")
                return _decode_object(existing["snapshot_json"], label="snapshot"), False

            initial_json = _encode_object(initial, label="initial")
            snapshot = _decode_object(initial_json, label="initial")
            job_id = snapshot.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise ValueError("initial must contain a non-empty string job_id")
            connection.execute(
                "INSERT INTO jobs(job_id, request_id, fingerprint, snapshot_json) VALUES (?, ?, ?, ?)",
                (job_id, request_id, fingerprint, initial_json),
            )
            self._append_event_locked(
                connection,
                job_id,
                "accepted",
                {"phase": snapshot.get("phase", "accepted")},
            )
            return snapshot, True

    def get(self, job_id: str) -> dict[str, Any]:
        """Return the current snapshot or raise ``KeyError`` for an unknown job."""
        with self._lock:
            self._ensure_open()
            row = self._connection.execute("SELECT snapshot_json FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            return _decode_object(row["snapshot_json"], label="snapshot")

    def find_request(self, request_id: str, fingerprint: str) -> dict[str, Any] | None:
        """Resolve a retry before admission; create() still enforces uniqueness."""
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT fingerprint, snapshot_json FROM jobs WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row is None:
                return None
            if row["fingerprint"] != fingerprint:
                raise RequestConflictError(f"request_id {request_id!r} has a different fingerprint")
            return _decode_object(row["snapshot_json"], label="snapshot")

    def update(
        self,
        job_id: str,
        changes: dict[str, Any],
        *,
        event_kind: str | None = None,
        event_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge snapshot fields and optionally append a matching event atomically."""
        _encode_object(changes, label="changes")
        if "job_id" in changes and changes["job_id"] != job_id:
            raise ValueError("job_id is immutable")
        if event_kind is None:
            if event_data is not None:
                raise ValueError("event_data requires event_kind")
            encoded_event_data = None
        else:
            self._validate_event_kind(event_kind)
            encoded_event_data = _encode_object(event_data if event_data is not None else {}, label="event_data")

        with self._transaction() as connection:
            row = connection.execute("SELECT snapshot_json FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            snapshot = _decode_object(row["snapshot_json"], label="snapshot")
            if snapshot.get("job_id") != job_id:
                raise ValueError("stored snapshot job_id does not match its database key")
            snapshot.update(changes)
            snapshot_json = _encode_object(snapshot, label="snapshot")
            connection.execute("UPDATE jobs SET snapshot_json = ? WHERE job_id = ?", (snapshot_json, job_id))
            if event_kind is not None:
                self._append_event_locked(
                    connection,
                    job_id,
                    event_kind,
                    _decode_object(encoded_event_data or "{}", label="event_data"),
                )
            return snapshot

    def append_event(self, job_id: str, kind: str, data: dict[str, Any]) -> dict[str, Any]:
        """Append an event and return its persisted representation."""
        self._validate_event_kind(kind)
        encoded_data = _encode_object(data, label="event_data")
        data_copy = _decode_object(encoded_data, label="event_data")
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone() is None:
                raise KeyError(job_id)
            return self._append_event_locked(connection, job_id, kind, data_copy)

    def read_events(self, job_id: str, after_seq: int = 0, limit: int = 100) -> dict[str, Any]:
        """Read a bounded, non-destructive page from one job's event stream."""
        if not isinstance(after_seq, int) or isinstance(after_seq, bool) or after_seq < 0:
            raise ValueError("after_seq must be a non-negative integer")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= _MAX_EVENT_LIMIT:
            raise ValueError(f"limit must be between 1 and {_MAX_EVENT_LIMIT}")

        with self._lock:
            self._ensure_open()
            if self._connection.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone() is None:
                raise KeyError(job_id)
            bounds = self._connection.execute(
                "SELECT MIN(seq) AS oldest, MAX(seq) AS latest FROM events WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            oldest = bounds["oldest"]
            rows = self._connection.execute(
                """
                SELECT job_id, seq, timestamp, kind, data_json
                FROM events WHERE job_id = ? AND seq > ? ORDER BY seq LIMIT ?
                """,
                (job_id, after_seq, limit),
            ).fetchall()
            events = [self._event_from_row(row) for row in rows]
            next_seq = events[-1]["seq"] if events else after_seq
            gap = oldest is not None and after_seq < oldest - 1
            return {
                "events": events,
                "next_seq": next_seq,
                "gap": gap,
                "oldest_available_seq": oldest,
            }

    def unfinished(self) -> list[dict[str, Any]]:
        """Return jobs whose snapshot phase does not indicate a terminal state."""
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute("SELECT snapshot_json FROM jobs ORDER BY rowid").fetchall()
            snapshots = [_decode_object(row["snapshot_json"], label="snapshot") for row in rows]
            return [snapshot for snapshot in snapshots if snapshot.get("phase") not in _TERMINAL_PHASES]

    def close(self) -> None:
        """Close the connection; repeated calls are harmless."""
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    @staticmethod
    def _validate_event_kind(kind: str) -> None:
        if not isinstance(kind, str) or not kind:
            raise ValueError("event kind must be a non-empty string")

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "seq": row["seq"],
            "timestamp": row["timestamp"],
            "kind": row["kind"],
            "data": _decode_object(row["data_json"], label="event data"),
        }

    def _append_event_locked(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        kind: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        self._validate_event_kind(kind)
        encoded_data = _encode_object(data, label="event_data")
        timestamp = _timestamp()
        row = connection.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE job_id = ?", (job_id,)).fetchone()
        seq = row[0]
        connection.execute(
            "INSERT INTO events(job_id, seq, timestamp, kind, data_json) VALUES (?, ?, ?, ?, ?)",
            (job_id, seq, timestamp, kind, encoded_data),
        )
        # Keep job snapshots and request IDs indefinitely. Only the event window
        # is bounded; readers see the resulting gap and can reload the snapshot.
        connection.execute("DELETE FROM events WHERE job_id = ? AND seq <= ?", (job_id, seq - 10000))
        return {"job_id": job_id, "seq": seq, "timestamp": timestamp, "kind": kind, "data": data}
