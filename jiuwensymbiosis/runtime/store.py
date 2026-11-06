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
import uuid
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
            CREATE TABLE IF NOT EXISTS invocations (
                invocation_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                resource_key TEXT NOT NULL,
                request_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                UNIQUE(resource_key, request_key)
            );
            CREATE INDEX IF NOT EXISTS invocations_job ON invocations(job_id);
            CREATE TABLE IF NOT EXISTS job_invocations (
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                invocation_id TEXT NOT NULL REFERENCES invocations(invocation_id),
                PRIMARY KEY (job_id, invocation_id)
            );
            INSERT OR IGNORE INTO job_invocations SELECT job_id, invocation_id FROM invocations;
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

    def begin_invocation(
        self,
        job_id: str,
        resource_key: str,
        request_key: str,
        fingerprint: str,
        initial: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Commit a device dispatch intent and event before the first hardware write.

        The stable request key is scoped to the physical resource, not the job,
        transport connection, action name or RPC ID. A retry can only read the
        original record; it never authorizes another dispatch.
        """
        for label, value in (
            ("resource_key", resource_key),
            ("request_key", request_key),
            ("fingerprint", fingerprint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        payload = _decode_object(_encode_object(initial, label="invocation"), label="invocation")
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT fingerprint, snapshot_json FROM invocations WHERE resource_key = ? AND request_key = ?",
                (resource_key, request_key),
            ).fetchone()
            if existing is not None:
                if existing["fingerprint"] != fingerprint:
                    raise RequestConflictError(
                        "device request key was reused with different action parameters or binding"
                    )
                snapshot = _decode_object(existing["snapshot_json"], label="invocation")
                self._link_invocation_locked(connection, job_id, snapshot)
                return snapshot, False
            if connection.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone() is None:
                raise KeyError(job_id)
            payload.update(
                invocation_id=uuid.uuid4().hex,
                job_id=job_id,
                resource_key=resource_key,
                request_key=request_key,
                fingerprint=fingerprint,
                status="intent",
                dispatch_intent_at=_timestamp(),
            )
            connection.execute(
                "INSERT INTO invocations VALUES (?, ?, ?, ?, ?, ?)",
                (
                    payload["invocation_id"],
                    job_id,
                    resource_key,
                    request_key,
                    fingerprint,
                    _encode_object(payload, label="invocation"),
                ),
            )
            connection.execute("INSERT INTO job_invocations VALUES (?, ?)", (job_id, payload["invocation_id"]))
            self._append_event_locked(connection, job_id, "invocation_intent", payload)
            return payload, True

    def find_invocation(self, resource_key: str, request_key: str, fingerprint: str) -> dict[str, Any] | None:
        """Look up an action retry without creating another dispatch intent."""
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT fingerprint, snapshot_json FROM invocations WHERE resource_key = ? AND request_key = ?",
                (resource_key, request_key),
            ).fetchone()
            if row is None:
                return None
            if row["fingerprint"] != fingerprint:
                raise RequestConflictError("device request key has a different fingerprint")
            return _decode_object(row["snapshot_json"], label="invocation")

    def link_invocation(self, job_id: str, invocation_id: str) -> None:
        """Associate a retrying job with the original authoritative action record."""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM invocations WHERE invocation_id = ?", (invocation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(invocation_id)
            self._link_invocation_locked(connection, job_id, _decode_object(row["snapshot_json"], label="invocation"))

    def _link_invocation_locked(self, connection: sqlite3.Connection, job_id: str, record: dict[str, Any]) -> None:
        linked = connection.execute(
            "INSERT OR IGNORE INTO job_invocations VALUES (?, ?)", (job_id, record["invocation_id"])
        )
        if linked.rowcount:
            self._append_event_locked(connection, job_id, "invocation_reused", record)

    def get_invocation(self, invocation_id: str) -> dict[str, Any]:
        """Return a detached durable action record, including completion evidence."""
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT snapshot_json FROM invocations WHERE invocation_id = ?", (invocation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(invocation_id)
            return _decode_object(row["snapshot_json"], label="invocation")

    def list_invocations(self, job_id: str | None = None) -> list[dict[str, Any]]:
        """Read action records in acceptance order without resuming hardware."""
        with self._lock:
            self._ensure_open()
            if job_id is None:
                rows = self._connection.execute("SELECT snapshot_json FROM invocations ORDER BY rowid").fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT i.snapshot_json FROM invocations AS i JOIN job_invocations AS j "
                    "ON i.invocation_id = j.invocation_id WHERE j.job_id = ? ORDER BY i.rowid",
                    (job_id,),
                ).fetchall()
            return [_decode_object(row["snapshot_json"], label="invocation") for row in rows]

    def unresolved_invocations(self, resource_key: str) -> list[dict[str, Any]]:
        """Find physical uncertainty even after a lease was externally reconciled."""
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT snapshot_json FROM invocations WHERE resource_key = ? ORDER BY rowid", (resource_key,)
            ).fetchall()
            records = [_decode_object(row["snapshot_json"], label="invocation") for row in rows]
            return [record for record in records if record["status"] not in {"succeeded", "failed"}]

    def update_invocation(
        self, invocation_id: str, changes: dict[str, Any], *, reconciled: bool = False
    ) -> dict[str, Any]:
        """Commit execution facts and their event in the same SQLite transaction.

        Only trusted reconciliation may turn an unknown result into a terminal
        fact. Runtime retries cannot change an uncertain record into a new intent.
        """
        immutable = {
            "invocation_id",
            "job_id",
            "resource_key",
            "request_key",
            "fingerprint",
            "binding_id",
            "contract_version",
            "action_name",
            "parameters_fingerprint",
            "owner_pid",
            "device_binding",
            "lease_operation_id",
            "principal",
            "dispatch_intent_at",
        }
        if immutable.intersection(changes):
            raise ValueError("invocation identity is immutable")
        _encode_object(changes, label="invocation changes")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM invocations WHERE invocation_id = ?", (invocation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(invocation_id)
            payload = _decode_object(row["snapshot_json"], label="invocation")
            status = changes.get("status", payload["status"])
            if status not in {"intent", "pending", "succeeded", "failed", "unknown"}:
                raise ValueError("invalid invocation status")
            if status == "intent" and payload["status"] != "intent":
                raise ValueError("an invocation cannot be dispatched again")
            if payload["status"] in {"succeeded", "failed"}:
                raise ValueError("terminal invocation records are immutable")
            if payload["status"] == "unknown" and status != "unknown" and not reconciled:
                raise ValueError("unknown invocation requires trusted reconciliation")
            if reconciled and (status not in {"succeeded", "failed"} or not changes.get("completion_evidence")):
                raise ValueError("reconciliation requires terminal status and completion evidence")
            payload.update(changes)
            payload["updated_at"] = _timestamp()
            connection.execute(
                "UPDATE invocations SET snapshot_json = ? WHERE invocation_id = ?",
                (_encode_object(payload, label="invocation"), invocation_id),
            )
            self._append_event_locked(
                connection, payload["job_id"], "invocation_reconciled" if reconciled else "invocation_updated", payload
            )
            return payload

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
