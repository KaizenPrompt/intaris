"""Durable indexer queue for the optional vector tier.

Vector ops only — lexical writes are automatic via Postgres generated
columns (or by reading canonical tables on SQLite). The outbox is
empty when ``vector_provider=disabled``.

Atomic claim semantics with a 5-minute lease so a crashed worker does
not stall the queue indefinitely.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)


OP_EMBED = "embed"
OP_DELETE_SESSION = "delete_session"
OP_DELETE_REF = "delete_ref"

VALID_OPS = frozenset({OP_EMBED, OP_DELETE_SESSION, OP_DELETE_REF})

MAX_ATTEMPTS = 8
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _format_ts(_db: Any, ts: datetime) -> str:
    return ts.isoformat()


def enqueue(
    db: Any,
    *,
    op: str,
    payload: dict[str, Any],
    cursor: Any | None = None,
) -> int:
    if op not in VALID_OPS:
        raise ValueError(f"invalid outbox op: {op}")
    payload_json = json.dumps(payload, ensure_ascii=False)
    next_attempt = _format_ts(db, _now())
    created_at = _format_ts(db, _now())

    if cursor is not None:
        return _insert(cursor, op, payload_json, next_attempt, created_at)
    with db.cursor() as cur:
        return _insert(cur, op, payload_json, next_attempt, created_at)


def _insert(
    cursor: Any,
    op: str,
    payload_json: str,
    next_attempt: str,
    created_at: str,
) -> int:
    cursor.execute(
        "INSERT INTO search_outbox (op, payload, next_attempt_at, created_at) "
        "VALUES (?, ?, ?, ?)",
        (op, payload_json, next_attempt, created_at),
    )
    return getattr(cursor, "lastrowid", 0) or 0


def claim_due(db: Any, *, limit: int) -> list[dict[str, Any]]:
    """Atomically claim up to ``limit`` due rows.

    Stale claims older than 5 minutes are recycled — a crashed worker
    will not stall the queue indefinitely.

    PostgreSQL: uses ``UPDATE ... WHERE id IN (SELECT ... FOR UPDATE
    SKIP LOCKED) RETURNING`` for true cross-worker atomicity.

    SQLite: a single thread runs the indexer, so the SELECT+UPDATE
    inside the same write transaction is sufficient. We still write
    them as a transaction-scoped pair to match the PG semantics.
    """
    now = _now()
    now_ts = _format_ts(db, now)
    stale_threshold = _format_ts(db, now - timedelta(minutes=5))

    rows: list[dict[str, Any]] = []
    if getattr(db, "backend", "sqlite") == "postgresql":
        with db.cursor() as cur:
            cur.execute(
                "UPDATE search_outbox AS o SET claimed_at = ? "
                "WHERE o.id IN ("
                "  SELECT id FROM search_outbox "
                "  WHERE next_attempt_at <= ? "
                "    AND (claimed_at IS NULL OR claimed_at <= ?) "
                "  ORDER BY next_attempt_at, id "
                "  LIMIT ? "
                "  FOR UPDATE SKIP LOCKED"
                ") "
                "RETURNING o.id, o.op, o.payload, o.attempts",
                (now_ts, now_ts, stale_threshold, limit),
            )
            candidates = cur.fetchall()
    else:
        with db.cursor() as cur:
            cur.execute(
                "SELECT id, op, payload, attempts FROM search_outbox "
                "WHERE next_attempt_at <= ? "
                "  AND (claimed_at IS NULL OR claimed_at <= ?) "
                "ORDER BY next_attempt_at, id "
                "LIMIT ?",
                (now_ts, stale_threshold, limit),
            )
            candidates = cur.fetchall()
            if candidates:
                ids = [int(r[0]) for r in candidates]
                placeholders = ",".join(["?"] * len(ids))
                cur.execute(
                    "UPDATE search_outbox SET claimed_at = ? "
                    f"WHERE id IN ({placeholders})",
                    (now_ts, *ids),
                )

    if not candidates:
        return []
    for row in candidates:
        try:
            payload = json.loads(row[2]) if row[2] else {}
        except Exception:
            payload = {}
        rows.append(
            {
                "id": int(row[0]),
                "op": str(row[1]),
                "payload": payload,
                "attempts": int(row[3] or 0),
            }
        )
    return rows


def mark_done(db: Any, ids: Iterable[int]) -> None:
    ids = list(ids)
    if not ids:
        return
    placeholders = ",".join(["?"] * len(ids))
    with db.cursor() as cur:
        cur.execute(
            f"DELETE FROM search_outbox WHERE id IN ({placeholders})",
            tuple(ids),
        )


def mark_failed(db: Any, *, row_id: int, attempts: int, error: str) -> None:
    if attempts >= MAX_ATTEMPTS:
        next_attempt = _now() + timedelta(days=30)
        logger.warning(
            "Search outbox row %s exceeded retry limit (%d attempts): %s",
            row_id,
            attempts,
            error,
        )
    else:
        backoff = min(
            BASE_BACKOFF_SECONDS * (2 ** max(attempts - 1, 0)),
            MAX_BACKOFF_SECONDS,
        )
        backoff *= 1 + (random.random() - 0.5) * 0.4
        next_attempt = _now() + timedelta(seconds=backoff)

    with db.cursor() as cur:
        cur.execute(
            "UPDATE search_outbox SET attempts = ?, next_attempt_at = ?, "
            "       error = ?, claimed_at = NULL "
            "WHERE id = ?",
            (attempts, _format_ts(None, next_attempt), error[:1024], row_id),
        )


def queue_depth(db: Any) -> int:
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM search_outbox")
        row = cur.fetchone()
    return int(row[0]) if row else 0


__all__ = [
    "OP_EMBED",
    "OP_DELETE_SESSION",
    "OP_DELETE_REF",
    "VALID_OPS",
    "MAX_ATTEMPTS",
    "enqueue",
    "claim_due",
    "mark_done",
    "mark_failed",
    "queue_depth",
]
