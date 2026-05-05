"""Persistence for the ``search_state`` singleton row.

Records the resolved vector-tier configuration so the indexer can
detect drift on startup and auto-trigger a backfill. Lexical search
runs against canonical tables and never has its own state.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SearchStateRow:
    lexical_schema_version: int = 0
    vector_provider: str = "disabled"
    vector_model: str | None = None
    vector_dim: int | None = None
    sparse_model: str | None = None
    backfill_status: str = "idle"
    backfill_job_id: str | None = None
    backfill_total: int | None = None
    backfill_processed: int | None = None
    backfill_error: str | None = None
    last_index_at: str | None = None
    notes: list[str] = field(default_factory=list)


def load_state(db: Any) -> SearchStateRow:
    with db.cursor() as cur:
        cur.execute(
            "SELECT lexical_schema_version, vector_provider, vector_model, "
            "       vector_dim, sparse_model, backfill_status, "
            "       backfill_job_id, backfill_total, backfill_processed, "
            "       backfill_error, last_index_at, notes "
            "FROM search_state WHERE id = 1"
        )
        row = cur.fetchone()
    if row is None:
        return SearchStateRow()
    notes_raw = row[11] if not isinstance(row, dict) else row.get("notes")
    notes: list[str] = []
    if notes_raw:
        try:
            parsed = json.loads(notes_raw)
            if isinstance(parsed, list):
                notes = [str(n) for n in parsed]
        except Exception:
            notes = []
    return SearchStateRow(
        lexical_schema_version=int(row[0] or 0),
        vector_provider=str(row[1] or "disabled"),
        vector_model=row[2] or None,
        vector_dim=int(row[3]) if row[3] is not None else None,
        sparse_model=row[4] or None,
        backfill_status=str(row[5] or "idle"),
        backfill_job_id=row[6] or None,
        backfill_total=int(row[7]) if row[7] is not None else None,
        backfill_processed=int(row[8]) if row[8] is not None else None,
        backfill_error=row[9] or None,
        last_index_at=row[10] or None,
        notes=notes,
    )


def save_resolved_config(
    db: Any,
    *,
    lexical_schema_version: int,
    vector_provider: str,
    vector_model: str | None,
    vector_dim: int | None,
    sparse_model: str | None,
    notes: list[str],
) -> None:
    with db.cursor() as cur:
        cur.execute(
            "UPDATE search_state SET "
            "  lexical_schema_version = ?, "
            "  vector_provider = ?, "
            "  vector_model = ?, "
            "  vector_dim = ?, "
            "  sparse_model = ?, "
            "  notes = ?, "
            "  updated_at = ? "
            "WHERE id = 1",
            (
                lexical_schema_version,
                vector_provider,
                vector_model,
                vector_dim,
                sparse_model,
                json.dumps(notes),
                _now(),
            ),
        )


def needs_backfill(prev: SearchStateRow, new: SearchStateRow) -> bool:
    return (
        prev.lexical_schema_version != new.lexical_schema_version
        or prev.vector_provider != new.vector_provider
        or (prev.vector_model or "") != (new.vector_model or "")
        or (prev.vector_dim or 0) != (new.vector_dim or 0)
        or (prev.sparse_model or "") != (new.sparse_model or "")
    )


def start_backfill(db: Any, *, total: int) -> str:
    job_id = str(uuid.uuid4())
    with db.cursor() as cur:
        cur.execute(
            "UPDATE search_state SET "
            "  backfill_status = 'queued', "
            "  backfill_job_id = ?, "
            "  backfill_total = ?, "
            "  backfill_processed = 0, "
            "  backfill_error = NULL, "
            "  updated_at = ? "
            "WHERE id = 1",
            (job_id, total, _now()),
        )
    return job_id


def update_backfill_progress(
    db: Any,
    *,
    processed: int,
    status: str | None = None,
    error: str | None = None,
) -> None:
    with db.cursor() as cur:
        sets = ["backfill_processed = ?", "updated_at = ?"]
        params: list[Any] = [processed, _now()]
        if status is not None:
            sets.append("backfill_status = ?")
            params.append(status)
        if error is not None:
            sets.append("backfill_error = ?")
            params.append(error)
        cur.execute(
            "UPDATE search_state SET " + ", ".join(sets) + " WHERE id = 1",
            tuple(params),
        )


def mark_indexed(db: Any) -> None:
    with db.cursor() as cur:
        cur.execute(
            "UPDATE search_state SET last_index_at = ?, updated_at = ? WHERE id = 1",
            (_now(), _now()),
        )


__all__ = [
    "SearchStateRow",
    "load_state",
    "save_resolved_config",
    "needs_backfill",
    "start_backfill",
    "update_backfill_progress",
    "mark_indexed",
]
