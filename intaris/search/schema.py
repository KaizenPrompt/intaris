"""Search subsystem schema bootstrap.

Idempotent. Adds ``tsvector`` generated columns and partial GIN indexes
directly to the canonical Intaris tables (``sessions``, ``audit_log``,
``session_summaries``, ``agent_summaries``). No projection table — the
canonical tables ARE the lexical search index.

When the vector tier is enabled we additionally create:

- ``search_vectors`` (pgvector backend only): per-row dense embeddings
  with an HNSW cosine index.
- ``search_outbox``: durable indexer queue of ``embed`` /
  ``delete_session`` ops.
- ``search_state``: singleton row recording the last-resolved vector
  config so the indexer can detect drift and auto-backfill.

Postgres extensions (``unaccent``, ``pg_trgm``, ``vector``) are probed
inside ``SAVEPOINT``s; failures are swallowed and the bootstrap
continues with the appropriate fallback path. Capability flags are
stored on the bootstrap object for the query layer to consult.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Bumped whenever the lexical schema changes in a way that requires a
# rebuild of generated columns. Stored in ``search_state`` so we can
# detect upgrades.
LEXICAL_SCHEMA_VERSION = 1


_PG_TSVECTOR_COLUMNS = [
    # (table, column, source_expr)
    ("sessions", "intention_tsv", "intention"),
    ("sessions", "title_tsv", "title"),
    ("audit_log", "intention_tsv", "intention"),
    ("audit_log", "content_tsv", "content"),
    ("session_summaries", "summary_tsv", "summary"),
    ("agent_summaries", "summary_tsv", "summary"),
]

# Partial GIN indexes — only on rows we actually search. Tool-call
# audit rows duplicate the parent intention; we surface that via the
# ``sessions.intention`` live value, not via every audit row.
_PG_GIN_INDEXES = [
    # (index_name, table, column, where_clause)
    ("ix_sessions_intention_tsv", "sessions", "intention_tsv", None),
    ("ix_sessions_title_tsv", "sessions", "title_tsv", None),
    (
        "ix_audit_intention_tsv",
        "audit_log",
        "intention_tsv",
        "intention IS NOT NULL",
    ),
    (
        "ix_audit_content_tsv",
        "audit_log",
        "content_tsv",
        "record_type IN ('reasoning','checkpoint')",
    ),
    ("ix_session_summaries_tsv", "session_summaries", "summary_tsv", None),
    ("ix_agent_summaries_tsv", "agent_summaries", "summary_tsv", None),
]

_PG_TRGM_INDEXES = [
    # (index_name, table, column)
    ("ix_sessions_intention_trgm", "sessions", "intention"),
    ("ix_audit_intention_trgm", "audit_log", "intention"),
    ("ix_session_summaries_summary_trgm", "session_summaries", "summary"),
    ("ix_agent_summaries_summary_trgm", "agent_summaries", "summary"),
]


def _split_pg_statements(script: str) -> list[str]:
    """Split a SQL script into individual statements for psycopg2.

    psycopg2's cursor cannot run multi-statement strings reliably, and
    the wrapper cursor inherits that limitation. This is a tiny
    splitter that handles single-line ``--`` comments and treats ``;``
    outside string literals as the statement boundary. Good enough
    for our DDL — keep the scripts conservative.
    """
    stmts: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    for line in script.splitlines():
        stripped = line.strip()
        if not in_single and not in_double and stripped.startswith("--"):
            continue
        for ch in line:
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            if ch == ";" and not in_single and not in_double:
                stmt = "".join(buf).strip()
                if stmt:
                    stmts.append(stmt)
                buf = []
                continue
            buf.append(ch)
        buf.append("\n")
    tail = "".join(buf).strip()
    if tail:
        stmts.append(tail)
    return stmts


_PG_VECTOR_TABLES = """
CREATE TABLE IF NOT EXISTS search_vectors (
    user_id      TEXT        NOT NULL,
    session_id   TEXT        NOT NULL,
    kind         TEXT        NOT NULL,
    ref_id       TEXT        NOT NULL,
    text         TEXT        NOT NULL,
    ts           TIMESTAMPTZ NOT NULL,
    agent_id     TEXT,
    model        TEXT        NOT NULL,
    dim          INTEGER     NOT NULL,
    PRIMARY KEY (user_id, session_id, kind, ref_id)
);

CREATE INDEX IF NOT EXISTS ix_search_vectors_user_ts
    ON search_vectors (user_id, ts DESC);
"""

_PG_OUTBOX_AND_STATE = """
CREATE TABLE IF NOT EXISTS search_outbox (
    id              BIGSERIAL PRIMARY KEY,
    op              TEXT        NOT NULL
        CHECK (op IN ('embed', 'delete_session', 'delete_ref')),
    payload         TEXT        NOT NULL,
    attempts        INTEGER     NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claimed_at      TIMESTAMPTZ,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_search_outbox_due
    ON search_outbox (next_attempt_at, id);

CREATE TABLE IF NOT EXISTS search_state (
    id                       INTEGER PRIMARY KEY DEFAULT 1
        CHECK (id = 1),
    lexical_schema_version   INTEGER     NOT NULL DEFAULT 0,
    vector_provider          TEXT        NOT NULL DEFAULT 'disabled',
    vector_model             TEXT,
    vector_dim               INTEGER,
    sparse_model             TEXT,
    backfill_status          TEXT        NOT NULL DEFAULT 'idle'
        CHECK (backfill_status IN ('idle','queued','running','failed','done')),
    backfill_job_id          TEXT,
    backfill_total           INTEGER,
    backfill_processed       INTEGER,
    backfill_error           TEXT,
    last_index_at            TIMESTAMPTZ,
    notes                    TEXT,
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO search_state (id) VALUES (1)
ON CONFLICT (id) DO NOTHING;
"""


_SQLITE_OUTBOX_AND_STATE = """
CREATE TABLE IF NOT EXISTS search_vectors (
    user_id      TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    ref_id       TEXT NOT NULL,
    text         TEXT NOT NULL,
    ts           TEXT NOT NULL,
    agent_id     TEXT,
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    PRIMARY KEY (user_id, session_id, kind, ref_id)
);

CREATE INDEX IF NOT EXISTS ix_search_vectors_user_ts
    ON search_vectors (user_id, ts DESC);

CREATE TABLE IF NOT EXISTS search_outbox (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    op              TEXT NOT NULL
        CHECK (op IN ('embed', 'delete_session', 'delete_ref')),
    payload         TEXT NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT NOT NULL,
    claimed_at      TEXT,
    error           TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_search_outbox_due
    ON search_outbox (next_attempt_at, id);

CREATE TABLE IF NOT EXISTS search_state (
    id                       INTEGER PRIMARY KEY
        CHECK (id = 1),
    lexical_schema_version   INTEGER NOT NULL DEFAULT 0,
    vector_provider          TEXT NOT NULL DEFAULT 'disabled',
    vector_model             TEXT,
    vector_dim               INTEGER,
    sparse_model             TEXT,
    backfill_status          TEXT NOT NULL DEFAULT 'idle'
        CHECK (backfill_status IN ('idle','queued','running','failed','done')),
    backfill_job_id          TEXT,
    backfill_total           INTEGER,
    backfill_processed       INTEGER,
    backfill_error           TEXT,
    last_index_at            TEXT,
    notes                    TEXT,
    updated_at               TEXT NOT NULL
);

INSERT OR IGNORE INTO search_state (id, updated_at)
    VALUES (1, datetime('now'));
"""


class SearchSchema:
    """Idempotent bootstrap for the search subsystem's storage."""

    def __init__(self, *, vector_enabled: bool, embedding_dim: int) -> None:
        self._vector_enabled = vector_enabled
        self._embedding_dim = max(1, int(embedding_dim or 1))
        self.lexical_backend: str = "unknown"
        self.has_unaccent: bool = False
        self.has_pg_trgm: bool = False
        self.has_pgvector: bool = False
        self.notes: list[str] = []

    def ensure(self, db: Any) -> None:
        if db.backend == "postgresql":
            self._ensure_pg(db)
        else:
            self._ensure_sqlite(db)

    # ── Postgres ──

    def _ensure_pg(self, db: Any) -> None:
        # Use the wrapper cursor (db.cursor()) so ``?`` placeholders
        # get translated to ``%s`` and the connection commits on exit.
        # Multi-statement DDL is split into individual ``cur.execute``
        # calls because psycopg2 has no ``executescript`` and
        # multi-statement strings are not portable across drivers.
        with db.cursor() as cur:
            self._probe_pg_extension(cur, "unaccent")
            self._probe_pg_extension(cur, "pg_trgm")

            self.has_unaccent = self._pg_extension_present(cur, "unaccent")
            self.has_pg_trgm = self._pg_extension_present(cur, "pg_trgm")

            self._ensure_pg_tsvector_columns(cur)
            self._ensure_pg_gin_indexes(cur)
            if self.has_pg_trgm:
                self._ensure_pg_trgm_indexes(cur)

            # Outbox + state tables — always created so toggling
            # vector tier on later doesn't require a fresh migration.
            for stmt in _split_pg_statements(_PG_OUTBOX_AND_STATE):
                cur.execute(stmt)

            # Add claimed_at column to outbox if missing (legacy upgrades).
            if not self._pg_column_exists(cur, "search_outbox", "claimed_at"):
                cur.execute(
                    "ALTER TABLE search_outbox ADD COLUMN claimed_at TIMESTAMPTZ"
                )

            if self._vector_enabled:
                self._ensure_pg_vector_tables(cur)

            self.lexical_backend = "postgres-fts"
            if not self.has_unaccent:
                self.notes.append(
                    "unaccent extension not enabled; fold via Python fallback only"
                )
            if not self.has_pg_trgm:
                self.notes.append(
                    "pg_trgm extension not enabled; substring fallback "
                    "uses LIKE on raw column"
                )

        logger.info(
            "Search schema (PG) ensured: lexical=%s unaccent=%s trgm=%s vector_enabled=%s",
            self.lexical_backend,
            self.has_unaccent,
            self.has_pg_trgm,
            self._vector_enabled,
        )

    def _ensure_pg_tsvector_columns(self, cur: Any) -> None:
        for table, column, source_expr in _PG_TSVECTOR_COLUMNS:
            if self._pg_column_exists(cur, table, column):
                continue
            expr_unaccent = (
                f"to_tsvector('simple', unaccent(coalesce({source_expr},'')))"
            )
            expr_plain = f"to_tsvector('simple', coalesce({source_expr},''))"
            expr = expr_unaccent if self.has_unaccent else expr_plain
            sp = f"add_{table}_{column}"
            cur.execute(f"SAVEPOINT {sp}")
            try:
                cur.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} tsvector "
                    f"GENERATED ALWAYS AS ({expr}) STORED"
                )
                cur.execute(f"RELEASE SAVEPOINT {sp}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Search: failed to add %s.%s with %s expression (%s); falling back",
                    table,
                    column,
                    "unaccent" if self.has_unaccent else "plain",
                    exc,
                )
                cur.execute(f"ROLLBACK TO SAVEPOINT {sp}")
                cur.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} tsvector "
                    f"GENERATED ALWAYS AS ({expr_plain}) STORED"
                )
                self.has_unaccent = False

    def _ensure_pg_gin_indexes(self, cur: Any) -> None:
        for index_name, table, column, where in _PG_GIN_INDEXES:
            sql = (
                f"CREATE INDEX IF NOT EXISTS {index_name} "
                f"ON {table} USING GIN ({column})"
            )
            if where:
                sql += f" WHERE {where}"
            try:
                cur.execute(sql)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Search: GIN index %s skipped (%s)", index_name, exc)

    def _ensure_pg_trgm_indexes(self, cur: Any) -> None:
        for index_name, table, column in _PG_TRGM_INDEXES:
            sql = (
                f"CREATE INDEX IF NOT EXISTS {index_name} "
                f"ON {table} USING GIN ({column} gin_trgm_ops)"
            )
            try:
                cur.execute(sql)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Search: trigram index %s skipped (%s)", index_name, exc)

    def _ensure_pg_vector_tables(self, cur: Any) -> None:
        cur.execute("SAVEPOINT vector_ext")
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("RELEASE SAVEPOINT vector_ext")
            self.has_pgvector = True
        except Exception as exc:  # noqa: BLE001
            cur.execute("ROLLBACK TO SAVEPOINT vector_ext")
            logger.warning(
                "pgvector: CREATE EXTENSION vector failed (%s); "
                "vector tier disabled until role has privileges",
                exc,
            )
            self.has_pgvector = False
            return

        for stmt in _split_pg_statements(_PG_VECTOR_TABLES):
            cur.execute(stmt)

        # Embedding column (added separately so we can react to
        # dimension drift).
        if not self._pg_column_exists(cur, "search_vectors", "embedding"):
            cur.execute(
                "ALTER TABLE search_vectors "
                f"ADD COLUMN embedding vector({self._embedding_dim})"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS ix_search_vectors_hnsw "
                "ON search_vectors USING hnsw (embedding vector_cosine_ops)"
            )

    @staticmethod
    def _probe_pg_extension(cur: Any, name: str) -> None:
        cur.execute(f"SAVEPOINT probe_{name}")
        try:
            cur.execute(f"CREATE EXTENSION IF NOT EXISTS {name}")
            cur.execute(f"RELEASE SAVEPOINT probe_{name}")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Search: CREATE EXTENSION %s skipped (%s)", name, exc)
            cur.execute(f"ROLLBACK TO SAVEPOINT probe_{name}")

    @staticmethod
    def _pg_extension_present(cur: Any, name: str) -> bool:
        cur.execute(
            "SELECT 1 FROM pg_extension WHERE extname = ?",
            (name,),
        )
        return cur.fetchone() is not None

    @staticmethod
    def _pg_column_exists(cur: Any, table: str, column: str) -> bool:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = ? AND column_name = ?",
            (table, column),
        )
        return cur.fetchone() is not None

    # ── SQLite ──

    def _ensure_sqlite(self, db: Any) -> None:
        with db.connection() as conn:
            conn.executescript(_SQLITE_OUTBOX_AND_STATE)
            # Add claimed_at column to outbox if missing (idempotent).
            cur = conn.execute("PRAGMA table_info(search_outbox)")
            cols = {row[1] for row in cur.fetchall()}
            if "claimed_at" not in cols:
                conn.execute("ALTER TABLE search_outbox ADD COLUMN claimed_at TEXT")
        self.lexical_backend = "sqlite-like"
        logger.info(
            "Search schema (SQLite) ensured: lexical=%s vector_enabled=%s",
            self.lexical_backend,
            self._vector_enabled,
        )


def lexical_schema_version() -> int:
    return LEXICAL_SCHEMA_VERSION


__all__ = [
    "SearchSchema",
    "LEXICAL_SCHEMA_VERSION",
    "lexical_schema_version",
]
