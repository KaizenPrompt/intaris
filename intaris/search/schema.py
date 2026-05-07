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


# Tables whose generated tsvector columns we will NOT auto-rebuild on a
# running deployment because the rewrite takes an ACCESS EXCLUSIVE lock
# and blocks the lifespan startup long enough to fail K8s liveness
# probes. For these, drift is logged and the operator must run the
# manual ``ALTER TABLE`` themselves during a maintenance window. The
# threshold is conservative: 50k rows of audit content is enough to
# rewrite slower than a 60-second probe budget on a busy DB.
_PG_AUTO_REBUILD_ROW_THRESHOLD = 50_000


def _extract_source(expr: str) -> str:
    """Return the inner ``coalesce(<col>,'')`` source from a tsvector
    expression so the fallback path can rebuild the plain form.

    Both expression flavors use the same ``coalesce(<source>,'')`` core,
    so we slice it out via simple string scanning. Robust for our
    constrained inputs (we never call this with arbitrary user SQL).
    """
    marker = "coalesce("
    start = expr.find(marker)
    if start < 0:
        return expr
    start += len(marker)
    depth = 1
    i = start
    while i < len(expr) and depth > 0:
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    inner = expr[start:i]
    # Drop the trailing ",''"
    return inner.rsplit(",", 1)[0].strip()


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

    def __init__(
        self,
        *,
        vector_enabled: bool,
        embedding_dim: int,
        pgvector_enabled: bool = False,
    ) -> None:
        self._vector_enabled = vector_enabled
        # Provider-specific gate. Only ``pgvector`` requires the
        # ``vector`` PG extension and the ``search_vectors`` table; the
        # ``qdrant`` provider stores vectors externally and must NOT
        # trigger ``CREATE EXTENSION vector`` (which logs a misleading
        # warning on PG clusters where the extension binary is absent).
        self._pgvector_enabled = pgvector_enabled
        self._embedding_dim = max(1, int(embedding_dim or 1))
        self.lexical_backend: str = "unknown"
        self.has_unaccent: bool = False
        self.has_pg_trgm: bool = False
        # ``unaccent(text)`` is declared STABLE because the dictionary
        # file can be reloaded at runtime, so PG rejects it inside
        # generated column expressions which require IMMUTABLE. We
        # work around it by creating a SQL wrapper function declared
        # IMMUTABLE that defers to ``unaccent('unaccent', $1)``. When
        # this flag is true, ``intaris_immutable_unaccent`` exists and
        # the tsvector expressions reference it.
        self.has_immutable_unaccent: bool = False
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

            # When ``unaccent`` is available, install an IMMUTABLE
            # wrapper so the generated tsvector columns can fold
            # diacritics without PG rejecting the expression.
            if self.has_unaccent:
                self.has_immutable_unaccent = self._create_immutable_unaccent_wrapper(
                    cur
                )

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

            # Outbox + state are created above unconditionally so the
            # vector provider can be toggled at runtime. The pgvector
            # extension + ``search_vectors`` table are only created
            # when the provider is actually pgvector — running them
            # for qdrant or disabled would either fail noisily on
            # clusters without the extension binary or create dead
            # storage that nothing writes to.
            if self._pgvector_enabled:
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
            "Search schema (PG) ensured: lexical=%s unaccent=%s trgm=%s "
            "vector_enabled=%s pgvector=%s",
            self.lexical_backend,
            self.has_unaccent,
            self.has_pg_trgm,
            self._vector_enabled,
            self._pgvector_enabled,
        )

    def _ensure_pg_tsvector_columns(self, cur: Any) -> None:
        """Add or rebuild ``tsvector`` generated columns on canonical tables.

        Expression selection:

        - When ``intaris_immutable_unaccent`` is available, use
          ``to_tsvector('simple', intaris_immutable_unaccent(coalesce(...,'')))``.
        - Otherwise use the plain ``to_tsvector('simple', coalesce(...,''))``
          form. Diacritic folding is not available on the tsvector
          path in that case; queries still fold via ``pg_trgm`` /
          Python normalization but recall is degraded.

        For existing columns, we read the current generation
        expression via ``pg_get_expr`` and check whether it agrees
        with the desired flavor (wrapper vs plain) on the same source
        column. This is intentionally a *semantic* check, not a
        textual one: ``pg_get_expr`` returns canonicalized SQL
        (``'simple'::regconfig``, ``COALESCE``, ``''::text``) that
        will never match a hand-authored expression byte-for-byte, so
        we'd rebuild on every startup if we compared text.

        When drift is detected on a table whose live row count exceeds
        ``_PG_AUTO_REBUILD_ROW_THRESHOLD``, we log a notice with the
        manual SQL the operator can run and continue without
        rebuilding. ``ALTER TABLE ... DROP COLUMN`` plus
        ``ADD COLUMN ... GENERATED STORED`` rewrites the entire table
        under an ACCESS EXCLUSIVE lock; on a busy deployment with a
        large ``audit_log`` this can run long enough that K8s liveness
        probes kill the pod, putting the cluster in a restart loop.
        """
        desired_expr = self._tsvector_expression
        for table, column, source_expr in _PG_TSVECTOR_COLUMNS:
            target = desired_expr(source_expr)
            existing_expr = self._pg_column_generation_expr(cur, table, column)
            if existing_expr is None:
                # Column doesn't exist — add fresh. We accept the
                # one-time rewrite cost here because there's no other
                # way to add a generated column. Operators of huge
                # tables can pre-create the column manually before
                # rolling out the search-enabled image to skip this.
                self._add_tsvector_column(cur, table, column, target)
                continue
            if self._tsvector_expr_matches(
                existing_expr, target, source_expr=source_expr
            ):
                continue
            # Column exists with a different flavor than desired.
            row_count = self._pg_table_rowcount(cur, table)
            if row_count is not None and row_count > _PG_AUTO_REBUILD_ROW_THRESHOLD:
                manual_sql = (
                    f"ALTER TABLE {table} DROP COLUMN {column} CASCADE; "
                    f"ALTER TABLE {table} ADD COLUMN {column} tsvector "
                    f"GENERATED ALWAYS AS ({target}) STORED;"
                )
                logger.warning(
                    "Search: %s.%s tsvector flavor mismatch (existing=%s desired=%s); "
                    "table has ~%d rows (>%d) — skipping auto-rebuild to avoid "
                    "blocking startup. Run during a maintenance window: %s",
                    table,
                    column,
                    "wrapper"
                    if "intaris_immutable_unaccent" in existing_expr.lower()
                    else "plain",
                    "wrapper" if self.has_immutable_unaccent else "plain",
                    row_count,
                    _PG_AUTO_REBUILD_ROW_THRESHOLD,
                    manual_sql,
                )
                self.notes.append(
                    f"{table}.{column} tsvector rebuild deferred ({row_count} rows)"
                )
                continue
            # Small table — safe to rebuild synchronously. Drop with
            # CASCADE so any dependent GIN index goes away too —
            # ``_ensure_pg_gin_indexes`` re-creates the index right
            # after this method returns.
            logger.info(
                "Search: rebuilding %s.%s (tsvector flavor change: %s -> %s, %d rows)",
                table,
                column,
                "wrapper"
                if "intaris_immutable_unaccent" in existing_expr.lower()
                else "plain",
                "wrapper" if self.has_immutable_unaccent else "plain",
                row_count or 0,
            )
            cur.execute(f"ALTER TABLE {table} DROP COLUMN {column} CASCADE")
            self._add_tsvector_column(cur, table, column, target)

    def _tsvector_expression(self, source_expr: str) -> str:
        if self.has_immutable_unaccent:
            return (
                "to_tsvector('simple', "
                f"intaris_immutable_unaccent(coalesce({source_expr},'')))"
            )
        return f"to_tsvector('simple', coalesce({source_expr},''))"

    @staticmethod
    def _tsvector_expr_matches(
        existing: str, desired: str, *, source_expr: str | None = None
    ) -> bool:
        """Decide whether an existing tsvector generation expression
        matches the desired one for our purposes.

        We do NOT do textual comparison. ``pg_get_expr`` returns
        canonicalized SQL that adds type casts (``'simple'::regconfig``,
        ``''::text``) and uppercases function names (``COALESCE``),
        which would never match the hand-authored ``desired`` string
        even after whitespace + case normalization — that bug caused
        every startup to drop and rebuild every tsvector column,
        triggering full table rewrites under ACCESS EXCLUSIVE locks.

        Instead we compare semantic features:

        1. Both must reference the same source column. We scan for the
           source identifier inside ``coalesce(...)`` on each side.
        2. Both must agree on whether to use the ``intaris_immutable_unaccent``
           wrapper. That is the only flavor difference our code
           generates.

        Anything else PG might canonicalize is fine — we wrote the
        original expression ourselves on a previous run.
        """

        def _has_wrapper(s: str) -> bool:
            return "intaris_immutable_unaccent" in s.lower()

        def _references_column(s: str, col: str) -> bool:
            # Match ``col`` as a whole word. Lowercased on both sides
            # because PG canonicalizes ``coalesce`` to ``COALESCE`` and
            # similar.
            text = s.lower()
            target = col.lower()
            idx = 0
            while True:
                idx = text.find(target, idx)
                if idx < 0:
                    return False
                left_ok = idx == 0 or not (
                    text[idx - 1].isalnum() or text[idx - 1] == "_"
                )
                end = idx + len(target)
                right_ok = end == len(text) or not (
                    text[end].isalnum() or text[end] == "_"
                )
                if left_ok and right_ok:
                    return True
                idx = end

        if _has_wrapper(existing) != _has_wrapper(desired):
            return False
        if source_expr is None:
            # Best-effort: extract the source column from the desired
            # expression by walking ``coalesce(<source>,...)``.
            source_expr = _extract_source(desired)
        if not source_expr:
            return True
        return _references_column(existing, source_expr)

    @staticmethod
    def _pg_table_rowcount(cur: Any, table: str) -> int | None:
        """Return the cached live-tuple count for ``table``, or None.

        Uses ``pg_stat_user_tables.n_live_tup`` which is fast (constant
        time) and good enough for a "is this table small or large"
        decision. ANALYZE keeps it accurate within a few percent of
        ``COUNT(*)`` on busy tables.
        """
        try:
            cur.execute(
                "SELECT n_live_tup FROM pg_stat_user_tables WHERE relname = ?",
                (table,),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Search: pg_stat_user_tables lookup failed for %s (%s)", table, exc
            )
            return None
        row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return None

    def _add_tsvector_column(
        self, cur: Any, table: str, column: str, expr: str
    ) -> None:
        """Add a tsvector generated column with the given expression.

        On failure (typically: PG complains the expression isn't
        IMMUTABLE because we didn't get the wrapper installed), we
        fall back to the plain form once and degrade
        ``has_immutable_unaccent`` so subsequent tables get the same
        treatment without re-trying.
        """
        sp = f"add_{table}_{column}"
        cur.execute(f"SAVEPOINT {sp}")
        try:
            cur.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} tsvector "
                f"GENERATED ALWAYS AS ({expr}) STORED"
            )
            cur.execute(f"RELEASE SAVEPOINT {sp}")
        except Exception as exc:  # noqa: BLE001
            cur.execute(f"ROLLBACK TO SAVEPOINT {sp}")
            plain = f"to_tsvector('simple', coalesce({_extract_source(expr)},''))"
            log = logger.warning if self.has_immutable_unaccent else logger.info
            log(
                "Search: %s.%s falls back to plain tsvector (%s)",
                table,
                column,
                exc if self.has_immutable_unaccent else "no immutable unaccent wrapper",
            )
            cur.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} tsvector "
                f"GENERATED ALWAYS AS ({plain}) STORED"
            )
            self.has_immutable_unaccent = False

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

    @staticmethod
    def _create_immutable_unaccent_wrapper(cur: Any) -> bool:
        """Create the ``intaris_immutable_unaccent`` SQL wrapper.

        ``unaccent(text)`` is declared STABLE in core PG; generated
        columns require IMMUTABLE. We declare a wrapper IMMUTABLE
        ourselves — operator's promise that the dictionary won't
        change in a way that invalidates stored values. This is the
        documented workaround.

        Returns ``True`` when the wrapper exists in the current
        database. Failures (typically: ``CREATE FUNCTION`` denied on
        managed PG) are swallowed and we return ``False``; callers
        fall back to the plain tsvector expression.
        """
        cur.execute("SAVEPOINT install_immutable_unaccent")
        try:
            cur.execute(
                "CREATE OR REPLACE FUNCTION "
                "intaris_immutable_unaccent(text) RETURNS text "
                "AS $$ SELECT public.unaccent('public.unaccent', $1) $$ "
                "LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE"
            )
            cur.execute("RELEASE SAVEPOINT install_immutable_unaccent")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "Search: intaris_immutable_unaccent wrapper not installed "
                "(%s); diacritic folding via tsvector disabled",
                type(exc).__name__,
            )
            cur.execute("ROLLBACK TO SAVEPOINT install_immutable_unaccent")
            return False

    @staticmethod
    def _pg_column_generation_expr(cur: Any, table: str, column: str) -> str | None:
        """Return the existing generation expression for a column, or None.

        Used to detect drift between the desired and stored expression
        when ``intaris_immutable_unaccent`` becomes available on a
        deployment that previously fell back to the plain form.
        """
        cur.execute(
            "SELECT pg_get_expr(d.adbin, d.adrelid) "
            "FROM pg_attribute a "
            "JOIN pg_attrdef d ON a.attrelid = d.adrelid "
            "  AND a.attnum = d.adnum "
            "WHERE a.attrelid = ?::regclass "
            "  AND a.attname = ? "
            "  AND a.attgenerated = 's'",
            (table, column),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return str(row[0]) if row[0] is not None else None

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
