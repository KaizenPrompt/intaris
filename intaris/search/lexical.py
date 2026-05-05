"""Lexical search over canonical Intaris tables.

Three per-kind helpers, each reading directly from the canonical
table that already holds the content. There is no projection table —
the canonical tables ARE the lexical search index, with a generated
``tsvector`` column added for Postgres and a Python ``LIKE``
fallback for SQLite.

| kind        | source                                | record filter                                   |
|-------------|---------------------------------------|-------------------------------------------------|
| summary     | session_summaries + agent_summaries   | (none)                                          |
| intention   | audit_log (deduped per session)       | record_type IN ('reasoning','checkpoint',...)   |
| reasoning   | audit_log                             | record_type IN ('reasoning','checkpoint')       |

Snippets use Postgres ``ts_headline`` on PG and a heuristic Python
fragmenter on SQLite. Queries always force ``user_id`` from the
caller — body-supplied scopes are never trusted.

SQLite path note: we register an ``intaris_fold`` UDF on the
underlying connection so the LIKE fallback can compare diacritic-
folded strings on both sides of the predicate (the query side is
folded in Python, the column side via the UDF). Connections that
don't expose ``create_function`` fall back to plain ``lower()`` —
acceptable for ASCII queries, weaker for accented content.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from intaris.search.types import (
    KIND_INTENTION,
    KIND_REASONING,
    KIND_SUMMARY,
    SearchMatch,
    fold_text,
)


def _ensure_sqlite_fold_udf(db: Any) -> str:
    """Register the ``intaris_fold`` UDF on the SQLite connection.

    Returns the SQL function name to use in queries: ``intaris_fold``
    when the UDF was registered, ``lower`` as a fallback. The UDF
    applies NFKD diacritic folding + casefold so that, combined with
    a ``fold_text``-folded query string, ``LIKE`` matches accented and
    unaccented text consistently.
    """
    if getattr(db, "backend", "sqlite") != "sqlite":
        return "lower"
    try:
        # Reach for the underlying sqlite3 connection. ``Database``
        # holds it on a thread-local (``_local.conn``) so we re-register
        # on whichever connection the current thread is using.
        local = getattr(db, "_local", None)
        conn = getattr(local, "conn", None) if local is not None else None
        if conn is None:
            return "lower"
        # ``deterministic=True`` lets SQLite use the function in
        # generated columns or expression indexes if we ever add them.
        conn.create_function("intaris_fold", 1, fold_text, deterministic=True)
        return "intaris_fold"
    except Exception as exc:  # noqa: BLE001
        logger.debug("SQLite intaris_fold UDF registration skipped (%s)", exc)
        return "lower"


logger = logging.getLogger(__name__)

SNIPPET_OPEN = "<mark>"
SNIPPET_CLOSE = "</mark>"
SNIPPET_MAX_FRAGMENTS = 2
SNIPPET_MAX_WORDS = 20
SNIPPET_MIN_WORDS = 5

_HEADLINE_OPTS = (
    f"StartSel={SNIPPET_OPEN},StopSel={SNIPPET_CLOSE},"
    f"MaxFragments={SNIPPET_MAX_FRAGMENTS},"
    f"MaxWords={SNIPPET_MAX_WORDS},MinWords={SNIPPET_MIN_WORDS}"
)


# ── Public API ────────────────────────────────────────────────────


def search_lexical(
    db: Any,
    *,
    user_id: str,
    q: str,
    kinds: Iterable[str],
    filters: dict[str, Any],
    limit: int,
    has_unaccent: bool = False,
    has_pg_trgm: bool = False,
) -> list[SearchMatch]:
    """Run lexical search across the requested kinds and merge results."""
    kind_set = {k for k in kinds if k in (KIND_SUMMARY, KIND_INTENTION, KIND_REASONING)}
    if not q or not q.strip() or not kind_set:
        return []

    rows: list[dict[str, Any]] = []
    if KIND_SUMMARY in kind_set:
        rows.extend(
            _search_summaries(
                db,
                user_id=user_id,
                q=q,
                filters=filters,
                limit=limit,
                has_unaccent=has_unaccent,
                has_pg_trgm=has_pg_trgm,
            )
        )
    if KIND_INTENTION in kind_set:
        rows.extend(
            _search_intentions(
                db,
                user_id=user_id,
                q=q,
                filters=filters,
                limit=limit,
                has_unaccent=has_unaccent,
                has_pg_trgm=has_pg_trgm,
            )
        )
    if KIND_REASONING in kind_set:
        rows.extend(
            _search_reasoning(
                db,
                user_id=user_id,
                q=q,
                filters=filters,
                limit=limit,
                has_unaccent=has_unaccent,
                has_pg_trgm=has_pg_trgm,
            )
        )

    rows.sort(key=lambda r: (r.get("score", 0.0), r.get("ts") or ""), reverse=True)
    out = [_to_match(r) for r in rows[:limit]]
    return out


def _to_match(row: dict[str, Any]) -> SearchMatch:
    return SearchMatch(
        session_id=str(row.get("session_id") or ""),
        kind=str(row.get("kind") or ""),
        ref_id=(str(row.get("ref_id")) if row.get("ref_id") is not None else None),
        role=row.get("role"),
        ts=row.get("ts"),
        snippet=row.get("snippet") or "",
        score=float(row.get("score") or 0.0),
        score_breakdown={"lexical": float(row.get("score") or 0.0)},
        agent_id=row.get("agent_id"),
    )


# ── Common filter helpers ────────────────────────────────────────


def _build_filters(
    filters: dict[str, Any], *, table_alias: str = ""
) -> tuple[list[str], list[Any]]:
    """Translate a filter dict into SQL fragments + params."""
    where: list[str] = []
    params: list[Any] = []
    prefix = f"{table_alias}." if table_alias else ""

    if filters.get("agent_id"):
        where.append(f"{prefix}agent_id = ?")
        params.append(filters["agent_id"])

    if filters.get("session_id"):
        where.append(f"{prefix}session_id = ?")
        params.append(filters["session_id"])

    sids = filters.get("session_ids")
    if sids:
        ph = ",".join(["?"] * len(sids))
        where.append(f"{prefix}session_id IN ({ph})")
        params.extend(sids)

    if filters.get("from_ts"):
        # session_summaries has window_end; audit_log has timestamp;
        # agent_summaries has created_at. The caller provides the
        # column name; we pass the from_ts/to_ts pair through and let
        # the per-kind helper inject the right column.
        where.append("__TS__ >= ?")
        params.append(filters["from_ts"])
    if filters.get("to_ts"):
        where.append("__TS__ <= ?")
        params.append(filters["to_ts"])

    return where, params


def _select(db: Any, sql: str, params: tuple[Any, ...]) -> list[Any]:
    with db.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ── Snippet helpers ──────────────────────────────────────────────


def _heuristic_snippet(text: str | None, query: str | None) -> str:
    if not text:
        return ""
    body = text.strip().replace("\n", " ")
    if not query:
        return body[:200]

    body_lower = body.lower()
    needle_lower = query.strip().lower()
    if not needle_lower:
        return body[:200]
    idx = body_lower.find(needle_lower)
    if idx < 0:
        # Try diacritic-folded match on both.
        folded_body = fold_text(body)
        folded_needle = fold_text(query)
        if folded_needle:
            idx = folded_body.find(folded_needle)
        if idx < 0:
            return body[:200]
    start = max(0, idx - 60)
    end = min(len(body), idx + len(needle_lower) + 60)
    return (
        ("..." if start > 0 else "")
        + body[start:idx]
        + SNIPPET_OPEN
        + body[idx : idx + len(needle_lower)]
        + SNIPPET_CLOSE
        + body[idx + len(needle_lower) : end]
        + ("..." if end < len(body) else "")
    )


# ── Per-kind queries ──────────────────────────────────────────────

# ── summary ──


def _search_summaries(
    db: Any,
    *,
    user_id: str,
    q: str,
    filters: dict[str, Any],
    limit: int,
    has_unaccent: bool,
    has_pg_trgm: bool,
) -> list[dict[str, Any]]:
    if db.backend == "postgresql":
        return _search_summaries_pg(
            db,
            user_id=user_id,
            q=q,
            filters=filters,
            limit=limit,
            has_pg_trgm=has_pg_trgm,
        )
    return _search_summaries_sqlite(
        db, user_id=user_id, q=q, filters=filters, limit=limit
    )


def _search_summaries_pg(
    db: Any,
    *,
    user_id: str,
    q: str,
    filters: dict[str, Any],
    limit: int,
    has_pg_trgm: bool,
) -> list[dict[str, Any]]:
    """``session_summaries`` and ``agent_summaries`` carry only the
    session reference; agent_id comes from the parent session via JOIN."""
    where_session, params_session = _build_filters(filters)
    where_session_with_ts = [
        w.replace("__TS__", "ss.window_end") for w in where_session
    ]
    where_session_with_ts = [
        w.replace("ss.session_id", "ss.session_id").replace("ss.agent_id", "s.agent_id")
        for w in where_session_with_ts
    ]
    # Build per-table filter strings using join aliases.
    sw_filters: list[str] = []
    sw_params: list[Any] = []
    for w, p in zip(where_session_with_ts, params_session, strict=False):
        # noop — we just keep both lists aligned.
        sw_filters.append(w.replace("session_id", "ss.session_id"))
        sw_params.append(p)

    sql = (
        "SELECT ss.id, ss.session_id, s.agent_id, ss.summary, "
        "       ss.window_end AS ts, "
        "       'window' AS variant, "
        "       ts_headline('simple', ss.summary, query, ?) AS snippet, "
        "       ts_rank(ss.summary_tsv, query) AS score "
        "FROM session_summaries AS ss "
        "JOIN sessions AS s "
        "  ON s.user_id = ss.user_id AND s.session_id = ss.session_id, "
        "  plainto_tsquery('simple', ?) AS query "
        "WHERE ss.user_id = ? AND ss.summary_tsv @@ query"
    )
    params: list[Any] = [_HEADLINE_OPTS, q, user_id]
    for w, p in zip(where_session_with_ts, params_session, strict=False):
        sql += " AND " + w
        params.append(p)
    sql += " ORDER BY score DESC, ts DESC LIMIT ?"
    params.append(limit)
    rows_session = _select(db, sql, tuple(params))

    sql_agent = (
        "SELECT ag.id, ag.session_id, s.agent_id, ag.summary, "
        "       ag.created_at AS ts, "
        "       'agent' AS variant, "
        "       ts_headline('simple', ag.summary, query, ?) AS snippet, "
        "       ts_rank(ag.summary_tsv, query) AS score "
        "FROM agent_summaries AS ag "
        "JOIN sessions AS s "
        "  ON s.user_id = ag.user_id AND s.session_id = ag.session_id, "
        "  plainto_tsquery('simple', ?) AS query "
        "WHERE ag.user_id = ? AND ag.summary_tsv @@ query"
    )
    params_a: list[Any] = [_HEADLINE_OPTS, q, user_id]
    where_agent_with_ts = [
        w.replace("__TS__", "ag.created_at").replace("session_id", "ag.session_id")
        for w in where_session
    ]
    for w, p in zip(where_agent_with_ts, params_session, strict=False):
        sql_agent += " AND " + w
        params_a.append(p)
    sql_agent += " ORDER BY score DESC, ts DESC LIMIT ?"
    params_a.append(limit)
    rows_agent = _select(db, sql_agent, tuple(params_a))

    rows = list(rows_session) + list(rows_agent)

    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "session_id": str(row[1]),
                "kind": KIND_SUMMARY,
                "ref_id": str(row[0]),
                "role": None,
                "ts": str(row[4]) if row[4] else None,
                "agent_id": (str(row[2]) if row[2] else None),
                "snippet": row[6] if row[6] else _heuristic_snippet(row[3], q),
                "score": float(row[7] or 0.0),
            }
        )
    return out


def _search_summaries_sqlite(
    db: Any,
    *,
    user_id: str,
    q: str,
    filters: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    """SQLite path. Joins to ``sessions`` for agent_id metadata.

    Uses the ``intaris_fold`` UDF (when registered) so accented text
    in the column matches an unaccented query and vice versa. Falls
    back to ``lower()`` when the UDF can't be registered.
    """
    fold = _ensure_sqlite_fold_udf(db)
    needle = fold_text(q)
    where_base, params_base = _build_filters(filters)
    where_session = [
        w.replace("__TS__", "ss.window_end")
        .replace("agent_id", "s.agent_id")
        .replace("session_id", "ss.session_id")
        for w in where_base
    ]
    where_agent = [
        w.replace("__TS__", "ag.created_at")
        .replace("agent_id", "s.agent_id")
        .replace("session_id", "ag.session_id")
        for w in where_base
    ]

    rows: list[dict[str, Any]] = []
    sql = (
        "SELECT ss.id, ss.session_id, s.agent_id, ss.summary, ss.window_end "
        "FROM session_summaries AS ss "
        "JOIN sessions AS s "
        "  ON s.user_id = ss.user_id AND s.session_id = ss.session_id "
        f"WHERE ss.user_id = ? AND {fold}(ss.summary) LIKE ?"
    )
    params: list[Any] = [user_id, f"%{needle}%"]
    if where_session:
        sql += " AND " + " AND ".join(where_session)
        params.extend(params_base)
    sql += " ORDER BY ss.window_end DESC LIMIT ?"
    params.append(limit)
    for row in _select(db, sql, tuple(params)):
        rows.append(
            {
                "session_id": str(row[1]),
                "kind": KIND_SUMMARY,
                "ref_id": str(row[0]),
                "role": None,
                "ts": str(row[4]) if row[4] else None,
                "agent_id": (str(row[2]) if row[2] else None),
                "snippet": _heuristic_snippet(str(row[3] or ""), q),
                "score": 0.5,
            }
        )

    sql_a = (
        "SELECT ag.id, ag.session_id, s.agent_id, ag.summary, ag.created_at "
        "FROM agent_summaries AS ag "
        "JOIN sessions AS s "
        "  ON s.user_id = ag.user_id AND s.session_id = ag.session_id "
        f"WHERE ag.user_id = ? AND {fold}(ag.summary) LIKE ?"
    )
    params_a: list[Any] = [user_id, f"%{needle}%"]
    if where_agent:
        sql_a += " AND " + " AND ".join(where_agent)
        params_a.extend(params_base)
    sql_a += " ORDER BY ag.created_at DESC LIMIT ?"
    params_a.append(limit)
    for row in _select(db, sql_a, tuple(params_a)):
        rows.append(
            {
                "session_id": str(row[1]),
                "kind": KIND_SUMMARY,
                "ref_id": str(row[0]),
                "role": None,
                "ts": str(row[4]) if row[4] else None,
                "agent_id": (str(row[2]) if row[2] else None),
                "snippet": _heuristic_snippet(str(row[3] or ""), q),
                "score": 0.4,
            }
        )

    return rows


# ── intention ──


def _search_intentions(
    db: Any,
    *,
    user_id: str,
    q: str,
    filters: dict[str, Any],
    limit: int,
    has_unaccent: bool,
    has_pg_trgm: bool,
) -> list[dict[str, Any]]:
    if db.backend == "postgresql":
        return _search_intentions_pg(
            db,
            user_id=user_id,
            q=q,
            filters=filters,
            limit=limit,
            has_pg_trgm=has_pg_trgm,
        )
    return _search_intentions_sqlite(
        db, user_id=user_id, q=q, filters=filters, limit=limit
    )


def _search_intentions_pg(
    db: Any,
    *,
    user_id: str,
    q: str,
    filters: dict[str, Any],
    limit: int,
    has_pg_trgm: bool,
) -> list[dict[str, Any]]:
    where, params = _build_filters(filters)
    where_with_ts = [w.replace("__TS__", "timestamp") for w in where]

    # DISTINCT ON (session_id, intention) collapses repeated snapshots
    # into one row per distinct intention text per session, picking
    # the earliest occurrence. We wrap the DISTINCT ON select in a
    # subquery so the outer LIMIT applies after relevance ranking.
    inner = (
        "SELECT DISTINCT ON (session_id, intention) "
        "  id, session_id, agent_id, intention, timestamp, "
        "  ts_headline('simple', intention, query, ?) AS snippet, "
        "  ts_rank(intention_tsv, query) AS score "
        "FROM audit_log, plainto_tsquery('simple', ?) AS query "
        "WHERE user_id = ? AND intention IS NOT NULL "
        "  AND intention_tsv @@ query"
    )
    qparams: list[Any] = [_HEADLINE_OPTS, q, user_id]
    if where_with_ts:
        inner += " AND " + " AND ".join(where_with_ts)
        qparams.extend(params)
    inner += " ORDER BY session_id, intention, timestamp ASC"

    sql = (
        f"SELECT * FROM ({inner}) AS deduped "
        "ORDER BY score DESC, timestamp DESC LIMIT ?"
    )
    qparams.append(limit)

    rows = _select(db, sql, tuple(qparams))
    if not rows and has_pg_trgm:
        rows = _search_intentions_pg_trgm(
            db, user_id=user_id, q=q, filters=filters, limit=limit
        )

    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "session_id": str(row[1]),
                "kind": KIND_INTENTION,
                "ref_id": str(row[0]),
                "role": None,
                "ts": str(row[4]) if row[4] else None,
                "agent_id": (str(row[2]) if row[2] else None),
                "snippet": row[5] if row[5] else _heuristic_snippet(row[3], q),
                "score": float(row[6] or 0.0),
            }
        )
    return out


def _search_intentions_pg_trgm(
    db: Any,
    *,
    user_id: str,
    q: str,
    filters: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    where, params = _build_filters(filters)
    where_with_ts = [w.replace("__TS__", "timestamp") for w in where]
    inner = (
        "SELECT DISTINCT ON (session_id, intention) "
        "  id, session_id, agent_id, intention, timestamp, NULL AS snippet, "
        "  similarity(intention, ?) AS score "
        "FROM audit_log "
        "WHERE user_id = ? AND intention IS NOT NULL AND intention % ?"
    )
    qparams: list[Any] = [q, user_id, q]
    if where_with_ts:
        inner += " AND " + " AND ".join(where_with_ts)
        qparams.extend(params)
    inner += " ORDER BY session_id, intention, timestamp ASC"
    sql = (
        f"SELECT * FROM ({inner}) AS deduped "
        "ORDER BY score DESC, timestamp DESC LIMIT ?"
    )
    qparams.append(limit)
    return _select(db, sql, tuple(qparams))


def _search_intentions_sqlite(
    db: Any,
    *,
    user_id: str,
    q: str,
    filters: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    fold = _ensure_sqlite_fold_udf(db)
    needle = fold_text(q)
    where_base, params_base = _build_filters(filters)
    where_with_ts = [w.replace("__TS__", "timestamp") for w in where_base]

    # SQLite has no DISTINCT ON; emulate with GROUP BY and MIN(timestamp).
    sql = (
        "SELECT MIN(id) AS id, session_id, agent_id, intention, "
        "       MIN(timestamp) AS ts "
        "FROM audit_log "
        "WHERE user_id = ? AND intention IS NOT NULL "
        f"  AND {fold}(intention) LIKE ?"
    )
    params: list[Any] = [user_id, f"%{needle}%"]
    if where_with_ts:
        sql += " AND " + " AND ".join(where_with_ts)
        params.extend(params_base)
    sql += " GROUP BY session_id, intention ORDER BY ts DESC LIMIT ?"
    params.append(limit)

    out: list[dict[str, Any]] = []
    for row in _select(db, sql, tuple(params)):
        out.append(
            {
                "session_id": str(row[1]),
                "kind": KIND_INTENTION,
                "ref_id": str(row[0]),
                "role": None,
                "ts": str(row[4]) if row[4] else None,
                "agent_id": (str(row[2]) if row[2] else None),
                "snippet": _heuristic_snippet(str(row[3] or ""), q),
                "score": 0.5,
            }
        )
    return out


# ── reasoning ──


def _search_reasoning(
    db: Any,
    *,
    user_id: str,
    q: str,
    filters: dict[str, Any],
    limit: int,
    has_unaccent: bool,
    has_pg_trgm: bool,
) -> list[dict[str, Any]]:
    if db.backend == "postgresql":
        return _search_reasoning_pg(
            db, user_id=user_id, q=q, filters=filters, limit=limit
        )
    return _search_reasoning_sqlite(
        db, user_id=user_id, q=q, filters=filters, limit=limit
    )


def _search_reasoning_pg(
    db: Any,
    *,
    user_id: str,
    q: str,
    filters: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    where, params = _build_filters(filters)
    where_with_ts = [w.replace("__TS__", "timestamp") for w in where]

    sql = (
        "SELECT id, session_id, agent_id, content, timestamp, record_type, "
        "       ts_headline('simple', content, query, ?) AS snippet, "
        "       ts_rank(content_tsv, query) AS score "
        "FROM audit_log, plainto_tsquery('simple', ?) AS query "
        "WHERE user_id = ? AND record_type IN ('reasoning','checkpoint') "
        "  AND content_tsv @@ query"
    )
    qparams: list[Any] = [_HEADLINE_OPTS, q, user_id]
    if where_with_ts:
        sql += " AND " + " AND ".join(where_with_ts)
        qparams.extend(params)
    sql += " ORDER BY score DESC, timestamp DESC LIMIT ?"
    qparams.append(limit)

    out: list[dict[str, Any]] = []
    for row in _select(db, sql, tuple(qparams)):
        # Distinguish user-message reasoning from agent reasoning by
        # the well-known prefix.
        content = str(row[3] or "")
        role = "user" if content.startswith("User message:") else "assistant"
        out.append(
            {
                "session_id": str(row[1]),
                "kind": KIND_REASONING,
                "ref_id": str(row[0]),
                "role": role,
                "ts": str(row[4]) if row[4] else None,
                "agent_id": (str(row[2]) if row[2] else None),
                "snippet": row[6] if row[6] else _heuristic_snippet(content, q),
                "score": float(row[7] or 0.0),
            }
        )
    return out


def _search_reasoning_sqlite(
    db: Any,
    *,
    user_id: str,
    q: str,
    filters: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    fold = _ensure_sqlite_fold_udf(db)
    needle = fold_text(q)
    where_base, params_base = _build_filters(filters)
    where_with_ts = [w.replace("__TS__", "timestamp") for w in where_base]

    sql = (
        "SELECT id, session_id, agent_id, content, timestamp, record_type "
        "FROM audit_log "
        "WHERE user_id = ? AND record_type IN ('reasoning','checkpoint') "
        f"  AND {fold}(content) LIKE ?"
    )
    params: list[Any] = [user_id, f"%{needle}%"]
    if where_with_ts:
        sql += " AND " + " AND ".join(where_with_ts)
        params.extend(params_base)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    out: list[dict[str, Any]] = []
    for row in _select(db, sql, tuple(params)):
        content = str(row[3] or "")
        role = "user" if content.startswith("User message:") else "assistant"
        out.append(
            {
                "session_id": str(row[1]),
                "kind": KIND_REASONING,
                "ref_id": str(row[0]),
                "role": role,
                "ts": str(row[4]) if row[4] else None,
                "agent_id": (str(row[2]) if row[2] else None),
                "snippet": _heuristic_snippet(content, q),
                "score": 0.5,
            }
        )
    return out


__all__ = [
    "SNIPPET_OPEN",
    "SNIPPET_CLOSE",
    "search_lexical",
]
