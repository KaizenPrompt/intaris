"""Tests for the search subsystem.

Coverage:

- ``test_schema_*``         schema bootstrap on SQLite
- ``test_outbox_*``         enqueue / claim / mark cycle (vector tier)
- ``test_lexical_*``        per-kind queries against canonical tables
- ``test_intention_*``      DISTINCT-ON dedup behavior
- ``test_service_*``        end-to-end query orchestration
- ``test_qdrant_local_*``   Qdrant local-mode + dense+sparse hybrid
- ``test_pgvector_*``       skipped here (require live PG container)

Lexical tier always works; vector tier is exercised through stubs and
local-mode Qdrant.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from intaris.audit import AuditStore
from intaris.config import DBConfig, SearchConfig
from intaris.db import Database
from intaris.search import outbox
from intaris.search.cursor import decode_cursor, encode_cursor
from intaris.search.fusion import rrf_fuse
from intaris.search.lexical import search_lexical
from intaris.search.schema import SearchSchema
from intaris.search.service import SearchService
from intaris.search.types import (
    KIND_INTENTION,
    KIND_REASONING,
    KIND_SUMMARY,
    fold_text,
    truncate_text,
)

TEST_USER = "alice@example.com"
OTHER_USER = "bob@example.com"


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    config = DBConfig()
    config.path = str(tmp_path / "test.db")
    return Database(config)


@pytest.fixture
def search_config():
    cfg = SearchConfig()
    cfg.enabled = True
    cfg.vector_provider = "disabled"
    cfg.embedding_model = ""
    cfg.indexer_poll_interval_seconds = 0.05
    return cfg


@pytest.fixture
def schema(db, search_config):
    s = SearchSchema(
        vector_enabled=search_config.vector_enabled(),
        embedding_dim=search_config.embedding_dim,
    )
    s.ensure(db)
    return s


@pytest.fixture
def service(db, search_config):
    return SearchService(db=db, config=search_config)


@pytest.fixture
def audit_store(db):
    AuditStore.set_search_service(None)
    return AuditStore(db)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_session(
    db,
    *,
    session_id: str,
    user_id: str = TEST_USER,
    agent_id: str | None = "agent-1",
    title: str = "rocket launch",
    intention: str = "track upcoming rocket launches",
):
    with db.cursor() as cur:
        now = _now()
        cur.execute(
            "INSERT INTO sessions "
            "  (user_id, session_id, intention, status, created_at, "
            "   updated_at, agent_id, last_activity_at, title) "
            "VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?)",
            (
                user_id,
                session_id,
                intention,
                now,
                now,
                agent_id,
                now,
                title,
            ),
        )


def _insert_summary(
    db, *, session_id: str, summary: str, user_id: str = TEST_USER
) -> str:
    sid = str(uuid.uuid4())
    now = _now()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO session_summaries "
            "  (id, user_id, session_id, window_start, window_end, "
            "   trigger, summary_type, summary, tools_used, "
            "   intent_alignment, risk_indicators, call_count, "
            "   created_at) "
            "VALUES (?, ?, ?, ?, ?, 'manual', 'window', ?, '[]', "
            "        'aligned', '[]', 0, ?)",
            (sid, user_id, session_id, now, now, summary, now),
        )
    return sid


# ── Types ──────────────────────────────────────────────────────────


def test_truncate_text_respects_byte_budget():
    text = "ä" * 100
    out = truncate_text(text, 100)
    assert len(out.encode("utf-8")) <= 100
    assert out


def test_fold_text_strips_diacritics():
    assert fold_text("Příliš žluťoučký") == "prilis zlutoucky"


# ── Schema ─────────────────────────────────────────────────────────


def test_schema_creates_outbox_and_state_on_sqlite(db, schema):
    with db.cursor() as cur:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('search_outbox','search_state','search_vectors')"
        )
        names = {row[0] for row in cur.fetchall()}
    assert {"search_outbox", "search_state", "search_vectors"} == names
    assert schema.lexical_backend == "sqlite-like"


def test_schema_is_idempotent(db, schema, search_config):
    SearchSchema(
        vector_enabled=False, embedding_dim=search_config.embedding_dim
    ).ensure(db)
    SearchSchema(
        vector_enabled=False, embedding_dim=search_config.embedding_dim
    ).ensure(db)
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM search_state")
        assert cur.fetchone()[0] == 1


def test_split_pg_statements_handles_quoted_semicolons():
    """Schema bootstrap relies on _split_pg_statements to break a DDL
    script into individual ``cur.execute`` calls. Verify it doesn't
    split inside string literals (e.g. CHECK constraint values)."""
    from intaris.search.schema import _split_pg_statements

    script = """
    -- a comment ; not a separator
    CREATE TABLE t (x text CHECK (x IN ('a;b','c')));
    INSERT INTO t (x) VALUES ('a;b');
    """
    stmts = _split_pg_statements(script)
    assert len(stmts) == 2
    assert "CHECK" in stmts[0]
    assert "INSERT" in stmts[1]


def test_split_pg_statements_handles_double_quotes():
    from intaris.search.schema import _split_pg_statements

    script = 'CREATE INDEX "ix;weird" ON t(x); CREATE TABLE u (a int);'
    stmts = _split_pg_statements(script)
    assert len(stmts) == 2
    assert '"ix;weird"' in stmts[0]


def test_extract_source_unwraps_coalesce_payload():
    """Used by the tsvector fallback to rebuild the plain form when
    the IMMUTABLE wrapper isn't installable."""
    from intaris.search.schema import _extract_source

    assert (
        _extract_source("to_tsvector('simple', coalesce(intention,''))") == "intention"
    )
    assert (
        _extract_source(
            "to_tsvector('simple', intaris_immutable_unaccent(coalesce(content,'')))"
        )
        == "content"
    )


def test_tsvector_expression_picks_wrapper_when_available():
    from intaris.search.schema import SearchSchema

    s = SearchSchema(vector_enabled=False, embedding_dim=1536)
    s.has_unaccent = True
    s.has_immutable_unaccent = True
    expr = s._tsvector_expression("intention")
    assert "intaris_immutable_unaccent" in expr
    assert "coalesce(intention,'')" in expr


def test_tsvector_expression_falls_back_when_wrapper_missing():
    from intaris.search.schema import SearchSchema

    s = SearchSchema(vector_enabled=False, embedding_dim=1536)
    s.has_unaccent = True
    s.has_immutable_unaccent = False
    expr = s._tsvector_expression("intention")
    assert "intaris_immutable_unaccent" not in expr
    assert "unaccent" not in expr
    assert "coalesce(intention,'')" in expr


def test_tsvector_expr_matches_collapses_whitespace():
    """``pg_get_expr`` returns canonicalized SQL — comparison must be
    whitespace/case insensitive."""
    from intaris.search.schema import SearchSchema

    a = "to_tsvector('simple', coalesce(intention,''))"
    b = "to_tsvector('simple',  COALESCE(intention, ''))"
    assert SearchSchema._tsvector_expr_matches(a, b)
    assert not SearchSchema._tsvector_expr_matches(
        a, "to_tsvector('simple', coalesce(content,''))"
    )


# ── Outbox ─────────────────────────────────────────────────────────


def test_outbox_enqueue_and_claim(db, schema):
    outbox.enqueue(db, op=outbox.OP_EMBED, payload={"text": "hello"})
    outbox.enqueue(db, op=outbox.OP_EMBED, payload={"text": "world"})
    rows = outbox.claim_due(db, limit=10)
    assert {r["payload"]["text"] for r in rows} == {"hello", "world"}


def test_outbox_claim_marks_rows_claimed(db, schema):
    outbox.enqueue(db, op=outbox.OP_EMBED, payload={"x": 1})
    outbox.enqueue(db, op=outbox.OP_EMBED, payload={"x": 2})
    first = outbox.claim_due(db, limit=10)
    assert len(first) == 2
    second = outbox.claim_due(db, limit=10)
    assert second == []


def test_outbox_invalid_op_raises(db, schema):
    with pytest.raises(ValueError):
        outbox.enqueue(db, op="bogus", payload={})


# ── Lexical: summary ──────────────────────────────────────────────


def test_lexical_summary_finds_match(db, schema):
    _seed_session(db, session_id="sess-1")
    _insert_summary(
        db, session_id="sess-1", summary="rocket launched at dawn over the pacific"
    )

    matches = search_lexical(
        db,
        user_id=TEST_USER,
        q="rocket",
        kinds=[KIND_SUMMARY],
        filters={},
        limit=10,
    )
    assert len(matches) == 1
    assert matches[0].kind == KIND_SUMMARY
    assert matches[0].session_id == "sess-1"
    assert "rocket" in matches[0].snippet.lower()


def test_lexical_summary_diacritic_folded_match(db, schema):
    """SQLite path uses an ``intaris_fold`` UDF so that an unfolded query
    matches accented stored content. Without the UDF this would miss."""
    _seed_session(db, session_id="sess-fold")
    _insert_summary(
        db,
        session_id="sess-fold",
        summary="Příliš žluťoučký kůň úpěl ďábelské ódy",
    )

    matches = search_lexical(
        db,
        user_id=TEST_USER,
        q="zlutoucky",  # no diacritics
        kinds=[KIND_SUMMARY],
        filters={},
        limit=10,
    )
    assert any(m.session_id == "sess-fold" for m in matches)


def test_lexical_summary_user_scoped(db, schema):
    _seed_session(db, session_id="shared", user_id=OTHER_USER)
    _insert_summary(db, session_id="shared", summary="bob rocket", user_id=OTHER_USER)
    matches = search_lexical(
        db,
        user_id=TEST_USER,
        q="rocket",
        kinds=[KIND_SUMMARY],
        filters={},
        limit=10,
    )
    assert matches == []


# ── Lexical: intention (audit_log + DISTINCT ON) ──────────────────


def test_lexical_intention_dedup_via_distinct_on(db, schema, audit_store):
    _seed_session(db, session_id="sess-2")
    # Three audit rows with the SAME intention text — should collapse
    # to one match.
    for i in range(3):
        audit_store.insert(
            call_id=f"call-{i}",
            user_id=TEST_USER,
            session_id="sess-2",
            agent_id="agent-1",
            tool="bash",
            args_redacted={},
            classification="read",
            evaluation_path="fast",
            decision="approve",
            risk=None,
            reasoning=None,
            latency_ms=10,
            record_type="tool_call",
            content=None,
            intention="rocket launch tracking system",
        )
    # And one with a DIFFERENT intention — should be a separate match.
    audit_store.insert(
        call_id="call-other",
        user_id=TEST_USER,
        session_id="sess-2",
        agent_id="agent-1",
        tool="bash",
        args_redacted={},
        classification="read",
        evaluation_path="fast",
        decision="approve",
        risk=None,
        reasoning=None,
        latency_ms=10,
        record_type="tool_call",
        content=None,
        intention="rocket diagnostics module",
    )

    matches = search_lexical(
        db,
        user_id=TEST_USER,
        q="rocket",
        kinds=[KIND_INTENTION],
        filters={},
        limit=10,
    )
    intentions = {m.snippet for m in matches}
    # SQLite emulates DISTINCT ON via GROUP BY; we get one row per
    # distinct intention text per session.
    assert len(matches) == 2
    # Both intentions should be visible in the snippets.
    assert any("tracking" in s for s in intentions)
    assert any("diagnostics" in s for s in intentions)


def test_lexical_intention_user_scoped(db, schema, audit_store):
    _seed_session(db, session_id="shared", user_id=OTHER_USER)
    audit_store.insert(
        call_id="other-call",
        user_id=OTHER_USER,
        session_id="shared",
        agent_id="agent-1",
        tool="bash",
        args_redacted={},
        classification="read",
        evaluation_path="fast",
        decision="approve",
        risk=None,
        reasoning=None,
        latency_ms=10,
        record_type="tool_call",
        content=None,
        intention="rocket launch",
    )
    matches = search_lexical(
        db,
        user_id=TEST_USER,
        q="rocket",
        kinds=[KIND_INTENTION],
        filters={},
        limit=10,
    )
    assert matches == []


# ── Lexical: reasoning ────────────────────────────────────────────


def test_lexical_reasoning_finds_user_messages_and_agent_reasoning(
    db, schema, audit_store
):
    _seed_session(db, session_id="sess-3")
    audit_store.insert(
        call_id="r1",
        user_id=TEST_USER,
        session_id="sess-3",
        agent_id="agent-1",
        tool=None,
        args_redacted=None,
        classification=None,
        evaluation_path="reasoning",
        decision="approve",
        risk=None,
        reasoning=None,
        latency_ms=0,
        record_type="reasoning",
        content="User message: please track the rocket launch",
    )
    audit_store.insert(
        call_id="r2",
        user_id=TEST_USER,
        session_id="sess-3",
        agent_id="agent-1",
        tool=None,
        args_redacted=None,
        classification=None,
        evaluation_path="reasoning",
        decision="approve",
        risk=None,
        reasoning=None,
        latency_ms=0,
        record_type="reasoning",
        content="Plan to monitor the rocket telemetry stream",
    )
    matches = search_lexical(
        db,
        user_id=TEST_USER,
        q="rocket",
        kinds=[KIND_REASONING],
        filters={},
        limit=10,
    )
    assert len(matches) == 2
    roles = {m.role for m in matches}
    # Caller can distinguish user-message reasoning from agent reasoning.
    assert roles == {"user", "assistant"}


def test_lexical_reasoning_ignores_tool_call_rows(db, schema, audit_store):
    _seed_session(db, session_id="sess-4")
    audit_store.insert(
        call_id="t1",
        user_id=TEST_USER,
        session_id="sess-4",
        agent_id="agent-1",
        tool="bash",
        args_redacted={"cmd": "rocket"},
        classification="read",
        evaluation_path="fast",
        decision="approve",
        risk=None,
        reasoning="rocket would only show up here in tool_call rows",
        latency_ms=10,
        record_type="tool_call",
        content=None,
    )
    matches = search_lexical(
        db,
        user_id=TEST_USER,
        q="rocket",
        kinds=[KIND_REASONING],
        filters={},
        limit=10,
    )
    # tool_call rows should not surface under the reasoning kind.
    assert matches == []


# ── Service integration ───────────────────────────────────────────


def test_service_disabled_returns_empty(db):
    cfg = SearchConfig()
    cfg.enabled = False
    svc = SearchService(db=db, config=cfg)
    matches, cursor, mode, degraded = svc.search(
        user_id=TEST_USER,
        q="rocket",
        kinds=None,
        filters={},
        mode="auto",
        limit=10,
        cursor=None,
    )
    assert matches == []
    assert mode == "disabled"
    assert degraded == "search_disabled"


def test_service_health_when_lexical_only(service):
    h = service.health()
    assert h.enabled is True
    assert h.lexical.backend in ("sqlite-like", "postgres-fts")
    assert h.vector.provider == "disabled"
    assert set(h.lexical.kinds) == {KIND_SUMMARY, KIND_INTENTION, KIND_REASONING}


def test_service_search_runs_without_indexer(db, schema, search_config, audit_store):
    """vector_provider=disabled means no outbox writes occur."""
    _seed_session(db, session_id="sess-5")
    svc = SearchService(db=db, config=search_config)
    audit_store.insert(
        call_id="r-x",
        user_id=TEST_USER,
        session_id="sess-5",
        agent_id="agent-1",
        tool=None,
        args_redacted=None,
        classification=None,
        evaluation_path="reasoning",
        decision="approve",
        risk=None,
        reasoning=None,
        latency_ms=0,
        record_type="reasoning",
        content="rocket info",
    )
    # Outbox stays empty (vector tier off).
    assert outbox.queue_depth(db) == 0
    # But search still finds the match via lexical.
    matches, _, mode_used, _ = svc.search(
        user_id=TEST_USER,
        q="rocket",
        kinds=None,
        filters={},
        mode="auto",
        limit=10,
        cursor=None,
    )
    assert mode_used == "lexical"
    assert any(m.kind == KIND_REASONING for m in matches)


def test_service_search_sessions_aggregates(db, schema, search_config, audit_store):
    _seed_session(db, session_id="sess-A")
    _seed_session(db, session_id="sess-B")
    _insert_summary(db, session_id="sess-A", summary="rocket payload telemetry")
    _insert_summary(db, session_id="sess-B", summary="rocket diagnostic reports")
    svc = SearchService(db=db, config=search_config)
    sessions, _, _, _ = svc.search_sessions(
        user_id=TEST_USER,
        q="rocket",
        kinds=None,
        filters={},
        mode="auto",
        limit=10,
        cursor=None,
    )
    assert {s.session_id for s in sessions} == {"sess-A", "sess-B"}


# ── Audit hook fan-out (vector tier) ──────────────────────────────


def test_audit_insert_fans_out_when_search_service_set(
    db, schema, search_config, audit_store
):
    """When a SearchService is wired in and vector tier is healthy,
    audit inserts enqueue embed ops; otherwise the outbox stays empty."""
    _seed_session(db, session_id="sess-fanout")

    class FakeService:
        def __init__(self):
            self.intentions = []
            self.reasonings = []

        def enqueue_audit_intention(self, **kwargs):
            self.intentions.append(kwargs)

        def enqueue_audit_reasoning(self, **kwargs):
            self.reasonings.append(kwargs)

    fake = FakeService()
    AuditStore.set_search_service(fake)
    try:
        audit_store.insert(
            call_id="r-fan",
            user_id=TEST_USER,
            session_id="sess-fanout",
            agent_id="agent-1",
            tool=None,
            args_redacted=None,
            classification=None,
            evaluation_path="reasoning",
            decision="approve",
            risk=None,
            reasoning=None,
            latency_ms=0,
            record_type="reasoning",
            content="rocket reasoning content",
            intention="rocket launch tracking",
        )
    finally:
        AuditStore.set_search_service(None)

    assert len(fake.intentions) == 1
    assert len(fake.reasonings) == 1
    assert fake.intentions[0]["intention"] == "rocket launch tracking"
    assert fake.reasonings[0]["content"] == "rocket reasoning content"


# ── Fusion ─────────────────────────────────────────────────────────


def test_rrf_fuses_two_lists():
    lex = [
        {"session_id": "S", "kind": "x", "ref_id": "1", "score": 0.9},
        {"session_id": "S", "kind": "x", "ref_id": "2", "score": 0.5},
    ]
    vec = [
        {"session_id": "S", "kind": "x", "ref_id": "2", "score": 0.99},
        {"session_id": "S", "kind": "x", "ref_id": "3", "score": 0.6},
    ]
    fused = rrf_fuse(lexical=lex, vector=vec, alpha=0.5)
    refs = [m["ref_id"] for m in fused]
    assert "3" in refs
    two = next(m for m in fused if m["ref_id"] == "2")
    assert "lexical" in two["score_breakdown"]
    assert "vector" in two["score_breakdown"]


# ── Cursor ─────────────────────────────────────────────────────────


def test_cursor_round_trip():
    payload = {"offset": 42, "tail": "abc"}
    encoded = encode_cursor(payload)
    assert decode_cursor(encoded) == payload


def test_cursor_handles_garbage():
    assert decode_cursor("not-base64!") == {}
    assert decode_cursor("") == {}
    assert decode_cursor(None) == {}


def test_service_clamps_huge_cursor_offset(db, schema, search_config):
    """A forged cursor must not let the caller request a 10M-row page."""
    svc = SearchService(db=db, config=search_config)
    huge_cursor = encode_cursor({"offset": 10_000_000})
    matches, _, _, _ = svc.search(
        user_id=TEST_USER,
        q="rocket",
        kinds=None,
        filters={},
        mode="auto",
        limit=10,
        cursor=huge_cursor,
    )
    # No crash, no allocation explosion. Offset is clamped server-side.
    assert matches == []


# ── Qdrant local-mode (skipped if package missing) ────────────────


@pytest.fixture
def qdrant_local(tmp_path, schema):
    pytest.importorskip("qdrant_client")
    from intaris.search.vector import QdrantVectorBackend

    backend = QdrantVectorBackend(
        url=str(tmp_path / "qdrant"),
        api_key=None,
        collection="test",
        model="text-embedding-3-small",
        dim=4,
        sparse_model="",  # disable sparse for simpler local-mode test
    )
    if not backend.healthy():
        pytest.skip("qdrant local-mode unhealthy in this environment")
    return backend


def test_qdrant_local_path_detection():
    from intaris.search.vector import _qdrant_local_path

    assert _qdrant_local_path("http://localhost:6333") is None
    assert _qdrant_local_path("https://qdrant.example.com") is None
    assert _qdrant_local_path("/abs/path") == "/abs/path"
    assert _qdrant_local_path("file:///abs/path") == "/abs/path"
    expanded = _qdrant_local_path("~/foo")
    assert expanded is not None and expanded.endswith("/foo")


def test_qdrant_upsert_and_search(qdrant_local):
    backend = qdrant_local
    backend.upsert(
        user_id=TEST_USER,
        session_id="s1",
        kind=KIND_INTENTION,
        ref_id="r1",
        text="rocket",
        ts="2026-04-01T00:00:00Z",
        agent_id="agent-1",
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    backend.upsert(
        user_id=TEST_USER,
        session_id="s2",
        kind=KIND_INTENTION,
        ref_id="r2",
        text="diagnostics",
        ts="2026-04-02T00:00:00Z",
        agent_id="agent-1",
        embedding=[0.0, 1.0, 0.0, 0.0],
    )
    matches = backend.search(
        user_id=TEST_USER,
        q="rocket",
        embedding=[1.0, 0.0, 0.0, 0.0],
        kinds=[KIND_INTENTION],
        filters={},
        limit=5,
    )
    refs = [m.ref_id for m in matches]
    assert refs and refs[0] == "r1"


def test_qdrant_user_scope(qdrant_local):
    backend = qdrant_local
    backend.upsert(
        user_id=TEST_USER,
        session_id="s1",
        kind=KIND_INTENTION,
        ref_id="r1",
        text="rocket",
        ts="2026-04-01T00:00:00Z",
        agent_id="agent-1",
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    other = backend.search(
        user_id=OTHER_USER,
        q="rocket",
        embedding=[1.0, 0.0, 0.0, 0.0],
        kinds=[KIND_INTENTION],
        filters={},
        limit=5,
    )
    assert other == []


# ── Hybrid orchestration with stub vector ─────────────────────────


def test_service_hybrid_with_stub_pgvector(db, schema, search_config, audit_store):
    """Wire a stub vector backend that mimics pgvector and confirm RRF
    fusion happens in the service.search() path."""
    from intaris.search.vector import VectorMatch

    _seed_session(db, session_id="sess-h1")
    _seed_session(db, session_id="sess-h2")
    _insert_summary(db, session_id="sess-h1", summary="rocket dawn")
    _insert_summary(db, session_id="sess-h2", summary="rocket noon")

    svc = SearchService(db=db, config=search_config)

    class StubVector:
        name = "pgvector"
        model = "stub"
        dim = 4
        sparse_model = None

        def healthy(self):
            return True

        def search(self, *, user_id, q, embedding, kinds, filters, limit):
            return [
                VectorMatch(
                    user_id=user_id,
                    session_id="sess-h2",
                    kind=KIND_SUMMARY,
                    ref_id="vec-2",
                    text="rocket noon",
                    ts=None,
                    agent_id=None,
                    score=0.99,
                ),
            ]

        def upsert(self, **_):
            pass

        def delete_session(self, **_):
            return 0

        def delete_ref(self, **_):
            pass

    class StubEmbeddings:
        model = "stub"

        def embed(self, texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    svc._vector = StubVector()
    svc._embeddings = StubEmbeddings()  # type: ignore[assignment]

    matches, _, mode_used, degraded = svc.search(
        user_id=TEST_USER,
        q="rocket",
        kinds=[KIND_SUMMARY],
        filters={},
        mode="hybrid",
        limit=10,
        cursor=None,
    )
    assert mode_used == "hybrid"
    assert degraded == ""
    # At minimum we get the lexical hits; vector contribution may
    # surface as an extra result.
    sids = {m.session_id for m in matches}
    assert {"sess-h1", "sess-h2"} <= sids
