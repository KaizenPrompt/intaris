"""Pydantic types and shared constants for the search subsystem."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Three canonical search kinds.
KIND_SUMMARY = "summary"
KIND_INTENTION = "intention"
KIND_REASONING = "reasoning"

SEARCH_KINDS: frozenset[str] = frozenset({KIND_SUMMARY, KIND_INTENTION, KIND_REASONING})

# Default kinds when caller passes none — all three high-signal kinds.
DEFAULT_KINDS: tuple[str, ...] = (KIND_SUMMARY, KIND_INTENTION, KIND_REASONING)


class SearchMatch(BaseModel):
    """One hit in the flat match list."""

    session_id: str
    kind: str
    ref_id: str | None = None
    role: str | None = None
    ts: str | None = None
    snippet: str
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    agent_id: str | None = None
    # Display metadata (joined from ``sessions`` for context).
    session_title: str | None = None
    session_intention: str | None = None


class SearchSessionMatch(BaseModel):
    """Aggregated match: one row per session with the top-scoring hit."""

    session_id: str
    agent_id: str | None = None
    title: str | None = None
    intention: str | None = None
    last_activity_at: str | None = None
    match_count: int
    top_match: SearchMatch


class SearchBackends(BaseModel):
    lexical: str
    vector: str
    mode_used: str


class SearchResponse(BaseModel):
    matches: list[SearchMatch]
    next_cursor: str | None = None
    total_estimated: int | None = None
    backend: SearchBackends


class SearchSessionsResponse(BaseModel):
    sessions: list[SearchSessionMatch]
    next_cursor: str | None = None
    total_estimated: int | None = None
    backend: SearchBackends


class SearchLexicalCapabilities(BaseModel):
    backend: str
    unaccent: bool = False
    pg_trgm: bool = False
    kinds: list[str] = Field(default_factory=list)


class SearchVectorState(BaseModel):
    provider: str
    model: str | None = None
    dim: int | None = None
    sparse_model: str | None = None
    queue_depth: int = 0
    last_index_at: str | None = None
    backfill_status: str = "idle"
    backfill_total: int | None = None
    backfill_processed: int | None = None
    backfill_job_id: str | None = None


class SearchHealth(BaseModel):
    enabled: bool
    lexical: SearchLexicalCapabilities
    vector: SearchVectorState
    notes: list[str] = Field(default_factory=list)


class SearchRequestFilters(BaseModel):
    agent_id: str | None = None
    session_id: str | None = None
    session_ids: list[str] | None = None
    from_ts: str | None = None
    to_ts: str | None = None


class SearchRequest(BaseModel):
    q: str
    filters: SearchRequestFilters = Field(default_factory=SearchRequestFilters)
    kinds: list[str] | None = None
    mode: str = "auto"  # "lexical" | "vector" | "hybrid" | "auto"
    limit: int = 50
    cursor: str | None = None


class SearchSessionsRequest(BaseModel):
    q: str
    filters: SearchRequestFilters = Field(default_factory=SearchRequestFilters)
    kinds: list[str] | None = None
    mode: str = "auto"
    limit: int = 25
    cursor: str | None = None


def truncate_text(text: str, max_bytes: int) -> str:
    if not text:
        return ""
    encoded = text.encode("utf-8", errors="ignore")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def fold_text(text: str) -> str:
    """NFKD diacritic-fold + casefold. Pure-Python LIKE fallback helper."""
    import unicodedata

    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in normalized if not unicodedata.combining(c))
    return folded.casefold()


__all__ = [
    "KIND_INTENTION",
    "KIND_REASONING",
    "KIND_SUMMARY",
    "SEARCH_KINDS",
    "DEFAULT_KINDS",
    "SearchMatch",
    "SearchSessionMatch",
    "SearchBackends",
    "SearchResponse",
    "SearchSessionsResponse",
    "SearchLexicalCapabilities",
    "SearchVectorState",
    "SearchHealth",
    "SearchRequest",
    "SearchRequestFilters",
    "SearchSessionsRequest",
    "truncate_text",
    "fold_text",
]
