"""Conversation search subsystem.

Three search kinds, queried directly from canonical Postgres tables:

- ``summary`` — ``session_summaries`` (window + compacted) and
  ``agent_summaries``. Highest signal when available; may not exist
  for new sessions.
- ``intention`` — ``audit_log.intention`` snapshots, deduped per
  session. Always present once a session has any tool calls.
- ``reasoning`` — ``audit_log.content`` for ``record_type``
  ``reasoning`` or ``checkpoint``. Most granular textual evidence:
  user messages plus agent reasoning.

The lexical tier (Postgres ``tsvector`` + GIN, or SQLite ``LIKE``) is
always available when the master flag is on. The vector tier is
optional and additive — it indexes the same three kinds and gets
RRF-fused with lexical results at query time.
"""

from __future__ import annotations

from intaris.search.types import (
    KIND_INTENTION,
    KIND_REASONING,
    KIND_SUMMARY,
    SEARCH_KINDS,
    SearchHealth,
    SearchMatch,
    SearchSessionMatch,
)

__all__ = [
    "KIND_INTENTION",
    "KIND_REASONING",
    "KIND_SUMMARY",
    "SEARCH_KINDS",
    "SearchHealth",
    "SearchMatch",
    "SearchSessionMatch",
]
