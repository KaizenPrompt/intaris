"""Reciprocal Rank Fusion for hybrid lexical + vector queries.

Used by the pgvector path. Qdrant fuses dense+sparse server-side and
needs no Python fusion.
"""

from __future__ import annotations

from typing import Any

RRF_K = 60.0


def rrf_fuse(
    *,
    lexical: list[dict[str, Any]],
    vector: list[dict[str, Any]],
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    """Combine two ranked lists into a single ordered list.

    Entries keyed by the unique ``ref`` tuple ``(session_id, kind, ref_id)``
    when present, falling back to ``id`` for backward compatibility.
    """
    if not lexical and not vector:
        return []
    if not vector:
        out: list[dict[str, Any]] = []
        for row in lexical:
            row = dict(row)
            row["score_breakdown"] = {"lexical": float(row.get("score", 0.0))}
            out.append(row)
        return out
    if not lexical:
        out_v: list[dict[str, Any]] = []
        for row in vector:
            row = dict(row)
            row["score_breakdown"] = {"vector": float(row.get("score", 0.0))}
            out_v.append(row)
        return out_v

    weight_lex = max(0.0, min(1.0, alpha))
    weight_vec = 1.0 - weight_lex

    fused: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("session_id"),
            row.get("kind"),
            row.get("ref_id") if row.get("ref_id") is not None else row.get("id"),
        )

    for rank, row in enumerate(lexical):
        contribution = weight_lex / (RRF_K + rank + 1)
        copy = dict(row)
        copy.setdefault("score_breakdown", {})["lexical"] = float(row.get("score", 0.0))
        copy["_rrf"] = contribution
        fused[_key(copy)] = copy

    for rank, row in enumerate(vector):
        contribution = weight_vec / (RRF_K + rank + 1)
        key = _key(row)
        existing = fused.get(key)
        if existing is None:
            copy = dict(row)
            copy.setdefault("score_breakdown", {})["vector"] = float(
                row.get("score", 0.0)
            )
            copy["_rrf"] = contribution
            fused[key] = copy
        else:
            existing["_rrf"] = float(existing.get("_rrf", 0.0)) + contribution
            existing.setdefault("score_breakdown", {})["vector"] = float(
                row.get("score", 0.0)
            )
            if not existing.get("snippet") and row.get("snippet"):
                existing["snippet"] = row["snippet"]

    ranked = sorted(
        fused.values(),
        key=lambda r: (r.get("_rrf", 0.0), r.get("ts") or ""),
        reverse=True,
    )
    out_f: list[dict[str, Any]] = []
    for row in ranked:
        score = float(row.pop("_rrf", 0.0))
        row["score"] = score
        out_f.append(row)
    return out_f


__all__ = ["rrf_fuse", "RRF_K"]
