"""Session policy helpers for evaluator-facing views."""

from __future__ import annotations

from typing import Any


def effective_policy_for_evaluator(
    policy: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a simplified policy view for LLM prompts.

    The stored policy remains unchanged. This only removes redundant child
    allow_paths already covered by broader simple subtree allow patterns, which
    keeps prompt context aligned with the deterministic matcher semantics.
    """

    if not policy:
        return policy

    effective = dict(policy)
    allow_paths = policy.get("allow_paths")
    if isinstance(allow_paths, list):
        effective["allow_paths"] = _reduce_allow_paths_for_prompt(allow_paths)
    return effective


def _reduce_allow_paths_for_prompt(allow_paths: list[Any]) -> list[Any]:
    """Drop simple child subtree allow_paths covered by broader parents."""

    simple_patterns: list[tuple[int, str]] = []
    for index, pattern in enumerate(allow_paths):
        if isinstance(pattern, str) and _is_simple_subtree_pattern(pattern):
            simple_patterns.append((index, _subtree_pattern_root(pattern)))

    removed: set[int] = set()
    for child_index, child_root in simple_patterns:
        for parent_index, parent_root in simple_patterns:
            if parent_index == child_index:
                continue
            if parent_root == child_root:
                if parent_index < child_index:
                    removed.add(child_index)
                    break
                continue
            if len(parent_root) > len(child_root):
                continue
            if child_root.startswith(parent_root + "/"):
                removed.add(child_index)
                break

    return [path for index, path in enumerate(allow_paths) if index not in removed]


def _is_simple_subtree_pattern(pattern: str) -> bool:
    """Return True for absolute `/path/*` patterns without other globs."""

    if not pattern.startswith("/") or not pattern.endswith("/*"):
        return False
    root = _subtree_pattern_root(pattern)
    return bool(root) and not any(char in root for char in "*?[")


def _subtree_pattern_root(pattern: str) -> str:
    """Return `/path` from `/path/*`, preserving root shape."""

    return pattern[:-2].rstrip("/") or "/"
