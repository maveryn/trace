"""Shared query-id sentinel constants and predicates."""

from __future__ import annotations

SINGLE_QUERY_ID = "single"
LEGACY_DEFAULT_QUERY_ID = "default"
NO_BRANCH_QUERY_IDS = frozenset({"", LEGACY_DEFAULT_QUERY_ID, SINGLE_QUERY_ID})


def is_no_branch_query_id(value: object) -> bool:
    """Return whether ``value`` is a no-semantic-branch query sentinel."""

    return str(value or "").strip() in NO_BRANCH_QUERY_IDS
