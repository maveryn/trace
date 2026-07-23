"""Shared deterministic sampling helpers for task-local support selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...core.seed import hash64


def resolve_selection_index(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> int:
    """Return a deterministic selection index.

    A namespaced hash keeps support selection deterministic while avoiding
    accidental coupling with unrelated decisions that also depend on
    `instance_seed`.
    """

    return abs(int(hash64(int(instance_seed), str(namespace), 0)))


def uniform_probability_map(values: list[int] | tuple[int, ...], *, selected: int | None = None) -> dict[str, float]:
    """Return a deterministic uniform probability map over a finite integer support."""

    support = tuple(int(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        return {str(int(selected)): 1.0}
    probability = 1.0 / float(len(support))
    return {str(int(value)): float(probability) for value in support}
