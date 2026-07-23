"""State-label helpers for graph automaton scenes."""

from __future__ import annotations

from typing import Sequence, Tuple

from ...shared.graph_sample_types import graph_label_sort_key


def state_labels(state_count: int) -> Tuple[str, ...]:
    """Return compact automaton state labels."""

    return tuple(chr(ord("A") + int(index)) for index in range(int(state_count)))


def sorted_state_label_tuple(values: Sequence[str]) -> Tuple[str, ...]:
    """Sort state labels with the graph-domain label ordering."""

    return tuple(sorted((str(value) for value in values), key=graph_label_sort_key))


__all__ = [
    "sorted_state_label_tuple",
    "state_labels",
]
