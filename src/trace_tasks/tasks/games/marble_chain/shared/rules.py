"""Marble-chain insertion and pop mechanics."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .state import MarbleOutcome


def marble_entity_id(index: int) -> str:
    """Return the stable entity id for one existing chain marble."""

    return f"marble_{int(index):02d}"


def compute_outcome(chain_colors: Sequence[str], *, shooter_color: str, slot_index: int) -> MarbleOutcome:
    """Compute the immediate no-cascade pop result for inserting one marble."""

    chain = tuple(str(color) for color in chain_colors)
    slot = max(0, min(int(slot_index), len(chain)))
    inserted_color = str(shooter_color)
    new_chain = list(chain[:slot]) + [inserted_color] + list(chain[slot:])
    inserted_index = int(slot)
    left = inserted_index
    while left - 1 >= 0 and str(new_chain[left - 1]) == inserted_color:
        left -= 1
    right = inserted_index
    while right + 1 < len(new_chain) and str(new_chain[right + 1]) == inserted_color:
        right += 1

    run_len = int(right - left + 1)
    affected_indices: List[int] = []
    popped_indices: List[int] = []
    if run_len >= 3:
        for new_index in range(left, right + 1):
            if new_index == inserted_index:
                continue
            original_index = int(new_index if new_index < inserted_index else new_index - 1)
            popped_indices.append(original_index)
            affected_indices.append(original_index)
        remaining_count = int(len(chain) - len(popped_indices))
    else:
        for original_index in (slot - 1, slot):
            if 0 <= int(original_index) < len(chain):
                affected_indices.append(int(original_index))
        remaining_count = int(len(chain) + 1)
    return MarbleOutcome(
        slot_index=int(slot),
        pop_count=int(len(popped_indices)),
        popped_indices=tuple(sorted(set(int(index) for index in popped_indices))),
        affected_indices=tuple(sorted(set(int(index) for index in affected_indices))),
        remaining_count=int(remaining_count),
    )


def all_outcomes(chain_colors: Sequence[str], *, shooter_color: str) -> Dict[int, MarbleOutcome]:
    """Compute every insertion gap outcome for one chain and shooter color."""

    return {
        int(slot): compute_outcome(chain_colors, shooter_color=str(shooter_color), slot_index=int(slot))
        for slot in range(len(chain_colors) + 1)
    }


def closure_pair_indices(chain_colors: Sequence[str], outcome: MarbleOutcome) -> Tuple[int, ...]:
    """Return original chain indices that touch after this pop closes the gap."""

    chain = tuple(str(color) for color in chain_colors)
    if int(outcome.pop_count) <= 0 or not outcome.popped_indices:
        return ()
    left_index = int(min(outcome.popped_indices)) - 1
    right_index = int(max(outcome.popped_indices)) + 1
    if left_index < 0 or right_index >= len(chain):
        return ()
    return (int(left_index), int(right_index))


def closure_pair_color(chain_colors: Sequence[str], outcome: MarbleOutcome) -> str | None:
    """Return the shared color when the closure boundary creates a color match."""

    pair = closure_pair_indices(chain_colors, outcome)
    if len(pair) != 2:
        return None
    left_color = str(chain_colors[int(pair[0])])
    right_color = str(chain_colors[int(pair[1])])
    if left_color != right_color:
        return None
    return left_color


def closure_creates_same_color_match(chain_colors: Sequence[str], outcome: MarbleOutcome) -> bool:
    """Return whether the immediate closure leaves two same-color marbles touching."""

    return closure_pair_color(chain_colors, outcome) is not None


def insertion_point_annotation_ids(slot_entity_id: str) -> Tuple[str, ...]:
    """Return the single shot-gap entity id used by direction-label tasks."""

    return (str(slot_entity_id),)


def popped_marble_annotation_ids(outcome: MarbleOutcome) -> Tuple[str, ...]:
    """Return entity ids for existing chain marbles removed by one shot."""

    return tuple(marble_entity_id(int(index)) for index in outcome.popped_indices)


__all__ = [
    "all_outcomes",
    "closure_creates_same_color_match",
    "closure_pair_color",
    "closure_pair_indices",
    "compute_outcome",
    "insertion_point_annotation_ids",
    "marble_entity_id",
    "popped_marble_annotation_ids",
]
