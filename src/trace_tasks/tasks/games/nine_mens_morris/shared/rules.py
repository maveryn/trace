"""Nine Men's Morris rule helpers independent of public task identity."""

from __future__ import annotations

from typing import FrozenSet, Mapping

from .state import (
    MILL_POSITION_INDICES,
    POSITION_LAYOUT,
    NineMensMorrisBoardState,
    NineMensMorrisOccupancyAnalysis,
    NineMensMorrisPieceInstance,
)


def piece_id_for_node(node_index: int, color: str) -> str:
    """Return a stable piece id for one occupied Morris node."""

    return f"{str(color)}_{POSITION_LAYOUT[int(node_index)][0]}"


def position_label_for_node(node_index: int) -> str:
    """Return the board-coordinate label for one Morris node."""

    return str(POSITION_LAYOUT[int(node_index)][0])


def opponent_color(color: str) -> str:
    """Return the opposite Morris piece color."""

    return "black" if str(color) == "white" else "white"


def completion_node_labels(occupancy_by_node: Mapping[int, str], *, color: str) -> tuple[str, ...]:
    """Return empty nodes where one piece of the color would complete a mill."""

    target_color = str(color)
    completion_nodes: set[int] = set()
    for node_index in range(len(POSITION_LAYOUT)):
        if int(node_index) in occupancy_by_node:
            continue
        for positions in MILL_POSITION_INDICES:
            if int(node_index) not in positions:
                continue
            other_positions = [int(position) for position in positions if int(position) != int(node_index)]
            if all(str(occupancy_by_node.get(int(position), "")) == target_color for position in other_positions):
                completion_nodes.add(int(node_index))
                break
    return tuple(position_label_for_node(int(node_index)) for node_index in sorted(completion_nodes))


def analyze_occupancy(occupancy_by_node: Mapping[int, str]) -> NineMensMorrisOccupancyAnalysis:
    """Return mill membership derived from one node occupancy map."""

    white_piece_ids: set[str] = set()
    black_piece_ids: set[str] = set()
    white_mill_ids: list[str] = []
    black_mill_ids: list[str] = []
    membership_count_by_piece_id: dict[str, int] = {}

    for mill_index, positions in enumerate(MILL_POSITION_INDICES):
        colors = [str(occupancy_by_node[position]) for position in positions if position in occupancy_by_node]
        if len(colors) != 3:
            continue
        color = colors[0]
        if not color or any(other != color for other in colors[1:]):
            continue
        mill_id = f"mill_{int(mill_index)}"
        if color == "white":
            white_mill_ids.append(str(mill_id))
        else:
            black_mill_ids.append(str(mill_id))
        for node_index in positions:
            piece_id = piece_id_for_node(int(node_index), color)
            membership_count_by_piece_id[str(piece_id)] = membership_count_by_piece_id.get(str(piece_id), 0) + 1
            if color == "white":
                white_piece_ids.add(str(piece_id))
            else:
                black_piece_ids.add(str(piece_id))

    all_piece_ids = sorted(white_piece_ids | black_piece_ids)
    return NineMensMorrisOccupancyAnalysis(
        white_piece_ids_in_mill=tuple(sorted(white_piece_ids)),
        black_piece_ids_in_mill=tuple(sorted(black_piece_ids)),
        all_piece_ids_in_mill=tuple(str(piece_id) for piece_id in all_piece_ids),
        white_mill_ids=tuple(str(value) for value in sorted(white_mill_ids)),
        black_mill_ids=tuple(str(value) for value in sorted(black_mill_ids)),
        white_mill_completion_node_labels=completion_node_labels(occupancy_by_node, color="white"),
        black_mill_completion_node_labels=completion_node_labels(occupancy_by_node, color="black"),
        mill_membership_count_by_piece_id={str(key): int(value) for key, value in membership_count_by_piece_id.items()},
    )


def compute_mill_union_sets_by_size() -> dict[int, tuple[FrozenSet[int], ...]]:
    """Precompute unique unions of complete-mill positions up to nine pieces."""

    union_sets: set[FrozenSet[int]] = {frozenset()}
    mill_count = len(MILL_POSITION_INDICES)
    for mask in range(1, 1 << mill_count):
        union: set[int] = set()
        for mill_index, positions in enumerate(MILL_POSITION_INDICES):
            if mask & (1 << mill_index):
                union.update(int(position) for position in positions)
        if len(union) <= 9:
            union_sets.add(frozenset(int(position) for position in union))
    by_size: dict[int, list[FrozenSet[int]]] = {}
    for union in union_sets:
        by_size.setdefault(len(union), []).append(frozenset(int(position) for position in union))
    return {
        int(size): tuple(sorted(values, key=lambda item: tuple(int(position) for position in sorted(item))))
        for size, values in by_size.items()
    }


MILL_UNION_SETS_BY_SIZE = compute_mill_union_sets_by_size()


def eligible_mill_union_sets(size: int, *, forbidden_nodes: FrozenSet[int]) -> tuple[FrozenSet[int], ...]:
    """Return precomputed mill unions of one size that avoid forbidden nodes."""

    eligible = [
        union
        for union in MILL_UNION_SETS_BY_SIZE.get(int(size), ())
        if frozenset(int(position) for position in union).isdisjoint(forbidden_nodes)
    ]
    return tuple(eligible)


def choose_mill_union_set(rng, *, size: int, forbidden_nodes: FrozenSet[int]) -> FrozenSet[int] | None:
    """Sample one complete-mill union of the requested size."""

    eligible = eligible_mill_union_sets(int(size), forbidden_nodes=forbidden_nodes)
    if not eligible:
        return None
    return frozenset(eligible[int(rng.randrange(len(eligible)))])


def board_state_from_occupancy(
    occupancy_by_node: Mapping[int, str],
    *,
    target_answer: int,
) -> NineMensMorrisBoardState:
    """Build a public board-state payload from finalized occupancy."""

    final_analysis = analyze_occupancy(occupancy_by_node)
    piece_specs: list[NineMensMorrisPieceInstance] = []
    for node_index in sorted(occupancy_by_node):
        color = str(occupancy_by_node[node_index])
        piece_specs.append(
            NineMensMorrisPieceInstance(
                piece_id=piece_id_for_node(int(node_index), color),
                node_index=int(node_index),
                node_label=position_label_for_node(int(node_index)),
                color=str(color),
            )
        )
    overlapping_piece_ids = [
        str(piece_id)
        for piece_id, count in final_analysis.mill_membership_count_by_piece_id.items()
        if int(count) >= 2
    ]
    return NineMensMorrisBoardState(
        piece_specs=tuple(piece_specs),
        white_piece_ids_in_mill=tuple(str(value) for value in final_analysis.white_piece_ids_in_mill),
        black_piece_ids_in_mill=tuple(str(value) for value in final_analysis.black_piece_ids_in_mill),
        all_piece_ids_in_mill=tuple(str(value) for value in final_analysis.all_piece_ids_in_mill),
        white_mill_ids=tuple(str(value) for value in final_analysis.white_mill_ids),
        black_mill_ids=tuple(str(value) for value in final_analysis.black_mill_ids),
        white_mill_completion_node_labels=tuple(str(value) for value in final_analysis.white_mill_completion_node_labels),
        black_mill_completion_node_labels=tuple(str(value) for value in final_analysis.black_mill_completion_node_labels),
        overlapping_piece_ids=tuple(sorted(overlapping_piece_ids)),
        target_answer=int(target_answer),
    )


__all__ = [
    "MILL_UNION_SETS_BY_SIZE",
    "analyze_occupancy",
    "board_state_from_occupancy",
    "choose_mill_union_set",
    "completion_node_labels",
    "eligible_mill_union_sets",
    "opponent_color",
    "piece_id_for_node",
    "position_label_for_node",
]
