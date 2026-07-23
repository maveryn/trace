"""Sampling primitives for Nine Men's Morris board states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Mapping

from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis

from .defaults import SUPPORTED_NINE_MENS_MORRIS_SCENE_VARIANTS
from .rules import (
    analyze_occupancy,
    board_state_from_occupancy,
    choose_mill_union_set,
    opponent_color,
)
from .state import POSITION_LAYOUT, NineMensMorrisBoardState


ALL_MILL_PIECE_COUNT_SUPPORT: tuple[int, ...] = (0, 3, 5, 6, 7, 8, 9)
MILL_COMPLETION_COUNT_SUPPORT: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
_COLOR_MILL_PIECE_SUPPORT: tuple[int, ...] = (0, 3, 5, 6, 7, 8, 9)


@dataclass(frozen=True)
class NineMensMorrisVisualAxes:
    """Resolved visual axes for one Nine Men's Morris board."""

    scene_variant: str
    style_variant: str
    scene_variant_probabilities: dict[str, float]
    style_variant_probabilities: dict[str, float]


def resolve_nine_mens_morris_visual_axes(
    instance_seed: int,
    *,
    gen_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    namespace: str,
    supported_style_variants: tuple[str, ...],
) -> NineMensMorrisVisualAxes:
    """Resolve scene and style axes shared by Morris objectives."""

    scene_variant, scene_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_variants=SUPPORTED_NINE_MENS_MORRIS_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_variants=tuple(str(value) for value in supported_style_variants),
    )
    return NineMensMorrisVisualAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
    )


def _sample_mill_sets(
    rng,
    *,
    target_answer: int,
) -> tuple[FrozenSet[int], FrozenSet[int]]:
    """Sample disjoint white/black complete-mill unions for one total count."""

    target = int(target_answer)
    feasible_pairs: list[tuple[int, int]] = []
    for white_size in _COLOR_MILL_PIECE_SUPPORT:
        for black_size in _COLOR_MILL_PIECE_SUPPORT:
            if int(white_size) + int(black_size) != int(target):
                continue
            if choose_mill_union_set(rng, size=int(white_size), forbidden_nodes=frozenset()) is not None:
                feasible_pairs.append((int(white_size), int(black_size)))
    rng.shuffle(feasible_pairs)
    for white_size, black_size in feasible_pairs:
        for _ in range(128):
            white_nodes = choose_mill_union_set(rng, size=int(white_size), forbidden_nodes=frozenset())
            if white_nodes is None:
                continue
            black_nodes = choose_mill_union_set(rng, size=int(black_size), forbidden_nodes=white_nodes)
            if black_nodes is not None:
                return frozenset(white_nodes), frozenset(black_nodes)
    raise ValueError(f"failed to sample disjoint mill sets for total target {target}")


def _try_add_fillers(
    rng,
    *,
    occupancy_by_node: dict[int, str],
    expected_white_piece_ids_in_mill: FrozenSet[str],
    expected_black_piece_ids_in_mill: FrozenSet[str],
) -> None:
    """Add non-mill filler pieces without changing counted mill witnesses."""

    for color in ("white", "black"):
        current_count = sum(1 for occupant in occupancy_by_node.values() if str(occupant) == str(color))
        remaining_capacity = max(0, 9 - int(current_count))
        desired_extra = int(rng.randint(0, min(2, remaining_capacity)))
        candidate_nodes = [node_index for node_index in range(len(POSITION_LAYOUT)) if node_index not in occupancy_by_node]
        rng.shuffle(candidate_nodes)
        added = 0
        for node_index in candidate_nodes:
            if added >= int(desired_extra):
                break
            occupancy_by_node[int(node_index)] = str(color)
            analysis = analyze_occupancy(occupancy_by_node)
            if frozenset(analysis.white_piece_ids_in_mill) == expected_white_piece_ids_in_mill and frozenset(
                analysis.black_piece_ids_in_mill
            ) == expected_black_piece_ids_in_mill:
                added += 1
            else:
                del occupancy_by_node[int(node_index)]


def _sample_completion_occupancy(
    rng,
    *,
    color: str,
    target_answer: int,
) -> dict[int, str]:
    """Sample one occupancy with an exact mill-completion count."""

    target = int(target_answer)
    target_color = str(color)
    other_color = opponent_color(target_color)
    for _ in range(8192):
        target_count_min = 0 if int(target) == 0 else 2
        target_piece_count = int(rng.randint(int(target_count_min), 9))
        other_piece_count = int(rng.randint(0, 9))
        nodes = list(range(len(POSITION_LAYOUT)))
        rng.shuffle(nodes)
        occupancy: dict[int, str] = {}
        for node_index in nodes[:target_piece_count]:
            occupancy[int(node_index)] = target_color
        for node_index in nodes[target_piece_count : target_piece_count + other_piece_count]:
            occupancy[int(node_index)] = other_color
        analysis = analyze_occupancy(occupancy)
        completion_labels = (
            analysis.white_mill_completion_node_labels
            if target_color == "white"
            else analysis.black_mill_completion_node_labels
        )
        if len(completion_labels) == int(target):
            return occupancy
    raise ValueError(f"failed to sample mill-completion board for {target_color} target {target}")


def sample_all_mill_piece_board(*, rng, target_answer: int) -> NineMensMorrisBoardState:
    """Build one board with an exact count of pieces inside any mill."""

    target = int(target_answer)
    if target not in ALL_MILL_PIECE_COUNT_SUPPORT:
        raise ValueError(f"unsupported all-color target_answer: {target}")

    for _ in range(512):
        white_mill_nodes, black_mill_nodes = _sample_mill_sets(
            rng,
            target_answer=int(target),
        )
        occupancy_by_node: dict[int, str] = {int(node): "white" for node in white_mill_nodes}
        occupancy_by_node.update({int(node): "black" for node in black_mill_nodes})

        base_analysis = analyze_occupancy(occupancy_by_node)
        expected_white_piece_ids_in_mill = frozenset(base_analysis.white_piece_ids_in_mill)
        expected_black_piece_ids_in_mill = frozenset(base_analysis.black_piece_ids_in_mill)
        _try_add_fillers(
            rng,
            occupancy_by_node=occupancy_by_node,
            expected_white_piece_ids_in_mill=expected_white_piece_ids_in_mill,
            expected_black_piece_ids_in_mill=expected_black_piece_ids_in_mill,
        )
        final_analysis = analyze_occupancy(occupancy_by_node)
        if len(final_analysis.all_piece_ids_in_mill) == int(target):
            return board_state_from_occupancy(occupancy_by_node, target_answer=int(target))

    raise ValueError(f"failed to build all-mill-piece board for target {target}")


def sample_mill_completion_board(*, rng, color: str, target_answer: int) -> NineMensMorrisBoardState:
    """Build one board with an exact count of completion points for a color."""

    target = int(target_answer)
    if target not in MILL_COMPLETION_COUNT_SUPPORT:
        raise ValueError(f"unsupported mill-completion target_answer: {target}")
    occupancy_by_node = _sample_completion_occupancy(
        rng,
        color=str(color),
        target_answer=int(target),
    )
    return board_state_from_occupancy(occupancy_by_node, target_answer=int(target))


__all__ = [
    "ALL_MILL_PIECE_COUNT_SUPPORT",
    "MILL_COMPLETION_COUNT_SUPPORT",
    "NineMensMorrisVisualAxes",
    "resolve_nine_mens_morris_visual_axes",
    "sample_all_mill_piece_board",
    "sample_mill_completion_board",
]
