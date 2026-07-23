"""Scene-neutral sampling helpers for Reversi tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import shuffled_support
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.games.shared.style import SUPPORTED_REVERSI_STYLE_VARIANTS
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .defaults import DEFAULTS
from .rules import coord_to_cell_id, frontier_disc_coords, legal_moves_with_flips, simulate_random_state
from .state import (
    BLACK,
    BOARD_SIZE_BY_SCENE_VARIANT,
    EMPTY,
    SUPPORTED_SCENE_VARIANTS,
    WHITE,
    Board,
    Coord,
    ReversiTargetAxis,
    ReversiVisualAxes,
    SampledReversiScene,
)


def resolve_reversi_visual_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace_root: str,
) -> ReversiVisualAxes:
    """Resolve scene and style axes without objective-specific branching."""

    scene_variant, scene_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_variants=[str(value) for value in SUPPORTED_SCENE_VARIANTS],
    )
    style_variant, style_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_variants=[str(value) for value in SUPPORTED_REVERSI_STYLE_VARIANTS],
    )
    return ReversiVisualAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        board_size=int(BOARD_SIZE_BY_SCENE_VARIANT[str(scene_variant)]),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
    )


def resolve_reversi_target_axis(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    use_instance_seed_cycle: bool = False,
) -> ReversiTargetAxis:
    """Resolve a task-owned integer target support and selected answer."""

    target_answer, target_answer_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
        use_instance_seed_cycle=bool(use_instance_seed_cycle),
        namespace_support_permutation=True,
    )
    target_answer_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    return ReversiTargetAxis(
        target_answer=int(target_answer),
        target_answer_support=tuple(int(value) for value in target_answer_support),
        target_answer_probabilities=dict(target_answer_probabilities),
    )


def resolve_current_player(rng: Any, *, params: Mapping[str, Any]) -> int:
    """Resolve the current player for one sampled board."""

    explicit = params.get("current_player")
    if explicit is None:
        return int(BLACK if int(rng.randrange(2)) == 0 else WHITE)
    text = str(explicit).strip().lower()
    if text in {"black", "b", "1"}:
        return int(BLACK)
    if text in {"white", "w", "-1"}:
        return int(WHITE)
    raise ValueError(f"unsupported current_player: {explicit}")


def _empty_board(board_size: int) -> List[List[int]]:
    """Return one mutable empty board."""

    return [[EMPTY for _ in range(int(board_size))] for _ in range(int(board_size))]


def _freeze_board(board: Sequence[Sequence[int]]) -> Board:
    """Freeze one mutable board into the canonical tuple form."""

    return tuple(tuple(int(cell) for cell in row) for row in board)


def _set_cell(board: List[List[int]], coord: Coord, value: int) -> None:
    """Write one board cell in-place."""

    board[int(coord[0])][int(coord[1])] = int(value)


def sample_legal_destination_scene(
    *,
    rng: Any,
    board_size: int,
    current_player: int,
    target_answer: int,
) -> SampledReversiScene:
    """Search for one reachable board with an exact number of legal moves."""

    max_plies = max(int(board_size) + 2, (int(board_size) * int(board_size)) - 4)
    if int(target_answer) == 0:
        min_plies = max(int(board_size) + 6, int(0.65 * max_plies))
    elif int(target_answer) >= 5:
        min_plies = max(4, int(0.22 * max_plies))
        max_plies = max(min_plies + 4, int(0.62 * max_plies))
    else:
        min_plies = max(4, int(0.35 * max_plies))

    for _ in range(192):
        board = simulate_random_state(
            rng=rng,
            board_size=int(board_size),
            min_plies=int(min_plies),
            max_plies=int(max_plies),
        )
        legal_moves = legal_moves_with_flips(board, int(current_player))
        if int(len(legal_moves)) != int(target_answer):
            continue
        annotation_coords = tuple(sorted((int(row), int(col)) for row, col in legal_moves.keys()))
        return SampledReversiScene(
            board=board,
            current_player=int(current_player),
            legal_moves=legal_moves,
            annotation_coords=annotation_coords,
            annotation_entity_ids=tuple(coord_to_cell_id(coord) for coord in annotation_coords),
            marked_move=None,
            marked_move_flips=tuple(),
            construction_mode="simulated_legal_count",
        )
    raise ValueError("failed to find a reachable board with the requested legal-move count")


def sample_marked_move_flip_scene(
    *,
    rng: Any,
    board_size: int,
    current_player: int,
    target_answer: int,
) -> SampledReversiScene:
    """Construct a simple marked move that flips an exact number of discs."""

    board = _empty_board(int(board_size))
    opponent = int(WHITE if int(current_player) == int(BLACK) else BLACK)
    edge_specs = (
        ((0, 0), (0, 1), (1, 0)),
        ((0, int(board_size) - 1), (1, 0), (0, -1)),
        ((int(board_size) - 1, int(board_size) - 1), (0, -1), (-1, 0)),
        ((int(board_size) - 1, 0), (-1, 0), (0, 1)),
    )
    marked_move, primary_direction, secondary_direction = edge_specs[int(rng.randrange(len(edge_specs)))]

    if int(target_answer) <= int(board_size) - 2:
        line_specs = ((primary_direction, int(target_answer)),)
    else:
        # Keep large flip counts as two edge-aligned lines so the witness discs
        # remain easy to scan on both compact and classic boards.
        line_specs = ((primary_direction, 3), (secondary_direction, int(target_answer) - 3))

    flipped_coords: List[Coord] = []
    for direction, length in line_specs:
        row_delta, col_delta = int(direction[0]), int(direction[1])
        for step in range(1, int(length) + 1):
            flipped_coord = (
                int(marked_move[0] + (step * row_delta)),
                int(marked_move[1] + (step * col_delta)),
            )
            _set_cell(board, flipped_coord, opponent)
            flipped_coords.append(flipped_coord)
        terminal_coord = (
            int(marked_move[0] + ((int(length) + 1) * row_delta)),
            int(marked_move[1] + ((int(length) + 1) * col_delta)),
        )
        _set_cell(board, terminal_coord, current_player)
    frozen_board = _freeze_board(board)
    legal_moves = legal_moves_with_flips(frozen_board, int(current_player))
    marked_flips = tuple(sorted(legal_moves.get(tuple(marked_move), ())))
    if len(marked_flips) != int(target_answer):
        raise ValueError("constructed flip-count board did not match the target answer")
    return SampledReversiScene(
        board=frozen_board,
        current_player=int(current_player),
        legal_moves=legal_moves,
        annotation_coords=marked_flips,
        annotation_entity_ids=tuple(coord_to_cell_id(coord) for coord in marked_flips),
        marked_move=tuple(int(value) for value in marked_move),
        marked_move_flips=marked_flips,
        construction_mode="marked_flip",
    )


def _frontier_ply_windows(*, board_size: int, target_answer: int) -> Tuple[Tuple[int, int], ...]:
    """Return target-aware simulation windows for reachable frontier boards."""

    size = int(board_size)
    target = int(target_answer)
    max_plies = max(size + 4, (size * size) - 4)
    if target == 0:
        return ((max(size + 8, int(0.75 * max_plies)), max_plies),)
    if target <= 2:
        return (
            (4, max(size + 4, int(0.30 * max_plies))),
            (max(size + 6, int(0.70 * max_plies)), max_plies),
        )
    if target <= 5:
        return (
            (4, max(size + 8, int(0.42 * max_plies))),
            (max(6, int(0.32 * max_plies)), max(size + 12, int(0.58 * max_plies))),
        )
    return (
        (max(4, int(0.22 * max_plies)), max(size + 14, int(0.62 * max_plies))),
        (max(6, int(0.35 * max_plies)), max(size + 18, int(0.78 * max_plies))),
    )


def sample_frontier_disc_scene(
    *,
    rng: Any,
    board_size: int,
    query_player: int,
    target_answer: int,
) -> SampledReversiScene:
    """Search for a reachable board with an exact frontier-disc count."""

    windows = _frontier_ply_windows(board_size=int(board_size), target_answer=int(target_answer))
    for _attempt_batch in range(210):
        for min_plies, max_plies in shuffled_support(rng, windows):
            board = simulate_random_state(
                rng=rng,
                board_size=int(board_size),
                min_plies=int(min_plies),
                max_plies=int(max_plies),
            )
            annotation_coords = frontier_disc_coords(board, int(query_player))
            if int(len(annotation_coords)) != int(target_answer):
                continue
            return SampledReversiScene(
                board=board,
                current_player=int(query_player),
                legal_moves=legal_moves_with_flips(board, int(query_player)),
                annotation_coords=tuple(annotation_coords),
                annotation_entity_ids=tuple(coord_to_cell_id(coord) for coord in annotation_coords),
                marked_move=None,
                marked_move_flips=tuple(),
                construction_mode="simulated_frontier_disc_count",
            )
    raise ValueError("failed to find a reachable board with the requested frontier-disc count")


__all__ = [
    "resolve_current_player",
    "resolve_reversi_target_axis",
    "resolve_reversi_visual_axes",
    "sample_frontier_disc_scene",
    "sample_legal_destination_scene",
    "sample_marked_move_flip_scene",
]
