"""Identity-free sampling primitives for 3D Tic-Tac-Toe boards."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import support_probability_map
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import (
    resolve_integer_choice,
    resolve_integer_support,
)

from .defaults import DEFAULTS, GEN_DEFAULTS
from .rules import (
    all_coords,
    board_get,
    board_set,
    completed_lines,
    empty_board,
    freeze_board,
    immediate_winning_cells,
    layer_index,
    opponent,
    WINNING_LINES,
)
from .state import (
    LAYERS,
    LAYOUT_VARIANTS,
    OPTION_LABELS,
    STYLE_VARIANTS,
    Coord,
    TicTacToe3DAxes,
    TicTacToe3DSample,
)


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> tuple[str, dict[str, float]]:
    """Resolve one named scene axis with a scene-local namespace."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        namespace=str(namespace),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=tuple(str(value) for value in supported),
    )


def _resolve_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
    use_instance_seed_cycle: bool = False,
) -> tuple[int, dict[str, float]]:
    """Resolve one integer axis with a scene-local namespace."""

    support = resolve_integer_support(
        params,
        gen_defaults=GEN_DEFAULTS,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(support):
            raise ValueError(f"unsupported {explicit_key}: {selected}")
        return int(selected), support_probability_map(
            support,
            selected=int(selected),
            sort_keys=True,
        )
    if bool(use_instance_seed_cycle) and params.get("_sample_cursor") is not None:
        selected = support[abs(int(params["_sample_cursor"])) % len(support)]
        return int(selected), support_probability_map(support, sort_keys=True)

    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(value) for value in support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        use_instance_seed_cycle=bool(use_instance_seed_cycle),
        namespace_support_permutation=True,
    )
    return int(value), dict(probabilities)


def resolve_tic_tac_toe_3d_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace_root: str,
) -> TicTacToe3DAxes:
    """Resolve all reusable visual and construction axes for this scene."""

    layout_variant, layout_probs = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{namespace_root}.layout_variant",
        explicit_key="layout_variant",
        weights_key="layout_variant_weights",
        balance_flag_key="balanced_layout_variant_sampling",
        supported=LAYOUT_VARIANTS,
    )
    style_variant, style_probs = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{namespace_root}.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=STYLE_VARIANTS,
    )
    option_count, option_probs = _resolve_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        support_key="option_count_support",
        explicit_key="option_count",
        fallback_support=DEFAULTS.option_count_support,
        namespace=f"{namespace_root}.option_count",
        balanced_flag_key="balanced_option_count_sampling",
    )
    answer_option_index, answer_option_probs = _resolve_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        support_key="answer_option_index_support",
        explicit_key="answer_option_index",
        fallback_support=tuple(range(int(option_count))),
        namespace=f"{namespace_root}.answer_option_index.{int(option_count)}",
        balanced_flag_key="balanced_answer_option_sampling",
        use_instance_seed_cycle=True,
    )
    target_layer, target_layer_probs = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{namespace_root}.target_layer",
        explicit_key="target_layer",
        weights_key="target_layer_weights",
        balance_flag_key="balanced_target_layer_sampling",
        supported=tuple(layer for layer, _label in LAYERS),
    )
    target_answer, target_answer_probs = _resolve_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        support_key="layer_piece_count_support",
        explicit_key="target_answer",
        fallback_support=DEFAULTS.layer_piece_count_support,
        namespace=f"{namespace_root}.layer_piece_count.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
    )
    return TicTacToe3DAxes(
        layout_variant=str(layout_variant),
        style_variant=str(style_variant),
        option_count=int(option_count),
        answer_option_index=int(answer_option_index),
        target_layer=str(target_layer),
        target_answer=int(target_answer),
        layout_variant_probabilities=dict(layout_probs),
        style_variant_probabilities=dict(style_probs),
        option_count_probabilities=dict(option_probs),
        answer_option_probabilities=dict(answer_option_probs),
        target_layer_probabilities=dict(target_layer_probs),
        target_answer_probabilities=dict(target_answer_probs),
    )


def axis_support_metadata(
    params: Mapping[str, Any], axes: TicTacToe3DAxes
) -> dict[str, Any]:
    """Return support metadata for trace query parameters."""

    option_support = tuple(
        int(value)
        for value in params.get(
            "option_count_support",
            group_default(
                GEN_DEFAULTS, "option_count_support", DEFAULTS.option_count_support
            ),
        )
    )
    target_support = tuple(
        int(value)
        for value in params.get(
            "layer_piece_count_support",
            group_default(
                GEN_DEFAULTS,
                "layer_piece_count_support",
                DEFAULTS.layer_piece_count_support,
            ),
        )
    )
    return {
        "layout_variant": str(axes.layout_variant),
        "layout_variant_probabilities": dict(axes.layout_variant_probabilities),
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "option_count": int(axes.option_count),
        "option_count_support": [int(value) for value in option_support],
        "option_count_probabilities": dict(axes.option_count_probabilities),
        "answer_option": int(axes.answer_option_index),
        "answer_option_index": int(axes.answer_option_index),
        "answer_option_index_support": [
            int(value) for value in range(int(axes.option_count))
        ],
        "answer_option_probabilities": dict(axes.answer_option_probabilities),
        "target_layer": str(axes.target_layer),
        "target_layer_probabilities": dict(axes.target_layer_probabilities),
        "target_answer": int(axes.target_answer),
        "target_answer_support": [int(value) for value in target_support],
        "target_answer_probabilities": dict(axes.target_answer_probabilities),
    }


def _random_empty_coords(
    board: Sequence[Sequence[Sequence[str]]], *, exclude: set[Coord] | None = None
) -> list[Coord]:
    """Return empty cells while respecting an optional exclusion set."""

    excluded = set(exclude or set())
    return [
        coord
        for coord in all_coords()
        if coord not in excluded and board_get(board, coord) == ""
    ]


def sample_winning_move_scene(
    *,
    rng: Any,
    target_player: str,
    option_count: int,
    answer_option_index: int,
) -> TicTacToe3DSample:
    """Construct a board with exactly one labeled immediate winning move."""

    other_player = opponent(str(target_player))
    line = tuple(rng.choice(tuple(WINNING_LINES)))
    answer_cell = tuple(rng.choice(line))
    support_cells = tuple(coord for coord in line if coord != answer_cell)
    board = empty_board()
    for coord in support_cells:
        board_set(board, coord, str(target_player))

    filler_target = int(rng.randint(6, 13))
    candidate_coords = _random_empty_coords(board, exclude={answer_cell})
    rng.shuffle(candidate_coords)
    for coord in candidate_coords:
        occupied_count = sum(
            1 for placed in all_coords() if board_get(board, placed) != ""
        )
        if occupied_count >= filler_target + len(support_cells):
            break
        mark = str(target_player) if rng.random() < 0.45 else str(other_player)
        board_set(board, coord, mark)
        if completed_lines(board, "X") or completed_lines(board, "O"):
            board_set(board, coord, "")

    frozen = freeze_board(board)
    winning_cells = set(immediate_winning_cells(frozen, str(target_player)))
    if answer_cell not in winning_cells:
        raise ValueError("constructed winning move no longer completes a line")
    distractors = [
        coord
        for coord in _random_empty_coords(frozen, exclude={answer_cell})
        if coord not in winning_cells
    ]
    if len(distractors) < int(option_count) - 1:
        raise ValueError("not enough non-winning distractor cells")
    rng.shuffle(distractors)
    option_cells = list(distractors[: int(option_count) - 1])
    insert_at = min(max(0, int(answer_option_index)), int(option_count) - 1)
    option_cells.insert(insert_at, answer_cell)
    label = OPTION_LABELS[insert_at]
    correct_options = [
        OPTION_LABELS[index]
        for index, coord in enumerate(option_cells)
        if coord in winning_cells
    ]
    if tuple(correct_options) != (label,):
        raise ValueError("winning-move options are not unique")
    return TicTacToe3DSample(
        board=frozen,
        answer=str(label),
        answer_type="string",
        target_player=str(target_player),
        target_layer="",
        option_cells=tuple(option_cells),
        answer_cell=answer_cell,
        support_cells=tuple(support_cells),
        annotation_coords=(answer_cell, *support_cells),
        metadata={
            "winning_line": [[int(c[0]), int(c[1]), int(c[2])] for c in line],
            "winning_cells": [
                [int(c[0]), int(c[1]), int(c[2])] for c in sorted(winning_cells)
            ],
            "correct_option_labels": list(correct_options),
        },
    )


def sample_blocking_move_scene(
    *,
    rng: Any,
    target_player: str,
    option_count: int,
    answer_option_index: int,
) -> TicTacToe3DSample:
    """Construct a board with exactly one labeled move blocking the opponent."""

    threat_player = opponent(str(target_player))
    line = tuple(rng.choice(tuple(WINNING_LINES)))
    answer_cell = tuple(rng.choice(line))
    support_cells = tuple(coord for coord in line if coord != answer_cell)
    board = empty_board()
    for coord in support_cells:
        board_set(board, coord, str(threat_player))

    filler_target = int(rng.randint(4, 10))
    candidate_coords = _random_empty_coords(board, exclude={answer_cell})
    rng.shuffle(candidate_coords)
    for coord in candidate_coords:
        occupied_count = sum(
            1 for placed in all_coords() if board_get(board, placed) != ""
        )
        if occupied_count >= filler_target + len(support_cells):
            break
        mark = str(target_player) if rng.random() < 0.52 else str(threat_player)
        board_set(board, coord, mark)
        threat_cells = set(immediate_winning_cells(board, str(threat_player)))
        target_wins = set(immediate_winning_cells(board, str(target_player)))
        if (
            completed_lines(board, "X")
            or completed_lines(board, "O")
            or threat_cells != {answer_cell}
            or target_wins
        ):
            board_set(board, coord, "")

    frozen = freeze_board(board)
    threat_cells = set(immediate_winning_cells(frozen, str(threat_player)))
    target_wins = set(immediate_winning_cells(frozen, str(target_player)))
    if threat_cells != {answer_cell}:
        raise ValueError(
            "constructed blocking position must have exactly one opponent threat"
        )
    if target_wins:
        raise ValueError(
            "blocking position should not also contain a target-player win"
        )
    distractors = [
        coord
        for coord in _random_empty_coords(frozen, exclude={answer_cell})
        if coord not in threat_cells
    ]
    if len(distractors) < int(option_count) - 1:
        raise ValueError("not enough non-blocking distractor cells")
    rng.shuffle(distractors)
    option_cells = list(distractors[: int(option_count) - 1])
    insert_at = min(max(0, int(answer_option_index)), int(option_count) - 1)
    option_cells.insert(insert_at, answer_cell)
    label = OPTION_LABELS[insert_at]
    correct_options = [
        OPTION_LABELS[index]
        for index, coord in enumerate(option_cells)
        if coord in threat_cells
    ]
    if tuple(correct_options) != (label,):
        raise ValueError("blocking-move options are not unique")
    return TicTacToe3DSample(
        board=frozen,
        answer=str(label),
        answer_type="string",
        target_player=str(target_player),
        target_layer="",
        option_cells=tuple(option_cells),
        answer_cell=answer_cell,
        support_cells=tuple(support_cells),
        annotation_coords=(answer_cell, *support_cells),
        metadata={
            "threat_player": str(threat_player),
            "threat_line": [[int(c[0]), int(c[1]), int(c[2])] for c in line],
            "opponent_threat_cells": [
                [int(c[0]), int(c[1]), int(c[2])] for c in sorted(threat_cells)
            ],
            "target_player_winning_cells": [
                [int(c[0]), int(c[1]), int(c[2])] for c in sorted(target_wins)
            ],
            "correct_option_labels": list(correct_options),
        },
    )


def sample_layer_piece_count_scene(
    *,
    rng: Any,
    target_player: str,
    target_layer: str,
    target_answer: int,
) -> TicTacToe3DSample:
    """Construct one layer with an exact target-player piece count."""

    other_player = opponent(str(target_player))
    target_layer_index = layer_index(str(target_layer))
    board = empty_board()
    layer_coords = [
        (target_layer_index, row, col) for row in range(3) for col in range(3)
    ]
    rng.shuffle(layer_coords)
    target_cells = tuple(layer_coords[: int(target_answer)])
    for coord in target_cells:
        board_set(board, coord, str(target_player))

    remaining_layer = [
        coord for coord in layer_coords if coord not in set(target_cells)
    ]
    rng.shuffle(remaining_layer)
    opponent_in_layer = int(rng.randint(0, min(3, len(remaining_layer))))
    for coord in remaining_layer[:opponent_in_layer]:
        board_set(board, coord, other_player)

    outside_coords = [
        coord for coord in all_coords() if int(coord[0]) != int(target_layer_index)
    ]
    rng.shuffle(outside_coords)
    outside_piece_count = int(rng.randint(5, min(14, len(outside_coords))))
    for coord in outside_coords[:outside_piece_count]:
        board_set(
            board, coord, str(target_player) if rng.random() < 0.5 else other_player
        )

    frozen = freeze_board(board)
    annotation_coords = tuple(
        coord
        for coord in layer_coords
        if board_get(frozen, coord) == str(target_player)
    )
    if len(annotation_coords) != int(target_answer):
        raise ValueError("layer piece-count construction mismatch")
    return TicTacToe3DSample(
        board=frozen,
        answer=int(target_answer),
        answer_type="integer",
        target_player=str(target_player),
        target_layer=str(target_layer),
        annotation_coords=tuple(annotation_coords),
        metadata={
            "target_layer_index": int(target_layer_index),
            "target_layer_name": str(target_layer),
        },
    )


__all__ = [
    "axis_support_metadata",
    "resolve_tic_tac_toe_3d_axes",
    "sample_blocking_move_scene",
    "sample_layer_piece_count_scene",
    "sample_winning_move_scene",
]
