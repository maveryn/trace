"""Sampling primitives for Tetris board objectives."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis

from .defaults import DEFAULTS, GEN_DEFAULTS
from .rules import (
    TETROMINOES,
    best_clear_outcomes,
    board_key,
    board_size,
    bottom_edge_below_cells,
    can_place,
    column_heights,
    drop_collision,
    evaluate_outcome,
    freeze,
    hard_drop_top,
    horizontal_sweep_cells,
    is_supported_stack_board,
    piece_cells,
    placement_trace,
    row_empty_count,
    shape_size,
    shift_instruction_text,
)
from .state import Board, Coord, EMPTY, OPTION_LABELS, PIECE_ORDER, Option, Placement, SceneAxes, TetrisSample, SUPPORTED_SCENE_VARIANTS, SUPPORTED_STYLE_VARIANTS


def sample_label(instance_seed: int, *, namespace: str, labels: Sequence[str] = OPTION_LABELS) -> Tuple[str, Dict[str, float]]:
    values = tuple(str(label) for label in labels)
    if not values:
        raise ValueError("labels must not be empty")
    rng = spawn_rng(int(instance_seed), str(namespace))
    label = str(uniform_choice(rng, values))
    probability = 1.0 / float(len(values))
    return str(label), {str(value): float(probability) for value in values}


def sample_named_axis(
    *,
    namespace_key: str,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        namespace=str(namespace),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=tuple(str(item) for item in supported),
    )


def sample_integer_axis(
    *,
    namespace_key: str,
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> Tuple[int, Dict[str, float]]:
    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(item) for item in fallback_support),
        namespace=f"{str(namespace_key)}.{str(namespace)}",
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    return int(value), {str(key): float(val) for key, val in probabilities.items()}


def sample_scene_axes(*, namespace_key: str, instance_seed: int, params: Mapping[str, Any]) -> SceneAxes:
    """Sample scene-wide visual/style/board axes before objective construction."""

    scene_variant, scene_variant_probabilities = sample_named_axis(
        namespace_key=str(namespace_key),
        instance_seed=int(instance_seed),
        params=params,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = sample_named_axis(
        namespace_key=str(namespace_key),
        instance_seed=int(instance_seed),
        params=params,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_STYLE_VARIANTS,
    )
    option_count, option_count_probabilities = sample_integer_axis(
        namespace_key=str(namespace_key),
        instance_seed=int(instance_seed),
        params=params,
        support_key="option_count_support",
        explicit_key="option_count",
        fallback_support=DEFAULTS.option_count_support,
        namespace="option_count",
        balanced_flag_key="balanced_option_count_sampling",
    )
    board_rows, board_row_probabilities = sample_integer_axis(
        namespace_key=str(namespace_key),
        instance_seed=int(instance_seed),
        params=params,
        support_key="board_row_count_support",
        explicit_key="board_rows",
        fallback_support=DEFAULTS.board_row_count_support,
        namespace="board_rows",
        balanced_flag_key="balanced_board_size_sampling",
    )
    board_cols, board_col_probabilities = sample_integer_axis(
        namespace_key=str(namespace_key),
        instance_seed=int(instance_seed),
        params=params,
        support_key="board_col_count_support",
        explicit_key="board_cols",
        fallback_support=DEFAULTS.board_col_count_support,
        namespace="board_cols",
        balanced_flag_key="balanced_board_size_sampling",
    )
    return SceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        option_count=int(option_count),
        board_rows=int(board_rows),
        board_cols=int(board_cols),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        option_count_probabilities=dict(option_count_probabilities),
        board_row_probabilities=dict(board_row_probabilities),
        board_col_probabilities=dict(board_col_probabilities),
    )


def supported_stack_generation_meta(*, strategy: str) -> Dict[str, Any]:
    return {
        "board_generation": {
            "mode": "natural_supported_stack",
            "strategy": str(strategy),
            "cell_support_policy": "every locked cell is supported by the bottom or a locked cell below",
        }
    }


def fill_supported_column(rows: List[List[str]], *, col: int, top_row: int, rng, protected_empty: set[Coord] | None = None) -> None:
    protected = set(protected_empty or set())
    board_rows = len(rows)
    if board_rows <= 0:
        return
    top = max(0, min(int(top_row), board_rows - 1))
    for row in range(board_rows - 1, top - 1, -1):
        coord = (int(row), int(col))
        if coord in protected:
            break
        if rows[int(row)][int(col)] == EMPTY:
            rows[int(row)][int(col)] = str(rng.choice(PIECE_ORDER))


def supported_stack_from_heights(
    rng,
    *,
    board_rows: int,
    board_cols: int,
    heights: Sequence[int],
    protected_empty: set[Coord] | None = None,
) -> Board:
    rows = [[EMPTY for _ in range(int(board_cols))] for _ in range(int(board_rows))]
    protected = set(protected_empty or set())
    for col in range(int(board_cols)):
        height = max(0, min(int(heights[int(col)]), int(board_rows)))
        if height <= 0:
            continue
        top = int(board_rows) - int(height)
        fill_supported_column(rows, col=int(col), top_row=int(top), rng=rng, protected_empty=protected)
    board = freeze(rows)
    if not is_supported_stack_board(board):
        raise ValueError("constructed unsupported Tetris stack")
    return board


def random_supported_heights(rng, *, scene_variant: str, board_rows: int, board_cols: int, force_gap_column: bool = True) -> List[int]:
    height_low, height_high = {
        "low_stack": (1, 3),
        "notched_stack": (2, 5),
        "high_stack": (3, 6),
    }.get(str(scene_variant), (2, 5))
    capped_high = min(int(height_high), max(1, int(board_rows) - 2))
    capped_low = min(int(height_low), capped_high)
    heights = [int(rng.randint(capped_low, capped_high)) for _ in range(int(board_cols))]
    if force_gap_column and heights:
        gap_col = int(rng.randrange(int(board_cols)))
        heights[gap_col] = 0
    return heights


def random_stack_board(rng, *, scene_variant: str, board_rows: int, board_cols: int) -> Board:
    heights = random_supported_heights(
        rng,
        scene_variant=str(scene_variant),
        board_rows=int(board_rows),
        board_cols=int(board_cols),
        force_gap_column=True,
    )
    return supported_stack_from_heights(rng, board_rows=int(board_rows), board_cols=int(board_cols), heights=heights)


def random_piece_with_min_rows(rng, *, min_rows: int) -> Tuple[str, int]:
    candidates: List[Tuple[str, int]] = []
    for piece, orientations in TETROMINOES.items():
        for orientation_index, piece_shape in enumerate(orientations):
            distinct_rows = len({int(r) for r, _c in piece_shape})
            if distinct_rows >= int(min_rows):
                candidates.append((str(piece), int(orientation_index)))
    if not candidates:
        raise ValueError("no tetromino can cover requested row count")
    return tuple(rng.choice(candidates))  # type: ignore[return-value]


def positive_clear_profile_candidates(*, target_clear_count: int, board_rows: int, board_cols: int) -> List[Tuple[str, int, int]]:
    """Return piece/orientation/column triples that can cover every target row."""

    target = int(target_clear_count)
    if target <= 0:
        return []
    candidates: List[Tuple[str, int, int]] = []
    for piece, orientations in TETROMINOES.items():
        for orientation_index, piece_shape in enumerate(orientations):
            height, width = shape_size(piece_shape)
            if int(height) > int(board_rows) or int(width) > int(board_cols) or int(target) > int(height):
                continue
            target_local_rows = set(range(int(height) - int(target), int(height)))
            piece_local_rows = {int(row) for row, _col in piece_shape}
            if not target_local_rows.issubset(piece_local_rows):
                continue
            for col in range(0, int(board_cols) - int(width) + 1):
                candidates.append((str(piece), int(orientation_index), int(col)))
    return candidates


def construct_supported_profile_clear_board(
    rng,
    *,
    piece: str,
    orientation_index: int,
    col: int,
    target_clear_count: int,
    scene_variant: str,
    board_rows: int,
    board_cols: int,
) -> Tuple[Board, Placement, Any]:
    """Build one supported stack profile for a candidate clear-producing placement."""

    piece_shape = TETROMINOES[str(piece)][int(orientation_index)]
    height, width = shape_size(piece_shape)
    if int(height) > int(board_rows) or int(width) > int(board_cols):
        raise ValueError("piece does not fit Tetris clear profile board")
    top = int(board_rows) - int(height)
    placement = Placement(str(piece), int(orientation_index), int(col), int(top))
    piece_cell_set = set(piece_cells(placement))
    target_rows = set(range(int(board_rows) - int(target_clear_count), int(board_rows)))
    if not target_rows.issubset({int(row) for row, _col in piece_cell_set}):
        raise ValueError("candidate piece does not occupy every target clear row")

    max_extra = {"low_stack": 0, "notched_stack": 1, "high_stack": 2}.get(str(scene_variant), 1)
    heights: List[int] = []
    for column in range(int(board_cols)):
        piece_rows_in_column = [int(row) for row, cell_col in piece_cell_set if int(cell_col) == int(column)]
        if piece_rows_in_column:
            heights.append(max(0, int(board_rows) - min(piece_rows_in_column)))
            continue
        heights.append(min(int(board_rows), int(target_clear_count) + int(rng.randint(0, int(max_extra)))))

    board = supported_stack_from_heights(
        rng,
        board_rows=int(board_rows),
        board_cols=int(board_cols),
        heights=heights,
        protected_empty=piece_cell_set,
    )
    if any(all(str(cell) != EMPTY for cell in row) for row in board):
        raise ValueError("constructed Tetris profile has a full row before locking")
    if not can_place(board, placement):
        raise ValueError("constructed Tetris profile overlaps the placement")
    dropped_top = hard_drop_top(board, piece=str(piece), orientation_index=int(orientation_index), col=int(col))
    if dropped_top != int(top):
        raise ValueError("constructed Tetris profile did not hard-drop to the planned placement")
    outcome = evaluate_outcome(board, placement)
    if int(outcome.clear_count) != int(target_clear_count):
        raise ValueError(f"profile Tetris clear construction produced {outcome.clear_count} clears instead of {target_clear_count}")
    return board, placement, outcome


def construct_board_with_target_clear(
    rng,
    *,
    target_clear_count: int,
    scene_variant: str,
    board_rows: int,
    board_cols: int,
) -> Tuple[Board, Placement, Any]:
    """Build a supported Tetris stack whose chosen placement clears exactly target rows."""

    target = int(target_clear_count)
    if target > 0:
        candidates = positive_clear_profile_candidates(
            target_clear_count=int(target),
            board_rows=int(board_rows),
            board_cols=int(board_cols),
        )
        rng.shuffle(candidates)
        for piece, orientation_index, col in candidates:
            for _profile_attempt in range(4):
                try:
                    return construct_supported_profile_clear_board(
                        rng,
                        piece=str(piece),
                        orientation_index=int(orientation_index),
                        col=int(col),
                        target_clear_count=int(target),
                        scene_variant=str(scene_variant),
                        board_rows=int(board_rows),
                        board_cols=int(board_cols),
                    )
                except ValueError:
                    continue
        raise ValueError(f"failed to construct positive Tetris clear profile for {target} rows")

    for _attempt in range(900):
        piece, orientation_index = random_piece_with_min_rows(rng, min_rows=1)
        piece_shape = TETROMINOES[str(piece)][int(orientation_index)]
        height, width = shape_size(piece_shape)
        if int(width) > int(board_cols) or int(height) > int(board_rows):
            continue
        col = int(rng.randint(0, int(board_cols) - int(width)))
        board = random_stack_board(rng, scene_variant=str(scene_variant), board_rows=int(board_rows), board_cols=int(board_cols))
        dropped_top = hard_drop_top(board, piece=str(piece), orientation_index=int(orientation_index), col=int(col))
        if dropped_top is None:
            continue
        placement = Placement(str(piece), int(orientation_index), int(col), int(dropped_top))
        outcome = evaluate_outcome(board, placement)
        if int(outcome.clear_count) == int(target):
            return board, placement, outcome
    raise ValueError(f"failed to construct Tetris board for clear count {target}")


def build_line_clear_sample(rng, *, scene_variant: str, board_rows: int, board_cols: int, target_clear_count: int) -> TetrisSample:
    """Construct a board where the next piece has a controlled maximum clear count."""

    target = int(target_clear_count)
    for _attempt in range(1200):
        board, base_placement, _base_outcome = construct_board_with_target_clear(
            rng,
            target_clear_count=target,
            scene_variant=str(scene_variant),
            board_rows=int(board_rows),
            board_cols=int(board_cols),
        )
        piece = str(base_placement.piece)
        best_clear, best_outcomes = best_clear_outcomes(board, piece=piece)
        if int(best_clear) != int(target) or not best_outcomes:
            continue
        best_outcome = sorted(
            best_outcomes,
            key=lambda outcome: (int(outcome.holes_after), int(outcome.max_height_after), int(outcome.placement.col)),
        )[0]
        return TetrisSample(
            answer=int(best_clear),
            answer_type="integer",
            board=board,
            piece=piece,
            preview_orientation_index=0,
            placement=best_outcome.placement,
            falling_placement=None,
            outcome=best_outcome,
            options=(),
            annotation_entity_ids=("main", "next_piece"),
            annotation_kind="board_and_next_piece",
            metadata={
                **supported_stack_generation_meta(strategy="line_clear_guided_profile" if target > 0 else "line_clear_supported_stack"),
                "target_clear_count": int(best_clear),
                "best_clear_count": int(best_clear),
                "max_clear_placement_count": len(best_outcomes),
            },
        )
    raise ValueError("failed to construct Tetris max-clear sample")


def uncleared_board_after_lock(outcome) -> Board:
    return outcome.locked_board


def clear_without_gravity(outcome) -> Board:
    rows = [list(row) for row in outcome.locked_board]
    for row in outcome.cleared_rows:
        rows[int(row)] = [EMPTY for _ in range(len(rows[int(row)]))]
    return freeze(rows)


def mutated_result_board(rng, board: Board) -> Board:
    board_rows, board_cols = board_size(board)
    return random_stack_board(
        rng,
        scene_variant=str(rng.choice(SUPPORTED_SCENE_VARIANTS)),
        board_rows=int(board_rows),
        board_cols=int(board_cols),
    )


def drop_result_distractor_boards(rng, board: Board, outcome) -> Tuple[Board, ...]:
    boards: List[Board] = []
    seen = {board_key(outcome.result_board)}

    def add(candidate: Board) -> None:
        key = board_key(candidate)
        if key not in seen:
            seen.add(key)
            boards.append(candidate)

    add(uncleared_board_after_lock(outcome))
    for delta in (-1, 1, -2, 2):
        alt_top = hard_drop_top(
            board,
            piece=str(outcome.placement.piece),
            orientation_index=int(outcome.placement.orientation_index),
            col=int(outcome.placement.col + delta),
        )
        if alt_top is not None:
            alt = Placement(str(outcome.placement.piece), int(outcome.placement.orientation_index), int(outcome.placement.col + delta), int(alt_top))
            if can_place(board, alt):
                alt_board = evaluate_outcome(board, alt).result_board
                if is_supported_stack_board(alt_board):
                    add(alt_board)
    orientations = TETROMINOES[str(outcome.placement.piece)]
    for orientation_index in range(len(orientations)):
        if int(orientation_index) == int(outcome.placement.orientation_index):
            continue
        alt_top = hard_drop_top(board, piece=str(outcome.placement.piece), orientation_index=int(orientation_index), col=int(outcome.placement.col))
        if alt_top is not None:
            alt = Placement(str(outcome.placement.piece), int(orientation_index), int(outcome.placement.col), int(alt_top))
            if can_place(board, alt):
                alt_board = evaluate_outcome(board, alt).result_board
                if is_supported_stack_board(alt_board):
                    add(alt_board)
    while len(boards) < 8:
        add(mutated_result_board(rng, boards[-1] if boards else outcome.locked_board))
    return tuple(boards)


def build_drop_result_sample(
    rng,
    *,
    scene_variant: str,
    board_rows: int,
    board_cols: int,
    target_clear_count: int,
    option_count: int,
    answer_label: str,
) -> TetrisSample:
    """Construct START and result-option boards for one fixed drop."""

    labels = OPTION_LABELS[: int(option_count)]
    board: Board | None = None
    placement: Placement | None = None
    outcome = None
    falling_placement: Placement | None = None
    for _attempt in range(900):
        board_candidate, placement_candidate, outcome_candidate = construct_board_with_target_clear(
            rng,
            target_clear_count=int(target_clear_count),
            scene_variant=str(scene_variant),
            board_rows=int(board_rows),
            board_cols=int(board_cols),
        )
        candidate_falling = Placement(str(placement_candidate.piece), int(placement_candidate.orientation_index), int(placement_candidate.col), 0)
        if not can_place(board_candidate, candidate_falling):
            continue
        board = board_candidate
        placement = placement_candidate
        outcome = outcome_candidate
        falling_placement = candidate_falling
        break
    if board is None or placement is None or outcome is None or falling_placement is None:
        raise ValueError("failed to construct Tetris fixed-drop sample")
    distractors = list(drop_result_distractor_boards(rng, board, outcome))
    rng.shuffle(distractors)
    options: List[Option] = []
    distractor_cursor = 0
    for label in labels:
        if str(label) == str(answer_label):
            options.append(Option(str(label), outcome.result_board, None, outcome, True))
        else:
            options.append(Option(str(label), distractors[distractor_cursor], None, None, False))
            distractor_cursor += 1
    return TetrisSample(
        answer=str(answer_label),
        answer_type="string",
        board=board,
        piece=str(placement.piece),
        preview_orientation_index=int(placement.orientation_index),
        placement=placement,
        falling_placement=falling_placement,
        outcome=outcome,
        options=tuple(options),
        annotation_entity_ids=(f"option_{str(answer_label).lower()}",),
        annotation_kind="option_panel",
        metadata={
            **supported_stack_generation_meta(strategy="drop_result_guided_profile" if int(outcome.clear_count) > 0 else "drop_result_supported_stack"),
            "target_clear_count": int(outcome.clear_count),
        },
    )


def build_active_piece_shape_sample(
    rng,
    *,
    scene_variant: str,
    board_rows: int,
    board_cols: int,
    target_piece: str,
) -> TetrisSample:
    """Construct a static Tetris board with one visible active falling piece."""

    piece = str(target_piece)
    if piece not in PIECE_ORDER:
        raise ValueError(f"unsupported Tetris target piece: {piece}")
    option_labels = tuple(OPTION_LABELS[:4])
    for _attempt in range(300):
        distractor_pieces = [str(candidate) for candidate in PIECE_ORDER if str(candidate) != str(piece)]
        rng.shuffle(distractor_pieces)
        option_pieces = distractor_pieces[:3]
        answer_index = int(rng.randrange(len(option_labels)))
        option_pieces.insert(answer_index, str(piece))
        option_entries = tuple(
            {"label": str(option_labels[index]), "piece": str(option_pieces[index])}
            for index in range(len(option_labels))
        )
        answer_label = str(option_labels[answer_index])
        orientation_index = int(rng.randrange(len(TETROMINOES[piece])))
        piece_shape = TETROMINOES[piece][orientation_index]
        height, width = shape_size(piece_shape)
        if int(width) > int(board_cols) or int(height) > int(board_rows):
            continue
        col = int(rng.randint(0, int(board_cols) - int(width)))
        falling = Placement(piece, int(orientation_index), int(col), 0)
        board = random_stack_board(
            rng,
            scene_variant=str(scene_variant),
            board_rows=int(board_rows),
            board_cols=int(board_cols),
        )
        if not can_place(board, falling):
            continue
        falling_ids = tuple(f"main_cell_{int(row)}_{int(col_index)}" for row, col_index in piece_cells(falling))
        return TetrisSample(
            answer=str(answer_label),
            answer_type="string",
            board=board,
            piece=str(piece),
            preview_orientation_index=int(orientation_index),
            placement=None,
            falling_placement=falling,
            outcome=None,
            options=(),
            annotation_entity_ids=tuple(falling_ids),
            annotation_kind="active_piece_bbox",
            metadata={
                **supported_stack_generation_meta(strategy="active_piece_supported_stack"),
                "target_piece": str(piece),
                "shape_options": [str(entry["piece"]) for entry in option_entries],
                "shape_option_entries": [dict(entry) for entry in option_entries],
                "correct_option_label": str(answer_label),
                "correct_shape": str(piece),
                "falling_placement": placement_trace(falling),
                "falling_piece_cell_ids": [str(entity_id) for entity_id in falling_ids],
            },
        )
    raise ValueError("failed to construct Tetris active-piece shape sample")


def row_qualifies_for_status(row: Sequence[str], *, row_status: str) -> bool:
    empty_count = row_empty_count(row)
    if str(row_status) == "full":
        return int(empty_count) == 0
    if str(row_status) == "one_gap":
        return int(empty_count) == 1
    raise ValueError(f"unsupported Tetris row occupancy status: {row_status}")


def qualifying_row_indices(board: Board, *, row_status: str) -> Tuple[int, ...]:
    return tuple(
        int(row_index)
        for row_index, row in enumerate(board)
        if row_qualifies_for_status(row, row_status=str(row_status))
    )


def build_row_occupancy_sample(
    rng,
    *,
    scene_variant: str,
    board_rows: int,
    board_cols: int,
    row_status: str,
    target_row_count: int,
) -> TetrisSample:
    """Construct a static board with an exact count of full or one-gap rows."""

    target = int(target_row_count)
    max_extra = {"low_stack": 2, "notched_stack": 4, "high_stack": 6}.get(str(scene_variant), 4)
    max_height = min(int(board_rows) - 2, int(target) + int(max_extra))
    if str(row_status) == "full":
        if target <= 0:
            heights = random_supported_heights(rng, scene_variant=str(scene_variant), board_rows=int(board_rows), board_cols=int(board_cols), force_gap_column=True)
            heights[int(rng.randrange(board_cols))] = 0
        else:
            heights = [int(rng.randint(int(target), max(int(target), int(max_height)))) for _ in range(int(board_cols))]
            heights[int(rng.randrange(board_cols))] = int(target)
        board = supported_stack_from_heights(rng, board_rows=int(board_rows), board_cols=int(board_cols), heights=heights)
    elif str(row_status) == "one_gap":
        if target <= 0:
            first_gap = int(rng.randrange(board_cols))
            second_gap = int((first_gap + 1 + rng.randrange(max(1, board_cols - 1))) % board_cols)
            heights = [int(rng.randint(1, max(1, int(max_height)))) for _ in range(int(board_cols))]
            heights[first_gap] = 0
            heights[second_gap] = 0
        else:
            gap_col = int(rng.randrange(board_cols))
            stopper_candidates = [col for col in range(int(board_cols)) if int(col) != int(gap_col)]
            stopper_col = int(rng.choice(stopper_candidates))
            heights = [int(rng.randint(int(target), max(int(target), int(max_height)))) for _ in range(int(board_cols))]
            heights[gap_col] = 0
            heights[stopper_col] = int(target)
        board = supported_stack_from_heights(rng, board_rows=int(board_rows), board_cols=int(board_cols), heights=heights)
    else:
        raise ValueError(f"unsupported Tetris row occupancy status: {row_status}")
    qualifying_rows = qualifying_row_indices(board, row_status=str(row_status))
    if len(qualifying_rows) != int(target):
        raise ValueError("constructed Tetris row-occupancy board has wrong answer")
    return TetrisSample(
        answer=int(target),
        answer_type="integer",
        board=board,
        piece="",
        preview_orientation_index=0,
        placement=None,
        falling_placement=None,
        outcome=None,
        options=(),
        annotation_entity_ids=tuple(f"main_row_{int(row)}" for row in qualifying_rows),
        annotation_kind="row_set",
        metadata={
            **supported_stack_generation_meta(strategy="row_occupancy_column_heights"),
            "target_row_count": int(target),
            "row_occupancy_status": str(row_status),
            "qualifying_rows": [int(row) for row in qualifying_rows],
            "row_empty_counts": [int(row_empty_count(row)) for row in board],
        },
    )


def build_drop_collision_time_sample(
    rng,
    *,
    scene_variant: str,
    board_rows: int,
    board_cols: int,
    shift_delta: int,
    target_drop_steps: int,
) -> TetrisSample:
    """Construct a falling-piece scene with a controlled downward timestep count."""

    target = int(target_drop_steps)
    for _attempt in range(1600):
        piece = str(rng.choice(PIECE_ORDER))
        orientation_index = int(rng.randrange(len(TETROMINOES[piece])))
        piece_shape = TETROMINOES[piece][orientation_index]
        height, width = shape_size(piece_shape)
        if int(width) > int(board_cols):
            continue
        final_top = int(target)
        if final_top < 0 or final_top >= int(board_rows) - int(height):
            continue
        shifted_col_candidates = []
        for shifted_col in range(0, int(board_cols) - int(width) + 1):
            start_col = int(shifted_col) - int(shift_delta)
            if 0 <= int(start_col) <= int(board_cols) - int(width):
                shifted_col_candidates.append((int(start_col), int(shifted_col)))
        if not shifted_col_candidates:
            continue
        start_col, shifted_col = tuple(rng.choice(shifted_col_candidates))
        start_placement = Placement(piece, int(orientation_index), int(start_col), 0)
        final_placement = Placement(piece, int(orientation_index), int(shifted_col), int(final_top))
        path_cells: set[Coord] = set()
        for top in range(0, int(final_top) + 1):
            path_cells.update(piece_cells(Placement(piece, int(orientation_index), int(shifted_col), int(top))))
        sweep_cells = set(horizontal_sweep_cells(start_placement, shift_delta=int(shift_delta)))
        blocker_cells = set(bottom_edge_below_cells(final_placement))
        blocker_cells = {(int(row), int(col)) for row, col in blocker_cells if 0 <= int(row) < int(board_rows)}
        if not blocker_cells:
            continue
        rows = [[EMPTY for _ in range(int(board_cols))] for _ in range(int(board_rows))]
        for row, col in blocker_cells:
            fill_supported_column(rows, col=int(col), top_row=int(row), rng=rng)
        protected = set(path_cells) | set(sweep_cells) | set(blocker_cells)
        height_cap = {"low_stack": 3, "notched_stack": 5, "high_stack": 7}.get(str(scene_variant), 5)
        blocker_cols = {int(col) for _row, col in blocker_cells}
        for col in range(int(board_cols)):
            if int(col) in blocker_cols:
                continue
            protected_rows = [int(row) for row, protected_col in protected if int(protected_col) == int(col)]
            top_limit = max(protected_rows) + 1 if protected_rows else 0
            max_height = max(0, int(board_rows) - int(top_limit))
            if max_height <= 0 or rng.random() > 0.70:
                continue
            height_value = int(rng.randint(1, min(int(height_cap), int(max_height))))
            top_row = int(board_rows) - int(height_value)
            if int(top_row) < int(top_limit):
                top_row = int(top_limit)
            fill_supported_column(rows, col=int(col), top_row=int(top_row), rng=rng, protected_empty=set(path_cells) | set(sweep_cells))
        board = freeze(rows)
        if not is_supported_stack_board(board):
            continue
        collision = drop_collision(board, start_placement, shift_delta=int(shift_delta))
        if collision is None or int(collision.drop_steps) != int(target):
            continue
        if str(collision.collision_kind) != "locked_block" or not collision.blocker_cells:
            continue
        start_ids = tuple(f"start_cell_{int(row)}_{int(col)}" for row, col in piece_cells(start_placement))
        stop_ids = tuple(f"start_cell_{int(row)}_{int(col)}" for row, col in collision.blocker_cells)
        return TetrisSample(
            answer=int(collision.drop_steps),
            answer_type="integer",
            board=board,
            piece=str(piece),
            preview_orientation_index=int(orientation_index),
            placement=collision.final_placement,
            falling_placement=start_placement,
            outcome=None,
            options=(),
            annotation_entity_ids=tuple(start_ids + stop_ids),
            annotation_kind="collision_keyed_cell_sets",
            metadata={
                **supported_stack_generation_meta(strategy="drop_collision_supported_blockers"),
                "target_drop_steps": int(target),
                "drop_steps": int(collision.drop_steps),
                "shift_delta": int(shift_delta),
                "shift_magnitude": abs(int(shift_delta)),
                "shift_instruction": shift_instruction_text(int(shift_delta)),
                "collision_kind": str(collision.collision_kind),
                "start_placement": placement_trace(collision.start_placement),
                "shifted_placement": placement_trace(collision.shifted_placement),
                "final_placement": placement_trace(collision.final_placement),
                "collision_blocker_cells": [[int(r), int(c)] for r, c in collision.blocker_cells],
                "bottom_contact_cells": [[int(r), int(c)] for r, c in collision.bottom_contact_cells],
                "annotation_entity_id_map": {
                    "start_piece": [str(entity_id) for entity_id in start_ids],
                    "stop_witness": [str(entity_id) for entity_id in stop_ids],
                },
            },
        )
    raise ValueError("failed to construct Tetris drop-collision-time sample")
