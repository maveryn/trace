"""Identity-free sampling primitives for sliding-block boards."""

from __future__ import annotations

from itertools import cycle
from string import ascii_uppercase
from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import shuffled_support, uniform_choice, weighted_support_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.sampling import get_games_int_param as _get_int
from trace_tasks.tasks.games.shared.sampling import get_games_int_range as _get_range
from trace_tasks.tasks.shared.config_defaults import group_default

from .defaults import GEN_DEFAULTS
from .rules import apply_move, block_ids_by_orientation, legal_moves, movable_block_ids, state_signature
from .state import BLOCK_FILLS, EXIT_SIDES, OPTION_LABELS, SCENE_VARIANTS, TARGET_FILL, BlockMoveSpec, BlockSpec


def _axis_probabilities(values: Sequence[str], selected: str) -> dict[str, float]:
    """Return a JSON-stable probability map for one selected categorical axis."""

    return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in values}


def _resolve_axis(
    params: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported_values: Sequence[str],
) -> tuple[str, dict[str, float]]:
    """Resolve one scene axis from params/defaults without public task identity."""

    supported = tuple(str(value) for value in supported_values)
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        selected = str(explicit)
        if selected not in set(supported):
            raise ValueError(f"unsupported sliding-block {explicit_key}: {selected}")
        return selected, _axis_probabilities(supported, selected)

    weights_raw = params.get(str(weights_key), group_default(defaults, str(weights_key), {}))
    weights = {str(value): float(weights_raw.get(str(value), 0.0)) for value in supported} if isinstance(weights_raw, Mapping) else {}
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = weighted_support_choice(
        rng,
        supported,
        weights=weights if weights else None,
    )
    return str(selected), dict(probabilities)


def resolve_scene_variant(params: Mapping[str, Any], *, instance_seed: int) -> tuple[str, dict[str, float]]:
    """Resolve the visual board-style axis used by all sliding-block tasks."""

    return _resolve_axis(
        params,
        defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.sliding_block.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_values=SCENE_VARIANTS,
    )


def resolve_exit_side(params: Mapping[str, Any], *, instance_seed: int) -> tuple[str, dict[str, float]]:
    """Resolve which board side the target block exits through."""

    return _resolve_axis(
        params,
        defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.sliding_block.exit_side",
        explicit_key="exit_side",
        weights_key="exit_side_weights",
        balance_flag_key="balanced_exit_side_sampling",
        supported_values=EXIT_SIDES,
    )


def integer_support(params: Mapping[str, Any], *, min_key: str, max_key: str, fallback_min: int, fallback_max: int) -> tuple[int, ...]:
    """Resolve an inclusive integer answer support from task-owned keys."""

    low = _get_int(params, GEN_DEFAULTS, str(min_key), int(fallback_min))
    high = _get_int(params, GEN_DEFAULTS, str(max_key), int(fallback_max))
    if int(low) > int(high):
        raise ValueError(f"{min_key} must be <= {max_key}")
    return tuple(range(int(low), int(high) + 1))


def select_target_from_support(
    params: Mapping[str, Any],
    *,
    support: Sequence[int | str],
    instance_seed: int,
    namespace: str,
) -> int | str:
    """Select a deterministic target from a task-owned answer support."""

    values = tuple(support)
    if not values:
        raise ValueError("sliding-block answer support must not be empty")
    rng = spawn_rng(int(instance_seed), str(namespace))
    return uniform_choice(rng, values)


def move_result_option_labels(params: Mapping[str, Any], *, instance_seed: int) -> tuple[tuple[str, ...], dict[str, float]]:
    """Resolve the visual option labels for the final-board task."""

    raw_support = params.get("move_result_option_count_support", group_default(GEN_DEFAULTS, "move_result_option_count_support", [4]))
    support = tuple(int(value) for value in raw_support)
    if not support:
        raise ValueError("move_result_option_count_support must not be empty")
    explicit = params.get("option_count", params.get("move_result_option_count"))
    if explicit is not None:
        count = int(explicit)
        if count not in set(support):
            raise ValueError(f"unsupported sliding-block option_count: {count}")
        return tuple(ascii_uppercase[:count]), {str(value): (1.0 if int(value) == count else 0.0) for value in support}
    rng = spawn_rng(int(instance_seed), "games.sliding_block.final_board.option_count")
    count = int(uniform_choice(rng, support))
    return tuple(ascii_uppercase[:count]), {str(value): 1.0 / float(len(support)) for value in support}


def target_layout(
    *,
    exit_side: str,
    rows: int,
    cols: int,
    path_support_count: int,
    rng,
) -> tuple[BlockSpec, list[tuple[int, int]]]:
    """Place the red target block and derive its straight exit-path cells."""

    target_length = 2
    if str(exit_side) in {"right", "left"}:
        row = int(rng.randint(1, max(1, int(rows) - 2)))
        if str(exit_side) == "right":
            max_col = max(0, int(cols) - target_length - int(path_support_count))
            col = int(rng.randint(0, int(max_col)))
            path_cells = [(row, cell_col) for cell_col in range(int(col + target_length), int(cols))]
        else:
            min_col = int(path_support_count)
            max_col = max(min_col, int(cols) - target_length)
            col = int(rng.randint(int(min_col), int(max_col)))
            path_cells = [(row, cell_col) for cell_col in range(0, int(col))]
        target = BlockSpec("target", "T", row, col, 1, target_length, "target", TARGET_FILL)
    else:
        col = int(rng.randint(1, max(1, int(cols) - 2)))
        if str(exit_side) == "bottom":
            max_row = max(0, int(rows) - target_length - int(path_support_count))
            row = int(rng.randint(0, int(max_row)))
            path_cells = [(cell_row, col) for cell_row in range(int(row + target_length), int(rows))]
        else:
            min_row = int(path_support_count)
            max_row = max(min_row, int(rows) - target_length)
            row = int(rng.randint(int(min_row), int(max_row)))
            path_cells = [(cell_row, col) for cell_row in range(0, int(row))]
        target = BlockSpec("target", "T", row, col, target_length, 1, "target", TARGET_FILL)
    return target, list(path_cells)


def perpendicular_block_for_path_cell(
    *,
    path_cell: tuple[int, int],
    exit_side: str,
    rows: int,
    cols: int,
    rng,
    block_id: str,
    role: str,
    fill_rgb: tuple[int, int, int],
) -> BlockSpec:
    """Create a block perpendicular to the exit path and covering one path cell."""

    row, col = int(path_cell[0]), int(path_cell[1])
    if str(exit_side) in {"right", "left"}:
        for height in ([3, 2] if int(rows) >= 6 else [2]):
            top_candidates = [
                top
                for top in range(int(row) - int(height) + 1, int(row) + 1)
                if 0 <= top and top + int(height) <= int(rows)
            ]
            if top_candidates:
                top = int(top_candidates[int(rng.randrange(len(top_candidates)))])
                return BlockSpec(str(block_id), "", top, col, int(height), 1, str(role), tuple(fill_rgb))
    else:
        for width in ([3, 2] if int(cols) >= 6 else [2]):
            left_candidates = [
                left
                for left in range(int(col) - int(width) + 1, int(col) + 1)
                if 0 <= left and left + int(width) <= int(cols)
            ]
            if left_candidates:
                left = int(left_candidates[int(rng.randrange(len(left_candidates)))])
                return BlockSpec(str(block_id), "", row, left, 1, int(width), str(role), tuple(fill_rgb))
    raise ValueError("failed to place a path-blocking sliding block")


def candidate_distractor(
    *,
    block_id: str,
    rows: int,
    cols: int,
    rng,
    role: str,
    fill_rgb: tuple[int, int, int],
) -> BlockSpec:
    """Sample one non-overlapping-candidate rectangular distractor shape."""

    horizontal = bool(rng.randrange(2))
    length = 3 if int(rng.randrange(4)) == 0 else 2
    width, height = (int(length), 1) if horizontal else (1, int(length))
    row = int(rng.randint(0, int(rows) - int(height)))
    col = int(rng.randint(0, int(cols) - int(width)))
    return BlockSpec(str(block_id), "", row, col, int(height), int(width), str(role), tuple(fill_rgb))


def assign_labels(blocks: Sequence[BlockSpec]) -> list[BlockSpec]:
    """Assign stable visible letters to every non-target block."""

    labels = list(ascii_uppercase)
    out: list[BlockSpec] = []
    cursor = 0
    for block in blocks:
        if str(block.block_id) == "target":
            out.append(block)
            continue
        out.append(
            BlockSpec(
                block.block_id,
                labels[cursor],
                block.row,
                block.col,
                block.height,
                block.width,
                block.role,
                block.fill_rgb,
            )
        )
        cursor += 1
    return out


def serialize_blocks(blocks: Sequence[BlockSpec]) -> list[dict[str, Any]]:
    """Serialize block specs into JSON-stable execution/render records."""

    return [
        {
            "block_id": str(block.block_id),
            "label": str(block.label),
            "row": int(block.row),
            "col": int(block.col),
            "height": int(block.height),
            "width": int(block.width),
            "role": str(block.role),
            "fill_rgb": [int(value) for value in block.fill_rgb],
            "cells": [[int(row), int(col)] for row, col in block.cells],
        }
        for block in blocks
    ]


def deserialize_blocks(blocks: Sequence[Mapping[str, Any]]) -> list[BlockSpec]:
    """Deserialize trace/render block records back into immutable specs."""

    return [
        BlockSpec(
            str(block["block_id"]),
            str(block["label"]),
            int(block["row"]),
            int(block["col"]),
            int(block["height"]),
            int(block["width"]),
            str(block["role"]),
            tuple(int(value) for value in block["fill_rgb"]),
        )
        for block in blocks
    ]


def build_exit_path_board(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    exit_side: str,
    blocker_target: int,
    namespace: str,
) -> dict[str, Any]:
    """Build a board with exactly the requested number of exit-path blockers."""

    rng = spawn_rng(int(instance_seed), str(namespace))
    rows_min, rows_max = _get_range(params, GEN_DEFAULTS, min_key="board_rows_min", max_key="board_rows_max", fallback_min=6, fallback_max=7)
    cols_min, cols_max = _get_range(params, GEN_DEFAULTS, min_key="board_cols_min", max_key="board_cols_max", fallback_min=6, fallback_max=7)
    blocker_count = int(blocker_target)
    rows = int(rng.randint(int(rows_min), int(rows_max)))
    cols = int(rng.randint(int(cols_min), int(cols_max)))
    if str(exit_side) in {"right", "left"}:
        cols = max(int(cols), int(blocker_count) + 2)
    else:
        rows = max(int(rows), int(blocker_count) + 2)
    rows = min(max(int(rows), int(rows_min)), int(rows_max))
    cols = min(max(int(cols), int(cols_min)), int(cols_max))

    target_block, path_cells = target_layout(
        exit_side=str(exit_side),
        rows=int(rows),
        cols=int(cols),
        path_support_count=int(blocker_count),
        rng=rng,
    )
    if len(path_cells) < int(blocker_count):
        raise ValueError("not enough path cells for requested blocker count")

    selected_path_cells = list(path_cells)
    rng.shuffle(selected_path_cells)
    selected_path_cells = selected_path_cells[: int(blocker_count)]

    blocks: list[BlockSpec] = [target_block]
    occupied = set(target_block.cells)
    blocker_ids: list[str] = []
    fill_cycle = cycle(shuffled_support(rng, BLOCK_FILLS))
    for index, cell in enumerate(selected_path_cells):
        block_id = f"blocker_{index}"
        block = perpendicular_block_for_path_cell(
            path_cell=cell,
            exit_side=str(exit_side),
            rows=int(rows),
            cols=int(cols),
            rng=rng,
            block_id=block_id,
            role="path_blocker",
            fill_rgb=next(fill_cycle),
        )
        if any(block_cell in occupied for block_cell in block.cells):
            raise ValueError("path blocker overlaps an existing block")
        occupied.update(block.cells)
        blocks.append(block)
        blocker_ids.append(str(block_id))

    non_target_min = _get_int(params, GEN_DEFAULTS, "non_target_block_count_min", 6)
    non_target_max = _get_int(params, GEN_DEFAULTS, "non_target_block_count_max", 10)
    desired_non_target = max(int(rng.randint(int(non_target_min), int(non_target_max))), int(blocker_count))
    path_cell_set = set(path_cells)
    attempts = 0
    while len(blocks) - 1 < int(desired_non_target) and attempts < 900:
        attempts += 1
        idx = len(blocks) - 1
        block = candidate_distractor(
            block_id=f"distractor_{idx}",
            rows=int(rows),
            cols=int(cols),
            rng=rng,
            role="distractor",
            fill_rgb=next(fill_cycle),
        )
        if any(block_cell in occupied for block_cell in block.cells):
            continue
        if any(block_cell in path_cell_set for block_cell in block.cells):
            continue
        occupied.update(block.cells)
        blocks.append(block)

    labeled_blocks = assign_labels(blocks)
    movable_ids = movable_block_ids(labeled_blocks, rows=int(rows), cols=int(cols))
    return {
        "rows": int(rows),
        "cols": int(cols),
        "exit_side": str(exit_side),
        "blocks": serialize_blocks(labeled_blocks),
        "target_block_id": "target",
        "blocking_block_ids": [str(block_id) for block_id in blocker_ids],
        "target_path_cells": [[int(row), int(col)] for row, col in path_cells],
        "movable_block_ids": [str(block_id) for block_id in movable_ids],
        "movable_count": int(len(movable_ids)),
        "blocker_count": int(blocker_count),
        "non_target_block_count": int(len(labeled_blocks) - 1),
    }


def build_neutral_board(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    desired_non_target_count: int | None = None,
) -> dict[str, Any]:
    """Build a plain sliding-block board without a target block or exit path."""

    rng = spawn_rng(int(instance_seed), str(namespace))
    rows_min, rows_max = _get_range(params, GEN_DEFAULTS, min_key="board_rows_min", max_key="board_rows_max", fallback_min=6, fallback_max=7)
    cols_min, cols_max = _get_range(params, GEN_DEFAULTS, min_key="board_cols_min", max_key="board_cols_max", fallback_min=6, fallback_max=7)
    rows = int(rng.randint(int(rows_min), int(rows_max)))
    cols = int(rng.randint(int(cols_min), int(cols_max)))
    non_target_min = _get_int(params, GEN_DEFAULTS, "non_target_block_count_min", 6)
    non_target_max = _get_int(params, GEN_DEFAULTS, "non_target_block_count_max", 10)
    desired_count = (
        int(desired_non_target_count)
        if desired_non_target_count is not None
        else int(rng.randint(int(non_target_min), int(non_target_max)))
    )
    desired_count = max(1, min(int(desired_count), int(non_target_max)))

    blocks: list[BlockSpec] = []
    occupied: set[tuple[int, int]] = set()
    fill_cycle = cycle(shuffled_support(rng, BLOCK_FILLS))
    attempts = 0
    while len(blocks) < int(desired_count) and attempts < 1200:
        attempts += 1
        idx = len(blocks)
        block = candidate_distractor(
            block_id=f"block_{idx}",
            rows=int(rows),
            cols=int(cols),
            rng=rng,
            role="distractor",
            fill_rgb=next(fill_cycle),
        )
        if any(cell in occupied for cell in block.cells):
            continue
        occupied.update(block.cells)
        blocks.append(block)

    if len(blocks) < int(desired_count):
        raise ValueError("failed to place requested number of neutral sliding blocks")

    labeled_blocks = assign_labels(blocks)
    movable_ids = movable_block_ids(labeled_blocks, rows=int(rows), cols=int(cols))
    return {
        "rows": int(rows),
        "cols": int(cols),
        "exit_side": "",
        "blocks": serialize_blocks(labeled_blocks),
        "target_block_id": "",
        "blocking_block_ids": [],
        "target_path_cells": [],
        "movable_block_ids": [str(block_id) for block_id in movable_ids],
        "movable_count": int(len(movable_ids)),
        "blocker_count": 0,
        "non_target_block_count": int(len(labeled_blocks)),
    }


def build_board_for_movable_target(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    movable_target: int,
) -> dict[str, Any]:
    """Search for a neutral board whose current movable count matches target."""

    max_attempts = _get_int(params, GEN_DEFAULTS, "movable_block_count_generation_attempts", 512)
    observed_counts: list[int] = []
    for attempt in range(int(max_attempts)):
        candidate_seed = int(instance_seed) + (7919 * (int(attempt) + 1))
        candidate = build_neutral_board(
            params=params,
            instance_seed=int(candidate_seed),
            namespace="games.sliding_block.movable_board",
        )
        count = int(candidate["movable_count"])
        observed_counts.append(count)
        if count != int(movable_target):
            continue
        return dict(candidate)
    raise ValueError(
        "failed to sample sliding-block board with requested movable count "
        f"{int(movable_target)}; observed={sorted(set(observed_counts))}"
    )


def build_board_for_orientation_target(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    orientation: str,
    orientation_target: int,
) -> dict[str, Any]:
    """Search for a neutral board with the requested block-orientation count."""

    max_attempts = _get_int(params, GEN_DEFAULTS, "block_orientation_count_generation_attempts", 512)
    observed_counts: list[int] = []
    for attempt in range(int(max_attempts)):
        candidate_seed = int(instance_seed) + (6151 * (int(attempt) + 1))
        non_target_max = _get_int(params, GEN_DEFAULTS, "non_target_block_count_max", 10)
        candidate = build_neutral_board(
            params=params,
            instance_seed=int(candidate_seed),
            namespace="games.sliding_block.orientation_board",
            desired_non_target_count=min(int(non_target_max), max(int(orientation_target) + 2, 3)),
        )
        blocks = deserialize_blocks(candidate["blocks"])
        matching_ids = block_ids_by_orientation(blocks, orientation=str(orientation))
        observed_counts.append(len(matching_ids))
        if len(matching_ids) != int(orientation_target):
            continue
        out = dict(candidate)
        out["orientation"] = str(orientation)
        out["orientation_block_ids"] = [str(block_id) for block_id in matching_ids]
        out["orientation_count"] = int(len(matching_ids))
        return out
    raise ValueError(
        "failed to sample sliding-block board with requested orientation count "
        f"{int(orientation_target)} for {orientation}; observed={sorted(set(observed_counts))}"
    )


def sample_move_sequence(
    *,
    blocks: Sequence[BlockSpec],
    rows: int,
    cols: int,
    move_count: int,
    max_distance: int,
    rng,
) -> tuple[list[BlockMoveSpec], list[list[BlockSpec]]]:
    """Sample an ordered sequence and return the board state after each move."""

    for _attempt in range(160):
        current = list(blocks)
        sequence: list[BlockMoveSpec] = []
        states: list[list[BlockSpec]] = [list(current)]
        moved_ids: list[str] = []
        for _step in range(int(move_count)):
            legal = legal_moves(current, rows=int(rows), cols=int(cols), max_distance=int(max_distance))
            fresh = [move for move in legal if str(move.block_id) not in set(moved_ids)]
            choices = fresh or legal
            if not choices:
                break
            move = choices[int(rng.randrange(len(choices)))]
            current = apply_move(current, move=move)
            sequence.append(move)
            states.append(list(current))
            if str(move.block_id) not in moved_ids:
                moved_ids.append(str(move.block_id))
        if len(sequence) == int(move_count):
            return sequence, states
    raise ValueError("failed to sample a valid sliding-block move sequence")


def move_sequence_text(sequence: Sequence[BlockMoveSpec], *, blocks: Sequence[BlockSpec]) -> str:
    """Return compact human-readable slide instructions for the prompt slot."""

    labels = {str(block.block_id): str(block.label) for block in blocks}
    parts = []
    for index, move in enumerate(sequence, start=1):
        noun = "cell" if int(move.distance) == 1 else "cells"
        parts.append(f"{index}. block {labels[str(move.block_id)]} slides {int(move.distance)} {noun} {move.direction}")
    return "; ".join(parts)


def build_move_result_options(
    *,
    initial_blocks: Sequence[BlockSpec],
    states: Sequence[Sequence[BlockSpec]],
    rows: int,
    cols: int,
    option_labels: Sequence[str],
    correct_label: str,
    max_distance: int,
    rng,
) -> list[dict[str, Any]]:
    """Build one correct final-board option plus unique plausible distractors."""

    correct_index = [str(label) for label in option_labels].index(str(correct_label))
    correct_blocks = list(states[-1])
    correct_signature = state_signature(correct_blocks)
    candidates: list[list[BlockSpec]] = []
    seen = {correct_signature}
    for state in list(states[:-1]):
        signature = state_signature(state)
        if signature not in seen:
            candidates.append(list(state))
            seen.add(signature)

    bases = [list(state) for state in states]
    for _attempt in range(260):
        base = list(bases[int(rng.randrange(len(bases)))])
        legal = legal_moves(base, rows=int(rows), cols=int(cols), max_distance=int(max_distance))
        if not legal:
            continue
        candidate = apply_move(base, move=legal[int(rng.randrange(len(legal)))])
        signature = state_signature(candidate)
        if signature in seen:
            continue
        candidates.append(list(candidate))
        seen.add(signature)
        if len(candidates) >= len(option_labels) - 1:
            break
    if len(candidates) < len(option_labels) - 1:
        raise ValueError("failed to build enough unique sliding-block result options")

    options: list[dict[str, Any]] = []
    distractor_cursor = 0
    for index, label in enumerate(option_labels):
        is_correct = int(index) == int(correct_index)
        option_blocks = correct_blocks if is_correct else candidates[distractor_cursor]
        if not is_correct:
            distractor_cursor += 1
        options.append(
            {
                "option_id": f"option_{label}",
                "label": str(label),
                "is_correct": bool(is_correct),
                "blocks": serialize_blocks(option_blocks),
            }
        )
    return options


def build_board_for_move_result(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    option_labels: Sequence[str],
    correct_option_label: str,
) -> dict[str, Any]:
    """Construct a source board, short slide sequence, and final-board options."""

    rng = spawn_rng(int(instance_seed), "games.sliding_block.final_board")
    board = build_neutral_board(
        params=params,
        instance_seed=int(instance_seed),
        namespace="games.sliding_block.final_board.source",
    )
    blocks = deserialize_blocks(board["blocks"])
    move_min = _get_int(params, GEN_DEFAULTS, "move_result_move_count_min", 1)
    move_max = _get_int(params, GEN_DEFAULTS, "move_result_move_count_max", 2)
    if int(move_min) > int(move_max):
        raise ValueError("move_result_move_count_min must be <= move_result_move_count_max")
    move_count = int(uniform_choice(rng, tuple(range(int(move_min), int(move_max) + 1))))
    max_distance = _get_int(params, GEN_DEFAULTS, "move_result_slide_distance_max", 3)
    sequence, states = sample_move_sequence(
        blocks=blocks,
        rows=int(board["rows"]),
        cols=int(board["cols"]),
        move_count=int(move_count),
        max_distance=int(max_distance),
        rng=rng,
    )
    options = build_move_result_options(
        initial_blocks=blocks,
        states=states,
        rows=int(board["rows"]),
        cols=int(board["cols"]),
        option_labels=option_labels,
        correct_label=str(correct_option_label),
        max_distance=int(max_distance),
        rng=rng,
    )
    moved_ids: list[str] = []
    move_trace: list[dict[str, Any]] = []
    labels = {str(block.block_id): str(block.label) for block in blocks}
    for move in sequence:
        if str(move.block_id) not in moved_ids:
            moved_ids.append(str(move.block_id))
        move_trace.append(
            {
                "block_id": str(move.block_id),
                "label": str(labels[str(move.block_id)]),
                "direction": str(move.direction),
                "distance": int(move.distance),
            }
        )
    out = dict(board)
    out.update(
        {
            "option_boards": [dict(option) for option in options],
            "correct_option_id": f"option_{correct_option_label}",
            "move_sequence": [dict(item) for item in move_trace],
            "move_sequence_description": move_sequence_text(sequence, blocks=blocks),
            "moved_block_ids": [str(block_id) for block_id in moved_ids],
        }
    )
    return out


__all__ = [
    "build_board_for_orientation_target",
    "build_board_for_movable_target",
    "build_board_for_move_result",
    "build_exit_path_board",
    "build_neutral_board",
    "deserialize_blocks",
    "integer_support",
    "move_result_option_labels",
    "resolve_exit_side",
    "resolve_scene_variant",
    "select_target_from_support",
    "serialize_blocks",
]
