"""Bubble-shooter board mechanics and validation helpers."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .state import Board, BubbleShooterState, BubbleShotOutcome, Coord


def board_from_mapping(*, rows: int, cols: int, values: Mapping[Coord, str]) -> Board:
    """Return an immutable board from sparse coordinate values."""

    return tuple(
        tuple(values.get((row, col)) for col in range(int(cols)))
        for row in range(int(rows))
    )


def board_value(board: Sequence[Sequence[str | None]], coord: Coord) -> str | None:
    """Return the board value at one coordinate."""

    return board[int(coord[0])][int(coord[1])]


def all_bubble_coords(rows: int, cols: int) -> tuple[Coord, ...]:
    """Return all row-major bubble-grid coordinates."""

    return tuple((row, col) for row in range(int(rows)) for col in range(int(cols)))


def occupied_coords(board: Sequence[Sequence[str | None]]) -> tuple[Coord, ...]:
    """Return every occupied board coordinate."""

    return tuple(
        (row, col)
        for row, row_values in enumerate(board)
        for col, value in enumerate(row_values)
        if value is not None
    )


def empty_coords(board: Sequence[Sequence[str | None]]) -> tuple[Coord, ...]:
    """Return every empty board coordinate."""

    return tuple(
        (row, col)
        for row, row_values in enumerate(board)
        for col, value in enumerate(row_values)
        if value is None
    )


def outside_reachable_empty_coords(
    board: Sequence[Sequence[str | None]],
) -> tuple[Coord, ...]:
    """Return empty slots connected to the open shooter-side exterior."""

    rows = len(board)
    cols = len(board[0]) if rows else 0
    if rows <= 0 or cols <= 0:
        return tuple()
    empty = set(empty_coords(board))
    starts = [
        coord
        for coord in empty
        if int(coord[0]) == rows - 1 or int(coord[1]) in (0, cols - 1)
    ]
    stack = list(starts)
    seen: set[Coord] = set()
    while stack:
        coord = stack.pop()
        if coord in seen or coord not in empty:
            continue
        seen.add(coord)
        for neighbor in bubble_neighbors(coord, rows=rows, cols=cols):
            if neighbor not in seen and neighbor in empty:
                stack.append(neighbor)
    return sorted_coords(seen)


def is_playable_landing_coord(
    board: Sequence[Sequence[str | None]], landing_coord: Coord
) -> bool:
    """Return whether a shot can plausibly land at an exposed empty slot."""

    rows = len(board)
    cols = len(board[0]) if rows else 0
    row, col = int(landing_coord[0]), int(landing_coord[1])
    if not (0 <= row < rows and 0 <= col < cols):
        return False
    if board_value(board, (row, col)) is not None:
        return False
    reachable_empty = set(outside_reachable_empty_coords(board))
    if (row, col) not in reachable_empty:
        return False
    return any(
        board_value(board, neighbor) is not None
        for neighbor in bubble_neighbors((row, col), rows=rows, cols=cols)
    )


def playable_landing_coords(board: Sequence[Sequence[str | None]]) -> tuple[Coord, ...]:
    """Return exposed empty slots where a shot can attach to the bubble cluster."""

    return tuple(
        coord
        for coord in outside_reachable_empty_coords(board)
        if is_playable_landing_coord(board, coord)
    )


def sorted_coords(coords: Iterable[Coord]) -> tuple[Coord, ...]:
    """Return canonical sorted coordinates."""

    return tuple(sorted((int(row), int(col)) for row, col in coords))


def bubble_neighbors(coord: Coord, *, rows: int, cols: int) -> tuple[Coord, ...]:
    """Return in-bounds neighbors in an odd-row offset bubble grid."""

    row, col = int(coord[0]), int(coord[1])
    if row % 2 == 0:
        candidates = (
            (row, col - 1),
            (row, col + 1),
            (row - 1, col - 1),
            (row - 1, col),
            (row + 1, col - 1),
            (row + 1, col),
        )
    else:
        candidates = (
            (row, col - 1),
            (row, col + 1),
            (row - 1, col),
            (row - 1, col + 1),
            (row + 1, col),
            (row + 1, col + 1),
        )
    return tuple(
        (int(next_row), int(next_col))
        for next_row, next_col in candidates
        if 0 <= int(next_row) < int(rows) and 0 <= int(next_col) < int(cols)
    )


def connected_component(
    board: Sequence[Sequence[str | None]],
    *,
    start: Coord,
    allowed_colors: Iterable[str] | None = None,
) -> tuple[Coord, ...]:
    """Return one connected occupied component."""

    rows = len(board)
    cols = len(board[0]) if rows else 0
    start_value = board_value(board, start)
    if start_value is None:
        return tuple()
    allowed = (
        set(str(value) for value in allowed_colors)
        if allowed_colors is not None
        else {str(start_value)}
    )
    stack = [tuple(start)]
    seen: set[Coord] = set()
    while stack:
        coord = stack.pop()
        if coord in seen:
            continue
        if board_value(board, coord) not in allowed:
            continue
        seen.add(coord)
        for neighbor in bubble_neighbors(coord, rows=rows, cols=cols):
            if neighbor not in seen and board_value(board, neighbor) in allowed:
                stack.append(neighbor)
    return sorted_coords(seen)


def same_color_component_from_landing(
    board: Sequence[Sequence[str | None]],
    *,
    landing_coord: Coord,
    color_key: str,
) -> tuple[Coord, ...]:
    """Return existing same-color bubbles connected to a placed bubble."""

    rows = len(board)
    cols = len(board[0]) if rows else 0
    color = str(color_key)
    seen: set[Coord] = set()
    stack = [
        neighbor
        for neighbor in bubble_neighbors(landing_coord, rows=rows, cols=cols)
        if board_value(board, neighbor) == color
    ]
    while stack:
        coord = stack.pop()
        if coord in seen:
            continue
        if board_value(board, coord) != color:
            continue
        seen.add(coord)
        for neighbor in bubble_neighbors(coord, rows=rows, cols=cols):
            if neighbor not in seen and board_value(board, neighbor) == color:
                stack.append(neighbor)
    return sorted_coords(seen)


def top_connected_occupied(
    board: Sequence[Sequence[str | None]],
    *,
    removed: Iterable[Coord] = (),
) -> tuple[Coord, ...]:
    """Return occupied cells connected to the top row after removals."""

    rows = len(board)
    cols = len(board[0]) if rows else 0
    removed_set = {tuple(coord) for coord in removed}
    starts = [
        (0, col)
        for col in range(cols)
        if (0, col) not in removed_set and board_value(board, (0, col)) is not None
    ]
    stack = list(starts)
    seen: set[Coord] = set()
    while stack:
        coord = stack.pop()
        if coord in seen or coord in removed_set:
            continue
        if board_value(board, coord) is None:
            continue
        seen.add(coord)
        for neighbor in bubble_neighbors(coord, rows=rows, cols=cols):
            if (
                neighbor not in seen
                and neighbor not in removed_set
                and board_value(board, neighbor) is not None
            ):
                stack.append(neighbor)
    return sorted_coords(seen)


def compute_shot_outcome(
    board: Sequence[Sequence[str | None]],
    *,
    landing_coord: Coord,
    color_key: str,
) -> BubbleShotOutcome:
    """Compute pop and drop effects for one placed bubble."""

    if board_value(board, landing_coord) is not None:
        raise ValueError("bubble shooter landing coordinate must be empty")
    same_color = same_color_component_from_landing(
        board,
        landing_coord=landing_coord,
        color_key=str(color_key),
    )
    popped = same_color if len(same_color) + 1 >= 3 else tuple()
    occupied_after_pop = set(occupied_coords(board)) - set(popped)
    anchored = set(top_connected_occupied(board, removed=popped))
    dropped = sorted_coords(
        coord for coord in occupied_after_pop if coord not in anchored
    )
    return BubbleShotOutcome(
        landing_coord=tuple(landing_coord),
        color_key=str(color_key),
        connected_same_color_coords=tuple(same_color),
        popped_coords=tuple(popped),
        dropped_coords=tuple(dropped),
    )


def validate_bubble_shooter_state(state: BubbleShooterState) -> None:
    """Validate one generated Bubble-shooter board state."""

    if int(state.row_count) <= 0 or int(state.col_count) <= 0:
        raise ValueError("bubble shooter board dimensions must be positive")
    if len(state.board) != int(state.row_count) or any(
        len(row) != int(state.col_count) for row in state.board
    ):
        raise ValueError(
            "bubble shooter board dimensions do not match row_count/col_count"
        )
    if board_value(state.board, state.landing_coord) is not None:
        raise ValueError("bubble shooter landing slot must be empty")
    if not is_playable_landing_coord(state.board, state.landing_coord):
        raise ValueError(
            "bubble shooter landing slot must be exposed to the shooter side"
        )

    outcome = compute_shot_outcome(
        state.board,
        landing_coord=state.landing_coord,
        color_key=str(state.outcome.color_key),
    )
    if outcome != state.outcome:
        raise ValueError(
            "bubble shooter stored outcome does not match board computation"
        )

    if state.option_specs:
        answer_options = [option for option in state.option_specs if option.is_answer]
        if len(answer_options) != 1:
            raise ValueError(
                "bubble shooter option scenes require exactly one answer option"
            )
        pop_positive = [
            option
            for option in state.option_specs
            if len(
                compute_shot_outcome(
                    state.board,
                    landing_coord=state.landing_coord,
                    color_key=str(option.color_key),
                ).popped_coords
            )
            > 0
        ]
        if len(pop_positive) != 1 or str(pop_positive[0].label) != str(
            answer_options[0].label
        ):
            raise ValueError(
                "bubble shooter option scenes must have exactly one displayed popping color"
            )

    if state.landing_option_specs:
        answer_options = [
            option for option in state.landing_option_specs if option.is_answer
        ]
        if len(answer_options) != 1:
            raise ValueError(
                "bubble shooter landing-option scenes require exactly one answer option"
            )
        if tuple(answer_options[0].landing_coord) != tuple(state.landing_coord):
            raise ValueError(
                "bubble shooter answer landing option must match state landing coordinate"
            )
        seen_labels: set[str] = set()
        seen_coords: set[Coord] = set()
        pop_positive_labels: list[str] = []
        for option in state.landing_option_specs:
            label = str(option.label)
            coord = tuple(option.landing_coord)
            if label in seen_labels:
                raise ValueError("bubble shooter landing-option labels must be unique")
            if coord in seen_coords:
                raise ValueError(
                    "bubble shooter landing-option coordinates must be unique"
                )
            if not is_playable_landing_coord(state.board, coord):
                raise ValueError("bubble shooter landing option must be playable")
            seen_labels.add(label)
            seen_coords.add(coord)
            option_outcome = compute_shot_outcome(
                state.board,
                landing_coord=coord,
                color_key=str(state.outcome.color_key),
            )
            if option_outcome.popped_coords:
                pop_positive_labels.append(label)
        if pop_positive_labels != [str(answer_options[0].label)]:
            raise ValueError(
                "bubble shooter landing-option scenes must have exactly one displayed popping target"
            )


__all__ = [
    "all_bubble_coords",
    "board_from_mapping",
    "board_value",
    "bubble_neighbors",
    "compute_shot_outcome",
    "connected_component",
    "empty_coords",
    "is_playable_landing_coord",
    "occupied_coords",
    "outside_reachable_empty_coords",
    "playable_landing_coords",
    "same_color_component_from_landing",
    "sorted_coords",
    "top_connected_occupied",
    "validate_bubble_shooter_state",
]
