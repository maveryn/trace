"""Pure Bingo card construction and line-rule helpers."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .state import (
    BINGO_BOARD_SIZE,
    BINGO_COLUMN_LABELS,
    BINGO_COLUMN_RANGES,
    BingoCardState,
    BingoCellInstance,
    cell_id,
)


def build_bingo_number_grid(rng) -> Tuple[Tuple[int, ...], ...]:
    """Return one `5 x 5` visible bingo number grid with realistic column ranges."""

    columns: List[List[int]] = []
    for start, end in BINGO_COLUMN_RANGES:
        sampled = [int(value) for value in rng.sample(range(int(start), int(end) + 1), BINGO_BOARD_SIZE)]
        rng.shuffle(sampled)
        columns.append(sampled)
    return tuple(
        tuple(int(columns[column_index][row_index]) for column_index in range(BINGO_BOARD_SIZE))
        for row_index in range(BINGO_BOARD_SIZE)
    )


def _build_row_count_marks(
    *,
    rng,
    target_answer: int,
    distractor_mark_prob: float,
) -> Tuple[Tuple[bool, ...], ...]:
    completed_rows = set(int(value) for value in rng.sample(range(BINGO_BOARD_SIZE), int(target_answer)))
    marks: List[List[bool]] = [[False] * BINGO_BOARD_SIZE for _ in range(BINGO_BOARD_SIZE)]
    for row_index in range(BINGO_BOARD_SIZE):
        if row_index in completed_rows:
            marks[row_index] = [True] * BINGO_BOARD_SIZE
            continue
        gap_column = int(rng.randrange(BINGO_BOARD_SIZE))
        row_marks = [bool(rng.random() < float(distractor_mark_prob)) for _ in range(BINGO_BOARD_SIZE)]
        row_marks[gap_column] = False
        marks[row_index] = row_marks
    return tuple(tuple(bool(value) for value in row) for row in marks)


def _build_column_count_marks(
    *,
    rng,
    target_answer: int,
    distractor_mark_prob: float,
) -> Tuple[Tuple[bool, ...], ...]:
    completed_columns = set(int(value) for value in rng.sample(range(BINGO_BOARD_SIZE), int(target_answer)))
    marks: List[List[bool]] = [[False] * BINGO_BOARD_SIZE for _ in range(BINGO_BOARD_SIZE)]
    gap_rows = {
        column_index: int(rng.randrange(BINGO_BOARD_SIZE))
        for column_index in range(BINGO_BOARD_SIZE)
        if column_index not in completed_columns
    }
    for row_index in range(BINGO_BOARD_SIZE):
        for column_index in range(BINGO_BOARD_SIZE):
            if column_index in completed_columns:
                marks[row_index][column_index] = True
                continue
            if gap_rows[column_index] == row_index:
                marks[row_index][column_index] = False
                continue
            marks[row_index][column_index] = bool(rng.random() < float(distractor_mark_prob))
    return tuple(tuple(bool(value) for value in row) for row in marks)


def _build_single_completed_column_marks(
    *,
    rng,
    target_column_index: int,
    distractor_mark_prob: float,
) -> Tuple[Tuple[bool, ...], ...]:
    """Mark one full column while blocking every other row and column from completion."""

    target_column = int(target_column_index)
    mark_prob = max(0.0, min(1.0, float(distractor_mark_prob)))
    marks: List[List[bool]] = [[False] * BINGO_BOARD_SIZE for _ in range(BINGO_BOARD_SIZE)]
    other_columns = [int(column_index) for column_index in range(BINGO_BOARD_SIZE) if int(column_index) != target_column]
    if not other_columns:
        raise ValueError("completed-column label scenes require non-target columns")

    for row_index in range(BINGO_BOARD_SIZE):
        row_gap_column = int(rng.choice(other_columns))
        for column_index in range(BINGO_BOARD_SIZE):
            if int(column_index) == target_column:
                marks[row_index][column_index] = True
            elif int(column_index) == row_gap_column:
                marks[row_index][column_index] = False
            else:
                marks[row_index][column_index] = bool(rng.random() < mark_prob)

    for column_index in other_columns:
        if all(bool(marks[row_index][column_index]) for row_index in range(BINGO_BOARD_SIZE)):
            marks[int(rng.randrange(BINGO_BOARD_SIZE))][column_index] = False

    return tuple(tuple(bool(value) for value in row) for row in marks)


def _build_single_completed_line_marks(
    *,
    rng,
    line_axis: str,
    target_line_index: int,
    distractor_mark_prob: float,
) -> Tuple[Tuple[bool, ...], ...]:
    """Mark exactly one requested row or column while preventing accidental lines."""

    axis = str(line_axis)
    target_index = int(target_line_index)
    if target_index < 0 or target_index >= BINGO_BOARD_SIZE:
        raise ValueError("single completed-line target index is outside the card")
    if axis == "column":
        return _build_single_completed_column_marks(
            rng=rng,
            target_column_index=int(target_index),
            distractor_mark_prob=float(distractor_mark_prob),
        )
    if axis != "row":
        raise ValueError(f"unsupported bingo line axis: {line_axis}")

    mark_prob = max(0.0, min(1.0, float(distractor_mark_prob)))
    marks: List[List[bool]] = [[False] * BINGO_BOARD_SIZE for _ in range(BINGO_BOARD_SIZE)]
    other_rows = [int(row_index) for row_index in range(BINGO_BOARD_SIZE) if int(row_index) != target_index]
    if not other_rows:
        raise ValueError("completed-line scenes require non-target rows")

    marks[target_index] = [True] * BINGO_BOARD_SIZE
    for row_index in other_rows:
        gap_column = int(rng.randrange(BINGO_BOARD_SIZE))
        for column_index in range(BINGO_BOARD_SIZE):
            marks[row_index][column_index] = (
                False if int(column_index) == gap_column else bool(rng.random() < mark_prob)
            )

    for column_index in range(BINGO_BOARD_SIZE):
        if all(bool(marks[row_index][column_index]) for row_index in range(BINGO_BOARD_SIZE)):
            marks[int(rng.choice(other_rows))][column_index] = False

    return tuple(tuple(bool(value) for value in row) for row in marks)


def _build_near_complete_row_marks(
    *,
    rng,
    target_answer: int,
    distractor_mark_prob: float,
) -> Tuple[Tuple[bool, ...], ...]:
    target_rows = set(int(value) for value in rng.sample(range(BINGO_BOARD_SIZE), int(target_answer)))
    marks: List[List[bool]] = [[False] * BINGO_BOARD_SIZE for _ in range(BINGO_BOARD_SIZE)]
    for row_index in range(BINGO_BOARD_SIZE):
        if row_index in target_rows:
            gap_column = int(rng.randrange(BINGO_BOARD_SIZE))
            row_marks = [True] * BINGO_BOARD_SIZE
            row_marks[gap_column] = False
            marks[row_index] = row_marks
            continue
        false_count = int(rng.randint(2, BINGO_BOARD_SIZE))
        false_columns = set(int(value) for value in rng.sample(range(BINGO_BOARD_SIZE), false_count))
        row_marks = [
            False if column_index in false_columns else bool(rng.random() < float(distractor_mark_prob))
            for column_index in range(BINGO_BOARD_SIZE)
        ]
        if sum(1 for value in row_marks if not bool(value)) == 1:
            row_marks[int(rng.choice(tuple(false_columns)))] = False
        marks[row_index] = row_marks
    return tuple(tuple(bool(value) for value in row) for row in marks)


def _build_near_complete_column_marks(
    *,
    rng,
    target_answer: int,
    distractor_mark_prob: float,
) -> Tuple[Tuple[bool, ...], ...]:
    target_columns = set(int(value) for value in rng.sample(range(BINGO_BOARD_SIZE), int(target_answer)))
    marks: List[List[bool]] = [[False] * BINGO_BOARD_SIZE for _ in range(BINGO_BOARD_SIZE)]
    for column_index in range(BINGO_BOARD_SIZE):
        if column_index in target_columns:
            gap_row = int(rng.randrange(BINGO_BOARD_SIZE))
            for row_index in range(BINGO_BOARD_SIZE):
                marks[row_index][column_index] = row_index != gap_row
            continue
        false_count = int(rng.randint(2, BINGO_BOARD_SIZE))
        false_rows = set(int(value) for value in rng.sample(range(BINGO_BOARD_SIZE), false_count))
        for row_index in range(BINGO_BOARD_SIZE):
            if row_index in false_rows:
                marks[row_index][column_index] = False
            else:
                marks[row_index][column_index] = bool(rng.random() < float(distractor_mark_prob))
    return tuple(tuple(bool(value) for value in row) for row in marks)


def _build_called_number_match_state(
    *,
    rng,
    numbers_grid: Sequence[Sequence[int]],
    present_called_count: int,
    called_number_count: int,
) -> Tuple[Tuple[Tuple[bool, ...], ...], Tuple[int, ...], Tuple[str, ...], Tuple[int, ...]]:
    """Construct a CALLED list with an exact count of values present on the card."""

    called_count = int(called_number_count)
    answer = int(present_called_count)
    if called_count < 1 or called_count > BINGO_BOARD_SIZE * BINGO_BOARD_SIZE:
        raise ValueError("called-number count must fit on the visible bingo card")
    if answer < 0 or answer > called_count:
        raise ValueError("called-number match target must be between zero and called-number count")

    coordinates = [
        (int(row_index), int(column_index))
        for row_index in range(BINGO_BOARD_SIZE)
        for column_index in range(BINGO_BOARD_SIZE)
    ]
    present_coordinates = [tuple(value) for value in rng.sample(coordinates, answer)]
    present_cell_ids = tuple(
        cell_id(row_index=int(row_index), column_index=int(column_index))
        for row_index, column_index in present_coordinates
    )
    present_numbers = [
        int(numbers_grid[int(row_index)][int(column_index)])
        for row_index, column_index in present_coordinates
    ]
    visible_number_set = {
        int(numbers_grid[row_index][column_index])
        for row_index in range(BINGO_BOARD_SIZE)
        for column_index in range(BINGO_BOARD_SIZE)
    }
    absent_count = int(called_count - answer)
    absent_pool = [int(value) for value in range(1, 76) if int(value) not in visible_number_set]
    if absent_count > len(absent_pool):
        raise ValueError("not enough absent Bingo numbers for called-number matching")
    absent_numbers = [int(value) for value in rng.sample(absent_pool, absent_count)]
    called_numbers = [int(value) for value in present_numbers + absent_numbers]
    rng.shuffle(called_numbers)
    marks: List[List[bool]] = [[False] * BINGO_BOARD_SIZE for _ in range(BINGO_BOARD_SIZE)]
    return (
        tuple(tuple(bool(value) for value in row) for row in marks),
        tuple(int(value) for value in called_numbers),
        tuple(str(value) for value in present_cell_ids),
        tuple(int(value) for value in absent_numbers),
    )


def completed_rows(mark_grid: Sequence[Sequence[bool]]) -> Tuple[int, ...]:
    """Return row indices whose five cells are marked."""

    return tuple(
        int(row_index)
        for row_index, row in enumerate(mark_grid)
        if all(bool(value) for value in row)
    )


def completed_columns(mark_grid: Sequence[Sequence[bool]]) -> Tuple[int, ...]:
    """Return column indices whose five cells are marked."""

    completed: List[int] = []
    for column_index in range(BINGO_BOARD_SIZE):
        if all(bool(mark_grid[row_index][column_index]) for row_index in range(BINGO_BOARD_SIZE)):
            completed.append(int(column_index))
    return tuple(int(value) for value in completed)


def near_complete_rows(mark_grid: Sequence[Sequence[bool]]) -> Tuple[int, ...]:
    """Return row indices with exactly one unmarked cell."""

    return tuple(
        int(row_index)
        for row_index, row in enumerate(mark_grid)
        if sum(1 for value in row if not bool(value)) == 1
    )


def near_complete_columns(mark_grid: Sequence[Sequence[bool]]) -> Tuple[int, ...]:
    """Return column indices with exactly one unmarked cell."""

    near_complete: List[int] = []
    for column_index in range(BINGO_BOARD_SIZE):
        unmarked_count = sum(
            1
            for row_index in range(BINGO_BOARD_SIZE)
            if not bool(mark_grid[row_index][column_index])
        )
        if int(unmarked_count) == 1:
            near_complete.append(int(column_index))
    return tuple(int(value) for value in near_complete)


def _near_complete_gap_cell_ids(
    *,
    mark_grid: Sequence[Sequence[bool]],
    line_axis: str,
    line_indices: Sequence[int],
) -> Tuple[str, ...]:
    gap_ids: List[str] = []
    axis = str(line_axis)
    for line_index in line_indices:
        if axis == "row":
            gap_columns = [
                int(column_index)
                for column_index in range(BINGO_BOARD_SIZE)
                if not bool(mark_grid[int(line_index)][column_index])
            ]
            if len(gap_columns) != 1:
                raise ValueError("near-complete row annotation requires exactly one gap")
            gap_ids.append(cell_id(row_index=int(line_index), column_index=int(gap_columns[0])))
        elif axis == "column":
            gap_rows = [
                int(row_index)
                for row_index in range(BINGO_BOARD_SIZE)
                if not bool(mark_grid[row_index][int(line_index)])
            ]
            if len(gap_rows) != 1:
                raise ValueError("near-complete column annotation requires exactly one gap")
            gap_ids.append(cell_id(row_index=int(gap_rows[0]), column_index=int(line_index)))
        else:
            raise ValueError(f"unsupported bingo line axis: {line_axis}")
    return tuple(str(value) for value in gap_ids)


def _has_unmarked_cell(mark_grid: Sequence[Sequence[bool]]) -> bool:
    return any(not bool(value) for row in mark_grid for value in row)


def _completed_line_sums_for_axis(
    *,
    numbers_grid: Sequence[Sequence[int]],
    line_axis: str,
    completed_line_indices: Sequence[int],
) -> Tuple[Tuple[str, int, int], ...]:
    axis = str(line_axis)
    sums: List[Tuple[str, int, int]] = []
    for line_index in completed_line_indices:
        if axis == "row":
            value = sum(int(numbers_grid[int(line_index)][column_index]) for column_index in range(BINGO_BOARD_SIZE))
        elif axis == "column":
            value = sum(int(numbers_grid[row_index][int(line_index)]) for row_index in range(BINGO_BOARD_SIZE))
        else:
            raise ValueError(f"unsupported bingo line axis: {line_axis}")
        sums.append((axis, int(line_index), int(value)))
    return tuple(sums)


def _state_from_marks(
    *,
    numbers_grid: Sequence[Sequence[int]],
    mark_grid: Sequence[Sequence[bool]],
    near_complete_gap_cell_ids: Sequence[str] = (),
    called_numbers: Sequence[int] = (),
    called_number_cell_ids: Sequence[str] = (),
    line_sum_target_axis: str | None = None,
    line_sum_target_line_index: int | None = None,
    line_sum_target_cell_ids: Sequence[str] = (),
    line_sum_target_value: int | None = None,
    completed_line_sums: Sequence[Tuple[str, int, int]] = (),
) -> BingoCardState:
    """Build passive Bingo state and validate that the visible card is not complete."""

    if not _has_unmarked_cell(mark_grid):
        raise ValueError("bingo card scenes require at least one unmarked cell")

    completed_row_indices = completed_rows(mark_grid)
    completed_column_indices = completed_columns(mark_grid)
    near_complete_row_indices = near_complete_rows(mark_grid)
    near_complete_column_indices = near_complete_columns(mark_grid)

    cells: List[BingoCellInstance] = []
    for row_index in range(BINGO_BOARD_SIZE):
        for column_index in range(BINGO_BOARD_SIZE):
            cells.append(
                BingoCellInstance(
                    cell_id=cell_id(row_index=int(row_index), column_index=int(column_index)),
                    row_index=int(row_index),
                    column_index=int(column_index),
                    column_label=str(BINGO_COLUMN_LABELS[column_index]),
                    number=int(numbers_grid[row_index][column_index]),
                    is_marked=bool(mark_grid[row_index][column_index]),
                )
            )

    return BingoCardState(
        cells=tuple(cells),
        numbers_grid=tuple(tuple(int(value) for value in row) for row in numbers_grid),
        mark_grid=tuple(tuple(bool(value) for value in row) for row in mark_grid),
        completed_row_indices=tuple(int(value) for value in completed_row_indices),
        completed_column_indices=tuple(int(value) for value in completed_column_indices),
        near_complete_row_indices=tuple(int(value) for value in near_complete_row_indices),
        near_complete_column_indices=tuple(int(value) for value in near_complete_column_indices),
        near_complete_gap_cell_ids=tuple(str(value) for value in near_complete_gap_cell_ids),
        called_numbers=tuple(int(value) for value in called_numbers),
        called_number_cell_ids=tuple(str(value) for value in called_number_cell_ids),
        line_sum_target_axis=None if line_sum_target_axis is None else str(line_sum_target_axis),
        line_sum_target_line_index=None if line_sum_target_line_index is None else int(line_sum_target_line_index),
        line_sum_target_cell_ids=tuple(str(value) for value in line_sum_target_cell_ids),
        line_sum_target_value=None if line_sum_target_value is None else int(line_sum_target_value),
        completed_line_sums=tuple(
            (str(axis_name), int(line_index), int(value))
            for axis_name, line_index, value in completed_line_sums
        ),
    )


def build_completed_line_card_state(
    *,
    rng,
    line_axis: str,
    completed_line_count: int,
    distractor_mark_prob: float = 0.45,
) -> BingoCardState:
    """Construct a card with exactly the requested completed rows or columns."""

    axis = str(line_axis)
    mark_prob = max(0.0, min(1.0, float(distractor_mark_prob)))
    if axis == "row":
        mark_grid = _build_row_count_marks(
            rng=rng,
            target_answer=int(completed_line_count),
            distractor_mark_prob=float(mark_prob),
        )
    elif axis == "column":
        mark_grid = _build_column_count_marks(
            rng=rng,
            target_answer=int(completed_line_count),
            distractor_mark_prob=float(mark_prob),
        )
    else:
        raise ValueError(f"unsupported bingo line axis: {line_axis}")

    numbers_grid = build_bingo_number_grid(rng)
    state = _state_from_marks(numbers_grid=numbers_grid, mark_grid=mark_grid)
    actual = len(state.completed_row_indices) if axis == "row" else len(state.completed_column_indices)
    if int(actual) != int(completed_line_count):
        raise ValueError("constructed bingo completed-line scene drifted from the target answer")
    return state


def build_completed_column_label_card_state(
    *,
    rng,
    target_column_label: str,
    distractor_mark_prob: float = 0.45,
) -> BingoCardState:
    """Construct a card with exactly one completed column named by B/I/N/G/O."""

    label = str(target_column_label).strip().upper()
    if label not in BINGO_COLUMN_LABELS:
        raise ValueError(f"unsupported Bingo column label: {target_column_label}")
    target_column_index = int(BINGO_COLUMN_LABELS.index(label))
    mark_grid = _build_single_completed_column_marks(
        rng=rng,
        target_column_index=int(target_column_index),
        distractor_mark_prob=float(distractor_mark_prob),
    )
    state = _state_from_marks(numbers_grid=build_bingo_number_grid(rng), mark_grid=mark_grid)
    if tuple(state.completed_column_indices) != (int(target_column_index),):
        raise ValueError("completed-column label scene must have exactly one completed column")
    if state.completed_row_indices:
        raise ValueError("completed-column label scene must not accidentally complete a row")
    return state


def build_completed_line_sum_card_state(
    *,
    rng,
    line_axis: str,
    line_index: int,
    distractor_mark_prob: float = 0.20,
) -> BingoCardState:
    """Construct a card with one completed row or column to sum."""

    axis = str(line_axis)
    target_index = int(line_index)
    mark_grid = _build_single_completed_line_marks(
        rng=rng,
        line_axis=axis,
        target_line_index=int(target_index),
        distractor_mark_prob=float(distractor_mark_prob),
    )
    numbers_grid = build_bingo_number_grid(rng)
    completed_line_sums = _completed_line_sums_for_axis(
        numbers_grid=numbers_grid,
        line_axis=axis,
        completed_line_indices=(int(target_index),),
    )
    _selected_axis, selected_index, selected_value = completed_line_sums[0]
    if axis == "row":
        target_cell_ids = tuple(
            cell_id(row_index=int(selected_index), column_index=int(column_index))
            for column_index in range(BINGO_BOARD_SIZE)
        )
    else:
        target_cell_ids = tuple(
            cell_id(row_index=int(row_index), column_index=int(selected_index))
            for row_index in range(BINGO_BOARD_SIZE)
        )

    state = _state_from_marks(
        numbers_grid=numbers_grid,
        mark_grid=mark_grid,
        line_sum_target_axis=axis,
        line_sum_target_line_index=int(selected_index),
        line_sum_target_cell_ids=target_cell_ids,
        line_sum_target_value=int(selected_value),
        completed_line_sums=completed_line_sums,
    )
    if axis == "row":
        if tuple(state.completed_row_indices) != (int(target_index),) or state.completed_column_indices:
            raise ValueError("completed-line sum scene must have exactly one completed row")
    elif tuple(state.completed_column_indices) != (int(target_index),) or state.completed_row_indices:
        raise ValueError("completed-line sum scene must have exactly one completed column")
    return state


def build_near_complete_line_card_state(
    *,
    rng,
    line_axis: str,
    near_complete_line_count: int,
    distractor_mark_prob: float = 0.45,
) -> BingoCardState:
    """Construct a card with exactly the requested near-complete rows or columns."""

    axis = str(line_axis)
    mark_prob = max(0.0, min(1.0, float(distractor_mark_prob)))
    if axis == "row":
        mark_grid = _build_near_complete_row_marks(
            rng=rng,
            target_answer=int(near_complete_line_count),
            distractor_mark_prob=float(mark_prob),
        )
    elif axis == "column":
        mark_grid = _build_near_complete_column_marks(
            rng=rng,
            target_answer=int(near_complete_line_count),
            distractor_mark_prob=float(mark_prob),
        )
    else:
        raise ValueError(f"unsupported bingo line axis: {line_axis}")

    near_rows = near_complete_rows(mark_grid)
    near_columns = near_complete_columns(mark_grid)
    actual = len(near_rows) if axis == "row" else len(near_columns)
    if int(actual) != int(near_complete_line_count):
        raise ValueError("constructed bingo near-complete line scene drifted from the target answer")
    gap_ids = _near_complete_gap_cell_ids(
        mark_grid=mark_grid,
        line_axis=axis,
        line_indices=near_rows if axis == "row" else near_columns,
    )
    return _state_from_marks(
        numbers_grid=build_bingo_number_grid(rng),
        mark_grid=mark_grid,
        near_complete_gap_cell_ids=gap_ids,
    )


def build_called_number_match_card_state(
    *,
    rng,
    present_called_count: int,
    called_number_count: int,
) -> BingoCardState:
    """Construct a card with an exact number of called values present."""

    numbers_grid = build_bingo_number_grid(rng)
    mark_grid, called_numbers, called_cell_ids, _absent_numbers = _build_called_number_match_state(
        rng=rng,
        numbers_grid=numbers_grid,
        present_called_count=int(present_called_count),
        called_number_count=int(called_number_count),
    )
    if len(called_cell_ids) != int(present_called_count):
        raise ValueError("constructed bingo called-number scene drifted from the target answer")
    if len(called_numbers) != int(called_number_count):
        raise ValueError("constructed bingo called-number scene drifted from the called-number count")
    return _state_from_marks(
        numbers_grid=numbers_grid,
        mark_grid=mark_grid,
        called_numbers=called_numbers,
        called_number_cell_ids=called_cell_ids,
    )


__all__ = [
    "build_bingo_number_grid",
    "build_called_number_match_card_state",
    "build_completed_column_label_card_state",
    "build_completed_line_card_state",
    "build_completed_line_sum_card_state",
    "build_near_complete_line_card_state",
    "completed_columns",
    "completed_rows",
    "near_complete_columns",
    "near_complete_rows",
]
