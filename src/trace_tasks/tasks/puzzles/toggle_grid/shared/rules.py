"""Toggle-grid state transition rules."""

from __future__ import annotations

from .state import Cell, GridState


def state_signature(state: GridState) -> GridState:
    """Normalize a nested grid into immutable integer rows."""

    return tuple(tuple(int(value) for value in row) for row in state)


def all_cells(rows: int, cols: int) -> tuple[Cell, ...]:
    """Return every row/column cell in row-major order."""

    return tuple((row, col) for row in range(int(rows)) for col in range(int(cols)))


def toggle_once(state: GridState, cell: Cell) -> GridState:
    """Press one switch, flipping that cell and its orthogonal neighbors."""

    rows = len(state)
    cols = len(state[0])
    mutable = [list(row) for row in state]
    row, col = int(cell[0]), int(cell[1])
    for rr, cc in (
        (row, col),
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    ):
        if 0 <= rr < rows and 0 <= cc < cols:
            mutable[rr][cc] = 1 - int(mutable[rr][cc])
    return state_signature(tuple(tuple(row_values) for row_values in mutable))


def apply_toggles(state: GridState, cells: tuple[Cell, ...]) -> GridState:
    """Apply a sequence of visible switch presses to one start grid."""

    current = state_signature(state)
    for cell in cells:
        current = toggle_once(current, tuple(cell))
    return current
