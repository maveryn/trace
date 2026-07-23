"""Pure match-3 board mechanics shared by the scene tasks."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .state import Board, Coord, MoveOutcome, SwapMove


def would_make_run(board_rows: Sequence[Sequence[str]], row: int, col: int, color: str) -> bool:
    """Return whether placing one color would create an initial run."""

    if int(col) >= 2 and str(board_rows[row][col - 1]) == str(color) and str(board_rows[row][col - 2]) == str(color):
        return True
    if int(row) >= 2 and str(board_rows[row - 1][col]) == str(color) and str(board_rows[row - 2][col]) == str(color):
        return True
    return False


def generate_board(rng, *, rows: int, cols: int, gem_keys: Sequence[str]) -> Board:
    """Create a board without any pre-existing horizontal or vertical runs."""

    board: List[List[str]] = []
    keys = [str(key) for key in gem_keys]
    for row in range(int(rows)):
        values: List[str] = []
        board.append(values)
        for col in range(int(cols)):
            choices = list(keys)
            rng.shuffle(choices)
            selected = None
            for choice in choices:
                if not would_make_run(board, int(row), int(col), str(choice)):
                    selected = str(choice)
                    break
            values.append(str(selected if selected is not None else choices[0]))
    return tuple(tuple(str(value) for value in row) for row in board)


def find_runs(board: Board) -> Tuple[Tuple[Coord, ...], ...]:
    """Find every horizontal and vertical run of at least three equal gems."""

    rows = len(board)
    cols = len(board[0]) if rows else 0
    runs: List[Tuple[Coord, ...]] = []
    for row in range(rows):
        col = 0
        while col < cols:
            end = col + 1
            while end < cols and str(board[row][end]) == str(board[row][col]):
                end += 1
            if int(end - col) >= 3:
                runs.append(tuple((int(row), int(c)) for c in range(col, end)))
            col = end
    for col in range(cols):
        row = 0
        while row < rows:
            end = row + 1
            while end < rows and str(board[end][col]) == str(board[row][col]):
                end += 1
            if int(end - row) >= 3:
                runs.append(tuple((int(r), int(col)) for r in range(row, end)))
            row = end
    return tuple(runs)


def swap_board(board: Board, move: SwapMove) -> Board:
    """Return a new board after swapping the two cells in one move."""

    rows = [list(row) for row in board]
    ar, ac = move.a
    br, bc = move.b
    rows[ar][ac], rows[br][bc] = rows[br][bc], rows[ar][ac]
    return tuple(tuple(str(value) for value in row) for row in rows)


def simulate_move(board: Board, move: SwapMove) -> MoveOutcome:
    """Evaluate one adjacent swap using immediate-clear-only match-3 rules."""

    after = swap_board(board, move)
    runs = find_runs(after)
    cleared = sorted({coord for run in runs for coord in run})
    return MoveOutcome(
        move=move,
        clear_count=int(len(cleared)),
        run_count=int(len(runs)),
        cleared_cells=tuple((int(row), int(col)) for row, col in cleared),
        runs=tuple(tuple((int(row), int(col)) for row, col in run) for run in runs),
    )


def external_same_color_neighbors_for_clear(board: Board, outcome: MoveOutcome) -> Tuple[Coord, ...]:
    """Return same-color neighbors touching cleared cells but not clearing."""

    cleared = set(outcome.cleared_cells)
    if not cleared:
        return tuple()
    after = swap_board(board, outcome.move)
    rows = len(after)
    cols = len(after[0]) if rows else 0
    external: set[Coord] = set()
    for row, col in cleared:
        color = str(after[int(row)][int(col)])
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (int(row + dr), int(col + dc))
            nr, nc = neighbor
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if neighbor in cleared:
                continue
            if str(after[nr][nc]) == color:
                external.add(neighbor)
    return tuple(sorted(external))


def all_move_outcomes(board: Board) -> Tuple[MoveOutcome, ...]:
    """Evaluate all distinct adjacent swaps that exchange unlike neighboring gems."""

    rows = len(board)
    cols = len(board[0]) if rows else 0
    outcomes: List[MoveOutcome] = []
    for row in range(rows):
        for col in range(cols):
            for dr, dc in ((0, 1), (1, 0)):
                other = (int(row + dr), int(col + dc))
                if other[0] >= rows or other[1] >= cols:
                    continue
                if str(board[row][col]) == str(board[other[0]][other[1]]):
                    continue
                outcomes.append(simulate_move(board, SwapMove(a=(int(row), int(col)), b=other)))
    return tuple(outcomes)


def cell_entity_id(coord: Coord) -> str:
    """Return the render entity id for one board cell/gem."""

    return f"gem_r{int(coord[0]) + 1}_c{int(coord[1]) + 1}"


def gem_count_matches(
    board: Board,
    *,
    scope: str,
    color_name: str,
    row_index: int | None,
    col_index: int | None,
) -> Tuple[Coord, ...]:
    """Return gem coordinates matching color within the selected spatial scope."""

    matches: List[Coord] = []
    for row, values in enumerate(board):
        for col, value in enumerate(values):
            if str(value) != str(color_name):
                continue
            if str(scope) == "row" and int(row) != int(row_index if row_index is not None else -1):
                continue
            if str(scope) == "column" and int(col) != int(col_index if col_index is not None else -1):
                continue
            matches.append((int(row), int(col)))
    return tuple(matches)


def histogram(values: Sequence[int]) -> dict[str, int]:
    """Return a JSON-stable histogram over integer values."""

    out: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        out[key] = int(out.get(key, 0) + 1)
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


__all__ = [
    "all_move_outcomes",
    "cell_entity_id",
    "external_same_color_neighbors_for_clear",
    "find_runs",
    "gem_count_matches",
    "generate_board",
    "histogram",
    "simulate_move",
    "swap_board",
    "would_make_run",
]
