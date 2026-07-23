"""Rule predicates for legal Tents candidate cells."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .state import Cell


def neighbors4(cell: Cell, rows: int, cols: int) -> List[Cell]:
    """Return orthogonally adjacent in-bounds cells."""

    row, col = int(cell[0]), int(cell[1])
    candidates = (
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    )
    return [
        (int(r), int(c))
        for r, c in candidates
        if 0 <= int(r) < int(rows) and 0 <= int(c) < int(cols)
    ]


def neighbors8(cell: Cell, rows: int, cols: int) -> List[Cell]:
    """Return all side- or corner-touching in-bounds cells."""

    row, col = int(cell[0]), int(cell[1])
    out: List[Cell] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = int(row + dr)
            cc = int(col + dc)
            if 0 <= rr < int(rows) and 0 <= cc < int(cols):
                out.append((rr, cc))
    return out


def touches_any_tent(cell: Cell, tents: Iterable[Cell], rows: int, cols: int) -> bool:
    """Return whether a candidate cell touches any visible tent, including diagonals."""

    near = set(neighbors8(cell, int(rows), int(cols)))
    return any(tuple(tent) in near for tent in tents)


def count_by_axis(cells: Iterable[Cell], size: int, axis: int) -> List[int]:
    """Count cells by row when axis=0 or by column when axis=1."""

    counts = [0 for _ in range(int(size))]
    for row, col in cells:
        index = int(row if int(axis) == 0 else col)
        counts[int(index)] += 1
    return counts


def legal_candidate_cells(
    *,
    marked_tree: Cell,
    candidate_cells: Sequence[Cell],
    visible_tents: Sequence[Cell],
    tree_cells: Sequence[Cell],
    row_clues: Sequence[int],
    col_clues: Sequence[int],
    rows: int,
    cols: int,
) -> List[Cell]:
    """Filter labeled cells that could legally hold the marked tree's tent."""

    visible_row_counts = count_by_axis(visible_tents, int(rows), 0)
    visible_col_counts = count_by_axis(visible_tents, int(cols), 1)
    occupied = {tuple(cell) for cell in visible_tents} | {
        tuple(cell) for cell in tree_cells
    }
    marked_neighbors = set(neighbors4(tuple(marked_tree), int(rows), int(cols)))
    legal: List[Cell] = []
    for cell in [tuple(candidate) for candidate in candidate_cells]:
        row, col = int(cell[0]), int(cell[1])
        if cell not in marked_neighbors:
            continue
        if cell in occupied:
            continue
        if touches_any_tent(cell, visible_tents, int(rows), int(cols)):
            continue
        if int(visible_row_counts[row]) + 1 > int(row_clues[row]):
            continue
        if int(visible_col_counts[col]) + 1 > int(col_clues[col]):
            continue
        legal.append(cell)
    return legal


__all__ = [
    "count_by_axis",
    "legal_candidate_cells",
    "neighbors4",
    "neighbors8",
    "touches_any_tent",
]
