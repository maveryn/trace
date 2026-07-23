"""Star Battle grid rules, scope helpers, and item identifiers."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple

from .state import Cell


def cell_key(cell: Cell) -> str:
    """Return the render-map key for one grid cell."""

    return f"cell_{int(cell[0])}_{int(cell[1])}"


def candidate_key(label: str) -> str:
    """Return the render-map key for one visible candidate label."""

    return f"candidate_{str(label)}"


def region_key(region_index: int) -> str:
    """Return the render-map key for one colored region."""

    return f"region_{int(region_index)}"


def neighbors4(cell: Cell, size: int) -> List[Cell]:
    """Return orthogonal neighbors inside the square Star Battle board."""

    row, col = int(cell[0]), int(cell[1])
    out: List[Cell] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = row + dr
        cc = col + dc
        if 0 <= rr < int(size) and 0 <= cc < int(size):
            out.append((int(rr), int(cc)))
    return out


def neighbors8(cell: Cell, size: int) -> List[Cell]:
    """Return edge and corner neighbors inside the square board."""

    row, col = int(cell[0]), int(cell[1])
    out: List[Cell] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = row + dr
            cc = col + dc
            if 0 <= rr < int(size) and 0 <= cc < int(size):
                out.append((int(rr), int(cc)))
    return out


def sample_solution(size: int, *, rng) -> List[Cell]:
    """Sample one non-touching permutation: one star in each row and column."""

    cols = list(range(int(size)))
    for _ in range(512):
        rng.shuffle(cols)
        stars = [(row, int(cols[row])) for row in range(int(size))]
        if all(abs(stars[row][1] - stars[row - 1][1]) > 1 for row in range(1, int(size))):
            return stars
    raise RuntimeError("failed to sample Star Battle solution")


def grow_regions(size: int, stars: Sequence[Cell], *, rng) -> List[List[int]]:
    """Grow connected colored regions from solution-star seed cells."""

    region_grid = [[-1 for _ in range(int(size))] for _ in range(int(size))]
    frontiers: Dict[int, List[Cell]] = {}
    for index, cell in enumerate(stars):
        row, col = int(cell[0]), int(cell[1])
        region_grid[row][col] = int(index)
        frontiers[int(index)] = [
            nbr for nbr in neighbors4((row, col), int(size))
            if region_grid[nbr[0]][nbr[1]] < 0
        ]

    unassigned = int(size * size - len(stars))
    while unassigned > 0:
        expandable = [idx for idx, frontier in frontiers.items() if frontier]
        if not expandable:
            raise RuntimeError("region growth became disconnected")
        region_index = int(expandable[int(rng.randrange(len(expandable)))])
        rng.shuffle(frontiers[region_index])
        cell = tuple(frontiers[region_index].pop())
        if region_grid[int(cell[0])][int(cell[1])] >= 0:
            continue
        region_grid[int(cell[0])][int(cell[1])] = int(region_index)
        unassigned -= 1
        for nbr in neighbors4(cell, int(size)):
            if region_grid[int(nbr[0])][int(nbr[1])] < 0:
                frontiers[int(region_index)].append(tuple(nbr))

    return region_grid


def region_cells(region_grid: Sequence[Sequence[int]]) -> Dict[int, List[Cell]]:
    """Group board cells by colored region index."""

    regions: Dict[int, List[Cell]] = {}
    for row, values in enumerate(region_grid):
        for col, value in enumerate(values):
            regions.setdefault(int(value), []).append((int(row), int(col)))
    return regions


def connected_region(region_grid: Sequence[Sequence[int]], region_index: int) -> bool:
    """Return whether all cells for one region form an orthogonally connected set."""

    cells = set(region_cells(region_grid).get(int(region_index), []))
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    queue: deque[Cell] = deque([start])
    size = len(region_grid)
    while queue:
        cur = queue.popleft()
        for nbr in neighbors4(cur, int(size)):
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    return seen == cells


def star_counts_by_region(
    stars: Iterable[Cell],
    region_grid: Sequence[Sequence[int]],
    size: int,
) -> List[int]:
    """Count solution stars in each colored region."""

    counts = [0 for _ in range(int(size))]
    for row, col in stars:
        counts[int(region_grid[int(row)][int(col)])] += 1
    return counts


def legal_cells(
    *,
    size: int,
    region_grid: Sequence[Sequence[int]],
    visible_stars: Sequence[Cell],
) -> List[Cell]:
    """Return cells where another star can be placed under Star Battle rules."""

    star_set = {tuple(cell) for cell in visible_stars}
    filled_rows = {int(row) for row, _col in star_set}
    filled_cols = {int(col) for _row, col in star_set}
    filled_regions = {int(region_grid[int(row)][int(col)]) for row, col in star_set}
    legal: List[Cell] = []
    for row in range(int(size)):
        if row in filled_rows:
            continue
        for col in range(int(size)):
            cell = (int(row), int(col))
            if col in filled_cols:
                continue
            if int(region_grid[row][col]) in filled_regions:
                continue
            if any(star in set(neighbors8(cell, int(size))) for star in star_set):
                continue
            legal.append(cell)
    return legal


def scope_cells(
    *,
    scope_kind: str,
    size: int,
    regions: Dict[str, Sequence[Cell]],
    marked_region_index: int | None = None,
    marked_row_index: int | None = None,
    marked_col_index: int | None = None,
) -> List[Cell]:
    """Return the cells covered by a task-owned scope kind."""

    if str(scope_kind) == "marked_region":
        if marked_region_index is None:
            raise ValueError("marked_region scope requires marked_region_index")
        return [tuple(cell) for cell in regions[str(int(marked_region_index))]]
    if str(scope_kind) == "marked_row":
        if marked_row_index is None:
            raise ValueError("marked_row scope requires marked_row_index")
        return [(int(marked_row_index), col) for col in range(int(size))]
    if str(scope_kind) == "marked_column":
        if marked_col_index is None:
            raise ValueError("marked_column scope requires marked_col_index")
        return [(row, int(marked_col_index)) for row in range(int(size))]
    if str(scope_kind) == "whole_board":
        return [(row, col) for row in range(int(size)) for col in range(int(size))]
    raise ValueError(f"unsupported Star Battle scope kind: {scope_kind}")


def scope_item_ids(
    *,
    scope_kind: str,
    marked_region_index: int | None = None,
    marked_row_index: int | None = None,
    marked_col_index: int | None = None,
) -> List[str]:
    """Return render-map item ids for the visible scope marker, if any."""

    if str(scope_kind) == "marked_region":
        if marked_region_index is None:
            return []
        return [region_key(int(marked_region_index))]
    if str(scope_kind) == "marked_row":
        if marked_row_index is None:
            return []
        return [f"row_{int(marked_row_index)}"]
    if str(scope_kind) == "marked_column":
        if marked_col_index is None:
            return []
        return [f"col_{int(marked_col_index)}"]
    return []


__all__ = [
    "candidate_key",
    "cell_key",
    "connected_region",
    "grow_regions",
    "legal_cells",
    "neighbors4",
    "neighbors8",
    "region_cells",
    "region_key",
    "sample_solution",
    "scope_cells",
    "scope_item_ids",
    "star_counts_by_region",
]
