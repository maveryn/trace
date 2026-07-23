"""Life automaton rules and neutral sampling primitives."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from ....shared.deterministic_sampling import resolve_selection_index
from ....shared.mcq import option_label_for_index

from .state import LifeOptionSpec


SCENE_ID = "life_automaton"
SCENE_VARIANTS: Tuple[str, ...] = ("clean_grid", "lab_panel", "notebook_grid")
BOARD_STYLES: Tuple[str, ...] = (
    "classic_grid",
    "rounded_tiles",
    "inset_tiles",
    "lab_matrix",
    "notebook_cells",
    "terminal_cells",
)
CELL_PALETTES: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "mono_ink": {
        "dead": (247, 248, 250),
        "alive": (35, 42, 54),
        "grid": (90, 101, 116),
        "edge": (68, 79, 94),
        "mark": (230, 73, 82),
        "accent": (188, 201, 218),
    },
    "blueprint_cells": {
        "dead": (239, 247, 254),
        "alive": (19, 55, 96),
        "grid": (73, 111, 150),
        "edge": (36, 82, 126),
        "mark": (230, 94, 55),
        "accent": (171, 203, 232),
    },
    "forest_cells": {
        "dead": (240, 249, 243),
        "alive": (20, 76, 58),
        "grid": (82, 129, 105),
        "edge": (44, 97, 77),
        "mark": (205, 69, 88),
        "accent": (180, 216, 194),
    },
    "plum_cells": {
        "dead": (250, 244, 251),
        "alive": (72, 35, 88),
        "grid": (126, 94, 141),
        "edge": (94, 63, 110),
        "mark": (35, 135, 168),
        "accent": (218, 193, 226),
    },
    "sepia_cells": {
        "dead": (252, 247, 236),
        "alive": (78, 51, 34),
        "grid": (142, 114, 82),
        "edge": (105, 82, 58),
        "mark": (202, 70, 61),
        "accent": (224, 204, 170),
    },
    "teal_cells": {
        "dead": (238, 250, 248),
        "alive": (16, 78, 85),
        "grid": (72, 132, 136),
        "edge": (36, 100, 106),
        "mark": (211, 70, 92),
        "accent": (172, 219, 218),
    },
    "burgundy_cells": {
        "dead": (252, 243, 245),
        "alive": (96, 30, 48),
        "grid": (145, 85, 100),
        "edge": (113, 55, 71),
        "mark": (0, 128, 158),
        "accent": (228, 192, 201),
    },
    "carbon_cells": {
        "dead": (246, 246, 242),
        "alive": (19, 22, 26),
        "grid": (92, 96, 101),
        "edge": (61, 66, 73),
        "mark": (226, 78, 66),
        "accent": (199, 202, 202),
    },
}


def simulate_life(grid: Sequence[Sequence[int]], *, steps: int) -> Tuple[Tuple[int, ...], ...]:
    """Run Conway-style Life updates for a finite grid without wraparound."""

    current = tuple(tuple(int(value) for value in row) for row in grid)
    rows = len(current)
    cols = len(current[0])
    for _ in range(int(steps)):
        next_rows: List[Tuple[int, ...]] = []
        for row in range(rows):
            values: List[int] = []
            for col in range(cols):
                neighbors = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr = row + dr
                        cc = col + dc
                        if 0 <= rr < rows and 0 <= cc < cols:
                            neighbors += int(current[rr][cc])
                alive = int(current[row][col])
                values.append(1 if (neighbors == 3 or (alive and neighbors == 2)) else 0)
            next_rows.append(tuple(values))
        current = tuple(next_rows)
    return current


def grid_live_count(grid: Sequence[Sequence[int]], cells: Sequence[Tuple[int, int]] | None = None) -> int:
    """Count live cells in a grid or in a supplied cell subset."""

    if cells is None:
        return int(sum(int(value) for row in grid for value in row))
    return int(sum(int(grid[int(row)][int(col)]) for row, col in cells))


def sample_life_grid(rng, *, rows: int, cols: int, live_prob: float) -> Tuple[Tuple[int, ...], ...]:
    """Sample a binary Life grid."""

    values: List[Tuple[int, ...]] = []
    for _row in range(int(rows)):
        row_values: List[int] = []
        for _col in range(int(cols)):
            row_values.append(1 if float(rng.random()) < float(live_prob) else 0)
        values.append(tuple(row_values))
    return tuple(values)


def sample_square_grid_size(
    rng,
    *,
    grid_size_min: int,
    grid_size_max: int,
    rows_min: int,
    rows_max: int,
    cols_min: int,
    cols_max: int,
) -> int:
    """Sample a square Life grid size within all configured bounds."""

    size_min = max(int(grid_size_min), int(rows_min), int(cols_min))
    size_max = min(int(grid_size_max), int(rows_max), int(cols_max))
    if size_min > size_max:
        raise ValueError(
            "Life automaton requires overlapping square grid-size support; "
            f"got grid={grid_size_min}..{grid_size_max}, rows={rows_min}..{rows_max}, cols={cols_min}..{cols_max}"
        )
    return int(rng.randint(size_min, size_max))


def line_candidates_by_count(grid: Sequence[Sequence[int]]) -> Dict[int, List[Tuple[Tuple[int, int], ...]]]:
    """Return all rows and columns grouped by their live-cell count."""

    rows = len(grid)
    cols = len(grid[0])
    candidates: Dict[int, List[Tuple[Tuple[int, int], ...]]] = {}
    for row in range(rows):
        cells = tuple((row, col) for col in range(cols))
        candidates.setdefault(grid_live_count(grid, cells), []).append(cells)
    for col in range(cols):
        cells = tuple((row, col) for row in range(rows))
        candidates.setdefault(grid_live_count(grid, cells), []).append(cells)
    return candidates


def choose_life_options(
    *,
    params: Mapping[str, object],
    instance_seed: int,
    namespace: str,
    future_grid: Sequence[Sequence[int]],
    option_count: int,
    rng,
) -> tuple[Tuple[LifeOptionSpec, ...], str]:
    """Build unique future-grid options and return the correct option label."""

    rows = len(future_grid)
    cols = len(future_grid[0])
    correct = tuple(tuple(int(value) for value in row) for row in future_grid)
    options = {correct}
    while len(options) < int(option_count):
        candidate = [list(row) for row in correct]
        flips = int(rng.randint(1, 4))
        for _ in range(flips):
            row = int(rng.randrange(rows))
            col = int(rng.randrange(cols))
            candidate[row][col] = 1 - int(candidate[row][col])
        options.add(tuple(tuple(int(value) for value in row) for row in candidate))

    distractors = [option for option in options if option != correct]
    rng.shuffle(distractors)
    correct_index = int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        % int(option_count)
    )
    ordered: List[Tuple[Tuple[int, ...], ...]] = []
    for index in range(int(option_count)):
        if index == correct_index:
            ordered.append(correct)
        else:
            ordered.append(distractors.pop())

    specs: List[LifeOptionSpec] = []
    answer_label = "A"
    for index, option_grid in enumerate(ordered):
        label = option_label_for_index(index)
        is_correct = bool(option_grid == correct)
        if is_correct:
            answer_label = str(label)
        specs.append(
            LifeOptionSpec(
                option_id=f"option_{label}",
                label=str(label),
                grid=tuple(tuple(int(value) for value in row) for row in option_grid),
                is_correct=is_correct,
            )
        )
    return tuple(specs), str(answer_label)
