"""Identity-free sampling primitives for cell-board puzzle tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds
from trace_tasks.tasks.shared.color_distance import color_distance
from trace_tasks.tasks.shared.named_colors import available_named_colors, sample_named_color_palette

from .state import Coord, NamedColor
from .topology import connected_components, coords_with_neighbors, four_neighbors, sort_coords


@dataclass(frozen=True)
class ComponentBoardSample:
    """Constructed target-color component board with verification metadata."""

    board: Mapping[Coord, NamedColor]
    target_color: NamedColor
    components: Sequence[Sequence[Coord]]
    witness_coords: Sequence[Coord]


def sample_dimensions(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    fallback_rows_min: int = 5,
    fallback_rows_max: int = 8,
    fallback_cols_min: int = 5,
    fallback_cols_max: int = 8,
) -> tuple[int, int]:
    """Sample rows and columns from task params/defaults."""

    rows_min, rows_max = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="rows_min",
        max_key="rows_max",
        fallback_min=fallback_rows_min,
        fallback_max=fallback_rows_max,
        context="cell-board rows",
    )
    cols_min, cols_max = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="cols_min",
        max_key="cols_max",
        fallback_min=fallback_cols_min,
        fallback_max=fallback_cols_max,
        context="cell-board cols",
    )
    rng = spawn_rng(int(instance_seed), str(namespace))
    return int(rng.randint(rows_min, rows_max)), int(rng.randint(cols_min, cols_max))


def sample_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    fallback_min: int,
    fallback_max: int,
    min_key: str = "answer_min",
    max_key: str = "answer_max",
) -> tuple[int, list[int]]:
    """Sample one integer answer target with uniform support metadata."""

    lo, hi = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context="cell-board answer support",
    )
    support = list(range(int(lo), int(hi) + 1))
    rng = spawn_rng(int(instance_seed), str(namespace))
    return int(rng.choice(support)), [int(value) for value in support]


def sample_palette(
    *,
    rng,
    palette_size: int,
) -> list[NamedColor]:
    """Sample one canonical named-color palette for board cells."""

    palette = sample_named_color_palette(rng, palette_size=int(palette_size))
    if len(palette) < int(palette_size):
        raise ValueError("not enough named colors for requested palette size")
    return [
        (str(name), (int(rgb[0]), int(rgb[1]), int(rgb[2])))
        for name, rgb in palette
    ]


def sample_target_separated_palette(
    *,
    rng,
    palette_size: int,
    min_target_filler_distance: float,
    distance_space: str,
) -> list[NamedColor]:
    """Sample a named palette whose filler colors are separated from the target."""

    candidates = [
        (str(name), (int(rgb[0]), int(rgb[1]), int(rgb[2])))
        for name, rgb in available_named_colors()
    ]
    if not candidates:
        raise ValueError("no named colors available")
    size = max(1, min(int(palette_size), len(candidates)))
    target = rng.choice(candidates)
    threshold = max(0.0, float(min_target_filler_distance))
    separated = [
        entry
        for entry in candidates
        if str(entry[0]) != str(target[0])
        and float(
            color_distance(
                target[1],
                entry[1],
                distance_space=str(distance_space),
            )
        )
        >= threshold
    ]
    if len(separated) < size - 1:
        separated = sorted(
            (entry for entry in candidates if str(entry[0]) != str(target[0])),
            key=lambda entry: float(
                color_distance(
                    target[1],
                    entry[1],
                    distance_space=str(distance_space),
                )
            ),
            reverse=True,
        )
    if len(separated) < size - 1:
        raise ValueError("not enough named colors for requested separated palette")
    fillers = list(rng.sample(separated, k=size - 1))
    return [target, *fillers]


def sample_palette_size(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    fallback_min: int = 3,
    fallback_max: int = 6,
) -> int:
    """Sample a palette-size axis from task params/defaults."""

    lo, hi = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="palette_size_min",
        max_key="palette_size_max",
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context="cell-board palette size",
    )
    rng = spawn_rng(int(instance_seed), str(namespace))
    return int(rng.randint(int(lo), int(hi)))


def all_coords(*, rows: int, cols: int) -> list[Coord]:
    """Return all coordinates for one rectangular board."""

    return [(int(row), int(col)) for row in range(int(rows)) for col in range(int(cols))]


def choose_non_adjacent_cells(
    *,
    rng,
    rows: int,
    cols: int,
    count: int,
    blocked: Iterable[Coord] = (),
) -> list[Coord]:
    """Choose cells that do not touch each other orthogonally."""

    selected: list[Coord] = []
    unavailable = {(int(row), int(col)) for row, col in blocked}
    candidates = all_coords(rows=int(rows), cols=int(cols))
    rng.shuffle(candidates)
    for coord in candidates:
        if coord in unavailable:
            continue
        selected.append(coord)
        unavailable.add(coord)
        unavailable.update(four_neighbors(coord, rows=int(rows), cols=int(cols)))
        if len(selected) >= int(count):
            return sort_coords(selected)
    raise ValueError("not enough non-adjacent cells")


def grow_connected_region(
    *,
    rng,
    rows: int,
    cols: int,
    size: int,
    start: Coord | None = None,
    blocked: Iterable[Coord] = (),
) -> list[Coord]:
    """Grow one connected cell set with exactly the requested size."""

    blocked_set = {(int(row), int(col)) for row, col in blocked}
    candidates = [coord for coord in all_coords(rows=rows, cols=cols) if coord not in blocked_set]
    if not candidates or int(size) > len(candidates):
        raise ValueError("connected region size exceeds available cells")
    origin = (int(start[0]), int(start[1])) if start is not None else rng.choice(candidates)
    if origin in blocked_set:
        raise ValueError("start cell is blocked")
    region = {origin}
    frontier = list(four_neighbors(origin, rows=rows, cols=cols))
    while len(region) < int(size):
        valid = [coord for coord in frontier if coord not in blocked_set and coord not in region]
        if not valid:
            raise ValueError("cannot grow connected region")
        coord = rng.choice(valid)
        region.add(coord)
        frontier.extend(four_neighbors(coord, rows=rows, cols=cols))
    return sort_coords(region)


def color_board_from_components(
    *,
    rows: int,
    cols: int,
    target_color: NamedColor,
    filler_colors: Sequence[NamedColor],
    components: Sequence[Sequence[Coord]],
) -> dict[Coord, NamedColor]:
    """Build a color board with target-color components over filler colors."""

    board: dict[Coord, NamedColor] = {}
    fillers = list(filler_colors)
    if not fillers:
        raise ValueError("filler colors must not be empty")
    for row in range(int(rows)):
        for col in range(int(cols)):
            board[(row, col)] = cycled_named_color(fillers, offset=int(row) + int(col))
    for component in components:
        for coord in component:
            board[(int(coord[0]), int(coord[1]))] = target_color
    return board


def cycled_named_color(colors: Sequence[NamedColor], *, offset: int) -> NamedColor:
    """Return a deterministic cyclic color from an already-sampled palette."""

    palette = tuple(colors)
    if not palette:
        raise ValueError("cycled color palette must not be empty")
    offset_index = int(offset)
    if offset_index < 0:
        raise ValueError("cycled color offset must be non-negative")
    while offset_index >= len(palette):
        offset_index -= len(palette)
    return palette[offset_index]


def target_color_cells(
    board: Mapping[Coord, NamedColor],
    *,
    color_name: str,
) -> list[Coord]:
    """Return cells whose named color matches the requested color."""

    return sort_coords(
        coord
        for coord, (name, _rgb) in board.items()
        if str(name).casefold() == str(color_name).casefold()
    )


def component_cells_for_color(
    board: Mapping[Coord, NamedColor],
    *,
    rows: int,
    cols: int,
    color_name: str,
) -> list[list[Coord]]:
    """Return target-color 4-neighbor connected components."""

    return connected_components(
        target_color_cells(board, color_name=str(color_name)),
        rows=int(rows),
        cols=int(cols),
    )


def sample_unique_largest_component_board(
    *,
    rng,
    rows: int,
    cols: int,
    palette_size: int,
    largest_size: int,
    target_filler_min_color_distance: float = 0.0,
    color_distance_space: str = "lab",
) -> ComponentBoardSample:
    """Build a target-color board with one unique largest component."""

    if float(target_filler_min_color_distance) > 0.0:
        palette = sample_target_separated_palette(
            rng=rng,
            palette_size=int(palette_size),
            min_target_filler_distance=float(target_filler_min_color_distance),
            distance_space=str(color_distance_space),
        )
    else:
        palette = sample_palette(rng=rng, palette_size=int(palette_size))
    target_color = palette[0]
    largest = grow_connected_region(
        rng=rng,
        rows=int(rows),
        cols=int(cols),
        size=int(largest_size),
    )
    small_seeds = choose_non_adjacent_cells(
        rng=rng,
        rows=int(rows),
        cols=int(cols),
        count=int(rng.randint(1, 3)),
        blocked=coords_with_neighbors(largest, rows=int(rows), cols=int(cols)),
    )
    board = color_board_from_components(
        rows=int(rows),
        cols=int(cols),
        target_color=target_color,
        filler_colors=palette[1:],
        components=[largest] + [[coord] for coord in small_seeds],
    )
    resolved = component_cells_for_color(
        board,
        rows=int(rows),
        cols=int(cols),
        color_name=str(target_color[0]),
    )
    sizes = [len(component) for component in resolved]
    if sizes.count(int(largest_size)) != 1 or max(sizes) != int(largest_size):
        raise ValueError("largest component was not unique")
    return ComponentBoardSample(
        board=board,
        target_color=target_color,
        components=tuple(tuple(component) for component in resolved),
        witness_coords=tuple(largest),
    )


__all__ = [
    "all_coords",
    "choose_non_adjacent_cells",
    "color_board_from_components",
    "ComponentBoardSample",
    "component_cells_for_color",
    "cycled_named_color",
    "grow_connected_region",
    "sample_answer",
    "sample_dimensions",
    "sample_palette",
    "sample_palette_size",
    "sample_target_separated_palette",
    "sample_unique_largest_component_board",
    "target_color_cells",
]
