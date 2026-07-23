"""Pure polyomino transforms and exact two-piece tiling rules."""

from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

from .state import Cell, Cells


def canonicalize_cells(cells: Iterable[Cell]) -> Cells:
    """Return cells shifted to the origin and sorted."""

    values = sorted((int(x), int(y)) for x, y in cells)
    if not values:
        raise ValueError("polyomino cells cannot be empty")
    min_x = min(x for x, _y in values)
    min_y = min(y for _x, y in values)
    return tuple(sorted((int(x - min_x), int(y - min_y)) for x, y in values))


def rotate_cells_90(cells: Cells) -> Cells:
    """Rotate one canonical cell set 90 degrees clockwise."""

    return canonicalize_cells((int(y), int(-x)) for x, y in cells)


def reflect_cells_horizontal(cells: Cells) -> Cells:
    """Reflect one canonical cell set across a vertical axis."""

    return canonicalize_cells((int(-x), int(y)) for x, y in cells)


def unique_rotations(cells: Iterable[Cell]) -> tuple[Cells, ...]:
    """Return unique rotations of one polyomino, without reflection."""

    current = canonicalize_cells(cells)
    rotations: list[Cells] = []
    seen: set[Cells] = set()
    for _index in range(4):
        if current not in seen:
            rotations.append(current)
            seen.add(current)
        current = rotate_cells_90(current)
    return tuple(rotations)


def unique_orientations_with_reflections(cells: Iterable[Cell]) -> tuple[Cells, ...]:
    """Return unique rotations and mirrored rotations of one polyomino."""

    canonical = canonicalize_cells(cells)
    orientations: list[Cells] = []
    seen: set[Cells] = set()
    for seed in (canonical, reflect_cells_horizontal(canonical)):
        current = seed
        for _index in range(4):
            if current not in seen:
                orientations.append(current)
                seen.add(current)
            current = rotate_cells_90(current)
    return tuple(orientations)


def rotation_signature(cells: Iterable[Cell]) -> Cells:
    """Return a rotation-invariant shape signature."""

    return min(unique_rotations(canonicalize_cells(cells)))


def reflection_signature(cells: Iterable[Cell]) -> Cells:
    """Return a rotation-and-reflection-invariant shape signature."""

    return min(unique_orientations_with_reflections(canonicalize_cells(cells)))


def pair_rotation_signature(piece_a: Iterable[Cell], piece_b: Iterable[Cell]) -> tuple[Cells, Cells]:
    """Return an order-invariant signature for a two-piece option."""

    signatures = sorted((rotation_signature(piece_a), rotation_signature(piece_b)))
    return tuple(signatures)  # type: ignore[return-value]


def translate_cells(cells: Iterable[Cell], dx: int, dy: int) -> Cells:
    """Translate cells by integer grid offsets."""

    return tuple(sorted((int(x + dx), int(y + dy)) for x, y in cells))


def shape_bbox_dims(cells: Iterable[Cell]) -> tuple[int, int]:
    """Return `(width, height)` for one canonical cell set."""

    canonical = canonicalize_cells(cells)
    max_x = max(int(x) for x, _y in canonical)
    max_y = max(int(y) for _x, y in canonical)
    return int(max_x + 1), int(max_y + 1)


def is_connected(cells: Iterable[Cell]) -> bool:
    """Return whether cells form one 4-connected component."""

    remaining = {(int(x), int(y)) for x, y in cells}
    if not remaining:
        return False
    stack = [next(iter(remaining))]
    seen: set[Cell] = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        x, y = cell
        for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if neighbor in remaining and neighbor not in seen:
                stack.append(neighbor)
    return len(seen) == len(remaining)


def edge_neighbors(cell: Cell) -> tuple[Cell, ...]:
    """Return the four edge-neighbor cells."""

    x, y = int(cell[0]), int(cell[1])
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def can_two_pieces_tile_target(
    piece_a: Iterable[Cell],
    piece_b: Iterable[Cell],
    target: Iterable[Cell],
) -> bool:
    """Return whether two pieces can tile the target using rotations only."""

    return first_tiling_placement(piece_a, piece_b, target) is not None


def first_tiling_placement(
    piece_a: Iterable[Cell],
    piece_b: Iterable[Cell],
    target: Iterable[Cell],
) -> dict[str, Cells] | None:
    """Return one exact tiling placement, or `None` if no tiling exists."""

    piece_a_cells = canonicalize_cells(piece_a)
    piece_b_cells = canonicalize_cells(piece_b)
    target_cells = set(canonicalize_cells(target))
    if len(target_cells) != len(piece_a_cells) + len(piece_b_cells):
        return None
    for first, second, swapped in (
        (piece_a_cells, piece_b_cells, False),
        (piece_b_cells, piece_a_cells, True),
    ):
        placement = _try_ordered_piece_placement(first, second, target_cells)
        if placement is None:
            continue
        if swapped:
            return {
                "piece_a": placement["piece_b"],
                "piece_b": placement["piece_a"],
            }
        return placement
    return None


def _try_ordered_piece_placement(
    first_piece: Iterable[Cell],
    second_piece: Iterable[Cell],
    target_cells: set[Cell],
) -> dict[str, Cells] | None:
    """Try placing `first_piece`, then matching the remaining target cells."""

    first_rotations = unique_rotations(canonicalize_cells(first_piece))
    second_rotations = set(unique_rotations(second_piece))
    for rotated_first in first_rotations:
        for piece_anchor, target_anchor in product(rotated_first, target_cells):
            dx = int(target_anchor[0] - piece_anchor[0])
            dy = int(target_anchor[1] - piece_anchor[1])
            placed_first = translate_cells(rotated_first, dx, dy)
            placed_first_set = set(placed_first)
            if not placed_first_set <= target_cells:
                continue
            remaining = tuple(sorted(target_cells - placed_first_set))
            if not remaining:
                continue
            remaining_canonical = canonicalize_cells(remaining)
            if remaining_canonical not in second_rotations:
                continue
            return {
                "piece_a": canonicalize_cells(placed_first),
                "piece_b": canonicalize_cells(remaining),
            }
    return None


def json_cells(cells: Iterable[Cell]) -> list[list[int]]:
    """Return cells as JSON-ready coordinates."""

    return [[int(x), int(y)] for x, y in canonicalize_cells(cells)]


def json_cell_sequence(cells: Sequence[Cell]) -> list[list[int]]:
    """Return already placed cells as JSON-ready sorted coordinates."""

    return [[int(x), int(y)] for x, y in sorted((int(x), int(y)) for x, y in cells)]


__all__ = [
    "can_two_pieces_tile_target",
    "canonicalize_cells",
    "edge_neighbors",
    "first_tiling_placement",
    "reflection_signature",
    "reflect_cells_horizontal",
    "is_connected",
    "json_cell_sequence",
    "json_cells",
    "pair_rotation_signature",
    "rotation_signature",
    "shape_bbox_dims",
    "translate_cells",
    "unique_orientations_with_reflections",
    "unique_rotations",
]
