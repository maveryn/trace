"""Voxel-cube scene rules and projection math."""

from __future__ import annotations

from typing import Iterable, Tuple

from .state import CubeStack, GridCell, HeightGrid, ProjectionGrid

Cube = Tuple[int, int, int]


def cube_coordinates(stack: CubeStack) -> Tuple[Cube, ...]:
    """Return all occupied unit-cube coordinates in one height grid."""

    coords: list[Cube] = []
    for row, values in enumerate(stack.heights):
        for col, height in enumerate(values):
            for level in range(int(height)):
                coords.append((int(row), int(col), int(level)))
    return tuple(coords)


def cube_count(stack: CubeStack) -> int:
    """Return the number of unit cubes in one stack."""

    return len(cube_coordinates(stack))


def max_height(stack: CubeStack) -> int:
    """Return the highest column height in one stack."""

    return max((int(value) for row in stack.heights for value in row), default=0)


def with_height_at(stack: CubeStack, row: int, col: int, height: int) -> CubeStack:
    """Return a copy with one footprint height changed."""

    heights = [list(values) for values in stack.heights]
    heights[int(row)][int(col)] = max(0, int(height))
    return CubeStack(tuple(tuple(int(value) for value in row) for row in heights))


def complete_cuboid(rows: int, cols: int, height: int) -> CubeStack:
    """Return a solid rectangular cuboid height grid."""

    return CubeStack(
        tuple(
            tuple(int(height) for _col in range(int(cols))) for _row in range(int(rows))
        )
    )


def remove_top_cubes(stack: CubeStack, removals: Iterable[GridCell]) -> CubeStack:
    """Remove one top cube from each selected footprint cell."""

    result = stack
    for row, col in removals:
        current = int(result.heights[int(row)][int(col)])
        if current <= 0:
            raise ValueError("cannot remove a cube from an empty column")
        result = with_height_at(result, int(row), int(col), current - 1)
    return result


def projection_grid(stack: CubeStack, direction: str) -> ProjectionGrid:
    """Project one voxel stack into a top/front/right occupied-cell grid."""

    view = str(direction)
    if view not in {"top", "front", "right"}:
        raise ValueError(f"unsupported voxel projection direction: {direction}")
    height = max_height(stack)
    if view == "top":
        filled = [
            (row, col)
            for row, values in enumerate(stack.heights)
            for col, column_height in enumerate(values)
            if int(column_height) > 0
        ]
        return ProjectionGrid(
            direction=view,
            rows=int(stack.rows),
            cols=int(stack.cols),
            filled_cells=tuple(filled),
        )

    if view == "front":
        filled = []
        for level in range(height):
            out_row = height - 1 - level
            for col in range(stack.cols):
                if any(
                    int(stack.heights[row][col]) > level for row in range(stack.rows)
                ):
                    filled.append((out_row, col))
        return ProjectionGrid(
            direction=view,
            rows=int(height),
            cols=int(stack.cols),
            filled_cells=tuple(filled),
        )

    filled = []
    for level in range(height):
        out_row = height - 1 - level
        for row in range(stack.rows):
            if any(int(stack.heights[row][col]) > level for col in range(stack.cols)):
                filled.append((out_row, row))
    return ProjectionGrid(
        direction=view,
        rows=int(height),
        cols=int(stack.rows),
        filled_cells=tuple(filled),
    )


def corrupted_projection(
    projection: ProjectionGrid,
    *,
    rng,
    target_count_delta: int | None = None,
) -> ProjectionGrid:
    """Return a nearby but distinct projection grid."""

    all_cells = {
        (row, col)
        for row in range(int(projection.rows))
        for col in range(int(projection.cols))
    }
    filled = set(projection.filled_cells)
    if target_count_delta is None:
        target_count_delta = int(rng.choice((-1, 1)))

    if int(target_count_delta) > 0:
        candidates = sorted(all_cells - filled)
        if candidates:
            filled.add(tuple(rng.choice(candidates)))
    else:
        candidates = sorted(filled)
        if len(candidates) > 1:
            filled.remove(tuple(rng.choice(candidates)))
        elif sorted(all_cells - filled):
            filled.add(tuple(rng.choice(sorted(all_cells - filled))))

    if set(filled) == set(projection.filled_cells):
        raise ValueError("projection corruption did not change the projection")
    return ProjectionGrid(
        direction=str(projection.direction),
        rows=int(projection.rows),
        cols=int(projection.cols),
        filled_cells=tuple(sorted(filled)),
    )


def projection_signature(projection: ProjectionGrid) -> tuple[object, ...]:
    """Return a hashable signature for projection option uniqueness."""

    return (
        str(projection.direction),
        int(projection.rows),
        int(projection.cols),
        tuple(sorted((int(row), int(col)) for row, col in projection.filled_cells)),
    )
