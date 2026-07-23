"""Shared graph-paper canvas/grid sampling helpers for geometry tasks."""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Sequence

from ...shared.geometry_primitives import Point
from ...shared.config_defaults import group_default, resolve_required_int_bounds


def resolve_square_canvas_size(
    rng,
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_min: int,
    fallback_max: int,
) -> int:
    """Resolve square canvas size with deterministic random range sampling."""
    explicit_size = params.get("canvas_size", render_defaults.get("canvas_size"))
    if explicit_size is not None:
        value = int(explicit_size)
        if value < 64:
            raise ValueError("canvas_size must be >= 64")
        return int(value)

    size_min, size_max = resolve_required_int_bounds(
        params,
        render_defaults,
        min_key="canvas_size_min",
        max_key="canvas_size_max",
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context="graph-paper canvas size defaults",
    )
    if size_min < 64:
        raise ValueError("canvas_size_min must be >= 64")
    return int(rng.randint(int(size_min), int(size_max)))


def graph_spacing_from_cells(*, canvas_size: int, graph_cells: int, min_spacing_px: int = 4) -> int:
    """Compute graph-paper spacing from canvas size and sampled cells-per-side."""
    size_px = max(1, int(canvas_size))
    cells = max(2, int(graph_cells))
    spacing = int(round(float(size_px) / float(cells)))
    return max(int(min_spacing_px), int(spacing))


def resolve_graph_cells_per_side(
    rng,
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    canvas_size: int,
    fallback_min: int,
    fallback_max: int,
    min_spacing_px: int = 4,
) -> int:
    """Resolve graph-paper cells-per-side with deterministic random range sampling."""
    explicit_cells = params.get("graph_cells", render_defaults.get("graph_cells"))
    if explicit_cells is not None:
        value = int(explicit_cells)
        if value < 2:
            raise ValueError("graph_cells must be >= 2")
        spacing = graph_spacing_from_cells(canvas_size=int(canvas_size), graph_cells=int(value), min_spacing_px=min_spacing_px)
        if int(spacing) < int(min_spacing_px):
            raise ValueError("graph_cells is too large for current canvas_size")
        return int(value)

    cells_min, cells_max = resolve_required_int_bounds(
        params,
        render_defaults,
        min_key="graph_cells_min",
        max_key="graph_cells_max",
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context="graph-paper cell-count defaults",
    )
    feasible = [
        cells
        for cells in range(max(2, int(cells_min)), int(cells_max) + 1)
        if graph_spacing_from_cells(canvas_size=int(canvas_size), graph_cells=int(cells), min_spacing_px=min_spacing_px)
        >= int(min_spacing_px)
    ]
    if not feasible:
        raise ValueError("no feasible graph_cells in range for current canvas_size")
    return int(rng.choice(feasible))


def resolve_graph_cell_capacity(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_min: int,
    fallback_max: int,
) -> tuple[int | None, int]:
    """Return explicit graph-cell count (if fixed) and max allowed graph-cell count."""
    explicit_cells = params.get("graph_cells", render_defaults.get("graph_cells"))
    if explicit_cells is not None:
        value = int(explicit_cells)
        if int(value) < 2:
            raise ValueError("graph_cells must be >= 2")
        return int(value), int(value)
    _cells_min, cells_max = resolve_required_int_bounds(
        params,
        render_defaults,
        min_key="graph_cells_min",
        max_key="graph_cells_max",
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context="graph-paper cell-count defaults",
    )
    return None, int(cells_max)


def lattice_axis_coordinates_for_offsets(
    *,
    canvas_size: int,
    spacing: int,
    offsets: Sequence[int],
    lattice_origin: float = 0.0,
    padding: int = 0,
) -> List[float]:
    """Return feasible lattice coordinates for one axis under offset constraints.

    Each sampled anchor coordinate `c` must satisfy:
      `padding <= c + (offset * spacing) <= canvas_size - padding`
    for every provided integer `offset`.

    Candidate anchor coordinates are restricted to one lattice
    `lattice_origin + k * spacing`, where `k` is an integer.
    """
    spacing_px = max(1, int(spacing))
    canvas_px = int(canvas_size)
    pad_px = max(0, int(padding))
    origin_px = float(lattice_origin)
    offset_values = [int(value) for value in offsets]
    if not offset_values:
        raise ValueError("offsets must be non-empty")

    lower_bound = max(pad_px - (offset * spacing_px) for offset in offset_values)
    upper_bound = min(((canvas_px - 1 - pad_px) - (offset * spacing_px)) for offset in offset_values)
    if lower_bound > upper_bound:
        return []

    k_lo = int(math.ceil((float(lower_bound) - float(origin_px)) / float(spacing_px)))
    k_hi = int(math.floor((float(upper_bound) - float(origin_px)) / float(spacing_px)))
    if int(k_lo) > int(k_hi):
        return []
    return [float(origin_px + (int(k) * int(spacing_px))) for k in range(int(k_lo), int(k_hi) + 1)]


def sample_lattice_point_with_offsets(
    rng,
    *,
    canvas_size: int,
    spacing: int,
    x_offsets: Sequence[int],
    y_offsets: Sequence[int],
    lattice_origin: Point | None = None,
    padding: int = 0,
) -> Point:
    """Sample one graph-lattice point that keeps all offset points inside canvas."""
    origin = (
        (float(lattice_origin[0]), float(lattice_origin[1]))
        if lattice_origin is not None
        else (0.0, 0.0)
    )
    x_values = lattice_axis_coordinates_for_offsets(
        canvas_size=int(canvas_size),
        spacing=int(spacing),
        offsets=list(x_offsets),
        lattice_origin=float(origin[0]),
        padding=int(padding),
    )
    y_values = lattice_axis_coordinates_for_offsets(
        canvas_size=int(canvas_size),
        spacing=int(spacing),
        offsets=list(y_offsets),
        lattice_origin=float(origin[1]),
        padding=int(padding),
    )
    if not x_values or not y_values:
        raise ValueError("no feasible lattice anchor for provided offsets")
    return (float(rng.choice(x_values)), float(rng.choice(y_values)))


def offset_point_by_grid_vector(
    point: Point,
    offset: Sequence[int],
    *,
    spacing: int,
) -> Point:
    """Translate one point by an integer-grid vector scaled by `spacing` pixels."""
    dx = int(offset[0])
    dy = int(offset[1])
    spacing_px = int(spacing)
    return (
        float(point[0]) + float(dx * spacing_px),
        float(point[1]) + float(dy * spacing_px),
    )
