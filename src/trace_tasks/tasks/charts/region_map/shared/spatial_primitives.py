"""Spatial primitives for region-map chart tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.sampling import uniform_choice
from .....core.seed import hash64, spawn_rng
from ....shared.bbox_projection import round_bbox as _round_bbox


Point = Tuple[float, float]
CHOROPLETH_TASK_NAMESPACE = "charts_region_map"


def _polygon_bbox(points: Sequence[Point]) -> List[float]:
    return _round_bbox(
        [
            min(float(point[0]) for point in points),
            min(float(point[1]) for point in points),
            max(float(point[0]) for point in points),
            max(float(point[1]) for point in points),
        ]
    )


def _polygon_center(points: Sequence[Point]) -> Point:
    return (
        sum(float(point[0]) for point in points) / float(len(points)),
        sum(float(point[1]) for point in points) / float(len(points)),
    )


def _shrink_polygon(points: Sequence[Point], *, gap_px: float) -> List[Point]:
    cx, cy = _polygon_center(points)
    out: List[Point] = []
    for x, y in points:
        dx = float(x - cx)
        dy = float(y - cy)
        distance = max(1.0, math.hypot(dx, dy))
        scale = max(0.0, (float(distance) - float(gap_px)) / float(distance))
        out.append((float(cx + (dx * scale)), float(cy + (dy * scale))))
    return out


def _grid_points(
    *,
    rows: int,
    cols: int,
    map_bbox: Sequence[float],
    instance_seed: int,
) -> Dict[Tuple[int, int], Point]:
    left, top, right, bottom = [float(value) for value in map_bbox]
    width = float(right - left)
    height = float(bottom - top)
    jitter_x = min(18.0, width / max(1.0, float(cols)) * 0.16)
    jitter_y = min(16.0, height / max(1.0, float(rows)) * 0.16)
    rng = spawn_rng(int(instance_seed), f"{CHOROPLETH_TASK_NAMESPACE}.cartogram_points")
    points: Dict[Tuple[int, int], Point] = {}
    for y in range(int(rows) + 1):
        for x in range(int(cols) + 1):
            px = float(left + (width * float(x) / float(cols)))
            py = float(top + (height * float(y) / float(rows)))
            if 0 < int(x) < int(cols):
                px += float(rng.uniform(-jitter_x, jitter_x))
            if 0 < int(y) < int(rows):
                py += float(rng.uniform(-jitter_y, jitter_y))
            points[(int(x), int(y))] = (float(px), float(py))
    return points


def _region_polygon(
    *,
    row: int,
    col: int,
    grid_points: Mapping[Tuple[int, int], Point],
) -> List[Point]:
    return [
        grid_points[(int(col), int(row))],
        grid_points[(int(col) + 1, int(row))],
        grid_points[(int(col) + 1, int(row) + 1)],
        grid_points[(int(col), int(row) + 1)],
    ]


def _neighbors(cell: Tuple[int, int], *, rows: int, cols: int) -> List[Tuple[int, int]]:
    row, col = int(cell[0]), int(cell[1])
    out: List[Tuple[int, int]] = []
    for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        next_row = int(row + delta_row)
        next_col = int(col + delta_col)
        if 0 <= next_row < int(rows) and 0 <= next_col < int(cols):
            out.append((int(next_row), int(next_col)))
    return out


def _reading_order_region_ids(region_ids: Sequence[str], regions_by_id: Mapping[str, Mapping[str, Any]]) -> List[str]:
    def sort_key(item: str) -> Tuple[float, float, str]:
        region = regions_by_id[str(item)]
        if "row" in region and "col" in region:
            return (float(region["row"]), float(region["col"]), str(item))
        centroid = region.get("centroid_lonlat")
        if isinstance(centroid, Sequence) and not isinstance(centroid, (str, bytes)) and len(centroid) >= 2:
            return (-float(centroid[1]), float(centroid[0]), str(item))
        return (0.0, 0.0, str(item))

    return [
        str(region_id)
        for region_id in sorted(
            [str(region_id) for region_id in region_ids],
            key=sort_key,
        )
    ]


def _grid_pair_support(
    *,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    region_min: int,
) -> List[Tuple[int, int]]:
    return [
        (int(rows), int(cols))
        for rows in range(int(row_min), int(row_max) + 1)
        for cols in range(int(col_min), int(col_max) + 1)
        if int(rows) * int(cols) >= int(region_min)
    ]


def _balanced_int(
    support: Sequence[int],
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> int:
    ordered = [int(value) for value in support]
    if not ordered:
        raise ValueError(f"empty support for {namespace}")
    sample_cursor = params.get("_sample_cursor")
    if sample_cursor is not None:
        offset = abs(int(hash64(0, f"{namespace}.cursor_offset", 0))) % len(ordered)
        return int(ordered[(abs(int(sample_cursor)) + int(offset)) % len(ordered)])
    return int(
        uniform_choice(
            spawn_rng(int(instance_seed), str(namespace)),
            ordered,
            sort_keys=True,
        )
    )


def _choose_random(items: Sequence[Any], *, rng) -> Any:
    if not items:
        raise ValueError("cannot sample from an empty sequence")
    return items[int(rng.randrange(len(items)))]


def _region_sort_key(region_id: str) -> Tuple[int, int | str]:
    text = str(region_id)
    suffix = text.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, text)


def _sample_connected_cells(
    *,
    rows: int,
    cols: int,
    target_count: int,
    rng,
) -> List[Tuple[int, int]]:
    if int(target_count) > int(rows) * int(cols):
        raise ValueError("target_count exceeds grid capacity")
    center = (int(rows) // 2, int(cols) // 2)
    active = {center}
    while len(active) < int(target_count):
        frontier = sorted(
            {
                neighbor
                for cell in active
                for neighbor in _neighbors(cell, rows=int(rows), cols=int(cols))
                if neighbor not in active
            }
        )
        if not frontier:
            raise RuntimeError("failed to grow connected choropleth shape")

        def score(cell: Tuple[int, int]) -> float:
            dr = abs(float(cell[0]) - ((float(rows) - 1.0) / 2.0))
            dc = abs(float(cell[1]) - ((float(cols) - 1.0) / 2.0))
            return float(dr + dc + rng.random() * 1.75)

        frontier = sorted(frontier, key=score)
        pool_size = min(len(frontier), max(2, 1 + len(frontier) // 2))
        active.add(frontier[int(rng.randrange(pool_size))])
    return sorted(active, key=lambda item: (int(item[0]), int(item[1])))


__all__ = [
    "CHOROPLETH_TASK_NAMESPACE",
    "Point",
    "_balanced_int",
    "_choose_random",
    "_grid_pair_support",
    "_grid_points",
    "_neighbors",
    "_polygon_bbox",
    "_polygon_center",
    "_reading_order_region_ids",
    "_region_polygon",
    "_region_sort_key",
    "_sample_connected_cells",
    "_shrink_polygon",
]
