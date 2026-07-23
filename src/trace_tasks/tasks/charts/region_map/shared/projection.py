"""Geographic projection and adjacency helpers for region-map charts."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .assets import load_geographic_map_asset as _load_geographic_map_asset


WORLD_FILTERED_CONTINENTS: Tuple[str, ...] = (
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "South America",
)


def _centroid_lonlat_from_rings(rings: Sequence[Sequence[Sequence[float]]]) -> List[float]:
    points = [
        (float(point[0]), float(point[1]))
        for ring in rings
        for point in ring
        if len(point) >= 2
    ]
    if not points:
        return [0.0, 0.0]
    return [
        round(sum(point[0] for point in points) / float(len(points)), 3),
        round(sum(point[1] for point in points) / float(len(points)), 3),
    ]


def _world_filtered_region_candidates(regions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    allowed = set(WORLD_FILTERED_CONTINENTS)
    return [
        dict(region)
        for region in regions
        if str(region.get("continent") or "") in allowed
    ]


def _border_segment_key(point_a: Sequence[float], point_b: Sequence[float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    a = (round(float(point_a[0]), 3), round(float(point_a[1]), 3))
    b = (round(float(point_b[0]), 3), round(float(point_b[1]), 3))
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def _border_segment_length(segment: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    (x0, y0), (x1, y1) = segment
    return float(math.hypot(float(x1 - x0), float(y1 - y0)))


def _region_boundary_segments(region: Mapping[str, Any]) -> Dict[Tuple[Tuple[float, float], Tuple[float, float]], float]:
    segments: Dict[Tuple[Tuple[float, float], Tuple[float, float]], float] = {}
    for ring in region.get("rings", []):
        if not isinstance(ring, (list, tuple)):
            continue
        points = [point for point in ring if isinstance(point, (list, tuple)) and len(point) >= 2]
        if len(points) < 2:
            continue
        for point_a, point_b in zip(points, points[1:] + points[:1]):
            segment = _border_segment_key(point_a, point_b)
            length = _border_segment_length(segment)
            if float(length) > 0.0:
                segments[segment] = float(length)
    return dict(segments)


@lru_cache(maxsize=None)
def _geographic_shared_border_lengths(map_variant_or_asset_id: str = "world_countries") -> Dict[str, Dict[str, float]]:
    """Return visible shared-boundary lengths between regions in a bundled map."""

    asset = _load_geographic_map_asset(str(map_variant_or_asset_id))
    regions = [
        dict(region)
        for region in asset.get("regions", [])
        if isinstance(region, Mapping)
    ]
    segments_by_id = {
        str(region["region_id"]): _region_boundary_segments(region)
        for region in regions
    }
    adjacency: Dict[str, Dict[str, float]] = {str(region["region_id"]): {} for region in regions}
    for index, region_a in enumerate(regions):
        region_id_a = str(region_a["region_id"])
        segments_a = segments_by_id[region_id_a]
        if not segments_a:
            continue
        keys_a = set(segments_a.keys())
        for region_b in regions[index + 1:]:
            region_id_b = str(region_b["region_id"])
            shared = keys_a.intersection(segments_by_id[region_id_b].keys())
            if not shared:
                continue
            shared_length = float(sum(segments_a[segment] for segment in shared))
            if float(shared_length) <= 0.0:
                continue
            adjacency[region_id_a][region_id_b] = float(shared_length)
            adjacency[region_id_b][region_id_a] = float(shared_length)
    return {str(key): dict(value) for key, value in adjacency.items()}


def _geographic_border_neighbors(
    map_variant_or_asset_id: str = "world_countries",
    *,
    min_shared_length_deg: float,
) -> Dict[str, List[str]]:
    lengths = _geographic_shared_border_lengths(str(map_variant_or_asset_id))
    threshold = float(min_shared_length_deg)
    return {
        str(region_id): sorted(
            str(neighbor_id)
            for neighbor_id, shared_length in neighbor_lengths.items()
            if float(shared_length) >= float(threshold)
        )
        for region_id, neighbor_lengths in lengths.items()
    }


def _world_country_shared_border_lengths() -> Dict[str, Dict[str, float]]:
    return _geographic_shared_border_lengths("world_countries")


def _synthetic_region_adjacency(regions_by_id: Mapping[str, Mapping[str, Any]]) -> Dict[str, List[str]]:
    region_id_by_cell = {
        (int(region["row"]), int(region["col"])): str(region_id)
        for region_id, region in regions_by_id.items()
        if "row" in region and "col" in region
    }
    if not region_id_by_cell:
        return {str(region_id): [] for region_id in regions_by_id}
    max_row = max(row for row, _col in region_id_by_cell)
    max_col = max(col for _row, col in region_id_by_cell)
    adjacency: Dict[str, set[str]] = {str(region_id): set() for region_id in regions_by_id}
    for cell, region_id in region_id_by_cell.items():
        row, col = int(cell[0]), int(cell[1])
        neighbor_cells = [
            (row + delta_row, col + delta_col)
            for delta_row in (-1, 0, 1)
            for delta_col in (-1, 0, 1)
            if (delta_row, delta_col) != (0, 0)
            and 0 <= row + delta_row <= int(max_row)
            and 0 <= col + delta_col <= int(max_col)
        ]
        for neighbor_cell in neighbor_cells:
            neighbor_id = region_id_by_cell.get(neighbor_cell)
            if neighbor_id is not None:
                adjacency[str(region_id)].add(str(neighbor_id))
    return {str(region_id): sorted(values) for region_id, values in adjacency.items()}


def _selected_geographic_region_adjacency(
    regions_by_id: Mapping[str, Mapping[str, Any]],
    *,
    map_variant: str,
    min_shared_length_deg: float,
) -> Dict[str, List[str]]:
    region_id_by_asset_id = {
        str(region.get("asset_region_id")): str(region_id)
        for region_id, region in regions_by_id.items()
        if str(region.get("asset_region_id") or "")
    }
    asset_id_by_region_id = {str(region_id): str(asset_id) for asset_id, region_id in region_id_by_asset_id.items()}
    border_neighbors = _geographic_border_neighbors(
        str(map_variant or "world_countries"),
        min_shared_length_deg=float(min_shared_length_deg),
    )
    adjacency: Dict[str, List[str]] = {}
    for region_id, asset_id in asset_id_by_region_id.items():
        adjacency[str(region_id)] = sorted(
            str(region_id_by_asset_id[str(neighbor_asset_id)])
            for neighbor_asset_id in border_neighbors.get(str(asset_id), [])
            if str(neighbor_asset_id) in region_id_by_asset_id
        )
    return dict(adjacency)


__all__ = [
    "WORLD_FILTERED_CONTINENTS",
    "_border_segment_key",
    "_border_segment_length",
    "_centroid_lonlat_from_rings",
    "_geographic_border_neighbors",
    "_geographic_shared_border_lengths",
    "_region_boundary_segments",
    "_selected_geographic_region_adjacency",
    "_synthetic_region_adjacency",
    "_world_country_shared_border_lengths",
    "_world_filtered_region_candidates",
]
