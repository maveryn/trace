"""Helpers for checking public tile pixel annotation in tests."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _anchor_points(trace: Mapping[str, Any]) -> dict[str, list[float]]:
    anchors = dict(dict(trace.get("render_map", {})).get("anchors", {}))
    out: dict[str, list[float]] = {}
    for anchor_id, anchor in anchors.items():
        if not str(anchor_id).startswith("cell_") or not isinstance(anchor, Mapping):
            continue
        point = anchor.get("point")
        if isinstance(point, (list, tuple)) and len(point) == 2:
            out[str(anchor_id)] = [float(point[0]), float(point[1])]
    return out


def tile_ids_from_points(trace: Mapping[str, Any], points: Sequence[Sequence[float]]) -> list[str]:
    """Map public tile-center pixel points back to stable cell ids for assertions."""

    anchors = _anchor_points(trace)
    ids: list[str] = []
    for point in points:
        px = [float(point[0]), float(point[1])]
        match = None
        for anchor_id, anchor_point in anchors.items():
            if abs(float(anchor_point[0]) - px[0]) <= 1e-3 and abs(float(anchor_point[1]) - px[1]) <= 1e-3:
                match = str(anchor_id)
                break
        if match is None:
            raise AssertionError(f"public tile annotation point has no matching cell anchor: {point}")
        ids.append(match)
    return ids


def tile_coords_from_points(trace: Mapping[str, Any], points: Sequence[Sequence[float]]) -> list[list[int]]:
    """Map public tile-center pixel points back to row/column coords for assertions."""

    coords: list[list[int]] = []
    for tile_id in tile_ids_from_points(trace, points):
        _, row, col = str(tile_id).split("_", 2)
        coords.append([int(row), int(col)])
    return coords
