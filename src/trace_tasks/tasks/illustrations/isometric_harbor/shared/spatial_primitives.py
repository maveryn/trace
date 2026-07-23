"""Shared spatial helpers for isometric harbor tasks."""

from __future__ import annotations

from typing import Sequence

from .state import IsoHarborScene, IsoHarborTile


def rounded_bbox(bbox: Sequence[float]) -> list[float]:
    """Return one rounded bbox in final pixel coordinates."""

    return [round(float(value), 3) for value in bbox[:4]]


def bbox_inside_canvas(bbox: Sequence[float], *, width: int, height: int) -> bool:
    """Return true when a bbox is fully visible inside the canvas."""

    return (
        0.0 <= float(bbox[0])
        and float(bbox[2]) <= float(width)
        and 0.0 <= float(bbox[1])
        and float(bbox[3]) <= float(height)
        and float(bbox[0]) < float(bbox[2])
        and float(bbox[1]) < float(bbox[3])
    )


def dock_tiles(scene: IsoHarborScene) -> tuple[IsoHarborTile, ...]:
    """Return all dock tiles in deterministic top-to-bottom order."""

    return tuple(
        sorted(
            (tile for tile in scene.tiles if str(tile.terrain) == "dock"),
            key=lambda tile: (int(tile.row), int(tile.col)),
        )
    )


def dock_is_connected(scene: IsoHarborScene) -> bool:
    """Return whether all dock tiles form one 4-neighbor component."""

    tiles = dock_tiles(scene)
    if not tiles:
        return False
    cells = {(int(tile.col), int(tile.row)) for tile in tiles}
    stack = [next(iter(cells))]
    visited = {stack[0]}
    while stack:
        col, row = stack.pop()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (int(col) + dc, int(row) + dr)
            if neighbor in cells and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return visited == cells


__all__ = [
    "bbox_inside_canvas",
    "dock_is_connected",
    "dock_tiles",
    "rounded_bbox",
]
