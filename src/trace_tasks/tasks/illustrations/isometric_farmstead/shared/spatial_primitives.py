"""Shared spatial primitives for isometric farmstead tasks."""

from __future__ import annotations

from typing import Mapping, Sequence

from .state import IsoFarmsteadScene, IsoFarmsteadTile


DEFAULT_SUPPORTED_LEVELS = (0, 1, 2)


def tile_inside_canvas(tile: IsoFarmsteadTile, *, width: int, height: int) -> bool:
    """Return true when the rendered tile bbox is fully visible."""

    return (
        0.0 <= float(tile.bbox_xyxy[0])
        and float(tile.bbox_xyxy[2]) <= float(width)
        and 0.0 <= float(tile.bbox_xyxy[1])
        and float(tile.bbox_xyxy[3]) <= float(height)
    )


def tile_label_bbox(tile: IsoFarmsteadTile) -> tuple[float, float, float, float]:
    """Approximate the letter label bbox anchored to a terrain tile."""

    cx, cy = tile.center_xy
    return (float(cx) - 18.0, float(cy) - 15.0, float(cx) + 18.0, float(cy) + 13.0)


def farmer_reference_bbox(tile: IsoFarmsteadTile) -> tuple[float, float, float, float]:
    """Approximate the farmer reference sprite bbox for clearance checks."""

    cx, cy = tile.center_xy
    tile_w = float(tile.bbox_xyxy[2]) - float(tile.bbox_xyxy[0])
    tile_h = float(tile.bbox_xyxy[3]) - float(tile.bbox_xyxy[1])
    width = tile_w * 0.5
    height = tile_w * 0.7
    return (cx - width * 0.5, cy - height + tile_h * 0.15, cx + width * 0.5, cy + tile_h * 0.15)


def boxes_intersect(left: Sequence[float], right: Sequence[float], *, pad: float = 0.0) -> bool:
    """Return true when two bboxes overlap after optional padding."""

    return (
        float(left[0]) - float(pad) < float(right[2])
        and float(left[2]) + float(pad) > float(right[0])
        and float(left[1]) - float(pad) < float(right[3])
        and float(left[3]) + float(pad) > float(right[1])
    )


def label_clear_of_context(
    scene: IsoFarmsteadScene,
    tile: IsoFarmsteadTile,
    *,
    blocking_boxes: Sequence[Sequence[float]] = (),
) -> bool:
    """Return true when a tile letter will not overlap context objects or extra boxes."""

    label_box = tile_label_bbox(tile)
    if any(boxes_intersect(label_box, box, pad=8.0) for box in blocking_boxes):
        return False
    return not any(boxes_intersect(label_box, entity.bbox_xyxy, pad=8.0) for entity in scene.entities)


def same_level_component_sizes(scene: IsoFarmsteadScene) -> dict[str, int]:
    """Compute connected component sizes for same-elevation terrain tiles."""

    tiles_by_cell = {(int(tile.col), int(tile.row)): tile for tile in scene.tiles}
    component_sizes: dict[str, int] = {}
    visited: set[str] = set()
    for tile in scene.tiles:
        if str(tile.tile_id) in visited:
            continue
        stack = [tile]
        component: list[IsoFarmsteadTile] = []
        visited.add(str(tile.tile_id))
        while stack:
            current = stack.pop()
            component.append(current)
            for delta_col, delta_row in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = tiles_by_cell.get((int(current.col) + delta_col, int(current.row) + delta_row))
                if neighbor is None:
                    continue
                if int(neighbor.level) != int(current.level):
                    continue
                if str(neighbor.tile_id) in visited:
                    continue
                visited.add(str(neighbor.tile_id))
                stack.append(neighbor)
        for component_tile in component:
            component_sizes[str(component_tile.tile_id)] = len(component)
    return component_sizes


def has_same_level_neighbor_support(
    tiles_by_cell: Mapping[tuple[int, int], IsoFarmsteadTile],
    tile: IsoFarmsteadTile,
) -> bool:
    """Require same-level support on both grid axes so labels sit on visually stable terrain."""

    has_horizontal = any(
        (neighbor := tiles_by_cell.get((int(tile.col) + delta_col, int(tile.row)))) is not None
        and int(neighbor.level) == int(tile.level)
        for delta_col in (-1, 1)
    )
    has_vertical = any(
        (neighbor := tiles_by_cell.get((int(tile.col), int(tile.row) + delta_row))) is not None
        and int(neighbor.level) == int(tile.level)
        for delta_row in (-1, 1)
    )
    return bool(has_horizontal and has_vertical)


def eligible_label_tiles_by_level(
    scene: IsoFarmsteadScene,
    *,
    blocking_boxes: Sequence[Sequence[float]] = (),
    exclude_tile_ids: Sequence[str] = (),
    exclude_unsafe_low_adjacent_higher: bool = False,
    min_component_size: int = 6,
) -> dict[int, list[IsoFarmsteadTile]]:
    """Return label-safe grass candidate tiles grouped by elevation level."""

    eligible_ids = {str(value) for value in scene.trace.get("eligible_tile_ids", [])}
    unsafe_ids = {str(value) for value in scene.trace.get("object_unsafe_low_adjacent_higher_tile_ids", [])}
    active_levels = tuple(int(level) for level in scene.trace.get("levels", DEFAULT_SUPPORTED_LEVELS))
    excluded = {str(value) for value in exclude_tile_ids}
    by_level: dict[int, list[IsoFarmsteadTile]] = {int(level): [] for level in active_levels}
    width, height = scene.image.size
    tiles_by_cell = {(int(tile.col), int(tile.row)): tile for tile in scene.tiles}
    component_sizes = same_level_component_sizes(scene)
    for tile in scene.tiles:
        tile_id = str(tile.tile_id)
        if tile_id in excluded:
            continue
        if bool(exclude_unsafe_low_adjacent_higher) and tile_id in unsafe_ids:
            continue
        if tile_id not in eligible_ids:
            continue
        if str(tile.terrain) != "grass" or not bool(tile.metadata.get("candidate_allowed", False)):
            continue
        if not tile_inside_canvas(tile, width=width, height=height):
            continue
        if not label_clear_of_context(scene, tile, blocking_boxes=blocking_boxes):
            continue
        if int(component_sizes.get(tile_id, 0)) < int(min_component_size):
            continue
        if not has_same_level_neighbor_support(tiles_by_cell, tile):
            continue
        by_level[int(tile.level)].append(tile)
    return {level: sorted(tiles, key=lambda item: (item.row, item.col)) for level, tiles in by_level.items()}


def label_clear_of_candidate_tiles(candidate_tiles: Sequence[IsoFarmsteadTile], tile: IsoFarmsteadTile) -> bool:
    """Return true when a tile label will not collide with existing candidate labels."""

    label_box = tile_label_bbox(tile)
    return not any(boxes_intersect(label_box, tile_label_bbox(candidate), pad=6.0) for candidate in candidate_tiles)


__all__ = [
    "boxes_intersect",
    "eligible_label_tiles_by_level",
    "farmer_reference_bbox",
    "has_same_level_neighbor_support",
    "label_clear_of_candidate_tiles",
    "label_clear_of_context",
    "same_level_component_sizes",
    "tile_inside_canvas",
    "tile_label_bbox",
]
