"""Movement-cost relations for the RPG tactical map scene."""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from typing import Mapping

from .state import RpgTacticalTile, TileCoord


TERRAIN_GRASS = "grass"
TERRAIN_ROAD = "road"
TERRAIN_FOREST = "forest"
TERRAIN_WATER = "water"
TERRAIN_BRIDGE = "bridge"
TERRAIN_MOUNTAIN = "mountain"

TERRAIN_MOVEMENT_COSTS: Mapping[str, int] = {
    TERRAIN_GRASS: 1,
    TERRAIN_ROAD: 1,
    TERRAIN_FOREST: 2,
    TERRAIN_BRIDGE: 1,
    TERRAIN_MOUNTAIN: 3,
}
BLOCKED_TERRAINS: frozenset[str] = frozenset({TERRAIN_WATER})


def movement_cost_for_terrain(terrain: str) -> int | None:
    """Return the movement cost for one terrain, or None when blocked."""

    terrain_id = str(terrain)
    if terrain_id in BLOCKED_TERRAINS:
        return None
    return TERRAIN_MOVEMENT_COSTS.get(terrain_id)


def is_passable_terrain(terrain: str) -> bool:
    """Return whether a unit can enter one terrain type."""

    return movement_cost_for_terrain(str(terrain)) is not None


def orthogonal_neighbors(coord: TileCoord) -> tuple[TileCoord, TileCoord, TileCoord, TileCoord]:
    """Return up/down/left/right neighbor coordinates."""

    row, col = int(coord[0]), int(coord[1])
    return ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))


def connected_passable_tile_ids(
    tiles_by_coord: Mapping[TileCoord, RpgTacticalTile],
    *,
    start_coord: TileCoord,
) -> set[str]:
    """Return passable tile ids connected to start by orthogonal moves."""

    start = (int(start_coord[0]), int(start_coord[1]))
    if start not in tiles_by_coord:
        raise ValueError(f"start tile {start} is not in the tactical map")
    if not bool(tiles_by_coord[start].passable):
        raise ValueError(f"start tile {start} is not passable")
    seen: set[TileCoord] = {start}
    queue: deque[TileCoord] = deque([start])
    while queue:
        coord = queue.popleft()
        for neighbor in orthogonal_neighbors(coord):
            if neighbor in seen:
                continue
            tile = tiles_by_coord.get(neighbor)
            if tile is None or not bool(tile.passable):
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return {str(tiles_by_coord[coord].tile_id) for coord in seen}


def shortest_movement_costs(
    tiles_by_coord: Mapping[TileCoord, RpgTacticalTile],
    *,
    start_coord: TileCoord,
) -> dict[str, int]:
    """Run Dijkstra over orthogonal moves using tile-entry movement costs."""

    start = (int(start_coord[0]), int(start_coord[1]))
    if start not in tiles_by_coord:
        raise ValueError(f"start tile {start} is not in the tactical map")
    if not bool(tiles_by_coord[start].passable):
        raise ValueError(f"start tile {start} is not passable")

    best_by_coord: dict[TileCoord, int] = {start: 0}
    heap: list[tuple[int, TileCoord]] = [(0, start)]
    while heap:
        current_cost, coord = heappop(heap)
        if int(current_cost) != int(best_by_coord.get(coord, current_cost)):
            continue
        for neighbor in orthogonal_neighbors(coord):
            tile = tiles_by_coord.get(neighbor)
            if tile is None or not bool(tile.passable) or tile.movement_cost is None:
                continue
            new_cost = int(current_cost) + int(tile.movement_cost)
            if new_cost < int(best_by_coord.get(neighbor, 10**9)):
                best_by_coord[neighbor] = new_cost
                heappush(heap, (new_cost, neighbor))
    return {
        str(tiles_by_coord[coord].tile_id): int(cost)
        for coord, cost in sorted(best_by_coord.items(), key=lambda item: (item[0][0], item[0][1]))
    }


def shortest_movement_costs_and_paths(
    tiles_by_coord: Mapping[TileCoord, RpgTacticalTile],
    *,
    start_coord: TileCoord,
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Run Dijkstra and return costs plus one deterministic shortest path per reachable tile."""

    start = (int(start_coord[0]), int(start_coord[1]))
    if start not in tiles_by_coord:
        raise ValueError(f"start tile {start} is not in the tactical map")
    if not bool(tiles_by_coord[start].passable):
        raise ValueError(f"start tile {start} is not passable")

    best_by_coord: dict[TileCoord, int] = {start: 0}
    previous_by_coord: dict[TileCoord, TileCoord] = {}
    heap: list[tuple[int, TileCoord]] = [(0, start)]
    while heap:
        current_cost, coord = heappop(heap)
        if int(current_cost) != int(best_by_coord.get(coord, current_cost)):
            continue
        for neighbor in orthogonal_neighbors(coord):
            tile = tiles_by_coord.get(neighbor)
            if tile is None or not bool(tile.passable) or tile.movement_cost is None:
                continue
            new_cost = int(current_cost) + int(tile.movement_cost)
            if new_cost < int(best_by_coord.get(neighbor, 10**9)):
                best_by_coord[neighbor] = new_cost
                previous_by_coord[neighbor] = coord
                heappush(heap, (new_cost, neighbor))

    costs_by_id = {
        str(tiles_by_coord[coord].tile_id): int(cost)
        for coord, cost in sorted(best_by_coord.items(), key=lambda item: (item[0][0], item[0][1]))
    }
    paths_by_id: dict[str, list[str]] = {}
    for coord in sorted(best_by_coord, key=lambda item: (item[0], item[1])):
        path_coords = [coord]
        current = coord
        while current != start:
            current = previous_by_coord[current]
            path_coords.append(current)
        path_coords.reverse()
        paths_by_id[str(tiles_by_coord[coord].tile_id)] = [
            str(tiles_by_coord[path_coord].tile_id)
            for path_coord in path_coords
        ]
    return costs_by_id, paths_by_id


__all__ = [
    "BLOCKED_TERRAINS",
    "TERRAIN_BRIDGE",
    "TERRAIN_FOREST",
    "TERRAIN_GRASS",
    "TERRAIN_MOUNTAIN",
    "TERRAIN_MOVEMENT_COSTS",
    "TERRAIN_ROAD",
    "TERRAIN_WATER",
    "is_passable_terrain",
    "connected_passable_tile_ids",
    "movement_cost_for_terrain",
    "orthogonal_neighbors",
    "shortest_movement_costs",
    "shortest_movement_costs_and_paths",
]
