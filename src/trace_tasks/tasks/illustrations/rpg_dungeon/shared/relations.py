"""Graph relations for top-down RPG dungeon layouts."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

from .state import Tile


def reachable_tiles(
    floor_tiles: Iterable[Tile],
    *,
    blocked_tiles: Iterable[Tile],
    start_tile: Tile,
) -> tuple[Tile, ...]:
    """Return floor tiles reachable from ``start_tile`` by four-neighbor movement."""

    floor = {tuple(int(value) for value in tile) for tile in floor_tiles}
    blocked = {tuple(int(value) for value in tile) for tile in blocked_tiles}
    start = (int(start_tile[0]), int(start_tile[1]))
    if start not in floor or start in blocked:
        return tuple()
    visited: set[Tile] = {start}
    queue: deque[Tile] = deque([start])
    while queue:
        x, y = queue.popleft()
        for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if neighbor in visited or neighbor not in floor or neighbor in blocked:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return tuple(sorted(visited, key=lambda tile: (tile[1], tile[0])))


def reachable_entity_ids(
    *,
    entity_tile_map: dict[str, Tile],
    reachable_tile_set: Sequence[Tile],
) -> tuple[str, ...]:
    """Return ids whose anchor tile is in the reachable tile set."""

    reachable = {tuple(int(value) for value in tile) for tile in reachable_tile_set}
    return tuple(
        str(entity_id)
        for entity_id, tile in sorted(entity_tile_map.items())
        if tuple(int(value) for value in tile) in reachable
    )


__all__ = ["reachable_entity_ids", "reachable_tiles"]

