"""Scene-local room and door relation primitives for RPG house layouts."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from .state import RpgHouseDoor

DoorEdgeMap = Mapping[str, Sequence[tuple[str, str]]]


def door_edges(doors: Sequence[Any]) -> dict[str, list[tuple[str, str]]]:
    """Return room adjacency with door ids for each undirected doorway."""

    edges: dict[str, list[tuple[str, str]]] = {}
    for door in doors:
        room_a = str(door.room_a_id)
        room_b = str(door.room_b_id)
        door_id = str(door.door_id)
        edges.setdefault(room_a, []).append((room_b, door_id))
        edges.setdefault(room_b, []).append((room_a, door_id))
    return {room_id: sorted(values) for room_id, values in edges.items()}


def connected_component(edges: DoorEdgeMap, *, start_room_id: str) -> set[str]:
    """Return all rooms connected to a start room through the provided edge map."""

    seen = {str(start_room_id)}
    queue = [str(start_room_id)]
    while queue:
        room_id = queue.pop(0)
        for other_room_id, _door_id in edges.get(room_id, ()):
            if str(other_room_id) in seen:
                continue
            seen.add(str(other_room_id))
            queue.append(str(other_room_id))
    return seen


def room_graph(doors: Sequence[RpgHouseDoor]) -> dict[str, list[dict[str, str]]]:
    """Serialize door adjacency for trace metadata."""

    graph: dict[str, list[dict[str, str]]] = {}
    state_by_door_id = {str(door.door_id): str(door.state) for door in doors}
    for room_id, neighbors in door_edges(doors).items():
        graph[str(room_id)] = [
            {
                "room_id": str(other_room_id),
                "door_id": str(door_id),
                "state": str(state_by_door_id[str(door_id)]),
            }
            for other_room_id, door_id in neighbors
        ]
    return graph


def reachable_room_ids(doors: Sequence[RpgHouseDoor], *, start_room_id: str) -> tuple[str, ...]:
    """Return rooms reachable from a start room through currently open doors only."""

    open_edges = door_edges([door for door in doors if str(door.state) == "open"])
    return tuple(sorted(connected_component(open_edges, start_room_id=str(start_room_id))))


def adjacent_room_ids(doors: Sequence[RpgHouseDoor], *, start_room_id: str) -> tuple[str, ...]:
    """Return rooms that share any direct doorway with the start room."""

    return tuple(sorted(str(room_id) for room_id, _door_id in door_edges(doors).get(str(start_room_id), ())))


def door_id_between(edges: DoorEdgeMap, *, room_a_id: str, room_b_id: str) -> str:
    """Return the door id connecting two adjacent rooms."""

    target = str(room_b_id)
    for other_room_id, door_id in edges.get(str(room_a_id), ()):
        if str(other_room_id) == target:
            return str(door_id)
    raise ValueError(f"rooms are not adjacent: {room_a_id!r}, {room_b_id!r}")


def grow_reachable_subset(
    *,
    edges: DoorEdgeMap,
    start_room_id: str,
    target_count: int,
    rng: random.Random,
) -> tuple[tuple[str, ...], frozenset[str]]:
    """Pick a connected room subset and the doors that make exactly that subset reachable."""

    selected_rooms = {str(start_room_id)}
    open_door_ids: set[str] = set()
    while len(selected_rooms) < int(target_count) + 1:
        frontier = sorted(
            (str(room_id), str(other_room_id), str(door_id))
            for room_id in selected_rooms
            for other_room_id, door_id in edges.get(room_id, ())
            if str(other_room_id) not in selected_rooms
        )
        if not frontier:
            raise ValueError(f"could not grow reachable subset to count {target_count}")
        _room_id, other_room_id, door_id = frontier[int(rng.randrange(len(frontier)))]
        selected_rooms.add(str(other_room_id))
        open_door_ids.add(str(door_id))
    reachable_ids = tuple(sorted(room_id for room_id in selected_rooms if room_id != str(start_room_id)))
    return reachable_ids, frozenset(open_door_ids)


__all__ = [
    "adjacent_room_ids",
    "connected_component",
    "door_edges",
    "door_id_between",
    "grow_reachable_subset",
    "reachable_room_ids",
    "room_graph",
]
