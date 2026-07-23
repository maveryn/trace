"""Movement graph rules for irregular-link-board scenes."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from .state import Coord, Edge


def all_coords(board_size: int) -> Tuple[Coord, ...]:
    """Return all playable point coordinates for a square point lattice."""

    return tuple((row, col) for row in range(int(board_size)) for col in range(int(board_size)))


def point_id(coord: Coord) -> str:
    """Return the stable rendered-point id for one coordinate."""

    return f"point_r{int(coord[0])}_c{int(coord[1])}"


def piece_id(coord: Coord) -> str:
    """Return the stable rendered-piece id for one coordinate."""

    return f"piece_r{int(coord[0])}_c{int(coord[1])}"


def edge(a: Coord, b: Coord) -> Edge:
    """Normalize an undirected edge endpoint pair."""

    ordered = tuple(sorted((tuple(a), tuple(b))))  # type: ignore[arg-type]
    return ordered  # type: ignore[return-value]


def base_lattice_edges(board_size: int) -> Tuple[Edge, ...]:
    """Return a Fanorona-like link lattice with no non-node diagonal crossings."""

    size = int(board_size)
    seen: set[Edge] = set()
    for row in range(size):
        for col in range(size):
            if col + 1 < size:
                seen.add(edge((row, col), (row, col + 1)))
            if row + 1 < size:
                seen.add(edge((row, col), (row + 1, col)))

    for row in range(size - 1):
        for col in range(size - 1):
            if (row + col) % 2 == 0:
                seen.add(edge((row, col), (row + 1, col + 1)))
            else:
                seen.add(edge((row + 1, col), (row, col + 1)))
    return tuple(sorted(seen))


def neighbors(coord: Coord, board_size: int) -> Tuple[Coord, ...]:
    """Return every lattice neighbor connected in the full base graph."""

    target = (int(coord[0]), int(coord[1]))
    out: list[Coord] = []
    for a, b in base_lattice_edges(int(board_size)):
        if a == target:
            out.append(b)
        elif b == target:
            out.append(a)
    return tuple(sorted(out))


def all_possible_edges(board_size: int) -> Tuple[Edge, ...]:
    """Return every possible drawn link before scene-specific removal."""

    return base_lattice_edges(int(board_size))


def legal_destinations(
    *,
    marked_coord: Coord,
    occupied_coords: Sequence[Coord],
    edges: Sequence[Edge],
    board_size: int,
) -> Tuple[Coord, ...]:
    """Return empty one-step destinations linked to the marked piece."""

    occupied = {tuple(coord) for coord in occupied_coords}
    edge_set = {tuple(link) for link in edges}
    out: list[Coord] = []
    for neighbor in neighbors(marked_coord, int(board_size)):
        if tuple(neighbor) in occupied:
            continue
        if edge(marked_coord, neighbor) in edge_set:
            out.append(neighbor)
    return tuple(sorted(out))


def capture_paths(marked_coord: Coord, board_size: int) -> Tuple[Tuple[Coord, Coord], ...]:
    """Return all geometric jump paths as destination/captured coordinate pairs."""

    base_edges = set(all_possible_edges(int(board_size)))
    out: list[Tuple[Coord, Coord]] = []
    for captured in neighbors(marked_coord, int(board_size)):
        dr = int(captured[0]) - int(marked_coord[0])
        dc = int(captured[1]) - int(marked_coord[1])
        destination = (int(captured[0]) + dr, int(captured[1]) + dc)
        if not (0 <= destination[0] < int(board_size) and 0 <= destination[1] < int(board_size)):
            continue
        if edge(captured, destination) not in base_edges:
            continue
        out.append((destination, captured))
    return tuple(sorted(out))


def capture_destinations(
    *,
    marked_coord: Coord,
    occupied_coords: Sequence[Coord],
    edges: Sequence[Edge],
    board_size: int,
) -> Tuple[Coord, ...]:
    """Return empty landing points where the marked piece can jump-capture."""

    occupied = {tuple(coord) for coord in occupied_coords}
    edge_set = {tuple(link) for link in edges}
    out: list[Coord] = []
    for destination, captured in capture_paths(marked_coord, int(board_size)):
        if tuple(destination) in occupied:
            continue
        if tuple(captured) not in occupied:
            continue
        if edge(marked_coord, captured) not in edge_set:
            continue
        if edge(captured, destination) not in edge_set:
            continue
        out.append(destination)
    return tuple(sorted(out))


def serialize_coords(coords: Iterable[Coord]) -> list[list[int]]:
    """Serialize row/column coordinates for trace payloads."""

    return [[int(row), int(col)] for row, col in coords]


__all__ = [
    "all_coords",
    "all_possible_edges",
    "base_lattice_edges",
    "capture_destinations",
    "capture_paths",
    "edge",
    "legal_destinations",
    "neighbors",
    "piece_id",
    "point_id",
    "serialize_coords",
]
