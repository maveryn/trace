"""Radial hunt board topology and move rules."""

from __future__ import annotations

from typing import Sequence, Tuple

from .state import CENTER, Coord, Edge, RadialHuntBoardSample


def all_coords() -> Tuple[Coord, ...]:
    """Return every playable point on the three-ring radial board."""

    return (CENTER,) + tuple((ring, spoke) for ring in range(1, 4) for spoke in range(6))


def circle_lines() -> Tuple[Tuple[Coord, ...], ...]:
    """Return concentric ring cycles used by movement and captures."""

    return tuple(tuple((ring, spoke) for spoke in range(6)) for ring in range(1, 4))


def diameter_lines() -> Tuple[Tuple[Coord, ...], ...]:
    """Return the three straight diameter paths through the board center."""

    return tuple(
        (
            (3, (axis + 3) % 6),
            (2, (axis + 3) % 6),
            (1, (axis + 3) % 6),
            CENTER,
            (1, axis),
            (2, axis),
            (3, axis),
        )
        for axis in range(3)
    )


def all_lines() -> Tuple[Tuple[Coord, ...], ...]:
    """Return every line family used by radial board rules."""

    return circle_lines() + diameter_lines()


def point_id(coord: Coord) -> str:
    """Return the stable point entity id for one board coordinate."""

    if tuple(coord) == CENTER:
        return "point_center"
    return f"point_ring{int(coord[0])}_spoke{int(coord[1])}"


def piece_id(coord: Coord) -> str:
    """Return the stable non-marked piece entity id for one coordinate."""

    if tuple(coord) == CENTER:
        return "piece_center"
    return f"piece_ring{int(coord[0])}_spoke{int(coord[1])}"


def edge(a: Coord, b: Coord) -> Edge:
    """Return one undirected edge in stable coordinate order."""

    ordered = tuple(sorted((tuple(a), tuple(b))))  # type: ignore[arg-type]
    return ordered  # type: ignore[return-value]


def all_possible_edges() -> Tuple[Edge, ...]:
    """Return every adjacency edge drawn on the radial board."""

    seen: set[Edge] = set()
    for line in circle_lines():
        for index, coord in enumerate(line):
            seen.add(edge(coord, line[(index + 1) % len(line)]))
    for line in diameter_lines():
        for index in range(len(line) - 1):
            seen.add(edge(line[index], line[index + 1]))
    return tuple(sorted(seen))


def neighbors(coord: Coord) -> Tuple[Coord, ...]:
    """Return empty-move neighbor coordinates for one board point."""

    resolved = tuple(coord)
    out: set[Coord] = set()
    for candidate in all_possible_edges():
        if candidate[0] == resolved:
            out.add(candidate[1])
        elif candidate[1] == resolved:
            out.add(candidate[0])
    return tuple(sorted(out))


def legal_destinations(*, marked_coord: Coord, occupied_coords: Sequence[Coord]) -> Tuple[Coord, ...]:
    """Return empty adjacent points reachable by one ordinary move."""

    occupied = {tuple(coord) for coord in occupied_coords}
    return tuple(sorted(coord for coord in neighbors(marked_coord) if coord not in occupied))


def capture_paths(marked_coord: Coord) -> Tuple[Tuple[Coord, Coord], ...]:
    """Return all possible landing/captured coordinate pairs from one point."""

    resolved = tuple(marked_coord)
    out: set[Tuple[Coord, Coord]] = set()
    for line in circle_lines():
        if resolved not in line:
            continue
        index = line.index(resolved)
        for direction in (-1, 1):
            captured = line[(index + direction) % len(line)]
            destination = line[(index + (2 * direction)) % len(line)]
            out.add((destination, captured))
    for line in diameter_lines():
        if resolved not in line:
            continue
        index = line.index(resolved)
        for direction in (-1, 1):
            captured_index = index + direction
            destination_index = index + (2 * direction)
            if 0 <= captured_index < len(line) and 0 <= destination_index < len(line):
                out.add((line[destination_index], line[captured_index]))
    return tuple(sorted(out))


def capture_destinations(*, marked_coord: Coord, occupied_coords: Sequence[Coord]) -> Tuple[Coord, ...]:
    """Return empty landing points that capture an adjacent occupied point."""

    occupied = {tuple(coord) for coord in occupied_coords}
    out: list[Coord] = []
    for destination, captured in capture_paths(marked_coord):
        if tuple(captured) in occupied and tuple(destination) not in occupied:
            out.append(destination)
    return tuple(sorted(out))


def validate_radial_hunt_board_sample(sample: RadialHuntBoardSample) -> None:
    """Validate symbolic sample consistency before rendering and output."""

    coords = set(all_coords())
    occupied = {tuple(coord) for coord in sample.occupied_coords}
    annotations = {tuple(coord) for coord in sample.annotation_coords}
    if tuple(sample.marked_coord) not in coords:
        raise ValueError("marked radial hunt board coordinate is not playable")
    if tuple(sample.marked_coord) not in occupied:
        raise ValueError("marked radial hunt board coordinate must be occupied")
    if not occupied <= coords:
        raise ValueError("radial hunt board occupied coordinates include non-playable points")
    if not annotations <= coords:
        raise ValueError("radial hunt board annotation coordinates include non-playable points")
    if int(sample.answer) != len(sample.annotation_coords):
        raise ValueError("radial hunt board answer must equal annotation count")


__all__ = [
    "all_coords",
    "all_lines",
    "all_possible_edges",
    "capture_destinations",
    "capture_paths",
    "circle_lines",
    "diameter_lines",
    "edge",
    "legal_destinations",
    "neighbors",
    "piece_id",
    "point_id",
    "validate_radial_hunt_board_sample",
]
