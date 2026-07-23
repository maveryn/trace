"""Pure Battleship geometry and validation helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

from .state import (
    FLEET_SHAPES,
    SHAPE_OPTION_LABELS,
    BattleshipSample,
    Coord,
    sorted_coords,
)


def normalize_offsets(coords: Iterable[Coord]) -> Tuple[Coord, ...]:
    """Normalize a set of shape cells to top-left origin."""

    items = [(int(row), int(col)) for row, col in coords]
    if not items:
        return tuple()
    min_row = min(row for row, _col in items)
    min_col = min(col for _row, col in items)
    return tuple(sorted((row - min_row, col - min_col) for row, col in items))


def shape_orientations(offsets: Sequence[Coord]) -> Tuple[Tuple[Coord, ...], ...]:
    """Return unique right-angle rotations for one fleet shape."""

    base = normalize_offsets(offsets)
    orientations = set()
    current = tuple(base)
    for _ in range(4):
        normalized = normalize_offsets(current)
        orientations.add(normalized)
        current = tuple((col, -row) for row, col in normalized)
    return tuple(sorted(orientations))


def fleet_shape_by_id() -> Dict[str, object]:
    """Return fleet shapes keyed by stable id."""

    return {shape.shape_id: shape for shape in FLEET_SHAPES}


def fleet_orientation_lookup() -> Dict[Tuple[Coord, ...], Tuple[str, ...]]:
    """Return normalized shape offsets mapped to matching fleet shape ids."""

    lookup: Dict[Tuple[Coord, ...], list[str]] = {}
    for shape in FLEET_SHAPES:
        for oriented in shape_orientations(shape.offsets):
            lookup.setdefault(oriented, []).append(str(shape.shape_id))
    return {key: tuple(values) for key, values in lookup.items()}


def matching_fleet_shape_ids(coords: Sequence[Coord]) -> Tuple[str, ...]:
    """Return fleet shape ids whose geometry exactly matches the given cells."""

    return fleet_orientation_lookup().get(normalize_offsets(coords), tuple())


def ship_size_for_shape_id(shape_id: str) -> int:
    """Return the number of cells in one fleet shape."""

    shape = fleet_shape_by_id()[str(shape_id)]
    return len(tuple(shape.offsets))  # type: ignore[attr-defined]


def cell_status_answer_support_for_ship(shape_id: str) -> Tuple[int, ...]:
    """Return feasible hit/unhit cell counts for one queried fleet shape."""

    return tuple(range(ship_size_for_shape_id(str(shape_id)) + 1))


def candidate_completes_target_shape(
    *,
    candidate: Coord,
    target_hit_coords: Sequence[Coord],
    target_shape_id: str,
) -> bool:
    """Return whether adding candidate to target hits completes the target ship shape."""

    completed = sorted_coords([*tuple(target_hit_coords), (int(candidate[0]), int(candidate[1]))])
    return str(target_shape_id) in matching_fleet_shape_ids(completed)


def validate_battleship_scene(sample: BattleshipSample) -> None:
    """Validate query-neutral Battleship scene geometry and shot state."""

    ship_ids = [ship.ship_id for ship in sample.ship_placements]
    if len(ship_ids) != len(set(ship_ids)):
        raise ValueError("Battleship ship ids must be unique")
    if len(sample.ship_placements) != len(FLEET_SHAPES):
        raise ValueError("Battleship sample must place every fleet ship exactly once")
    expected_shape_ids = {shape.shape_id for shape in FLEET_SHAPES}
    placed_shape_ids = {ship.shape_id for ship in sample.ship_placements}
    if placed_shape_ids != expected_shape_ids:
        raise ValueError("Battleship placed fleet shape ids must match the fleet exactly")

    ship_cells: set[Coord] = set()
    for ship in sample.ship_placements:
        if set(ship.coords) & ship_cells:
            raise ValueError("Battleship ships cannot overlap")
        ship_cells.update(ship.coords)
        if str(ship.shape_id) not in matching_fleet_shape_ids(ship.coords):
            raise ValueError("Battleship ship placement geometry does not match its declared shape")
        hit_subset = set(ship.hit_coords)
        if not hit_subset <= set(ship.coords):
            raise ValueError("Battleship ship hits must be a subset of that ship's cells")
        if bool(ship.is_sunk) != (hit_subset == set(ship.coords)):
            raise ValueError("Battleship sunk flag must mean every ship cell is hit")

    hit_set = set(sample.hit_coords)
    if len(hit_set) != len(sample.hit_coords):
        raise ValueError("Battleship hit coordinates must be unique")
    if not hit_set <= ship_cells:
        raise ValueError("Battleship hits must occur only on placed ship cells")
    if set(sample.miss_coords) & hit_set:
        raise ValueError("Battleship miss coordinates cannot overlap hit coordinates")
    if set(sample.miss_coords) & ship_cells:
        raise ValueError("Battleship misses must occur only on water cells")

    sunk_ships = [ship for ship in sample.ship_placements if bool(ship.is_sunk)]
    partial_ships = [
        ship
        for ship in sample.ship_placements
        if bool(ship.hit_coords) and not bool(ship.is_sunk)
    ]
    untouched_ships = [ship for ship in sample.ship_placements if not bool(ship.hit_coords)]
    if len(sunk_ships) != int(sample.sunk_ship_count):
        raise ValueError("Battleship sunk_ship_count does not equal sunk ships")
    if len(partial_ships) != int(sample.partial_ship_count):
        raise ValueError("Battleship partial_ship_count does not equal partially hit ships")
    if len(untouched_ships) != int(sample.untouched_ship_count):
        raise ValueError("Battleship untouched_ship_count does not equal untouched ships")

    if sample.candidate_options:
        labels = [str(option.label) for option in sample.candidate_options]
        expected_labels = ["A", "B", "C", "D", "E", "F"][: len(labels)]
        if labels != expected_labels:
            raise ValueError("Battleship candidate labels must be ordered A-prefix labels")
        candidate_coords = [(int(option.coord[0]), int(option.coord[1])) for option in sample.candidate_options]
        if len(candidate_coords) != len(set(candidate_coords)):
            raise ValueError("Battleship candidate coords must be unique")
        for row, col in candidate_coords:
            if not (0 <= int(row) < int(sample.board_size) and 0 <= int(col) < int(sample.board_size)):
                raise ValueError("Battleship candidate coords must be inside the board")

    if sample.shape_options:
        labels = [str(option.label) for option in sample.shape_options]
        expected_labels = list(SHAPE_OPTION_LABELS[: len(labels)])
        if labels != expected_labels:
            raise ValueError("Battleship shape-option labels must be ordered A-prefix labels")
        shape_ids = [str(option.shape_id) for option in sample.shape_options]
        valid_shape_ids = {shape.shape_id for shape in FLEET_SHAPES}
        if len(shape_ids) != len(set(shape_ids)):
            raise ValueError("Battleship shape-option shape ids must be unique")
        if not set(shape_ids) <= valid_shape_ids:
            raise ValueError("Battleship shape-option shape ids must belong to the fleet")
        if sum(1 for option in sample.shape_options if bool(option.is_answer)) != 1:
            raise ValueError("Battleship shape-option scenes require exactly one answer")


__all__ = [
    "candidate_completes_target_shape",
    "cell_status_answer_support_for_ship",
    "fleet_orientation_lookup",
    "fleet_shape_by_id",
    "matching_fleet_shape_ids",
    "normalize_offsets",
    "shape_orientations",
    "ship_size_for_shape_id",
    "validate_battleship_scene",
]
