"""State contracts for the Battleship games scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


Coord = Tuple[int, int]
SCENE_ID = "battleship"

SUPPORTED_BATTLESHIP_SCENE_VARIANTS: Tuple[str, ...] = ("standard_fleet",)
SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS: Tuple[str, ...] = (
    "line5",
    "line4",
    "line3",
    "square4",
    "elbow3",
)
LAST_CELL_OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
SHAPE_OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class FleetShapeSpec:
    """One visible fleet shape and its canonical cell offsets."""

    shape_id: str
    display_name: str
    offsets: Tuple[Coord, ...]


@dataclass(frozen=True)
class BattleshipShipPlacement:
    """One placed fleet ship on the Battleship grid."""

    ship_id: str
    shape_id: str
    display_name: str
    coords: Tuple[Coord, ...]
    hit_coords: Tuple[Coord, ...]
    is_sunk: bool


@dataclass(frozen=True)
class BattleshipCandidateOption:
    """One labeled candidate cell in a Battleship option-label task."""

    label: str
    coord: Coord
    is_answer: bool


@dataclass(frozen=True)
class BattleshipShapeOption:
    """One labeled fleet-shape answer option."""

    label: str
    shape_id: str
    display_name: str
    is_answer: bool


@dataclass(frozen=True)
class BattleshipSample:
    """Generated Battleship tracking-grid scene state.

    This state is intentionally public-task agnostic. Public task files bind
    the final answer, annotation, and query metadata from this rendered scene.
    """

    board_size: int
    scene_variant: str
    ship_placements: Tuple[BattleshipShipPlacement, ...]
    hit_coords: Tuple[Coord, ...]
    miss_coords: Tuple[Coord, ...]
    sunk_ship_count: int
    partial_ship_count: int
    untouched_ship_count: int
    construction_mode: str
    candidate_options: Tuple[BattleshipCandidateOption, ...] = tuple()
    shape_options: Tuple[BattleshipShapeOption, ...] = tuple()


FLEET_SHAPES: Tuple[FleetShapeSpec, ...] = (
    FleetShapeSpec(
        shape_id="line5",
        display_name="Line 5",
        offsets=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    ),
    FleetShapeSpec(
        shape_id="line4",
        display_name="Line 4",
        offsets=((0, 0), (0, 1), (0, 2), (0, 3)),
    ),
    FleetShapeSpec(
        shape_id="line3",
        display_name="Line 3",
        offsets=((0, 0), (0, 1), (0, 2)),
    ),
    FleetShapeSpec(
        shape_id="square4",
        display_name="Square 2x2",
        offsets=((0, 0), (0, 1), (1, 0), (1, 1)),
    ),
    FleetShapeSpec(
        shape_id="elbow3",
        display_name="L 3",
        offsets=((0, 0), (1, 0), (1, 1)),
    ),
)


def coord_to_cell_id(coord: Coord) -> str:
    """Return the stable render/entity id for one Battleship board cell."""

    row, col = coord
    return f"r{int(row)}_c{int(col)}"


def all_coords(size: int) -> Tuple[Coord, ...]:
    """Return all board coordinates in row-major order."""

    return tuple((row, col) for row in range(int(size)) for col in range(int(size)))


def sorted_coords(coords: Iterable[Coord]) -> Tuple[Coord, ...]:
    """Return canonical sorted coordinates."""

    return tuple(sorted((int(row), int(col)) for row, col in coords))


__all__ = [
    "BattleshipCandidateOption",
    "BattleshipSample",
    "BattleshipShapeOption",
    "BattleshipShipPlacement",
    "Coord",
    "FLEET_SHAPES",
    "FleetShapeSpec",
    "LAST_CELL_OPTION_LABELS",
    "SCENE_ID",
    "SHAPE_OPTION_LABELS",
    "SUPPORTED_BATTLESHIP_SCENE_VARIANTS",
    "SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS",
    "all_coords",
    "coord_to_cell_id",
    "sorted_coords",
]
