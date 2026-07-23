"""Passive state models and entity helpers for Pac-Man scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

from .defaults import SUPPORTED_PACMAN_SCENE_VARIANTS, SUPPORTED_PACMAN_STYLE_VARIANTS


Coord = Tuple[int, int]


@dataclass(frozen=True)
class PacmanItem:
    """One visible bonus item in a Pac-Man maze."""

    label: str
    item_id: str
    coord: Coord
    kind: str
    is_answer: bool = False
    score_value: int | None = None


@dataclass(frozen=True)
class PacmanGhost:
    """One visible ghost in a Pac-Man maze."""

    ghost_id: str
    coord: Coord
    color_key: str
    is_stop_ghost: bool = False


@dataclass(frozen=True)
class PacmanSceneState:
    """Rendered Pac-Man maze state without task answer ownership."""

    row_count: int
    col_count: int
    scene_variant: str
    style_variant: str
    open_cells: Tuple[Coord, ...]
    wall_cells: Tuple[Coord, ...]
    pacman_coord: Coord
    route_coords: Tuple[Coord, ...]
    pellets: Tuple[Coord, ...]
    items: Tuple[PacmanItem, ...]
    ghosts: Tuple[PacmanGhost, ...]
    construction_mode: str


def sorted_coords(coords: Iterable[Coord]) -> Tuple[Coord, ...]:
    """Return canonical row-major coordinates."""

    return tuple(sorted((int(row), int(col)) for row, col in coords))


def pellet_entity_id(coord: Coord) -> str:
    """Return the stable entity id for one normal pellet."""

    row, col = coord
    return f"pellet_r{int(row)}_c{int(col)}"


def item_entity_id(label: str) -> str:
    """Return the stable entity id for one labeled bonus item."""

    return f"item_{str(label)}"


def ghost_entity_id(name: int | str) -> str:
    """Return the stable entity id for one visible ghost."""

    if isinstance(name, int):
        return f"ghost_{int(name):02d}"
    text = str(name).strip()
    if text.startswith("ghost_"):
        return text
    return f"ghost_{text}"


def pacman_entity_id() -> str:
    """Return the stable entity id for the visible Pac-Man marker."""

    return "pacman"


def route_entity_id(index: int) -> str:
    """Return the stable entity id for one highlighted route cell."""

    return f"route_{int(index):02d}"


def all_grid_coords(rows: int, cols: int) -> Tuple[Coord, ...]:
    """Return all grid coordinates."""

    return tuple((row, col) for row in range(int(rows)) for col in range(int(cols)))


def grid_neighbors(coord: Coord, *, rows: int, cols: int) -> Tuple[Coord, ...]:
    """Return orthogonal in-bounds grid neighbors excluding the outer wall ring."""

    row, col = int(coord[0]), int(coord[1])
    candidates = (
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    )
    return tuple(
        (int(next_row), int(next_col))
        for next_row, next_col in candidates
        if 1 <= int(next_row) < int(rows) - 1 and 1 <= int(next_col) < int(cols) - 1
    )


def coord_from_entity_id(entity_id: str) -> Coord:
    """Decode a pellet-like entity id into a coordinate."""

    text = str(entity_id)
    if "_r" not in text or "_c" not in text:
        raise ValueError(f"entity id does not encode a coordinate: {entity_id}")
    row_text, col_text = text.split("_r", 1)[1].split("_c", 1)
    return int(row_text), int(col_text)


def validate_pacman_scene_state(scene: PacmanSceneState) -> None:
    """Validate scene-level Pac-Man invariants independent of the objective."""

    if str(scene.scene_variant) not in SUPPORTED_PACMAN_SCENE_VARIANTS:
        raise ValueError(f"unsupported Pac-Man scene_variant: {scene.scene_variant}")
    if str(scene.style_variant) not in SUPPORTED_PACMAN_STYLE_VARIANTS:
        raise ValueError(f"unsupported Pac-Man style_variant: {scene.style_variant}")
    open_cells = {tuple(coord) for coord in scene.open_cells}
    wall_cells = {tuple(coord) for coord in scene.wall_cells}
    if open_cells & wall_cells:
        raise ValueError("Pac-Man open and wall cells overlap")
    if tuple(scene.pacman_coord) not in open_cells:
        raise ValueError("Pac-Man coordinate must be an open cell")
    if not scene.route_coords or tuple(scene.route_coords[0]) != tuple(scene.pacman_coord):
        raise ValueError("Pac-Man route must start at Pac-Man")
    if any(tuple(coord) not in open_cells for coord in scene.route_coords):
        raise ValueError("Pac-Man route includes a non-open cell")
    pellet_set = {tuple(coord) for coord in scene.pellets}
    if len(pellet_set) != len(scene.pellets):
        raise ValueError("Pac-Man pellets must be unique")
    if any(coord not in open_cells for coord in pellet_set):
        raise ValueError("Pac-Man pellet must be on an open cell")
    item_labels = [str(item.label) for item in scene.items]
    if len(set(item_labels)) != len(item_labels):
        raise ValueError("Pac-Man item labels must be unique")
    item_coords = [tuple(item.coord) for item in scene.items]
    if len(set(item_coords)) != len(item_coords):
        raise ValueError("Pac-Man item coordinates must be unique")
    if any(coord not in open_cells for coord in item_coords):
        raise ValueError("Pac-Man item must be on an open cell")
    if any(coord in pellet_set for coord in item_coords):
        raise ValueError("Pac-Man items and pellets must not overlap")
    ghost_ids = [str(ghost.ghost_id) for ghost in scene.ghosts]
    if len(set(ghost_ids)) != len(ghost_ids):
        raise ValueError("Pac-Man ghost ids must be unique")
    ghost_coords = [tuple(ghost.coord) for ghost in scene.ghosts]
    if len(set(ghost_coords)) != len(ghost_coords):
        raise ValueError("Pac-Man ghost coordinates must be unique")
    if any(coord not in open_cells for coord in ghost_coords):
        raise ValueError("Pac-Man ghost must be on an open cell")
    if any(coord in pellet_set for coord in ghost_coords):
        raise ValueError("Pac-Man ghosts and pellets must not overlap")
    if any(coord in set(item_coords) for coord in ghost_coords):
        raise ValueError("Pac-Man ghosts and items must not overlap")
    if tuple(scene.pacman_coord) in set(ghost_coords):
        raise ValueError("Pac-Man ghost must not overlap Pac-Man")


def visible_pellet_trace(pellets: Sequence[Coord]) -> Tuple[Mapping[str, object], ...]:
    """Return row-major trace rows for visible pellets."""

    return tuple(
        {
            "coord": [int(coord[0]), int(coord[1])],
            "entity_id": pellet_entity_id(coord),
        }
        for coord in sorted_coords(pellets)
    )


def visible_ghost_trace(ghosts: Sequence[PacmanGhost]) -> Tuple[Mapping[str, object], ...]:
    """Return trace rows for visible ghosts."""

    return tuple(
        {
            "coord": [int(ghost.coord[0]), int(ghost.coord[1])],
            "entity_id": str(ghost.ghost_id),
            "color_key": str(ghost.color_key),
            "is_stop_ghost": bool(ghost.is_stop_ghost),
        }
        for ghost in ghosts
    )


def visible_item_trace(items: Sequence[PacmanItem]) -> Tuple[Mapping[str, object], ...]:
    """Return trace rows for visible bonus items."""

    rows = []
    for item in items:
        row = {
            "label": str(item.label),
            "kind": str(item.kind),
            "coord": [int(item.coord[0]), int(item.coord[1])],
            "entity_id": item_entity_id(str(item.label)),
            "is_answer": bool(item.is_answer),
        }
        if item.score_value is not None:
            row["score_value"] = int(item.score_value)
            row["display_text"] = str(int(item.score_value))
        else:
            row["display_text"] = str(item.label)
        rows.append(row)
    return tuple(rows)


__all__ = [
    "Coord",
    "PacmanGhost",
    "PacmanItem",
    "PacmanSceneState",
    "all_grid_coords",
    "coord_from_entity_id",
    "ghost_entity_id",
    "grid_neighbors",
    "item_entity_id",
    "pacman_entity_id",
    "pellet_entity_id",
    "route_entity_id",
    "sorted_coords",
    "validate_pacman_scene_state",
    "visible_ghost_trace",
    "visible_item_trace",
    "visible_pellet_trace",
]
