"""Pure movement rules for circular-chess games tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.games.shared.piece_board_rules import ChessPiece

from .defaults import RING_COUNT, SECTOR_COUNT
from .state import Board, CircularChessEvaluation, Coord


def circular_coord_to_cell_id(coord: Coord) -> str:
    """Return a stable id for one circular-chess cell."""

    return f"cell_ring{int(coord[0])}_sector{int(coord[1])}"


def circular_piece_to_entity_id(coord: Coord, piece: ChessPiece) -> str:
    """Return a stable id for one circular-chess piece."""

    return f"piece_{piece.color}_{piece.kind}_ring{int(coord[0])}_sector{int(coord[1])}"


def in_circular_bounds(ring: int, sector: int) -> bool:
    """Return whether one ring/sector coordinate is on the circular board."""

    return 0 <= int(ring) < RING_COUNT and 0 <= int(sector) < SECTOR_COUNT


def all_coords() -> tuple[Coord, ...]:
    """Return all circular board cells."""

    return tuple((ring, sector) for ring in range(RING_COUNT) for sector in range(SECTOR_COUNT))


def empty_board() -> Board:
    """Return an empty circular-chess board."""

    return tuple(tuple(None for _ in range(SECTOR_COUNT)) for _ in range(RING_COUNT))


def freeze_board(board: Sequence[Sequence[ChessPiece | None]]) -> Board:
    """Freeze one mutable board into the canonical tuple representation."""

    return tuple(tuple(cell for cell in row) for row in board)


def serialize_board(board: Board) -> list[list[str | None]]:
    """Return a JSON-friendly board state."""

    return [[None if piece is None else f"{piece.color}_{piece.kind}" for piece in row] for row in board]


def occupied_coords(board: Board) -> tuple[Coord, ...]:
    """Return occupied circular board coordinates."""

    return tuple(coord for coord in all_coords() if board[int(coord[0])][int(coord[1])] is not None)


def movement_units_for_piece(kind: str, coord: Coord) -> tuple[tuple[Coord, ...], ...]:
    """Return independent movement rays/jumps for a piece on an empty circular board."""

    ring, sector = int(coord[0]), int(coord[1])
    kind_text = str(kind)
    if kind_text == "knight":
        units = []
        for dr, ds in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
            target_ring = int(ring + dr)
            if not 0 <= int(target_ring) < RING_COUNT:
                continue
            units.append(((int(target_ring), int((sector + ds) % SECTOR_COUNT)),))
        return tuple(sorted(set(units)))

    max_steps = SECTOR_COUNT - 1
    if kind_text == "rook":
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    elif kind_text == "bishop":
        directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    elif kind_text == "queen":
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    elif kind_text == "king":
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
        max_steps = 1
    else:
        raise ValueError(f"unsupported Circular Chess piece kind: {kind}")

    units: list[tuple[Coord, ...]] = []
    for dr, ds in directions:
        ray: list[Coord] = []
        for step in range(1, int(max_steps) + 1):
            next_ring = int(ring + (int(dr) * step))
            if not 0 <= int(next_ring) < RING_COUNT:
                break
            next_sector = int((sector + (int(ds) * step)) % SECTOR_COUNT)
            ray.append((int(next_ring), int(next_sector)))
            if int(dr) == 0 and int(step) >= SECTOR_COUNT - 1:
                break
        if ray:
            units.append(tuple(ray))
    return tuple(units)


def legal_destinations(board: Board, coord: Coord) -> tuple[Coord, ...]:
    """Return pseudo-legal destinations for one circular-chess piece."""

    piece = board[int(coord[0])][int(coord[1])]
    if piece is None:
        return tuple()
    destinations: list[Coord] = []
    for unit in movement_units_for_piece(str(piece.kind), coord):
        for target in unit:
            occupant = board[int(target[0])][int(target[1])]
            if occupant is None:
                destinations.append(tuple(target))
                continue
            if str(occupant.color) != str(piece.color):
                destinations.append(tuple(target))
            break
    return tuple(sorted(set(destinations)))


def capture_destinations(board: Board, coord: Coord) -> tuple[Coord, ...]:
    """Return opponent-occupied legal destinations for one piece."""

    piece = board[int(coord[0])][int(coord[1])]
    if piece is None:
        return tuple()
    return tuple(
        target
        for target in legal_destinations(board, coord)
        if board[int(target[0])][int(target[1])] is not None
        and str(board[int(target[0])][int(target[1])].color) != str(piece.color)
    )


def target_reachers(board: Board, *, target_coord: Coord, target_color: str) -> tuple[Coord, ...]:
    """Return source pieces of `target_color` that can move to `target_coord`."""

    coords: list[Coord] = []
    for coord in occupied_coords(board):
        piece = board[int(coord[0])][int(coord[1])]
        if piece is None or str(piece.color) != str(target_color):
            continue
        if tuple(target_coord) in legal_destinations(board, coord):
            coords.append(tuple(coord))
    return tuple(sorted(coords))


def max_possible_marked_destination_answer(*, destination_mode: str, piece_kind: str) -> int:
    """Return the largest feasible answer for a marked-piece destination count."""

    origin = (1, 0)
    units = movement_units_for_piece(str(piece_kind), origin)
    if str(destination_mode) == "capture":
        return min(6, sum(1 for unit in units if unit))
    if str(destination_mode) == "move":
        return min(8, sum(len(unit) for unit in units))
    raise ValueError(f"unsupported destination mode: {destination_mode}")


def max_possible_target_reacher_answer() -> int:
    """Return the largest feasible answer for a target-cell reacher count."""

    return 6


def evaluate_marked_destinations(
    board: Board,
    *,
    destination_mode: str,
    marked_coord: Coord,
) -> CircularChessEvaluation:
    """Evaluate one marked-piece destination or capture count."""

    piece = board[int(marked_coord[0])][int(marked_coord[1])]
    if piece is None:
        raise ValueError("marked query requires a marked piece")
    if str(destination_mode) == "capture":
        coords = capture_destinations(board, marked_coord)
    elif str(destination_mode) == "move":
        coords = legal_destinations(board, marked_coord)
    else:
        raise ValueError(f"unsupported destination mode: {destination_mode}")
    return CircularChessEvaluation(
        answer=len(coords),
        annotation_coords=tuple(sorted(coords)),
        annotation_entity_ids=tuple(circular_coord_to_cell_id(coord) for coord in sorted(coords)),
        annotation_kind="destination_cell_centers",
        marked_coord=tuple(marked_coord),
        marked_piece=piece,
    )


def evaluate_target_reachers(
    board: Board,
    *,
    target_coord: Coord,
    target_color: str,
) -> CircularChessEvaluation:
    """Evaluate same-side pieces that can legally move to one marked target cell."""

    coords = target_reachers(board, target_coord=target_coord, target_color=str(target_color))
    return CircularChessEvaluation(
        answer=len(coords),
        annotation_coords=tuple(sorted(coords)),
        annotation_entity_ids=tuple(
            circular_piece_to_entity_id(coord, board[int(coord[0])][int(coord[1])])
            for coord in sorted(coords)
            if board[int(coord[0])][int(coord[1])] is not None
        ),
        annotation_kind="source_piece_centers",
        target_coord=tuple(target_coord),
        target_color=str(target_color),
    )


__all__ = [
    "all_coords",
    "capture_destinations",
    "circular_coord_to_cell_id",
    "circular_piece_to_entity_id",
    "color_name",
    "empty_board",
    "evaluate_marked_destinations",
    "evaluate_target_reachers",
    "freeze_board",
    "in_circular_bounds",
    "legal_destinations",
    "max_possible_marked_destination_answer",
    "max_possible_target_reacher_answer",
    "movement_units_for_piece",
    "occupied_coords",
    "serialize_board",
    "target_reachers",
]
