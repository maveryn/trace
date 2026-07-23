"""Pure movement rules for chess-variant games tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.games.shared.piece_board_rules import (
    BOARD_SIZE,
    Board,
    Coord,
    coord_to_cell_id,
    in_bounds,
)

from .state import ChessVariantEvaluation


def directions_for_rule(rule_family: str) -> tuple[tuple[int, int], ...]:
    """Return direction vectors for range-based visible rules."""

    if str(rule_family) == "straight_range":
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    if str(rule_family) == "diagonal_range":
        return ((-1, -1), (-1, 1), (1, -1), (1, 1))
    if str(rule_family) == "straight_or_diagonal_range":
        return ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    return ()


def leaper_offsets(rule_family: str) -> tuple[tuple[int, int], ...]:
    """Return leaper offsets for jump-based visible rules."""

    if str(rule_family) == "leaper_2_1":
        a, b = 2, 1
    elif str(rule_family) == "leaper_3_1":
        a, b = 3, 1
    else:
        return ()
    offsets = set()
    for sign_row in (-1, 1):
        for sign_col in (-1, 1):
            offsets.add((sign_row * a, sign_col * b))
            offsets.add((sign_row * b, sign_col * a))
    return tuple(sorted(offsets))


def ray_coords(origin: Coord, direction: tuple[int, int], *, max_steps: int) -> tuple[Coord, ...]:
    """Return in-bounds ray coordinates from one origin."""

    row, col = int(origin[0]), int(origin[1])
    delta_row, delta_col = int(direction[0]), int(direction[1])
    coords: list[Coord] = []
    for step in range(1, int(max_steps) + 1):
        coord = (row + (delta_row * step), col + (delta_col * step))
        if not in_bounds(*coord):
            break
        coords.append(coord)
    return tuple(coords)


def all_empty_board_destinations(rule_family: str, range_k: int, origin: Coord) -> tuple[Coord, ...]:
    """Return geometrically reachable destinations on an empty board."""

    if str(rule_family).endswith("_range"):
        coords: list[Coord] = []
        for direction in directions_for_rule(str(rule_family)):
            coords.extend(ray_coords(origin, direction, max_steps=int(range_k)))
        return tuple(coords)
    out: list[Coord] = []
    row, col = int(origin[0]), int(origin[1])
    for delta_row, delta_col in leaper_offsets(str(rule_family)):
        coord = (row + int(delta_row), col + int(delta_col))
        if in_bounds(*coord):
            out.append(coord)
    return tuple(sorted(out))


def destinations_for_piece_under_rule(
    board: Board,
    *,
    source_coord: Coord,
    rule_family: str,
    range_k: int,
) -> tuple[tuple[Coord, ...], tuple[Coord, ...]]:
    """Return legal destinations and capture destinations under the visible rule."""

    piece = board[int(source_coord[0])][int(source_coord[1])]
    if piece is None:
        return tuple(), tuple()
    destinations: list[Coord] = []
    captures: list[Coord] = []
    if str(rule_family).endswith("_range"):
        for direction in directions_for_rule(str(rule_family)):
            for coord in ray_coords(source_coord, direction, max_steps=int(range_k)):
                occupant = board[int(coord[0])][int(coord[1])]
                if occupant is None:
                    destinations.append(coord)
                    continue
                if str(occupant.color) != str(piece.color):
                    destinations.append(coord)
                    captures.append(coord)
                break
    else:
        for coord in all_empty_board_destinations(str(rule_family), int(range_k), source_coord):
            occupant = board[int(coord[0])][int(coord[1])]
            if occupant is None:
                destinations.append(coord)
            elif str(occupant.color) != str(piece.color):
                destinations.append(coord)
                captures.append(coord)
    return tuple(sorted(destinations)), tuple(sorted(captures))


def max_possible_marked_destination_answer(*, destination_mode: str, rule_family: str, range_k: int) -> int:
    """Return the largest feasible answer for a marked-piece count."""

    if str(destination_mode) == "capture" and str(rule_family).endswith("_range"):
        max_count = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                ray_count = sum(
                    1
                    for direction in directions_for_rule(str(rule_family))
                    if ray_coords((row, col), direction, max_steps=int(range_k))
                )
                max_count = max(int(max_count), int(ray_count))
        return int(max_count)

    max_count = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            max_count = max(
                int(max_count),
                len(all_empty_board_destinations(str(rule_family), int(range_k), (row, col))),
            )
    return int(max_count)


def evaluate_marked_piece_board(board: Board, *, marked_coord: Coord, rule_family: str, range_k: int) -> ChessVariantEvaluation:
    """Evaluate all legal destinations for one marked piece."""

    piece = board[int(marked_coord[0])][int(marked_coord[1])]
    if piece is None:
        raise ValueError("marked square is empty")
    legal_destinations, capture_coords = destinations_for_piece_under_rule(
        board,
        source_coord=marked_coord,
        rule_family=str(rule_family),
        range_k=int(range_k),
    )
    return ChessVariantEvaluation(
        answer=0,
        legal_destinations=legal_destinations,
        capture_coords=capture_coords,
        annotation_coords=(),
        annotation_entity_ids=(),
        annotation_kind="cell",
        marked_coord=marked_coord,
        marked_piece=piece,
    )


def with_destination_annotation(
    evaluation: ChessVariantEvaluation,
    *,
    destination_mode: str,
) -> ChessVariantEvaluation:
    """Attach answer and cell annotation for a marked-piece destination count."""

    if str(destination_mode) == "move":
        annotation_coords = tuple(evaluation.legal_destinations)
    elif str(destination_mode) == "empty":
        capture_set = set(evaluation.capture_coords)
        annotation_coords = tuple(coord for coord in evaluation.legal_destinations if coord not in capture_set)
    elif str(destination_mode) == "capture":
        annotation_coords = tuple(evaluation.capture_coords)
    else:
        raise ValueError(f"unsupported destination mode: {destination_mode}")

    return ChessVariantEvaluation(
        answer=len(annotation_coords),
        legal_destinations=tuple(evaluation.legal_destinations),
        capture_coords=tuple(evaluation.capture_coords),
        annotation_coords=annotation_coords,
        annotation_entity_ids=tuple(coord_to_cell_id(coord) for coord in annotation_coords),
        annotation_kind="cell",
        marked_coord=evaluation.marked_coord,
        marked_piece=evaluation.marked_piece,
        marker_role=str(evaluation.marker_role),
    )


def evaluate_by_semantic_query(
    board: Board,
    *,
    destination_mode: str | None,
    marked_coord: Coord | None,
    rule_family: str,
    range_k: int,
) -> ChessVariantEvaluation:
    """Evaluate a board using semantic task arguments, not public query ids."""

    if destination_mode is None or marked_coord is None:
        raise ValueError("marked-piece evaluation requires destination_mode and marked_coord")
    return with_destination_annotation(
        evaluate_marked_piece_board(
            board,
            marked_coord=marked_coord,
            rule_family=str(rule_family),
            range_k=int(range_k),
        ),
        destination_mode=str(destination_mode),
    )


__all__ = [
    "all_empty_board_destinations",
    "destinations_for_piece_under_rule",
    "directions_for_rule",
    "evaluate_by_semantic_query",
    "evaluate_marked_piece_board",
    "leaper_offsets",
    "max_possible_marked_destination_answer",
    "ray_coords",
    "with_destination_annotation",
]
