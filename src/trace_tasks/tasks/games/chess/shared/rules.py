"""Identity-free Chess rules and construction primitives."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

from trace_tasks.tasks.games.shared.piece_board_rules import (
    BLACK,
    BOARD_SIZE,
    PIECE_KINDS,
    WHITE,
    Board,
    ChessPiece,
    Coord,
    attackers_to_square,
    capturable_opponent_coords,
    coord_to_cell_id,
    coords_adjacent,
    empty_board,
    freeze_board,
    in_bounds,
    king_escape_squares,
    occupied_coords,
    occupied_piece_count,
    opponent,
    piece_attacks_square,
    piece_capture_targets,
    piece_move_destinations,
    piece_to_entity_id,
    sample_material_piece,
    standard_pawn_row_valid,
    validate_square_chess_material,
)

SliderEvaluator = Callable[[Board], Tuple[Coord, ...]]


def random_empty_coord(rng, mutable: list[list[ChessPiece | None]]) -> Coord:
    """Sample one empty coordinate from a mutable board."""

    empties = [(row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE) if mutable[row][col] is None]
    if not empties:
        raise ValueError("no empty chess board cells remain")
    return tuple(rng.choice(empties))


def place_capped_piece(*, mutable: list[list[ChessPiece | None]], coord: Coord, piece: ChessPiece) -> bool:
    """Place one standard material-capped piece on an empty coordinate."""

    row, col = int(coord[0]), int(coord[1])
    if mutable[row][col] is not None:
        return False
    candidate = [list(board_row) for board_row in mutable]
    candidate[row][col] = piece
    if not validate_square_chess_material(
        freeze_board(candidate),
        require_both_kings=False,
        enforce_standard_pawn_rows=True,
        enforce_non_adjacent_kings=False,
    ):
        return False
    mutable[row][col] = piece
    return True


def place_capped_random_piece(
    *,
    rng,
    mutable: list[list[ChessPiece | None]],
    coord: Coord,
    colors: Sequence[str] = (WHITE, BLACK),
    kinds: Sequence[str] = PIECE_KINDS,
) -> bool:
    """Place one random standard material-capped piece on an empty coordinate."""

    row, col = int(coord[0]), int(coord[1])
    if mutable[row][col] is not None:
        return False
    piece = sample_material_piece(
        rng,
        freeze_board(mutable),
        colors=tuple(str(color) for color in colors),
        kinds=tuple(str(kind) for kind in kinds),
        row=int(row),
        enforce_standard_pawn_rows=True,
    )
    if piece is None:
        return False
    mutable[row][col] = piece
    return True


def place_display_piece(*, rng, mutable: list[list[ChessPiece | None]], color: str, kind: str) -> Coord:
    """Place one visible chess display piece without material-count restrictions."""

    coords: list[Coord] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if mutable[row][col] is not None:
                continue
            if str(kind) == "pawn" and not standard_pawn_row_valid(row):
                continue
            coords.append((int(row), int(col)))
    if not coords:
        raise ValueError("no legal display coordinate remains")
    coord = tuple(rng.choice(coords))
    mutable[int(coord[0])][int(coord[1])] = ChessPiece(str(color), str(kind))
    return coord


def piece_entity_ids(board: Board, coords: Tuple[Coord, ...]) -> Tuple[str, ...]:
    """Return piece entity ids for occupied coordinates."""

    out: list[str] = []
    for coord in coords:
        piece = board[int(coord[0])][int(coord[1])]
        if piece is None:
            raise ValueError("piece annotation coordinate is empty")
        out.append(piece_to_entity_id(coord, piece))
    return tuple(out)


def line_step_between(a: Coord, b: Coord) -> Coord | None:
    """Return one row/file/diagonal step from a to b, or None if unaligned."""

    ar, ac = int(a[0]), int(a[1])
    br, bc = int(b[0]), int(b[1])
    if (ar, ac) == (br, bc):
        return None
    if ar == br:
        return (0, 1 if bc > ac else -1)
    if ac == bc:
        return (1 if br > ar else -1, 0)
    if abs(ar - br) == abs(ac - bc):
        return (1 if br > ar else -1, 1 if bc > ac else -1)
    return None


def coords_between(a: Coord, b: Coord) -> Tuple[Coord, ...]:
    """Return coordinates strictly between two aligned coordinates."""

    step = line_step_between(a, b)
    if step is None:
        return ()
    coords: list[Coord] = []
    row, col = int(a[0]) + int(step[0]), int(a[1]) + int(step[1])
    while (int(row), int(col)) != (int(b[0]), int(b[1])):
        coords.append((int(row), int(col)))
        row += int(step[0])
        col += int(step[1])
    return tuple(coords)


def slider_directions(piece_kind: str) -> Tuple[Coord, ...]:
    """Return row/file/diagonal directions for a sliding piece kind."""

    rook_dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    bishop_dirs = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    if str(piece_kind) == "rook":
        return rook_dirs
    if str(piece_kind) == "bishop":
        return bishop_dirs
    if str(piece_kind) == "queen":
        return rook_dirs + bishop_dirs
    raise ValueError(f"unsupported sliding piece kind: {piece_kind}")


def line_allowed_for_slider(piece_kind: str, source: Coord, target: Coord) -> bool:
    """Return whether target lies on an allowed ray for a sliding piece kind."""

    step = line_step_between(source, target)
    return step is not None and tuple(step) in slider_directions(str(piece_kind))


def attacker_slots_for_square(target_coord: Coord, attacker_color: str) -> Tuple[Tuple[Coord, str], ...]:
    """Return candidate attacker placements around one marked target square."""

    row, col = int(target_coord[0]), int(target_coord[1])
    slots: list[Tuple[Coord, str]] = []
    for dr, dc, kind in (
        (-2, -1, "knight"), (-2, 1, "knight"), (-1, -2, "knight"), (-1, 2, "knight"),
        (1, -2, "knight"), (1, 2, "knight"), (2, -1, "knight"), (2, 1, "knight"),
    ):
        coord = (row + dr, col + dc)
        if in_bounds(*coord):
            slots.append((coord, kind))
    pawn_row = row - 1 if str(attacker_color) == BLACK else row + 1
    for dc in (-1, 1):
        coord = (pawn_row, col + dc)
        if in_bounds(*coord) and standard_pawn_row_valid(coord[0]):
            slots.append((coord, "pawn"))
    for dr, dc, kind in (
        (-1, 0, "rook"), (1, 0, "rook"), (0, -1, "rook"), (0, 1, "rook"),
        (-1, -1, "bishop"), (-1, 1, "bishop"), (1, -1, "bishop"), (1, 1, "bishop"),
    ):
        for distance in (2, 3, 4):
            coord = (row + (dr * distance), col + (dc * distance))
            if in_bounds(*coord):
                slots.append((coord, kind))
                break
    return tuple(slots)


def place_non_adjacent_kings(
    *,
    rng,
    mutable: list[list[ChessPiece | None]],
    first_color: str,
    second_color: str,
    forbidden: Sequence[Coord] = (),
) -> Tuple[Coord, Coord]:
    """Place two non-adjacent kings, respecting occupied and forbidden cells."""

    forbidden_set = {tuple(coord) for coord in forbidden}
    placed: list[Coord] = []
    for color in (str(first_color), str(second_color)):
        for _ in range(256):
            coord = random_empty_coord(rng, mutable)
            if coord in forbidden_set:
                continue
            if any(coords_adjacent(coord, other) for other in placed):
                continue
            mutable[int(coord[0])][int(coord[1])] = ChessPiece(color, "king")
            placed.append(coord)
            break
        else:
            raise ValueError("failed to place separated kings")
    return tuple(placed)  # type: ignore[return-value]


def evaluate_marked_destinations(board: Board, marked_coord: Coord, *, destination_mode: str) -> Tuple[Coord, ...]:
    """Return marked-piece destination or capture coordinates."""

    piece = board[int(marked_coord[0])][int(marked_coord[1])]
    if piece is None or str(piece.kind) == "king":
        return ()
    if str(destination_mode) == "capture":
        return tuple(sorted(piece_capture_targets(board, marked_coord)))
    if str(destination_mode) == "move":
        return tuple(
            sorted(
                coord
                for coord in piece_move_destinations(board, marked_coord)
                if board[int(coord[0])][int(coord[1])] is None
            )
        )
    raise ValueError(f"unsupported destination mode: {destination_mode}")


def evaluate_player_captures(board: Board, player_color: str) -> Tuple[Coord, ...]:
    """Return unique opponent-piece coordinates capturable by one side."""

    return tuple(sorted(capturable_opponent_coords(board, str(player_color))))


def evaluate_target_attackers(board: Board, target_coord: Coord, attacker_color: str) -> Tuple[Coord, ...]:
    """Return pieces of one color attacking a target square."""

    return tuple(sorted(attackers_to_square(board, target_coord, str(attacker_color))))


def evaluate_line_blockers(board: Board, source: Coord, target: Coord, *, slider_kind: str) -> Tuple[Coord, ...]:
    """Return occupied coordinates strictly between a slider and aligned target."""

    piece = board[int(source[0])][int(source[1])]
    if piece is None or str(piece.kind) != str(slider_kind):
        return ()
    if board[int(target[0])][int(target[1])] is not None:
        return ()
    if not line_allowed_for_slider(str(slider_kind), source, target):
        return ()
    return tuple(sorted(coord for coord in coords_between(source, target) if board[int(coord[0])][int(coord[1])] is not None))


def evaluate_king_escapes(board: Board, king_coord: Coord) -> Tuple[Coord, ...]:
    """Return safe king-escape coordinates for a marked king."""

    return tuple(sorted(king_escape_squares(board, king_coord)))


def evaluate_piece_matches(
    board: Board,
    *,
    target_kind: str,
    target_color: str | None = None,
) -> Tuple[Coord, ...]:
    """Return visible pieces matching a kind and optional color."""

    matches: list[Coord] = []
    for coord in occupied_coords(board):
        piece = board[int(coord[0])][int(coord[1])]
        if piece is None or str(piece.kind) != str(target_kind):
            continue
        if target_color is not None and str(piece.color) != str(target_color):
            continue
        matches.append(coord)
    return tuple(sorted(matches))


def opponent_attackers_after_king_move(board: Board, king_coord: Coord, dest: Coord, king_color: str) -> Tuple[Coord, ...]:
    """Return opponent attackers after moving the king to a candidate cell."""

    mutable = [list(row) for row in board]
    king = mutable[int(king_coord[0])][int(king_coord[1])]
    if king is None:
        return ()
    mutable[int(king_coord[0])][int(king_coord[1])] = None
    mutable[int(dest[0])][int(dest[1])] = king
    return attackers_to_square(freeze_board(mutable), dest, opponent(str(king_color)))


def same_color_slider_points_to_any(board: Board, *, color: str, targets: Tuple[Coord, ...]) -> bool:
    """Return whether a same-color slider attacks any target coordinate."""

    for coord in occupied_coords(board):
        piece = board[int(coord[0])][int(coord[1])]
        if piece is None or str(piece.color) != str(color) or str(piece.kind) not in {"queen", "rook", "bishop"}:
            continue
        if any(piece_attacks_square(board, coord, target) for target in targets):
            return True
    return False


def add_material_fillers_preserving(
    *,
    rng,
    board: Board,
    scene_variant: str,
    preserved_coords: Sequence[Coord],
    expected_coords: Tuple[Coord, ...],
    evaluator: SliderEvaluator,
    avoid_adjacent_to: Coord | None = None,
) -> Board:
    """Add random material fillers while preserving a coordinate-valued evaluator."""

    minimum, maximum = (6, 9) if str(scene_variant) == "sparse_board" else (9, 13)
    lower_bound = max(int(minimum), occupied_piece_count(board))
    upper_bound = max(int(maximum), int(lower_bound))
    desired = int(rng.randint(int(lower_bound), int(upper_bound)))
    mutable = [list(row) for row in board]
    protected = {tuple(coord) for coord in preserved_coords}
    attempts = 0
    while occupied_piece_count(mutable) < desired and attempts < 420:
        attempts += 1
        coord = random_empty_coord(rng, mutable)
        if coord in protected:
            continue
        if avoid_adjacent_to is not None and coords_adjacent(coord, avoid_adjacent_to):
            continue
        if not place_capped_random_piece(rng=rng, mutable=mutable, coord=coord, kinds=("queen", "rook", "bishop", "knight", "pawn")):
            continue
        frozen = freeze_board(mutable)
        if not validate_square_chess_material(frozen, require_both_kings=True):
            mutable[int(coord[0])][int(coord[1])] = None
            continue
        if tuple(sorted(evaluator(frozen))) != tuple(sorted(expected_coords)):
            mutable[int(coord[0])][int(coord[1])] = None
            continue
    frozen = freeze_board(mutable)
    if not validate_square_chess_material(frozen, require_both_kings=True):
        raise ValueError("filler-added chess board is not material-plausible")
    return frozen

__all__ = [
    "add_material_fillers_preserving",
    "attacker_slots_for_square",
    "coords_between",
    "evaluate_king_escapes",
    "evaluate_line_blockers",
    "evaluate_marked_destinations",
    "evaluate_piece_matches",
    "evaluate_player_captures",
    "evaluate_target_attackers",
    "line_allowed_for_slider",
    "line_step_between",
    "opponent_attackers_after_king_move",
    "piece_entity_ids",
    "place_capped_piece",
    "place_capped_random_piece",
    "place_display_piece",
    "place_non_adjacent_kings",
    "random_empty_coord",
    "same_color_slider_points_to_any",
    "slider_directions",
]
