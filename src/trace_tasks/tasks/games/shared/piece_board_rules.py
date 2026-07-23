"""Shared Chess rules helpers for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


BOARD_SIZE = 8
WHITE = "white"
BLACK = "black"
PIECE_KINDS: Tuple[str, ...] = ("king", "queen", "rook", "bishop", "knight", "pawn")
NON_KING_PIECE_KINDS: Tuple[str, ...] = ("queen", "rook", "bishop", "knight", "pawn")
STANDARD_CHESS_MATERIAL_CAPS: Mapping[str, int] = {
    "king": 1,
    "queen": 1,
    "rook": 2,
    "bishop": 2,
    "knight": 2,
    "pawn": 8,
}
CIRCULAR_CHESS_MATERIAL_CAPS: Mapping[str, int] = {
    "king": 1,
    "queen": 1,
    "rook": 2,
    "bishop": 2,
    "knight": 2,
}

Coord = Tuple[int, int]
Board = Tuple[Tuple["ChessPiece | None", ...], ...]


@dataclass(frozen=True)
class ChessPiece:
    """One visible chess piece."""

    color: str
    kind: str


def opponent(color: str) -> str:
    """Return the opposite chess color."""

    return BLACK if str(color) == WHITE else WHITE


def color_name(color: str) -> str:
    """Return a prompt-facing color name."""

    return "White" if str(color) == WHITE else "Black"


def piece_name(piece: ChessPiece) -> str:
    """Return a prompt-facing piece name."""

    return f"{color_name(piece.color)} {piece.kind}"


def coord_to_cell_id(coord: Coord) -> str:
    """Return one stable board-cell id for a coordinate."""

    return f"cell_r{int(coord[0])}_c{int(coord[1])}"


def coord_to_square_name(coord: Coord) -> str:
    """Return standard chess square notation for a row/column coordinate."""

    row, col = int(coord[0]), int(coord[1])
    if not in_bounds(row, col):
        raise ValueError(f"coordinate outside chess board: {coord}")
    return f"{chr(ord('A') + col)}{BOARD_SIZE - row}"


def piece_to_entity_id(coord: Coord, piece: ChessPiece) -> str:
    """Return one stable piece entity id for a coordinate."""

    return f"piece_{piece.color}_{piece.kind}_r{int(coord[0])}_c{int(coord[1])}"


def in_bounds(row: int, col: int) -> bool:
    """Return whether one board coordinate lies inside the 8 by 8 board."""

    return 0 <= int(row) < BOARD_SIZE and 0 <= int(col) < BOARD_SIZE


def empty_board() -> Board:
    """Return an empty 8 by 8 chess board."""

    return tuple(tuple(None for _ in range(BOARD_SIZE)) for _ in range(BOARD_SIZE))


def freeze_board(board: Sequence[Sequence[ChessPiece | None]]) -> Board:
    """Freeze one mutable board into the canonical tuple representation."""

    return tuple(tuple(cell for cell in row) for row in board)


def occupied_coords(board: Sequence[Sequence[ChessPiece | None]]) -> Tuple[Coord, ...]:
    """Return all occupied board coordinates in row-major order."""

    coords: List[Coord] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] is not None:
                coords.append((int(row), int(col)))
    return tuple(coords)


def occupied_piece_count(board: Sequence[Sequence[ChessPiece | None]]) -> int:
    """Return the total number of visible pieces on one board."""

    return len(occupied_coords(board))


def material_counts(board: Sequence[Sequence[ChessPiece | None]]) -> Dict[Tuple[str, str], int]:
    """Return visible material counts keyed by `(color, kind)`."""

    counts: Dict[Tuple[str, str], int] = {}
    for row in board:
        for piece in row:
            if piece is None:
                continue
            key = (str(piece.color), str(piece.kind))
            counts[key] = int(counts.get(key, 0)) + 1
    return counts


def material_count(board: Sequence[Sequence[ChessPiece | None]], *, color: str, kind: str) -> int:
    """Return the visible count for one color/kind material slot."""

    return int(material_counts(board).get((str(color), str(kind)), 0))


def can_add_material_piece(
    board: Sequence[Sequence[ChessPiece | None]],
    piece: ChessPiece,
    *,
    caps: Mapping[str, int] = STANDARD_CHESS_MATERIAL_CAPS,
) -> bool:
    """Return whether adding one piece would preserve material caps."""

    kind = str(piece.kind)
    if kind not in caps:
        return False
    return material_count(board, color=str(piece.color), kind=kind) < int(caps[kind])


def standard_pawn_row_valid(row: int) -> bool:
    """Return whether a pawn can be displayed on a non-promotion square."""

    return 1 <= int(row) <= 6


def coords_adjacent(a: Coord, b: Coord) -> bool:
    """Return whether two square-board coordinates are king-adjacent."""

    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= 1


def king_coords(board: Sequence[Sequence[ChessPiece | None]], color: str) -> Tuple[Coord, ...]:
    """Return all coordinates occupied by kings of one color."""

    coords: List[Coord] = []
    for row_index, row in enumerate(board):
        for col_index, piece in enumerate(row):
            if piece is not None and str(piece.color) == str(color) and str(piece.kind) == "king":
                coords.append((int(row_index), int(col_index)))
    return tuple(coords)


def legal_material_piece_choices(
    board: Sequence[Sequence[ChessPiece | None]],
    *,
    colors: Sequence[str] = (WHITE, BLACK),
    kinds: Sequence[str] = PIECE_KINDS,
    caps: Mapping[str, int] = STANDARD_CHESS_MATERIAL_CAPS,
    row: int | None = None,
    enforce_standard_pawn_rows: bool = False,
) -> Tuple[ChessPiece, ...]:
    """Return all material-capped pieces that can be added to one square."""

    choices: List[ChessPiece] = []
    for color in colors:
        for kind in kinds:
            if str(kind) == "pawn" and bool(enforce_standard_pawn_rows):
                if row is None or not standard_pawn_row_valid(int(row)):
                    continue
            piece = ChessPiece(str(color), str(kind))
            if can_add_material_piece(board, piece, caps=caps):
                choices.append(piece)
    return tuple(choices)


def sample_material_piece(
    rng,
    board: Sequence[Sequence[ChessPiece | None]],
    *,
    colors: Sequence[str] = (WHITE, BLACK),
    kinds: Sequence[str] = PIECE_KINDS,
    caps: Mapping[str, int] = STANDARD_CHESS_MATERIAL_CAPS,
    row: int | None = None,
    enforce_standard_pawn_rows: bool = False,
) -> ChessPiece | None:
    """Sample one material-capped piece, or return None if no legal choice remains."""

    choices = list(
        legal_material_piece_choices(
            board,
            colors=colors,
            kinds=kinds,
            caps=caps,
            row=row,
            enforce_standard_pawn_rows=bool(enforce_standard_pawn_rows),
        )
    )
    if not choices:
        return None
    return choices[int(rng.randrange(len(choices)))]


def validate_material_caps(
    board: Sequence[Sequence[ChessPiece | None]],
    *,
    caps: Mapping[str, int] = STANDARD_CHESS_MATERIAL_CAPS,
) -> bool:
    """Return whether all visible pieces respect color-wise material caps."""

    for (_color, kind), count in material_counts(board).items():
        if str(kind) not in caps or int(count) > int(caps[str(kind)]):
            return False
    return True


def validate_square_chess_material(
    board: Sequence[Sequence[ChessPiece | None]],
    *,
    require_both_kings: bool = True,
    enforce_standard_pawn_rows: bool = True,
    enforce_non_adjacent_kings: bool = True,
    caps: Mapping[str, int] = STANDARD_CHESS_MATERIAL_CAPS,
) -> bool:
    """Return whether one square-board scene is material-plausible."""

    if not validate_material_caps(board, caps=caps):
        return False
    for color in (WHITE, BLACK):
        kings = king_coords(board, color)
        if bool(require_both_kings) and len(kings) != 1:
            return False
        if not bool(require_both_kings) and len(kings) > int(caps.get("king", 1)):
            return False
    if bool(enforce_standard_pawn_rows):
        for row_index, row in enumerate(board):
            for piece in row:
                if piece is not None and str(piece.kind) == "pawn" and not standard_pawn_row_valid(int(row_index)):
                    return False
    if bool(enforce_non_adjacent_kings):
        for white_king in king_coords(board, WHITE):
            for black_king in king_coords(board, BLACK):
                if coords_adjacent(white_king, black_king):
                    return False
    return True


def validate_circular_chess_material(board: Sequence[Sequence[ChessPiece | None]]) -> bool:
    """Return whether one circular-chess scene respects its no-pawn material caps."""

    return validate_material_caps(board, caps=CIRCULAR_CHESS_MATERIAL_CAPS)


def serialize_board(board: Sequence[Sequence[ChessPiece | None]]) -> list[list[str | None]]:
    """Return a JSON-friendly board representation."""

    rows: list[list[str | None]] = []
    for row in board:
        rows.append([None if piece is None else f"{piece.color}_{piece.kind}" for piece in row])
    return rows


def _ray_destinations(
    board: Sequence[Sequence[ChessPiece | None]],
    coord: Coord,
    piece: ChessPiece,
    directions: Iterable[Tuple[int, int]],
    *,
    include_opponent_king: bool = False,
) -> Tuple[Coord, ...]:
    """Return pseudo-legal destinations along sliding-piece rays."""

    out: List[Coord] = []
    row, col = int(coord[0]), int(coord[1])
    for dr, dc in directions:
        r = int(row + dr)
        c = int(col + dc)
        while in_bounds(r, c):
            occupant = board[r][c]
            if occupant is None:
                out.append((int(r), int(c)))
            else:
                if str(occupant.color) != str(piece.color) and (
                    bool(include_opponent_king) or str(occupant.kind) != "king"
                ):
                    out.append((int(r), int(c)))
                break
            r += int(dr)
            c += int(dc)
    return tuple(out)


def _ray_attack_squares(
    board: Sequence[Sequence[ChessPiece | None]],
    coord: Coord,
    directions: Iterable[Tuple[int, int]],
) -> Tuple[Coord, ...]:
    """Return all squares attacked along sliding-piece rays."""

    out: List[Coord] = []
    row, col = int(coord[0]), int(coord[1])
    for dr, dc in directions:
        r = int(row + dr)
        c = int(col + dc)
        while in_bounds(r, c):
            out.append((int(r), int(c)))
            if board[r][c] is not None:
                break
            r += int(dr)
            c += int(dc)
    return tuple(out)


def piece_move_destinations(
    board: Sequence[Sequence[ChessPiece | None]],
    coord: Coord,
    *,
    include_pawn_double_step: bool = True,
    include_opponent_king: bool = False,
) -> Tuple[Coord, ...]:
    """Return normal one-move destinations for a piece, excluding castling and en passant."""

    row, col = int(coord[0]), int(coord[1])
    piece = board[row][col]
    if piece is None:
        return ()
    kind = str(piece.kind)
    if kind == "queen":
        return _ray_destinations(
            board,
            coord,
            piece,
            ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
            include_opponent_king=bool(include_opponent_king),
        )
    if kind == "rook":
        return _ray_destinations(
            board,
            coord,
            piece,
            ((-1, 0), (1, 0), (0, -1), (0, 1)),
            include_opponent_king=bool(include_opponent_king),
        )
    if kind == "bishop":
        return _ray_destinations(
            board,
            coord,
            piece,
            ((-1, -1), (-1, 1), (1, -1), (1, 1)),
            include_opponent_king=bool(include_opponent_king),
        )
    if kind == "knight":
        moves: List[Coord] = []
        for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
            r = int(row + dr)
            c = int(col + dc)
            if not in_bounds(r, c):
                continue
            occupant = board[r][c]
            if occupant is None or (
                str(occupant.color) != str(piece.color)
                and (bool(include_opponent_king) or str(occupant.kind) != "king")
            ):
                moves.append((int(r), int(c)))
        return tuple(moves)
    if kind == "king":
        moves = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if int(dr) == 0 and int(dc) == 0:
                    continue
                r = int(row + dr)
                c = int(col + dc)
                if not in_bounds(r, c):
                    continue
                occupant = board[r][c]
                if occupant is None or (
                    str(occupant.color) != str(piece.color)
                    and (bool(include_opponent_king) or str(occupant.kind) != "king")
                ):
                    moves.append((int(r), int(c)))
        return tuple(moves)
    if kind == "pawn":
        moves = []
        direction = -1 if str(piece.color) == WHITE else 1
        start_row = 6 if str(piece.color) == WHITE else 1
        one_row = int(row + direction)
        if in_bounds(one_row, col) and board[one_row][col] is None:
            moves.append((int(one_row), int(col)))
            two_row = int(row + (2 * direction))
            if (
                bool(include_pawn_double_step)
                and int(row) == int(start_row)
                and in_bounds(two_row, col)
                and board[two_row][col] is None
            ):
                moves.append((int(two_row), int(col)))
        for dc in (-1, 1):
            r = int(row + direction)
            c = int(col + dc)
            if not in_bounds(r, c):
                continue
            occupant = board[r][c]
            if (
                occupant is not None
                and str(occupant.color) != str(piece.color)
                and (bool(include_opponent_king) or str(occupant.kind) != "king")
            ):
                moves.append((int(r), int(c)))
        return tuple(moves)
    raise ValueError(f"unsupported chess piece kind: {kind}")


def pawn_attack_squares(coord: Coord, color: str) -> Tuple[Coord, ...]:
    """Return attacked squares for a pawn regardless of occupancy."""

    row, col = int(coord[0]), int(coord[1])
    direction = -1 if str(color) == WHITE else 1
    squares: List[Coord] = []
    for dc in (-1, 1):
        r = int(row + direction)
        c = int(col + dc)
        if in_bounds(r, c):
            squares.append((int(r), int(c)))
    return tuple(squares)


def piece_attack_squares(board: Sequence[Sequence[ChessPiece | None]], coord: Coord) -> Tuple[Coord, ...]:
    """Return squares attacked by a piece, independent of legal landing rules."""

    row, col = int(coord[0]), int(coord[1])
    piece = board[row][col]
    if piece is None:
        return ()
    kind = str(piece.kind)
    if kind == "pawn":
        return pawn_attack_squares(coord, str(piece.color))
    if kind == "queen":
        return _ray_attack_squares(
            board,
            coord,
            ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
        )
    if kind == "rook":
        return _ray_attack_squares(board, coord, ((-1, 0), (1, 0), (0, -1), (0, 1)))
    if kind == "bishop":
        return _ray_attack_squares(board, coord, ((-1, -1), (-1, 1), (1, -1), (1, 1)))
    if kind == "knight":
        squares: List[Coord] = []
        for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
            r = int(row + dr)
            c = int(col + dc)
            if in_bounds(r, c):
                squares.append((int(r), int(c)))
        return tuple(squares)
    if kind == "king":
        squares = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if int(dr) == 0 and int(dc) == 0:
                    continue
                r = int(row + dr)
                c = int(col + dc)
                if in_bounds(r, c):
                    squares.append((int(r), int(c)))
        return tuple(squares)
    raise ValueError(f"unsupported chess piece kind: {kind}")


def piece_attacks_square(
    board: Sequence[Sequence[ChessPiece | None]],
    origin: Coord,
    target: Coord,
) -> bool:
    """Return whether the origin piece attacks the target square."""

    row, col = int(origin[0]), int(origin[1])
    piece = board[row][col]
    if piece is None:
        return False
    return tuple(target) in piece_attack_squares(board, origin)


def piece_capture_targets(board: Sequence[Sequence[ChessPiece | None]], coord: Coord) -> Tuple[Coord, ...]:
    """Return opponent-occupied squares capturable by one piece in a single move."""

    row, col = int(coord[0]), int(coord[1])
    piece = board[row][col]
    if piece is None:
        return ()
    captures = []
    for dest in piece_move_destinations(board, coord):
        occupant = board[int(dest[0])][int(dest[1])]
        if occupant is not None and str(occupant.color) != str(piece.color) and str(occupant.kind) != "king":
            captures.append((int(dest[0]), int(dest[1])))
    return tuple(captures)


def capturable_opponent_coords(board: Sequence[Sequence[ChessPiece | None]], player_color: str) -> Tuple[Coord, ...]:
    """Return unique opponent-piece squares capturable by the requested side."""

    captures: set[Coord] = set()
    for coord in occupied_coords(board):
        piece = board[int(coord[0])][int(coord[1])]
        if piece is None or str(piece.color) != str(player_color):
            continue
        captures.update(piece_capture_targets(board, coord))
    return tuple(sorted(captures))


def attackers_to_square(
    board: Sequence[Sequence[ChessPiece | None]],
    target: Coord,
    attacker_color: str,
) -> Tuple[Coord, ...]:
    """Return all pieces of one color attacking a target square."""

    attackers: List[Coord] = []
    for coord in occupied_coords(board):
        piece = board[int(coord[0])][int(coord[1])]
        if piece is None or str(piece.color) != str(attacker_color):
            continue
        if piece_attacks_square(board, coord, target):
            attackers.append((int(coord[0]), int(coord[1])))
    return tuple(sorted(attackers))


def find_king(board: Sequence[Sequence[ChessPiece | None]], color: str) -> Coord | None:
    """Return the coordinate of the requested king, if present."""

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece is not None and str(piece.color) == str(color) and str(piece.kind) == "king":
                return (int(row), int(col))
    return None


def apply_chess_move(board: Sequence[Sequence[ChessPiece | None]], source: Coord, destination: Coord) -> Board:
    """Return a new board after moving one piece from source to destination."""

    sr, sc = int(source[0]), int(source[1])
    dr, dc = int(destination[0]), int(destination[1])
    piece = board[sr][sc]
    if piece is None:
        raise ValueError("cannot move from an empty chess square")
    occupant = board[dr][dc]
    if occupant is not None and str(occupant.color) == str(piece.color):
        raise ValueError("cannot move onto a friendly chess piece")
    if occupant is not None and str(occupant.kind) == "king":
        raise ValueError("cannot capture a king")
    mutable = [list(row) for row in board]
    mutable[sr][sc] = None
    mutable[dr][dc] = piece
    return freeze_board(mutable)


def is_king_in_check(board: Sequence[Sequence[ChessPiece | None]], color: str) -> bool:
    """Return whether the requested color's king is currently attacked."""

    king_coord = find_king(board, str(color))
    if king_coord is None:
        return False
    return bool(attackers_to_square(board, king_coord, opponent(str(color))))


def legal_chess_moves_for_color(board: Sequence[Sequence[ChessPiece | None]], color: str) -> Tuple[Tuple[Coord, Coord], ...]:
    """Return legal normal moves for one color under Trace's simplified chess rules."""

    moves: List[Tuple[Coord, Coord]] = []
    for source in occupied_coords(board):
        piece = board[int(source[0])][int(source[1])]
        if piece is None or str(piece.color) != str(color):
            continue
        for destination in piece_move_destinations(board, source, include_pawn_double_step=False):
            try:
                moved_board = apply_chess_move(board, source, destination)
            except ValueError:
                continue
            if not is_king_in_check(moved_board, str(color)):
                moves.append((source, (int(destination[0]), int(destination[1]))))
    return tuple(sorted(moves))


def move_checkmates(board: Sequence[Sequence[ChessPiece | None]], source: Coord, destination: Coord) -> bool:
    """Return whether a normal move immediately checkmates the opponent king."""

    piece = board[int(source[0])][int(source[1])]
    if piece is None:
        return False
    if (tuple(source), tuple(destination)) not in legal_chess_moves_for_color(board, str(piece.color)):
        return False
    moved_board = apply_chess_move(board, source, destination)
    defender_color = opponent(str(piece.color))
    return is_king_in_check(moved_board, defender_color) and not legal_chess_moves_for_color(moved_board, defender_color)


def king_escape_squares(board: Sequence[Sequence[ChessPiece | None]], king_coord: Coord) -> Tuple[Coord, ...]:
    """Return legal one-step king destinations that are not attacked after moving."""

    row, col = int(king_coord[0]), int(king_coord[1])
    piece = board[row][col]
    if piece is None or str(piece.kind) != "king":
        return ()
    escapes: List[Coord] = []
    for dest in piece_move_destinations(board, king_coord, include_pawn_double_step=False):
        dest_piece = board[int(dest[0])][int(dest[1])]
        if dest_piece is not None and str(dest_piece.kind) == "king":
            continue
        mutable = [list(board_row) for board_row in board]
        mutable[row][col] = None
        mutable[int(dest[0])][int(dest[1])] = piece
        moved_board = freeze_board(mutable)
        if not attackers_to_square(moved_board, dest, opponent(str(piece.color))):
            escapes.append((int(dest[0]), int(dest[1])))
    return tuple(sorted(escapes))


__all__ = [
    "BLACK",
    "BOARD_SIZE",
    "Board",
    "ChessPiece",
    "Coord",
    "CIRCULAR_CHESS_MATERIAL_CAPS",
    "NON_KING_PIECE_KINDS",
    "PIECE_KINDS",
    "STANDARD_CHESS_MATERIAL_CAPS",
    "WHITE",
    "attackers_to_square",
    "apply_chess_move",
    "capturable_opponent_coords",
    "color_name",
    "coord_to_cell_id",
    "coord_to_square_name",
    "coords_adjacent",
    "empty_board",
    "find_king",
    "freeze_board",
    "in_bounds",
    "is_king_in_check",
    "king_escape_squares",
    "king_coords",
    "legal_chess_moves_for_color",
    "legal_material_piece_choices",
    "material_count",
    "material_counts",
    "move_checkmates",
    "occupied_coords",
    "occupied_piece_count",
    "opponent",
    "pawn_attack_squares",
    "piece_attacks_square",
    "piece_attack_squares",
    "piece_capture_targets",
    "piece_move_destinations",
    "piece_name",
    "piece_to_entity_id",
    "sample_material_piece",
    "serialize_board",
    "standard_pawn_row_valid",
    "validate_circular_chess_material",
    "validate_material_caps",
    "validate_square_chess_material",
]
