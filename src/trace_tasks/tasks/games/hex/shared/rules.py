"""Hex-board mechanics and sample validation primitives."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


EMPTY = 0
RED = 1
BLUE = -1
HEX_MODE_WINNING_MOVE = "winning_move"
HEX_MODE_CONNECTION_GAP = "connection_gap"
HEX_MODE_NEIGHBOR_COUNT = "neighbor_count"
SUPPORTED_HEX_SCENE_VARIANTS: Tuple[str, ...] = (
    "open_board",
    "crowded_board",
)
SUPPORTED_HEX_PLAYER_COLORS: Tuple[str, ...] = ("red", "blue")
HEX_CANDIDATE_LABELS: Tuple[str, ...] = tuple("ABCDEFGH")

Coord = Tuple[int, int]
Board = Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class HexCandidateSpec:
    """One labeled empty Hex cell."""

    label: str
    coord: Coord
    is_answer: bool


@dataclass(frozen=True)
class HexSample:
    """Generated Hex board state and task answer."""

    board_size: int
    mode: str
    scene_variant: str
    player_color: str
    player_value: int
    board: Board
    answer: str | int
    target_answer: str | int
    candidate_specs: Tuple[HexCandidateSpec, ...]
    annotation_coords: Tuple[Coord, ...]
    winning_move_coord: Coord | None
    min_gap_path: Tuple[Coord, ...]
    min_gap_empty_coords: Tuple[Coord, ...]
    construction_mode: str
    reference_coord: Coord | None = None
    neighbor_target_state: str | None = None
    neighbor_match_coords: Tuple[Coord, ...] = tuple()


@dataclass(frozen=True)
class HexGapSetSearch:
    """Minimum connection gap sets discovered for a Hex position."""

    gap_count: int
    gap_sets: Tuple[Tuple[Coord, ...], ...]
    exhaustive: bool


def color_value(color: str) -> int:
    """Return the board value for one Hex player color."""

    normalized = str(color).lower()
    if normalized == "red":
        return RED
    if normalized == "blue":
        return BLUE
    raise ValueError(f"unsupported Hex player color: {color}")


def color_name(value: int) -> str:
    """Return the prompt-facing color for one Hex board value."""

    return "Red" if int(value) == int(RED) else "Blue"


def opponent(value: int) -> int:
    """Return the opponent value for one Hex player."""

    return BLUE if int(value) == int(RED) else RED


def coord_to_cell_id(coord: Coord) -> str:
    """Return the stable render/entity id for one Hex cell."""

    row, col = coord
    return f"hex_r{int(row)}_c{int(col)}"


def all_coords(size: int) -> Tuple[Coord, ...]:
    """Return all Hex-board coordinates in row-major order."""

    return tuple((row, col) for row in range(int(size)) for col in range(int(size)))


def sorted_coords(coords: Iterable[Coord]) -> Tuple[Coord, ...]:
    """Return canonical sorted Hex coordinates."""

    return tuple(sorted((int(row), int(col)) for row, col in coords))


def board_from_rows(rows: Sequence[Sequence[int]]) -> Board:
    """Return one immutable Hex board from row-major values."""

    return tuple(tuple(int(value) for value in row) for row in rows)


def neighbors(coord: Coord, *, board_size: int) -> Tuple[Coord, ...]:
    """Return in-bounds Hex neighbors for a rhombus board."""

    row, col = int(coord[0]), int(coord[1])
    candidates = (
        (row - 1, col),
        (row - 1, col + 1),
        (row, col - 1),
        (row, col + 1),
        (row + 1, col - 1),
        (row + 1, col),
    )
    return tuple(
        (int(next_row), int(next_col))
        for next_row, next_col in candidates
        if 0 <= int(next_row) < int(board_size) and 0 <= int(next_col) < int(board_size)
    )


def side_coords(*, board_size: int, player_value: int, side: str) -> Tuple[Coord, ...]:
    """Return coordinates touching one connecting side for a player."""

    size = int(board_size)
    if int(player_value) == int(RED):
        col = 0 if str(side) == "start" else size - 1
        return tuple((row, col) for row in range(size))
    row = 0 if str(side) == "start" else size - 1
    return tuple((row, col) for col in range(size))


def has_connection(board: Sequence[Sequence[int]], *, player_value: int) -> bool:
    """Return whether the player has a complete side-to-side Hex connection."""

    size = int(len(board))
    starts = [
        coord
        for coord in side_coords(board_size=size, player_value=int(player_value), side="start")
        if int(board[coord[0]][coord[1]]) == int(player_value)
    ]
    targets = set(side_coords(board_size=size, player_value=int(player_value), side="end"))
    seen: set[Coord] = set()
    stack: List[Coord] = list(starts)
    while stack:
        coord = stack.pop()
        if coord in seen:
            continue
        seen.add(coord)
        if coord in targets:
            return True
        for neighbor in neighbors(coord, board_size=size):
            if neighbor not in seen and int(board[neighbor[0]][neighbor[1]]) == int(player_value):
                stack.append(neighbor)
    return False


def winning_path_after_move(
    board: Sequence[Sequence[int]],
    *,
    player_value: int,
    move_coord: Coord,
) -> Tuple[Coord, ...]:
    """Return one winning path after placing a player stone at `move_coord`."""

    size = int(len(board))
    rows = [list(row) for row in board]
    rows[int(move_coord[0])][int(move_coord[1])] = int(player_value)
    starts = [
        coord
        for coord in side_coords(board_size=size, player_value=int(player_value), side="start")
        if int(rows[coord[0]][coord[1]]) == int(player_value)
    ]
    targets = set(side_coords(board_size=size, player_value=int(player_value), side="end"))
    parent: Dict[Coord, Coord | None] = {coord: None for coord in starts}
    queue: List[Coord] = list(starts)
    for coord in queue:
        if coord in targets:
            path: List[Coord] = [coord]
            current = coord
            while parent[current] is not None:
                current = parent[current]  # type: ignore[assignment]
                path.append(current)
            return tuple(reversed(path))
        for neighbor in neighbors(coord, board_size=size):
            if neighbor in parent:
                continue
            if int(rows[neighbor[0]][neighbor[1]]) != int(player_value):
                continue
            parent[neighbor] = coord
            queue.append(neighbor)
    return tuple()


def immediate_winning_moves(board: Sequence[Sequence[int]], *, player_value: int) -> Tuple[Coord, ...]:
    """Return every empty cell that gives the player an immediate Hex win."""

    if has_connection(board, player_value=int(player_value)):
        return tuple()
    winners: list[Coord] = []
    for coord in all_coords(len(board)):
        if int(board[coord[0]][coord[1]]) != EMPTY:
            continue
        rows = [list(row) for row in board]
        rows[coord[0]][coord[1]] = int(player_value)
        if has_connection(rows, player_value=int(player_value)):
            winners.append(coord)
    return tuple(winners)


def minimum_connection_path(
    board: Sequence[Sequence[int]],
    *,
    player_value: int,
) -> Tuple[int, Tuple[Coord, ...]]:
    """Return the minimum empty-cell fill count and one witness path."""

    size = int(len(board))
    blocked = int(opponent(int(player_value)))

    def cell_cost(coord: Coord) -> int | None:
        value = int(board[coord[0]][coord[1]])
        if value == blocked:
            return None
        return 0 if value == int(player_value) else 1

    starts = side_coords(board_size=size, player_value=int(player_value), side="start")
    targets = set(side_coords(board_size=size, player_value=int(player_value), side="end"))
    heap: list[tuple[int, int, Coord]] = []
    counter = 0
    best: Dict[Coord, int] = {}
    parent: Dict[Coord, Coord | None] = {}
    for coord in starts:
        cost = cell_cost(coord)
        if cost is None:
            continue
        best[coord] = int(cost)
        parent[coord] = None
        heapq.heappush(heap, (int(cost), counter, coord))
        counter += 1

    while heap:
        cost, _order, coord = heapq.heappop(heap)
        if int(cost) != int(best.get(coord, 10**9)):
            continue
        if coord in targets:
            path: list[Coord] = [coord]
            current = coord
            while parent[current] is not None:
                current = parent[current]  # type: ignore[assignment]
                path.append(current)
            return int(cost), tuple(reversed(path))
        for neighbor in neighbors(coord, board_size=size):
            step = cell_cost(neighbor)
            if step is None:
                continue
            next_cost = int(cost) + int(step)
            if next_cost < int(best.get(neighbor, 10**9)):
                best[neighbor] = int(next_cost)
                parent[neighbor] = coord
                heapq.heappush(heap, (int(next_cost), counter, neighbor))
                counter += 1
    return 10**9, tuple()


def minimum_connection_gap_sets(
    board: Sequence[Sequence[int]],
    *,
    player_value: int,
    max_sets: int = 2,
    max_states: int = 200_000,
) -> HexGapSetSearch:
    """Return distinct minimum empty-cell sets that can complete the connection.

    The scalar answer for the gap-count task is the minimum number of empty cells.
    Its public annotation is the set of empty cells that witnesses that count, so
    generation needs to reject boards with multiple distinct minimum gap sets.
    This search stops once `max_sets` distinct sets are found.
    """

    min_cost, _path = minimum_connection_path(board, player_value=int(player_value))
    if int(min_cost) >= 10**9:
        return HexGapSetSearch(gap_count=int(min_cost), gap_sets=tuple(), exhaustive=True)

    size = int(len(board))
    blocked = int(opponent(int(player_value)))

    def cell_cost(coord: Coord) -> int | None:
        value = int(board[coord[0]][coord[1]])
        if value == blocked:
            return None
        return 0 if value == int(player_value) else 1

    targets = set(side_coords(board_size=size, player_value=int(player_value), side="end"))

    # suffix[coord] is the minimum additional empty-cell cost needed to reach
    # an end side from coord, excluding coord itself.
    suffix: Dict[Coord, int] = {}
    heap: list[tuple[int, int, Coord]] = []
    counter = 0
    for coord in sorted(targets):
        if cell_cost(coord) is None:
            continue
        suffix[coord] = 0
        heapq.heappush(heap, (0, counter, coord))
        counter += 1
    while heap:
        cost, _order, coord = heapq.heappop(heap)
        if int(cost) != int(suffix.get(coord, 10**9)):
            continue
        step_into_coord = cell_cost(coord)
        if step_into_coord is None:
            continue
        for previous in neighbors(coord, board_size=size):
            if cell_cost(previous) is None:
                continue
            next_cost = int(cost) + int(step_into_coord)
            if next_cost < int(suffix.get(previous, 10**9)):
                suffix[previous] = int(next_cost)
                heapq.heappush(heap, (int(next_cost), counter, previous))
                counter += 1

    starts = tuple(
        coord
        for coord in side_coords(board_size=size, player_value=int(player_value), side="start")
        if cell_cost(coord) is not None
    )
    seen_gap_sets: set[Tuple[Coord, ...]] = set()
    gap_sets: list[Tuple[Coord, ...]] = []
    stack: list[tuple[Coord, int, Tuple[Coord, ...], Tuple[Coord, ...]]] = []
    for start in reversed(starts):
        start_cost = cell_cost(start)
        if start_cost is None:
            continue
        if int(start_cost) + int(suffix.get(start, 10**9)) > int(min_cost):
            continue
        empty_coords = (start,) if int(start_cost) == 1 else tuple()
        stack.append((start, int(start_cost), (start,), empty_coords))

    states_seen = 0
    exhaustive = True
    while stack:
        coord, cost, path, empty_coords = stack.pop()
        states_seen += 1
        if states_seen > int(max_states):
            exhaustive = False
            break
        if int(cost) + int(suffix.get(coord, 10**9)) > int(min_cost):
            continue
        if coord in targets:
            if int(cost) == int(min_cost):
                gap_set = sorted_coords(empty_coords)
                if gap_set not in seen_gap_sets:
                    seen_gap_sets.add(gap_set)
                    gap_sets.append(gap_set)
                    if len(gap_sets) >= int(max_sets):
                        break
            continue
        visited = set(path)
        for neighbor in reversed(neighbors(coord, board_size=size)):
            if neighbor in visited:
                continue
            step = cell_cost(neighbor)
            if step is None:
                continue
            next_cost = int(cost) + int(step)
            if next_cost > int(min_cost):
                continue
            if next_cost + int(suffix.get(neighbor, 10**9)) > int(min_cost):
                continue
            next_empty = empty_coords + ((neighbor,) if int(step) == 1 else tuple())
            stack.append((neighbor, int(next_cost), path + (neighbor,), next_empty))

    return HexGapSetSearch(
        gap_count=int(min_cost),
        gap_sets=tuple(gap_sets),
        exhaustive=bool(exhaustive),
    )


def make_connection_path(*, rng, board_size: int, player_value: int) -> Tuple[Coord, ...]:
    """Sample one monotone side-to-side path for the active player."""

    size = int(board_size)
    if int(player_value) == int(RED):
        row = int(rng.randint(0, size - 1))
        path = [(row, 0)]
        for col in range(1, size):
            if row > 0 and int(rng.randint(0, 1)) == 1:
                row -= 1
            elif row < size - 1 and int(rng.randint(0, 3)) == 0:
                row += 1
            path.append((row, col))
        return tuple(path)

    col = int(rng.randint(0, size - 1))
    path = [(0, col)]
    for row in range(1, size):
        if col > 0 and int(rng.randint(0, 1)) == 1:
            col -= 1
        elif col < size - 1 and int(rng.randint(0, 3)) == 0:
            col += 1
        path.append((row, col))
    return tuple(path)


def validate_hex_sample(sample: HexSample) -> None:
    """Validate task answer and annotation contracts for one Hex sample."""

    board = tuple(tuple(int(value) for value in row) for row in sample.board)
    if len(board) != int(sample.board_size) or any(len(row) != int(sample.board_size) for row in board):
        raise ValueError("Hex board dimensions do not match board_size")
    if int(sample.player_value) != int(color_value(sample.player_color)):
        raise ValueError("Hex player color and value disagree")
    if str(sample.mode) == HEX_MODE_WINNING_MOVE:
        if sample.winning_move_coord is None:
            raise ValueError("Hex winning-move sample is missing winning_move_coord")
        if int(board[sample.winning_move_coord[0]][sample.winning_move_coord[1]]) != EMPTY:
            raise ValueError("Hex winning move must be an empty cell")
        answer_specs = [spec for spec in sample.candidate_specs if bool(spec.is_answer)]
        if len(answer_specs) != 1:
            raise ValueError("Hex winning-move sample must have exactly one answer candidate")
        if str(answer_specs[0].label) != str(sample.answer):
            raise ValueError("Hex winning-move answer does not match candidate label")
        if tuple(answer_specs[0].coord) != tuple(sample.winning_move_coord):
            raise ValueError("Hex winning-move answer candidate coord mismatch")
        winning_moves = immediate_winning_moves(board, player_value=int(sample.player_value))
        if tuple(sample.winning_move_coord) not in winning_moves:
            raise ValueError("Hex labeled winning move does not win immediately")
        if len(winning_moves) != 1:
            raise ValueError("Hex winning-move board must have exactly one immediate winning cell")
        path = winning_path_after_move(
            board,
            player_value=int(sample.player_value),
            move_coord=sample.winning_move_coord,
        )
        if tuple(sample.annotation_coords) != tuple(path):
            raise ValueError("Hex winning-move annotation must be one completed winning path")
        return

    if str(sample.mode) == HEX_MODE_CONNECTION_GAP:
        gap_count, path = minimum_connection_path(board, player_value=int(sample.player_value))
        if int(sample.answer) != int(gap_count):
            raise ValueError("Hex gap-count answer does not match shortest path cost")
        gap_search = minimum_connection_gap_sets(board, player_value=int(sample.player_value), max_sets=2)
        if not bool(gap_search.exhaustive):
            raise ValueError("Hex gap-count minimum gap-set search did not finish")
        if int(gap_search.gap_count) != int(gap_count):
            raise ValueError("Hex gap-count gap-set search disagrees with shortest path cost")
        if len(gap_search.gap_sets) != 1:
            raise ValueError("Hex gap-count board must have exactly one minimum empty-cell annotation set")
        if tuple(sample.min_gap_path) != tuple(path):
            raise ValueError("Hex gap-count stored path does not match shortest path")
        empty_on_path = tuple(coord for coord in path if int(board[coord[0]][coord[1]]) == EMPTY)
        if sorted_coords(sample.min_gap_empty_coords) != tuple(gap_search.gap_sets[0]):
            raise ValueError("Hex gap-count empty-cell annotation set mismatch")
        if sorted_coords(empty_on_path) != tuple(gap_search.gap_sets[0]):
            raise ValueError("Hex gap-count stored path does not use the unique minimum gap set")
        if sorted_coords(sample.annotation_coords) != tuple(gap_search.gap_sets[0]):
            raise ValueError("Hex gap-count annotation must be the unique minimum empty-cell set")
        return

    if str(sample.mode) == HEX_MODE_NEIGHBOR_COUNT:
        if sample.reference_coord is None:
            raise ValueError("Hex neighbor-count sample is missing reference_coord")
        reference_coord = (int(sample.reference_coord[0]), int(sample.reference_coord[1]))
        if not (0 <= reference_coord[0] < int(sample.board_size) and 0 <= reference_coord[1] < int(sample.board_size)):
            raise ValueError("Hex neighbor-count reference_coord is out of bounds")
        adjacent = neighbors(reference_coord, board_size=int(sample.board_size))
        if len(adjacent) != 6:
            raise ValueError("Hex neighbor-count reference cell must have six adjacent cells")
        target_state = str(sample.neighbor_target_state or "").lower()
        target_value_by_state = {
            "red": RED,
            "blue": BLUE,
            "empty": EMPTY,
        }
        if target_state not in target_value_by_state:
            raise ValueError("Hex neighbor-count sample has unsupported target state")
        target_value = int(target_value_by_state[str(target_state)])
        expected = sorted_coords(
            coord
            for coord in adjacent
            if int(board[coord[0]][coord[1]]) == int(target_value)
        )
        if int(sample.answer) != len(expected):
            raise ValueError("Hex neighbor-count answer does not match adjacent-cell state count")
        if sorted_coords(sample.annotation_coords) != expected:
            raise ValueError("Hex neighbor-count annotation must be matching adjacent-cell centers")
        if sorted_coords(sample.neighbor_match_coords) != expected:
            raise ValueError("Hex neighbor-count stored match coords mismatch")
        if reference_coord in expected:
            raise ValueError("Hex neighbor-count annotation must not include the reference cell")
        return

    raise ValueError(f"unsupported Hex mode: {sample.mode}")


__all__ = [
    "BLUE",
    "EMPTY",
    "HEX_CANDIDATE_LABELS",
    "RED",
    "Board",
    "Coord",
    "HexCandidateSpec",
    "HexGapSetSearch",
    "HexSample",
    "HEX_MODE_CONNECTION_GAP",
    "HEX_MODE_NEIGHBOR_COUNT",
    "HEX_MODE_WINNING_MOVE",
    "SUPPORTED_HEX_PLAYER_COLORS",
    "SUPPORTED_HEX_SCENE_VARIANTS",
    "all_coords",
    "board_from_rows",
    "color_name",
    "color_value",
    "coord_to_cell_id",
    "has_connection",
    "immediate_winning_moves",
    "make_connection_path",
    "minimum_connection_path",
    "minimum_connection_gap_sets",
    "neighbors",
    "opponent",
    "side_coords",
    "sorted_coords",
    "validate_hex_sample",
    "winning_path_after_move",
]
