"""Shared Go board-state helpers for group-property games tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple


EMPTY = 0
BLACK = 1
WHITE = -1
BOARD_SIZE = 7
SUPPORTED_GO_PLAYER_COLORS: Tuple[str, ...] = ("black", "white")
GO_RULE_GROUP_LIBERTIES = "group_liberties"
GO_RULE_ADJACENT_ENEMY_STONES = "adjacent_enemy_stones"
GO_RULE_SHARED_LIBERTIES = "shared_liberties"
GO_RULE_MARKED_GROUP_STONES = "marked_group_stones"

Coord = Tuple[int, int]
Board = Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class GoStoneSpec:
    """One visible stone on the Go board."""

    stone_id: str
    point_id: str
    row: int
    col: int
    color: str
    is_marked_group: bool


@dataclass(frozen=True)
class GoBoardState:
    """One visible Go board plus the highlighted group and its liberties."""

    board: Board
    marked_group_color: int
    marked_group_coords: Tuple[Coord, ...]
    liberty_coords: Tuple[Coord, ...]
    adjacent_enemy_coords: Tuple[Coord, ...]
    shared_liberty_coords: Tuple[Coord, ...]
    stone_specs: Tuple[GoStoneSpec, ...]
    scene_variant: str


def color_name(color: int) -> str:
    """Return the canonical prompt-facing color name for one stone value."""

    return "Black" if int(color) == int(BLACK) else "White"


def coord_to_point_id(coord: Coord) -> str:
    """Return one stable entity id for a board intersection."""

    return f"point_r{int(coord[0])}_c{int(coord[1])}"


def coord_to_stone_id(coord: Coord) -> str:
    """Return one stable entity id for an occupied intersection."""

    return f"stone_r{int(coord[0])}_c{int(coord[1])}"


def opponent(color: int) -> int:
    """Return the opposite Go color."""

    return int(WHITE if int(color) == int(BLACK) else BLACK)


def neighbors(coord: Coord, *, board_size: int = BOARD_SIZE) -> Tuple[Coord, ...]:
    """Return orthogonal in-bounds neighbors for one intersection."""

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
        if 0 <= int(next_row) < int(board_size) and 0 <= int(next_col) < int(board_size)
    )


def _board_from_rows(rows: Sequence[Sequence[int]]) -> Board:
    """Return one immutable board from row-major cells."""

    return tuple(tuple(int(cell) for cell in row) for row in rows)


def connected_group(board: Sequence[Sequence[int]], start: Coord) -> Tuple[Coord, ...]:
    """Return the full same-color connected group containing `start`."""

    row, col = int(start[0]), int(start[1])
    color = int(board[row][col])
    if int(color) == int(EMPTY):
        return tuple()
    board_size = int(len(board))
    seen: Set[Coord] = set()
    stack: List[Coord] = [(int(row), int(col))]
    while stack:
        coord = stack.pop()
        if coord in seen:
            continue
        seen.add(coord)
        for neighbor in neighbors(coord, board_size=board_size):
            if int(board[neighbor[0]][neighbor[1]]) == int(color) and neighbor not in seen:
                stack.append(neighbor)
    return tuple(sorted(seen))


def group_liberties(board: Sequence[Sequence[int]], group: Iterable[Coord]) -> Tuple[Coord, ...]:
    """Return the empty orthogonal liberties for one connected group."""

    board_size = int(len(board))
    liberties: Set[Coord] = set()
    for row, col in group:
        for neighbor in neighbors((int(row), int(col)), board_size=board_size):
            if int(board[neighbor[0]][neighbor[1]]) == int(EMPTY):
                liberties.add((int(neighbor[0]), int(neighbor[1])))
    return tuple(sorted(liberties))


def all_groups_have_liberty(board: Sequence[Sequence[int]]) -> bool:
    """Return whether every occupied group on the board has at least one liberty."""

    board_size = int(len(board))
    seen: Set[Coord] = set()
    for row in range(board_size):
        for col in range(board_size):
            if int(board[row][col]) == int(EMPTY) or (row, col) in seen:
                continue
            group = connected_group(board, (int(row), int(col)))
            seen.update(group)
            if not group_liberties(board, group):
                return False
    return True


def stone_groups(board: Sequence[Sequence[int]], *, color: int) -> Tuple[Tuple[Coord, ...], ...]:
    """Return all same-color connected groups in stable row-major order."""

    board_size = int(len(board))
    target_color = int(color)
    if target_color == int(EMPTY):
        raise ValueError("stone_groups requires a non-empty stone color")
    seen: Set[Coord] = set()
    groups: List[Tuple[Coord, ...]] = []
    for row in range(board_size):
        for col in range(board_size):
            coord = (int(row), int(col))
            if coord in seen or int(board[row][col]) != int(target_color):
                continue
            group = connected_group(board, coord)
            seen.update(group)
            groups.append(tuple(sorted(group)))
    return tuple(sorted(groups, key=lambda group: group[0] if group else (9999, 9999)))


def liberty_point_ids(liberties: Iterable[Coord]) -> Tuple[str, ...]:
    """Return stable point ids for one liberty set."""

    return tuple(coord_to_point_id(coord) for coord in liberties)


def stone_ids_for_coords(coords: Iterable[Coord]) -> Tuple[str, ...]:
    """Return stable stone ids for occupied intersections."""

    return tuple(coord_to_stone_id(coord) for coord in coords)


def adjacent_enemy_coords(board: Sequence[Sequence[int]], group: Iterable[Coord]) -> Tuple[Coord, ...]:
    """Return unique opponent stones orthogonally adjacent to one group."""

    group_tuple = tuple((int(coord[0]), int(coord[1])) for coord in group)
    if not group_tuple:
        return tuple()
    first_row, first_col = group_tuple[0]
    group_color = int(board[int(first_row)][int(first_col)])
    if int(group_color) == int(EMPTY):
        return tuple()
    enemy_color = int(opponent(group_color))
    board_size = int(len(board))
    enemies: Set[Coord] = set()
    for coord in group_tuple:
        for neighbor in neighbors(coord, board_size=board_size):
            if int(board[neighbor[0]][neighbor[1]]) == int(enemy_color):
                enemies.add((int(neighbor[0]), int(neighbor[1])))
    return tuple(sorted(enemies))


def shared_liberty_coords(board: Sequence[Sequence[int]], group: Iterable[Coord]) -> Tuple[Coord, ...]:
    """Return liberties of one group that also touch at least one opponent stone."""

    group_tuple = tuple((int(coord[0]), int(coord[1])) for coord in group)
    if not group_tuple:
        return tuple()
    first_row, first_col = group_tuple[0]
    group_color = int(board[int(first_row)][int(first_col)])
    if int(group_color) == int(EMPTY):
        return tuple()
    enemy_color = int(opponent(group_color))
    board_size = int(len(board))
    shared: Set[Coord] = set()
    for liberty in group_liberties(board, group_tuple):
        for neighbor in neighbors(liberty, board_size=board_size):
            if int(board[neighbor[0]][neighbor[1]]) == int(enemy_color):
                shared.add((int(liberty[0]), int(liberty[1])))
                break
    return tuple(sorted(shared))


def supported_targets_for_mode(count_mode: str = GO_RULE_GROUP_LIBERTIES) -> Tuple[int, ...]:
    """Return supported count targets for one Go group-property query."""

    variant = str(count_mode)
    if variant == GO_RULE_GROUP_LIBERTIES:
        return (1, 2, 3, 4, 6)
    if variant == GO_RULE_ADJACENT_ENEMY_STONES:
        return (1, 2, 3, 4, 5, 6)
    if variant == GO_RULE_SHARED_LIBERTIES:
        return (1, 2, 3, 4, 5)
    if variant == GO_RULE_MARKED_GROUP_STONES:
        return (2, 3, 4, 5, 6)
    raise ValueError(f"unsupported Go count mode: {count_mode}")


def _sample_connected_group(rng, *, board_size: int, size: int, favor_center: bool) -> Tuple[Coord, ...]:
    """Sample one connected group footprint with optional center bias."""

    if bool(favor_center):
        row_support = tuple(range(1, int(board_size) - 1))
        col_support = tuple(range(1, int(board_size) - 1))
    else:
        row_support = tuple(range(int(board_size)))
        col_support = tuple(range(int(board_size)))
    start = (int(rng.choice(row_support)), int(rng.choice(col_support)))
    group: Set[Coord] = {start}
    frontier: Set[Coord] = set(neighbors(start, board_size=board_size))
    while len(group) < int(size) and frontier:
        ordered_frontier = sorted(frontier)
        coord = ordered_frontier[int(rng.randrange(len(ordered_frontier)))]
        frontier.remove(coord)
        group.add(coord)
        for neighbor in neighbors(coord, board_size=board_size):
            if neighbor not in group:
                frontier.add(neighbor)
    if len(group) != int(size):
        return tuple()
    return tuple(sorted(group))


def _boundary_neighbors(group: Iterable[Coord], *, board_size: int) -> Tuple[Coord, ...]:
    """Return the orthogonal boundary coordinates around a connected group."""

    group_set = {(
        int(coord[0]),
        int(coord[1]),
    ) for coord in group}
    boundary: Set[Coord] = set()
    for coord in group_set:
        for neighbor in neighbors(coord, board_size=board_size):
            if neighbor not in group_set:
                boundary.add(neighbor)
    return tuple(sorted(boundary))


def _minimum_group_size_for_target(target_answer: int) -> int:
    """Return a small connected-group size that can support the requested liberties."""

    return max(1, int(math.ceil((float(target_answer) - 2.0) / 2.0)))


def _minimum_group_size_for_adjacent_enemies(target_answer: int) -> int:
    """Return a small group size likely to expose enough boundary enemy slots."""

    return max(2, int(math.ceil((float(target_answer) - 2.0) / 2.0)))


def _minimum_group_size_for_shared_liberties(target_answer: int) -> int:
    """Return a small group size likely to expose enough shared-liberty slots."""

    return max(2, int(math.ceil((float(target_answer) + 1.0) / 2.0)))


def _stone_specs(board: Sequence[Sequence[int]], *, marked_group_coords: Iterable[Coord]) -> Tuple[GoStoneSpec, ...]:
    """Return visible stone specs in stable row-major order."""

    board_size = int(len(board))
    marked_group = {(int(coord[0]), int(coord[1])) for coord in marked_group_coords}
    specs: List[GoStoneSpec] = []
    for row in range(board_size):
        for col in range(board_size):
            color = int(board[row][col])
            if int(color) == int(EMPTY):
                continue
            coord = (int(row), int(col))
            specs.append(
                GoStoneSpec(
                    stone_id=coord_to_stone_id(coord),
                    point_id=coord_to_point_id(coord),
                    row=int(row),
                    col=int(col),
                    color=str(color_name(int(color)).lower()),
                    is_marked_group=coord in marked_group,
                )
            )
    return tuple(specs)


def build_go_board_state(
    *,
    rng,
    count_mode: str,
    scene_variant: str,
    target_answer: int,
    player_color: str | None = None,
    board_size: int = BOARD_SIZE,
    max_internal_attempts: int = 512,
) -> GoBoardState:
    """Construct one visible Go board with a marked group and exact query answer."""

    target = int(target_answer)
    if int(board_size) < 5:
        raise ValueError("Go liberty boards require board_size >= 5")
    variant = str(count_mode)
    if variant not in {
        GO_RULE_GROUP_LIBERTIES,
        GO_RULE_ADJACENT_ENEMY_STONES,
        GO_RULE_SHARED_LIBERTIES,
        GO_RULE_MARKED_GROUP_STONES,
    }:
        raise ValueError(f"unsupported Go count mode: {count_mode}")
    if target not in supported_targets_for_mode(variant):
        raise ValueError(f"unsupported Go target {target} for {variant}")
    color = str(player_color or "black")
    if color not in SUPPORTED_GO_PLAYER_COLORS:
        raise ValueError(f"unsupported Go player_color: {player_color}")
    if str(scene_variant) not in {"open_board", "crowded_board"}:
        raise ValueError(f"unsupported Go scene variant: {scene_variant}")

    marked_group_color = int(BLACK if color == "black" else WHITE)
    opponent_color = int(opponent(marked_group_color))
    extras_min, extras_max = (4, 10) if str(scene_variant) == "open_board" else (12, 20)

    for _ in range(max(1, int(max_internal_attempts))):
        if variant == GO_RULE_MARKED_GROUP_STONES:
            group_size = int(target)
            favor_center = bool(int(target) >= 5)
        elif variant == GO_RULE_ADJACENT_ENEMY_STONES:
            group_size_min = _minimum_group_size_for_adjacent_enemies(int(target))
            group_size_max = min(8, int(group_size_min) + 3)
            group_size = int(rng.randint(int(group_size_min), int(group_size_max)))
            favor_center = bool(int(target) >= 6)
        elif variant == GO_RULE_SHARED_LIBERTIES:
            group_size_min = _minimum_group_size_for_shared_liberties(int(target))
            group_size_max = min(8, int(group_size_min) + 3)
            group_size = int(rng.randint(int(group_size_min), int(group_size_max)))
            favor_center = True
        else:
            group_size_min = _minimum_group_size_for_target(int(target))
            group_size_max = min(6, int(group_size_min) + 2)
            group_size = int(rng.randint(int(group_size_min), int(group_size_max)))
            favor_center = bool(int(target) >= 6)
        group = _sample_connected_group(
            rng,
            board_size=int(board_size),
            size=int(group_size),
            favor_center=bool(favor_center),
        )
        if not group:
            continue
        boundary = _boundary_neighbors(group, board_size=int(board_size))
        if variant == GO_RULE_MARKED_GROUP_STONES:
            if not boundary:
                continue
            max_liberty_count = max(1, min(len(boundary), max(2, int(len(boundary) // 2))))
            liberty_count = int(rng.randint(1, int(max_liberty_count)))
            liberties = tuple(sorted(boundary[index] for index in rng.sample(range(len(boundary)), int(liberty_count))))
            liberty_set_for_enemies = set(liberties)
            enemy_candidates = tuple(coord for coord in boundary if coord not in liberty_set_for_enemies)
            max_enemy_count = min(len(enemy_candidates), 5)
            enemy_count = int(rng.randint(0, int(max_enemy_count))) if max_enemy_count > 0 else 0
            enemy_boundary = (
                tuple(sorted(enemy_candidates[index] for index in rng.sample(range(len(enemy_candidates)), int(enemy_count))))
                if enemy_count > 0
                else tuple()
            )
        elif variant == GO_RULE_GROUP_LIBERTIES:
            if len(boundary) < int(target):
                continue
            liberties = tuple(sorted(boundary[index] for index in rng.sample(range(len(boundary)), int(target))))
            enemy_boundary = tuple(sorted(coord for coord in boundary if coord not in set(liberties)))
        elif variant == GO_RULE_ADJACENT_ENEMY_STONES:
            if len(boundary) <= int(target):
                continue
            enemy_boundary = tuple(sorted(boundary[index] for index in rng.sample(range(len(boundary)), int(target))))
            enemy_set = set(enemy_boundary)
            liberties = tuple(sorted(coord for coord in boundary if coord not in enemy_set))
        else:
            if len(boundary) < int(target):
                continue
            liberties = tuple(sorted(boundary))
            shared_targets = tuple(sorted(boundary[index] for index in rng.sample(range(len(boundary)), int(target))))
            enemy_boundary = tuple()

        liberty_set = set(liberties)
        enemy_boundary_set = set(enemy_boundary)
        rows = [[int(EMPTY) for _ in range(int(board_size))] for _ in range(int(board_size))]
        for row, col in group:
            rows[int(row)][int(col)] = int(marked_group_color)
        for row, col in boundary:
            if (int(row), int(col)) in enemy_boundary_set:
                rows[int(row)][int(col)] = int(opponent_color)
        if variant == GO_RULE_SHARED_LIBERTIES:
            used_marker_cells: Set[Coord] = set()
            for liberty in shared_targets:
                marker_candidates = [
                    coord
                    for coord in neighbors(liberty, board_size=int(board_size))
                    if coord not in set(group)
                    and coord not in liberty_set
                    and coord not in used_marker_cells
                ]
                if not marker_candidates:
                    break
                marker = marker_candidates[int(rng.randrange(len(marker_candidates)))]
                rows[int(marker[0])][int(marker[1])] = int(opponent_color)
                used_marker_cells.add((int(marker[0]), int(marker[1])))
            if len(used_marker_cells) != int(target):
                continue
        board = _board_from_rows(rows)
        if not all_groups_have_liberty(board):
            continue

        empty_coords = [
            (int(row), int(col))
            for row in range(int(board_size))
            for col in range(int(board_size))
            if int(board[row][col]) == int(EMPTY) and (int(row), int(col)) not in liberty_set
        ]
        rng.shuffle(empty_coords)
        extras_target = int(rng.randint(int(extras_min), int(extras_max)))
        mutable_rows = [list(int(cell) for cell in row) for row in board]
        extras_added = 0
        for row, col in empty_coords:
            if extras_added >= int(extras_target):
                break
            stone_color = int(marked_group_color if float(rng.random()) < 0.45 else opponent_color)
            mutable_rows[int(row)][int(col)] = int(stone_color)
            candidate_board = _board_from_rows(mutable_rows)
            if not all_groups_have_liberty(candidate_board):
                mutable_rows[int(row)][int(col)] = int(EMPTY)
                continue
            candidate_group = connected_group(candidate_board, group[0])
            if set(candidate_group) != set(group):
                mutable_rows[int(row)][int(col)] = int(EMPTY)
                continue
            if variant == GO_RULE_SHARED_LIBERTIES:
                if tuple(sorted(group_liberties(candidate_board, candidate_group))) != tuple(sorted(liberties)):
                    mutable_rows[int(row)][int(col)] = int(EMPTY)
                    continue
                if len(shared_liberty_coords(candidate_board, candidate_group)) != int(target):
                    mutable_rows[int(row)][int(col)] = int(EMPTY)
                    continue
            extras_added += 1
        board = _board_from_rows(mutable_rows)
        marked_group = connected_group(board, group[0])
        liberties_now = group_liberties(board, marked_group)
        adjacent_enemies_now = adjacent_enemy_coords(board, marked_group)
        shared_liberties_now = shared_liberty_coords(board, marked_group)
        if set(marked_group) != set(group):
            continue
        if variant == GO_RULE_GROUP_LIBERTIES and len(liberties_now) != int(target):
            continue
        if variant == GO_RULE_ADJACENT_ENEMY_STONES and len(adjacent_enemies_now) != int(target):
            continue
        if variant == GO_RULE_SHARED_LIBERTIES and len(shared_liberties_now) != int(target):
            continue
        if variant == GO_RULE_MARKED_GROUP_STONES and len(marked_group) != int(target):
            continue
        if variant in {GO_RULE_GROUP_LIBERTIES, GO_RULE_SHARED_LIBERTIES} and tuple(sorted(liberties_now)) != tuple(sorted(liberties)):
            continue

        return GoBoardState(
            board=board,
            marked_group_color=int(marked_group_color),
            marked_group_coords=tuple(sorted(marked_group)),
            liberty_coords=tuple(sorted(liberties_now)),
            adjacent_enemy_coords=tuple(sorted(adjacent_enemies_now)),
            shared_liberty_coords=tuple(sorted(shared_liberties_now)),
            stone_specs=_stone_specs(board, marked_group_coords=marked_group),
            scene_variant=str(scene_variant),
        )

    raise RuntimeError(
        f"failed to construct a visible Go board with target {target} for {count_mode}/{color}/{scene_variant}"
    )


__all__ = [
    "BLACK",
    "BOARD_SIZE",
    "Board",
    "Coord",
    "EMPTY",
    "GO_RULE_ADJACENT_ENEMY_STONES",
    "GO_RULE_GROUP_LIBERTIES",
    "GO_RULE_MARKED_GROUP_STONES",
    "GO_RULE_SHARED_LIBERTIES",
    "GoBoardState",
    "GoStoneSpec",
    "SUPPORTED_GO_PLAYER_COLORS",
    "WHITE",
    "adjacent_enemy_coords",
    "all_groups_have_liberty",
    "build_go_board_state",
    "color_name",
    "connected_group",
    "coord_to_point_id",
    "coord_to_stone_id",
    "group_liberties",
    "liberty_point_ids",
    "neighbors",
    "opponent",
    "shared_liberty_coords",
    "stone_ids_for_coords",
    "stone_groups",
    "supported_targets_for_mode",
]
