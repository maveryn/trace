"""Sliding-block movement rules and board-state utilities."""

from __future__ import annotations

from typing import Iterable, Sequence

from .state import BlockMoveSpec, BlockSpec


def cells_for(row: int, col: int, height: int, width: int) -> tuple[tuple[int, int], ...]:
    """Return occupied grid cells for one rectangular block placement."""

    return tuple((int(row + dr), int(col + dc)) for dr in range(int(height)) for dc in range(int(width)))


def block_by_id(blocks: Sequence[BlockSpec]) -> dict[str, BlockSpec]:
    """Map block identifiers to immutable block specs."""

    return {str(block.block_id): block for block in blocks}


def state_signature(blocks: Sequence[BlockSpec]) -> tuple[tuple[str, int, int], ...]:
    """Return the location-only state signature used for option uniqueness."""

    return tuple(sorted((str(block.block_id), int(block.row), int(block.col)) for block in blocks))


def block_orientation(block: BlockSpec) -> str:
    """Return the legal slide axis implied by one rectangular block shape."""

    return "horizontal" if int(block.width) > int(block.height) else "vertical"


def block_ids_by_orientation(blocks: Sequence[BlockSpec], *, orientation: str) -> list[str]:
    """Return non-target block ids whose shape matches the requested orientation."""

    target_orientation = str(orientation)
    if target_orientation not in {"horizontal", "vertical"}:
        raise ValueError(f"unsupported sliding-block orientation: {target_orientation}")
    return [
        str(block.block_id)
        for block in blocks
        if str(block.block_id) != "target" and block_orientation(block) == target_orientation
    ]


def replace_block(blocks: Sequence[BlockSpec], moved: BlockSpec) -> list[BlockSpec]:
    """Return a board state with one block replaced by its moved version."""

    return [moved if str(block.block_id) == str(moved.block_id) else block for block in blocks]


def shift_block(block: BlockSpec, *, direction: str, distance: int) -> BlockSpec:
    """Slide one block in a cardinal direction without checking legality."""

    dr, dc = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[str(direction)]
    return BlockSpec(
        block.block_id,
        block.label,
        int(block.row) + (int(dr) * int(distance)),
        int(block.col) + (int(dc) * int(distance)),
        block.height,
        block.width,
        block.role,
        block.fill_rgb,
    )


def inside_board(block: BlockSpec, *, rows: int, cols: int) -> bool:
    """Return whether a block placement stays inside the board rectangle."""

    return (
        0 <= int(block.row)
        and 0 <= int(block.col)
        and int(block.row) + int(block.height) <= int(rows)
        and int(block.col) + int(block.width) <= int(cols)
    )


def can_apply_move(
    blocks: Sequence[BlockSpec],
    *,
    block_id: str,
    direction: str,
    distance: int,
    rows: int,
    cols: int,
) -> bool:
    """Check whether every intermediate slide step is in-bounds and collision-free."""

    block_map = block_by_id(blocks)
    block = block_map[str(block_id)]
    occupied_without_block = {
        cell
        for other in blocks
        if str(other.block_id) != str(block_id)
        for cell in other.cells
    }
    for step in range(1, int(distance) + 1):
        shifted = shift_block(block, direction=str(direction), distance=int(step))
        if not inside_board(shifted, rows=int(rows), cols=int(cols)):
            return False
        if any(cell in occupied_without_block for cell in shifted.cells):
            return False
    return True


def apply_move(blocks: Sequence[BlockSpec], *, move: BlockMoveSpec) -> list[BlockSpec]:
    """Apply one already-validated slide to a board state."""

    block = block_by_id(blocks)[str(move.block_id)]
    return replace_block(blocks, shift_block(block, direction=str(move.direction), distance=int(move.distance)))


def legal_moves(
    blocks: Sequence[BlockSpec],
    *,
    rows: int,
    cols: int,
    max_distance: int,
    exclude_block_ids: Iterable[str] = (),
) -> list[BlockMoveSpec]:
    """Enumerate legal slides for all non-target blocks."""

    excluded = {str(block_id) for block_id in exclude_block_ids}
    moves: list[BlockMoveSpec] = []
    for block in blocks:
        if str(block.block_id) == "target" or str(block.block_id) in excluded:
            continue
        directions = ("left", "right") if block_orientation(block) == "horizontal" else ("up", "down")
        for direction in directions:
            for distance in range(1, int(max_distance) + 1):
                if can_apply_move(
                    blocks,
                    block_id=str(block.block_id),
                    direction=str(direction),
                    distance=int(distance),
                    rows=int(rows),
                    cols=int(cols),
                ):
                    moves.append(BlockMoveSpec(str(block.block_id), str(direction), int(distance)))
                else:
                    break
    return moves


def movable_block_ids(blocks: Sequence[BlockSpec], *, rows: int, cols: int) -> list[str]:
    """Return non-target block ids that have at least one legal one-cell slide."""

    movable = {str(move.block_id) for move in legal_moves(blocks, rows=int(rows), cols=int(cols), max_distance=1)}
    return [
        str(block.block_id)
        for block in blocks
        if str(block.block_id) != "target" and str(block.block_id) in movable
    ]


__all__ = [
    "apply_move",
    "block_by_id",
    "block_ids_by_orientation",
    "block_orientation",
    "can_apply_move",
    "cells_for",
    "inside_board",
    "legal_moves",
    "movable_block_ids",
    "replace_block",
    "shift_block",
    "state_signature",
]
