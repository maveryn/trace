"""Passive state records for pool-table scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class PoolPocket:
    """One visible pool-table pocket."""

    pocket_id: str
    display_name: str
    center: Point


@dataclass(frozen=True)
class PoolBall:
    """One visible ball on the pool table."""

    ball_id: str
    number: int
    group: str
    center: Point
    is_cue: bool = False
    is_marked: bool = False


@dataclass(frozen=True)
class PoolSceneState:
    """Generated pool-table state before task-specific answer binding."""

    scene_variant: str
    balls: Tuple[PoolBall, ...]
    pockets: Tuple[PoolPocket, ...]
    cue_ball_id: str
    marked_ball_id: str | None
    marked_pocket_id: str | None
    current_player_group: str | None
    construction_mode: str


POOL_POCKETS: Tuple[PoolPocket, ...] = (
    PoolPocket("pocket_top_left", "top-left pocket", (0.035, 0.045)),
    PoolPocket("pocket_top_middle", "top-middle pocket", (0.500, 0.026)),
    PoolPocket("pocket_top_right", "top-right pocket", (0.965, 0.045)),
    PoolPocket("pocket_bottom_left", "bottom-left pocket", (0.035, 0.955)),
    PoolPocket("pocket_bottom_middle", "bottom-middle pocket", (0.500, 0.974)),
    PoolPocket("pocket_bottom_right", "bottom-right pocket", (0.965, 0.955)),
)


def validate_pool_scene_state(state: PoolSceneState) -> None:
    """Validate generic pool-table references without task-specific semantics."""

    ball_ids = [str(ball.ball_id) for ball in state.balls]
    if len(ball_ids) != len(set(ball_ids)):
        raise ValueError("pool ball ids must be unique")
    pocket_ids = [str(pocket.pocket_id) for pocket in state.pockets]
    if len(pocket_ids) != len(set(pocket_ids)):
        raise ValueError("pool pocket ids must be unique")
    if str(state.cue_ball_id) not in set(ball_ids):
        raise ValueError("pool state missing cue ball")
    if state.marked_ball_id is not None and str(state.marked_ball_id) not in set(ball_ids):
        raise ValueError("marked pool ball id is not present")
    if state.marked_pocket_id is not None and str(state.marked_pocket_id) not in set(pocket_ids):
        raise ValueError("marked pool pocket id is not present")
    if state.current_player_group is not None and str(state.current_player_group) not in {"solid", "stripe"}:
        raise ValueError("pool current player group must be solid or stripe")


__all__ = [
    "POOL_POCKETS",
    "Point",
    "PoolBall",
    "PoolPocket",
    "PoolSceneState",
    "validate_pool_scene_state",
]
