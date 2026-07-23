"""Pool-table geometry and rule helpers."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

from .state import Point, PoolBall, PoolPocket


TABLE_GEOM_ASPECT = 0.58
POOL_BALL_NUMBERS: Tuple[int, ...] = tuple(range(1, 16))
SUPPORTED_POOL_SCENE_VARIANTS: Tuple[str, ...] = ("standard_table",)


def ball_group(number: int) -> str:
    """Return the standard 8-ball group for one object-ball number."""

    value = int(number)
    if value == 0:
        return "cue"
    if value == 8:
        return "eight"
    if 1 <= value <= 7:
        return "solid"
    return "stripe"


def ball_entity_id(number: int) -> str:
    """Return a stable entity id for one ball number."""

    return "cue_ball" if int(number) == 0 else f"ball_{int(number)}"


def group_display_name(group: str) -> str:
    """Return prompt-facing pool group text."""

    return "solids" if str(group) == "solid" else "stripes"


def pocket_by_id(pockets: Sequence[PoolPocket], pocket_id: str) -> PoolPocket:
    """Return one pocket by id."""

    for pocket in pockets:
        if str(pocket.pocket_id) == str(pocket_id):
            return pocket
    raise KeyError(f"unknown pool pocket id: {pocket_id}")


def sorted_ids(values: Iterable[str]) -> Tuple[str, ...]:
    """Return stable id ordering for annotation and trace payloads."""

    return tuple(sorted(str(value) for value in values))


def _geom(point: Point) -> Point:
    """Map normalized table coordinates to aspect-correct geometry space."""

    x, y = point
    return (float(x), float(y) * float(TABLE_GEOM_ASPECT))


def point_distance(a: Point, b: Point) -> float:
    """Return aspect-correct distance between two normalized table points."""

    ax, ay = _geom(a)
    bx, by = _geom(b)
    return float(math.hypot(float(ax - bx), float(ay - by)))


def segment_distance(point: Point, start: Point, end: Point) -> float:
    """Return aspect-correct point-to-segment distance."""

    px, py = _geom(point)
    ax, ay = _geom(start)
    bx, by = _geom(end)
    vx = float(bx - ax)
    vy = float(by - ay)
    denom = float((vx * vx) + (vy * vy))
    if denom <= 1e-12:
        return float(math.hypot(float(px - ax), float(py - ay)))
    t = max(0.0, min(1.0, (((px - ax) * vx) + ((py - ay) * vy)) / denom))
    qx = float(ax + (t * vx))
    qy = float(ay + (t * vy))
    return float(math.hypot(float(px - qx), float(py - qy)))


def balls_on_segment(
    *,
    balls: Sequence[PoolBall],
    start: Point,
    end: Point,
    ignore_ball_ids: Iterable[str],
    clearance: float,
) -> Tuple[PoolBall, ...]:
    """Return balls whose centers block one straight shot lane."""

    ignored = {str(ball_id) for ball_id in ignore_ball_ids}
    blockers = [
        ball
        for ball in balls
        if str(ball.ball_id) not in ignored
        and segment_distance(ball.center, start, end) <= float(clearance)
        and point_distance(ball.center, start) > float(clearance) * 0.65
        and point_distance(ball.center, end) > float(clearance) * 0.65
    ]
    return tuple(sorted(blockers, key=lambda ball: str(ball.ball_id)))


def object_balls(balls: Sequence[PoolBall]) -> Tuple[PoolBall, ...]:
    """Return all non-cue balls."""

    return tuple(ball for ball in balls if not bool(ball.is_cue))


__all__ = [
    "POOL_BALL_NUMBERS",
    "SUPPORTED_POOL_SCENE_VARIANTS",
    "TABLE_GEOM_ASPECT",
    "ball_entity_id",
    "ball_group",
    "balls_on_segment",
    "group_display_name",
    "object_balls",
    "pocket_by_id",
    "point_distance",
    "segment_distance",
    "sorted_ids",
]
