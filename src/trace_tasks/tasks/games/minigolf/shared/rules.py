"""Mini-golf geometric motion rules used by scene samplers and tests."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from .defaults import HOLE_RADIUS_NORM
from .state import MinigolfObstacle


def normalize_angle(angle_rad: float) -> float:
    """Normalize an angle to [-pi, pi]."""

    value = float(angle_rad)
    while value <= -math.pi:
        value += 2.0 * math.pi
    while value > math.pi:
        value -= 2.0 * math.pi
    return float(value)


def unit_from_angle(angle_rad: float) -> Tuple[float, float]:
    """Return a unit vector for an angle in course-normalized coordinates."""

    return (math.cos(float(angle_rad)), math.sin(float(angle_rad)))


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Return Euclidean distance in normalized course coordinates."""

    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def ray_circle_intersection(
    *,
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    center: Tuple[float, float],
    radius: float,
) -> float | None:
    """Return first positive ray/circle intersection distance."""

    ox, oy = float(origin[0]), float(origin[1])
    dx, dy = float(direction[0]), float(direction[1])
    cx, cy = float(center[0]), float(center[1])
    fx = ox - cx
    fy = oy - cy
    a = (dx * dx) + (dy * dy)
    b = 2.0 * ((fx * dx) + (fy * dy))
    c = (fx * fx) + (fy * fy) - (float(radius) * float(radius))
    disc = (b * b) - (4.0 * a * c)
    if disc < 0.0 or a <= 1e-9:
        return None
    root = math.sqrt(disc)
    t1 = (-b - root) / (2.0 * a)
    t2 = (-b + root) / (2.0 * a)
    candidates = [float(t) for t in (t1, t2) if float(t) >= 1e-5]
    return None if not candidates else min(candidates)


def first_hit_obstacle_id(
    *,
    origin: Tuple[float, float],
    angle_rad: float,
    obstacles: Sequence[MinigolfObstacle],
) -> str | None:
    """Return the first obstacle intersected by the shot ray."""

    direction = unit_from_angle(float(angle_rad))
    best: tuple[float, str] | None = None
    for obstacle in obstacles:
        t = ray_circle_intersection(
            origin=origin,
            direction=direction,
            center=(float(obstacle.x_norm), float(obstacle.y_norm)),
            radius=float(obstacle.radius_norm),
        )
        if t is None:
            continue
        if best is None or float(t) < best[0]:
            best = (float(t), str(obstacle.obstacle_id))
    return None if best is None else str(best[1])


def trace_shot_path(
    *,
    ball_xy: Tuple[float, float],
    angle_rad: float,
    hole_xy: Tuple[float, float],
    obstacles: Sequence[MinigolfObstacle],
    max_bounces: int = 2,
) -> Tuple[bool, str | None, Tuple[Tuple[float, float], ...]]:
    """Trace one shot through wall bounces until it reaches the hole or fails."""

    x, y = float(ball_xy[0]), float(ball_xy[1])
    dx, dy = unit_from_angle(float(angle_rad))
    points: list[Tuple[float, float]] = [(float(x), float(y))]
    for _segment_index in range(int(max_bounces) + 1):
        wall_candidates: list[Tuple[float, str]] = []
        if dx > 1e-6:
            wall_candidates.append(((1.0 - x) / dx, "right"))
        elif dx < -1e-6:
            wall_candidates.append(((0.0 - x) / dx, "left"))
        if dy > 1e-6:
            wall_candidates.append(((1.0 - y) / dy, "bottom"))
        elif dy < -1e-6:
            wall_candidates.append(((0.0 - y) / dy, "top"))
        wall_candidates = [(float(t), side) for t, side in wall_candidates if float(t) > 1e-5]
        if not wall_candidates:
            break
        t_wall, wall_side = min(wall_candidates, key=lambda item: item[0])

        t_hole = ray_circle_intersection(
            origin=(x, y),
            direction=(dx, dy),
            center=(float(hole_xy[0]), float(hole_xy[1])),
            radius=HOLE_RADIUS_NORM,
        )
        obstacle_hits: list[Tuple[float, str]] = []
        for obstacle in obstacles:
            t_obstacle = ray_circle_intersection(
                origin=(x, y),
                direction=(dx, dy),
                center=(float(obstacle.x_norm), float(obstacle.y_norm)),
                radius=float(obstacle.radius_norm) + 0.005,
            )
            if t_obstacle is not None:
                obstacle_hits.append((float(t_obstacle), str(obstacle.obstacle_id)))
        first_obstacle = min(obstacle_hits, key=lambda item: item[0]) if obstacle_hits else None

        event_t = float(t_wall)
        event_kind = "wall"
        event_id: str | None = str(wall_side)
        if t_hole is not None and float(t_hole) < event_t:
            event_t = float(t_hole)
            event_kind = "hole"
            event_id = "hole"
        if first_obstacle is not None and float(first_obstacle[0]) < event_t:
            event_t = float(first_obstacle[0])
            event_kind = "obstacle"
            event_id = str(first_obstacle[1])

        x = float(x + (dx * event_t))
        y = float(y + (dy * event_t))
        points.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
        if event_kind == "hole":
            return True, None, tuple(points)
        if event_kind == "obstacle":
            return False, str(event_id), tuple(points)
        if str(wall_side) in {"left", "right"}:
            dx = -dx
        else:
            dy = -dy
        x = max(0.001, min(0.999, x))
        y = max(0.001, min(0.999, y))
    return False, None, tuple(points)


def target_angle_for_mode(*, ball: Tuple[float, float], hole: Tuple[float, float], mode: str) -> float:
    """Return the angle that reaches the hole directly or by one mirror-bank."""

    if str(mode) == "bank_left":
        aim = (-float(hole[0]), float(hole[1]))
    elif str(mode) == "bank_right":
        aim = (2.0 - float(hole[0]), float(hole[1]))
    elif str(mode) == "bank_top":
        aim = (float(hole[0]), -float(hole[1]))
    else:
        aim = (float(hole[0]), float(hole[1]))
    return math.atan2(float(aim[1]) - float(ball[1]), float(aim[0]) - float(ball[0]))


__all__ = [
    "distance",
    "first_hit_obstacle_id",
    "normalize_angle",
    "ray_circle_intersection",
    "target_angle_for_mode",
    "trace_shot_path",
    "unit_from_angle",
]
