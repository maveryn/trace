"""Passive state records for racing-track scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class RacingTrackCar:
    """One labeled car placed on a racing-track centerline."""

    car_id: str
    label: str
    progress: float
    center_px: Point
    tangent_px: Point
    remaining_distance: float


@dataclass(frozen=True)
class RacingTrackSceneState:
    """Track geometry and visible cars before task-specific answer binding."""

    scene_variant: str
    style_variant: str
    track_width_px: int
    track_height_px: int
    centerline_points_px: Tuple[Point, ...]
    finish_point_px: Point
    finish_tangent_px: Point
    cars: Tuple[RacingTrackCar, ...]
    construction_mode: str


def visible_car_trace(cars: Sequence[RacingTrackCar]) -> Tuple[dict, ...]:
    """Return trace-friendly records for every visible racing car."""

    return tuple(
        {
            "car_id": str(car.car_id),
            "label": str(car.label),
            "progress": round(float(car.progress), 6),
            "remaining_distance": round(float(car.remaining_distance), 6),
            "center_px_local": [
                round(float(car.center_px[0]), 3),
                round(float(car.center_px[1]), 3),
            ],
            "tangent_px": [
                round(float(car.tangent_px[0]), 6),
                round(float(car.tangent_px[1]), 6),
            ],
        }
        for car in cars
    )


__all__ = [
    "Point",
    "RacingTrackCar",
    "RacingTrackSceneState",
    "visible_car_trace",
]
