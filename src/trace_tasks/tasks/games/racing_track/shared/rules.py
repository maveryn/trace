"""Racing-track geometry and movement-order rules."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from .defaults import SUPPORTED_SCENE_VARIANTS
from .state import Point, RacingTrackSceneState


def car_entity_id(index: int) -> str:
    """Return stable entity id for one visible race car."""

    return f"car_{int(index):02d}"


def remaining_distance_to_finish(progress: float) -> float:
    """Return normalized forward distance to the finish line."""

    value = float(progress) % 1.0
    if value <= 1e-9:
        return 0.0
    return round(1.0 - float(value), 6)


def normalize_vector(vector: Sequence[float]) -> Point:
    """Return one normalized 2D vector."""

    x, y = float(vector[0]), float(vector[1])
    length = math.hypot(x, y)
    if length <= 1e-9:
        return (1.0, 0.0)
    return (round(x / length, 6), round(y / length, 6))


def circular_progress_gap(progress_a: float, progress_b: float) -> float:
    """Return shortest normalized progress gap on a closed loop."""

    raw = abs((float(progress_a) % 1.0) - (float(progress_b) % 1.0))
    return min(float(raw), 1.0 - float(raw))


def progress_is_ahead_of_reference(*, reference_progress: float, object_progress: float) -> bool:
    """Return whether object progress is after reference progress before finish."""

    reference = float(reference_progress) % 1.0
    obj = float(object_progress) % 1.0
    return bool(obj > reference)


def track_point(*, scene_variant: str, progress: float, track_width_px: int, track_height_px: int) -> Point:
    """Return a local centerline point for one normalized track progress."""

    width = float(track_width_px)
    height = float(track_height_px)
    cx = width * 0.5
    cy = height * 0.5
    angle = (-0.5 * math.pi) + (math.tau * (float(progress) % 1.0))
    if str(scene_variant) == "rounded_loop":
        rx = width * (0.34 + (0.028 * math.cos(4.0 * angle)))
        ry = height * (0.29 + (0.020 * math.sin(3.0 * angle)))
        x = cx + (rx * math.cos(angle))
        y = cy + (ry * math.sin(angle))
    elif str(scene_variant) == "kidney_loop":
        rx = width * 0.335
        ry = height * 0.285
        x = cx + (rx * (math.cos(angle) + (0.16 * math.sin(2.0 * angle))))
        y = cy + (ry * (math.sin(angle) + (0.12 * math.cos(2.0 * angle))))
    else:
        x = cx + ((width * 0.36) * math.cos(angle))
        y = cy + ((height * 0.30) * math.sin(angle))
    return (round(float(x), 3), round(float(y), 3))


def track_point_and_tangent(
    *,
    scene_variant: str,
    progress: float,
    track_width_px: int,
    track_height_px: int,
) -> Tuple[Point, Point]:
    """Return local centerline point and tangent for one progress value."""

    point = track_point(
        scene_variant=str(scene_variant),
        progress=float(progress),
        track_width_px=int(track_width_px),
        track_height_px=int(track_height_px),
    )
    delta = 1e-4
    before = track_point(
        scene_variant=str(scene_variant),
        progress=float(progress) - delta,
        track_width_px=int(track_width_px),
        track_height_px=int(track_height_px),
    )
    after = track_point(
        scene_variant=str(scene_variant),
        progress=float(progress) + delta,
        track_width_px=int(track_width_px),
        track_height_px=int(track_height_px),
    )
    tangent = normalize_vector((float(after[0]) - float(before[0]), float(after[1]) - float(before[1])))
    return point, tangent


def centerline_points(*, scene_variant: str, track_width_px: int, track_height_px: int, count: int = 180) -> Tuple[Point, ...]:
    """Return local centerline samples for rendering one loop."""

    return tuple(
        track_point(
            scene_variant=str(scene_variant),
            progress=float(index) / float(count),
            track_width_px=int(track_width_px),
            track_height_px=int(track_height_px),
        )
        for index in range(int(count))
    )


def validate_racing_track_state(state: RacingTrackSceneState) -> None:
    """Validate scene-only racing-track state before objective binding."""

    if str(state.scene_variant) not in SUPPORTED_SCENE_VARIANTS:
        raise ValueError(f"unsupported racing-track scene_variant: {state.scene_variant}")
    if int(state.track_width_px) <= 0 or int(state.track_height_px) <= 0:
        raise ValueError("racing-track dimensions must be positive")
    if len(state.centerline_points_px) < 48:
        raise ValueError("racing-track centerline needs enough draw samples")
    if not state.cars:
        raise ValueError("racing-track state must include cars")
    labels = [str(car.label) for car in state.cars]
    if len(labels) != len(set(labels)):
        raise ValueError("racing-track car labels must be unique")
    ids = [str(car.car_id) for car in state.cars]
    if len(ids) != len(set(ids)):
        raise ValueError("racing-track car ids must be unique")
    for car in state.cars:
        if not (0.0 < float(car.progress) < 1.0):
            raise ValueError(f"car progress must be in (0, 1): {car.car_id}")
        if abs(float(car.remaining_distance) - remaining_distance_to_finish(float(car.progress))) > 1e-5:
            raise ValueError(f"car remaining distance mismatch: {car.car_id}")


__all__ = [
    "car_entity_id",
    "centerline_points",
    "circular_progress_gap",
    "normalize_vector",
    "progress_is_ahead_of_reference",
    "remaining_distance_to_finish",
    "track_point",
    "track_point_and_tangent",
    "validate_racing_track_state",
]
